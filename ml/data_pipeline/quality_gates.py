#!/usr/bin/env python3
"""
Pikkit Quality Gates
Define and check quality thresholds before ML training
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class QualityThreshold:
    name: str
    threshold: float
    operator: str  # 'gte', 'lte', 'eq', 'between'
    upper_bound: Optional[float] = None


class QualityGate:
    """Define and check quality gates for training data"""

    DEFAULT_THRESHOLDS = {
        # Completeness
        'min_records': QualityThreshold('min_records', 1000, 'gte'),
        'min_settled_pct': QualityThreshold('min_settled_pct', 0.50, 'gte'),
        'max_null_sport_pct': QualityThreshold('max_null_sport_pct', 0.05, 'lte'),
        'max_null_market_pct': QualityThreshold('max_null_market_pct', 0.10, 'lte'),
        'max_null_odds_pct': QualityThreshold('max_null_odds_pct', 0.01, 'lte'),

        # Recency
        'max_data_age_days': QualityThreshold('max_data_age_days', 90, 'lte'),

        # Anomaly limits
        'max_critical_anomalies': QualityThreshold('max_critical_anomalies', 0, 'lte'),
        'max_high_anomalies_pct': QualityThreshold('max_high_anomalies_pct', 0.01, 'lte'),

        # Distribution checks
        'min_sports_count': QualityThreshold('min_sports_count', 3, 'gte'),
        'min_books_count': QualityThreshold('min_books_count', 2, 'gte'),
        'win_rate_range': QualityThreshold('win_rate_range', 0.35, 'between', 0.65),
    }

    def __init__(self, thresholds: Dict[str, QualityThreshold] = None):
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
        self.results = {}

    def check_completeness(self, df: pd.DataFrame) -> Dict:
        """Check data completeness metrics"""
        checks = {}
        n = len(df)

        if n == 0:
            return {'error': 'Empty dataframe'}

        # Record count
        checks['min_records'] = {
            'value': n,
            'threshold': self.thresholds['min_records'].threshold,
            'passed': n >= self.thresholds['min_records'].threshold
        }

        # Settled percentage
        if 'is_settled' in df.columns:
            settled_pct = df['is_settled'].sum() / n
            checks['min_settled_pct'] = {
                'value': settled_pct,
                'threshold': self.thresholds['min_settled_pct'].threshold,
                'passed': settled_pct >= self.thresholds['min_settled_pct'].threshold
            }

        # Null checks
        for col, threshold_name in [
            ('sport', 'max_null_sport_pct'),
            ('market', 'max_null_market_pct'),
            ('american_odds', 'max_null_odds_pct'),
        ]:
            if col in df.columns:
                null_pct = df[col].isna().sum() / n
                checks[threshold_name] = {
                    'value': null_pct,
                    'threshold': self.thresholds[threshold_name].threshold,
                    'passed': null_pct <= self.thresholds[threshold_name].threshold
                }

        return checks

    def check_recency(self, df: pd.DataFrame) -> Dict:
        """Check data recency"""
        checks = {}

        if 'created_at' in df.columns and len(df) > 0:
            df_copy = df.copy()
            df_copy['created_at'] = pd.to_datetime(df_copy['created_at'], errors='coerce')
            max_date = df_copy['created_at'].max()

            if pd.notna(max_date):
                now = pd.Timestamp.now(tz='UTC')
                if max_date.tzinfo is None:
                    max_date = max_date.tz_localize('UTC')
                age_days = (now - max_date).days

                checks['max_data_age_days'] = {
                    'value': age_days,
                    'threshold': self.thresholds['max_data_age_days'].threshold,
                    'passed': age_days <= self.thresholds['max_data_age_days'].threshold
                }

        return checks

    def check_distribution(self, df: pd.DataFrame) -> Dict:
        """Check data distribution characteristics"""
        checks = {}

        # Sports diversity
        if 'sport' in df.columns:
            sports_count = df['sport'].nunique()
            checks['min_sports_count'] = {
                'value': sports_count,
                'threshold': self.thresholds['min_sports_count'].threshold,
                'passed': sports_count >= self.thresholds['min_sports_count'].threshold
            }

        # Books diversity
        if 'institution_name' in df.columns:
            books_count = df['institution_name'].nunique()
            checks['min_books_count'] = {
                'value': books_count,
                'threshold': self.thresholds['min_books_count'].threshold,
                'passed': books_count >= self.thresholds['min_books_count'].threshold
            }

        # Win rate sanity check
        if 'is_win' in df.columns and 'is_settled' in df.columns:
            settled = df[df['is_settled'] == True]
            if len(settled) > 0:
                win_rate = settled['is_win'].sum() / len(settled)
                lower = self.thresholds['win_rate_range'].threshold
                upper = self.thresholds['win_rate_range'].upper_bound
                checks['win_rate_range'] = {
                    'value': win_rate,
                    'threshold': f"[{lower}, {upper}]",
                    'passed': lower <= win_rate <= upper
                }

        return checks

    def check_anomalies(self, anomaly_report: Dict) -> Dict:
        """Check anomaly counts against thresholds"""
        checks = {}

        critical_count = anomaly_report.get('by_severity', {}).get('critical', 0)
        checks['max_critical_anomalies'] = {
            'value': critical_count,
            'threshold': self.thresholds['max_critical_anomalies'].threshold,
            'passed': critical_count <= self.thresholds['max_critical_anomalies'].threshold
        }

        total = anomaly_report.get('total_records', 1)
        high_count = anomaly_report.get('by_severity', {}).get('high', 0)
        high_pct = high_count / total if total > 0 else 0
        checks['max_high_anomalies_pct'] = {
            'value': high_pct,
            'threshold': self.thresholds['max_high_anomalies_pct'].threshold,
            'passed': high_pct <= self.thresholds['max_high_anomalies_pct'].threshold
        }

        return checks

    def run_all_checks(self, df: pd.DataFrame, anomaly_report: Dict = None) -> Dict:
        """Run all quality gate checks"""
        self.results = {
            'completeness': self.check_completeness(df),
            'recency': self.check_recency(df),
            'distribution': self.check_distribution(df),
        }

        if anomaly_report:
            anomaly_report['total_records'] = len(df)
            self.results['anomalies'] = self.check_anomalies(anomaly_report)

        # Overall pass/fail
        all_checks = []
        for category, checks in self.results.items():
            if isinstance(checks, dict) and 'error' not in checks:
                for name, result in checks.items():
                    if isinstance(result, dict) and 'passed' in result:
                        all_checks.append(result['passed'])

        self.results['summary'] = {
            'total_checks': len(all_checks),
            'passed_checks': sum(all_checks),
            'failed_checks': len(all_checks) - sum(all_checks),
            'overall_passed': all(all_checks) if all_checks else False,
        }

        return self.results

    def generate_report(self) -> str:
        """Generate human-readable quality report"""
        lines = [
            "=" * 60,
            "DATA QUALITY GATE REPORT",
            "=" * 60,
        ]

        for category, checks in self.results.items():
            if category == 'summary':
                continue
            if not isinstance(checks, dict) or 'error' in checks:
                continue

            lines.append(f"\n{category.upper()}")
            lines.append("-" * 40)

            for name, result in checks.items():
                if isinstance(result, dict) and 'passed' in result:
                    status = "PASS" if result['passed'] else "FAIL"
                    value = result['value']
                    if isinstance(value, float):
                        value_str = f"{value:.4f}"
                    else:
                        value_str = str(value)
                    lines.append(f"  {name}: {value_str} (threshold: {result['threshold']}) [{status}]")

        summary = self.results.get('summary', {})
        lines.append(f"\n{'=' * 60}")
        lines.append(f"OVERALL: {'PASSED' if summary.get('overall_passed') else 'FAILED'}")
        lines.append(f"  Checks: {summary.get('passed_checks', 0)}/{summary.get('total_checks', 0)} passed")
        lines.append("=" * 60)

        return "\n".join(lines)


if __name__ == '__main__':
    # Test quality gates
    import pandas as pd
    import numpy as np

    # Create test data
    np.random.seed(42)
    n = 2000
    test_df = pd.DataFrame({
        'id': [f'bet_{i}' for i in range(n)],
        'sport': np.random.choice(['Basketball', 'Football', 'Baseball', 'Hockey'], n),
        'market': np.random.choice(['Spread', 'Moneyline', 'Total', 'Props'], n),
        'american_odds': np.random.choice([-110, -105, 100, 110, 120], n),
        'institution_name': np.random.choice(['DraftKings', 'FanDuel', 'BetMGM'], n),
        'is_settled': np.random.choice([True, False], n, p=[0.7, 0.3]),
        'is_win': np.random.choice([True, False], n, p=[0.48, 0.52]),
        'created_at': pd.date_range('2024-01-01', periods=n, freq='H'),
    })

    gate = QualityGate()
    results = gate.run_all_checks(test_df)
    print(gate.generate_report())
