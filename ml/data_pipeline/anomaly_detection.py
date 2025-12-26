#!/usr/bin/env python3
"""
Pikkit Anomaly Detection
Statistical anomaly detection for betting data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class AnomalyType(str, Enum):
    ODDS_OUTLIER = "odds_outlier"
    CLV_OUTLIER = "clv_outlier"
    AMOUNT_OUTLIER = "amount_outlier"
    ROI_OUTLIER = "roi_outlier"
    DUPLICATE = "duplicate"
    TEMPORAL_ANOMALY = "temporal_anomaly"
    MISSING_REQUIRED = "missing_required"


@dataclass
class Anomaly:
    record_id: str
    anomaly_type: AnomalyType
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    value: Optional[float] = None
    expected_range: Optional[Tuple[float, float]] = None


class AnomalyDetector:
    """Detect anomalies in betting data"""

    def __init__(self, historical_df: pd.DataFrame = None):
        self.historical_df = historical_df
        self.thresholds = {
            'odds_z_score': 3.5,
            'clv_z_score': 3.0,
            'amount_z_score': 4.0,
            'roi_z_score': 4.0,
            'iqr_multiplier': 2.5,
        }
        self.baselines = {}
        if historical_df is not None and len(historical_df) > 0:
            self._compute_baselines()

    def _compute_baselines(self):
        """Compute baseline statistics from historical data"""
        df = self.historical_df

        if 'american_odds' in df.columns:
            valid_odds = df['american_odds'].dropna()
            if len(valid_odds) > 0:
                self.baselines['odds'] = {
                    'mean': valid_odds.mean(),
                    'std': valid_odds.std(),
                    'q1': valid_odds.quantile(0.25),
                    'q3': valid_odds.quantile(0.75),
                }

        if 'clv_percentage' in df.columns:
            valid_clv = df['clv_percentage'].dropna()
            if len(valid_clv) > 0:
                self.baselines['clv'] = {
                    'mean': valid_clv.mean(),
                    'std': valid_clv.std() if valid_clv.std() > 0 else 1.0,
                }

        if 'amount' in df.columns:
            valid_amount = df['amount'].dropna()
            if len(valid_amount) > 0:
                self.baselines['amount'] = {
                    'mean': valid_amount.mean(),
                    'std': valid_amount.std() if valid_amount.std() > 0 else 1.0,
                    'p99': valid_amount.quantile(0.99),
                }

    def detect_odds_anomalies(self, df: pd.DataFrame) -> List[Anomaly]:
        """Detect outliers in odds values"""
        anomalies = []

        if 'american_odds' not in df.columns:
            return anomalies

        valid_df = df[df['american_odds'].notna()].copy()
        if len(valid_df) == 0:
            return anomalies

        # Z-score method using historical baseline
        if 'odds' in self.baselines:
            mean = self.baselines['odds']['mean']
            std = self.baselines['odds']['std']

            if std > 0:
                valid_df['z_score'] = (valid_df['american_odds'] - mean) / std
                outliers = valid_df[abs(valid_df['z_score']) > self.thresholds['odds_z_score']]

                for idx, row in outliers.iterrows():
                    z = row['z_score']
                    anomalies.append(Anomaly(
                        record_id=str(row['id']),
                        anomaly_type=AnomalyType.ODDS_OUTLIER,
                        severity='medium' if abs(z) < 5 else 'high',
                        description=f"Odds {row['american_odds']} is {abs(z):.1f} std from mean",
                        value=float(row['american_odds']),
                        expected_range=(mean - 3*std, mean + 3*std)
                    ))

        # IQR method for extreme values (use data itself if no baseline)
        q1 = valid_df['american_odds'].quantile(0.25)
        q3 = valid_df['american_odds'].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.thresholds['iqr_multiplier'] * iqr
        upper = q3 + self.thresholds['iqr_multiplier'] * iqr

        extreme = valid_df[(valid_df['american_odds'] < lower) | (valid_df['american_odds'] > upper)]
        existing_ids = {a.record_id for a in anomalies}

        for idx, row in extreme.iterrows():
            if str(row['id']) not in existing_ids:
                anomalies.append(Anomaly(
                    record_id=str(row['id']),
                    anomaly_type=AnomalyType.ODDS_OUTLIER,
                    severity='high',
                    description=f"Odds {row['american_odds']} outside IQR bounds [{lower:.0f}, {upper:.0f}]",
                    value=float(row['american_odds']),
                    expected_range=(lower, upper)
                ))

        return anomalies

    def detect_clv_anomalies(self, df: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies in CLV values"""
        anomalies = []

        if 'clv_percentage' not in df.columns:
            return anomalies

        clv_data = df[df['clv_percentage'].notna()].copy()
        if len(clv_data) == 0:
            return anomalies

        # Extreme CLV values (> 20% is suspicious)
        extreme_clv = clv_data[abs(clv_data['clv_percentage']) > 20]
        for idx, row in extreme_clv.iterrows():
            anomalies.append(Anomaly(
                record_id=str(row['id']),
                anomaly_type=AnomalyType.CLV_OUTLIER,
                severity='high' if abs(row['clv_percentage']) > 30 else 'medium',
                description=f"CLV {row['clv_percentage']:.1f}% is unusually high",
                value=float(row['clv_percentage']),
                expected_range=(-20, 20)
            ))

        # Statistical outliers using historical baseline
        if 'clv' in self.baselines and self.baselines['clv']['std'] > 0:
            mean = self.baselines['clv']['mean']
            std = self.baselines['clv']['std']
            clv_data['z_score'] = (clv_data['clv_percentage'] - mean) / std
            outliers = clv_data[abs(clv_data['z_score']) > self.thresholds['clv_z_score']]

            existing_ids = {a.record_id for a in anomalies}
            for idx, row in outliers.iterrows():
                if str(row['id']) not in existing_ids:
                    anomalies.append(Anomaly(
                        record_id=str(row['id']),
                        anomaly_type=AnomalyType.CLV_OUTLIER,
                        severity='medium',
                        description=f"CLV {row['clv_percentage']:.1f}% is statistical outlier",
                        value=float(row['clv_percentage'])
                    ))

        return anomalies

    def detect_amount_anomalies(self, df: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies in bet amounts"""
        anomalies = []

        if 'amount' not in df.columns:
            return anomalies

        valid_df = df[df['amount'].notna()].copy()

        # Extreme amounts (> $10,000 single bet is suspicious)
        extreme = valid_df[valid_df['amount'] > 10000]
        for idx, row in extreme.iterrows():
            anomalies.append(Anomaly(
                record_id=str(row['id']),
                anomaly_type=AnomalyType.AMOUNT_OUTLIER,
                severity='high',
                description=f"Bet amount ${row['amount']:.2f} exceeds $10,000 threshold",
                value=float(row['amount']),
                expected_range=(0, 10000)
            ))

        # Zero or negative amounts
        invalid = valid_df[valid_df['amount'] <= 0]
        for idx, row in invalid.iterrows():
            anomalies.append(Anomaly(
                record_id=str(row['id']),
                anomaly_type=AnomalyType.AMOUNT_OUTLIER,
                severity='critical',
                description=f"Invalid bet amount: ${row['amount']:.2f}",
                value=float(row['amount']),
                expected_range=(0.01, float('inf'))
            ))

        return anomalies

    def detect_temporal_anomalies(self, df: pd.DataFrame) -> List[Anomaly]:
        """Detect time-based anomalies"""
        anomalies = []

        if 'created_at' not in df.columns:
            return anomalies

        df_copy = df.copy()
        df_copy['created_at'] = pd.to_datetime(df_copy['created_at'], errors='coerce')

        now = pd.Timestamp.now(tz='UTC')

        # Future timestamps
        future_bets = df_copy[df_copy['created_at'] > now]
        for idx, row in future_bets.iterrows():
            anomalies.append(Anomaly(
                record_id=str(row['id']),
                anomaly_type=AnomalyType.TEMPORAL_ANOMALY,
                severity='critical',
                description=f"Bet has future timestamp: {row['created_at']}"
            ))

        # Very old timestamps (> 5 years)
        old_threshold = now - pd.Timedelta(days=365*5)
        old_bets = df_copy[df_copy['created_at'] < old_threshold]
        for idx, row in old_bets.iterrows():
            anomalies.append(Anomaly(
                record_id=str(row['id']),
                anomaly_type=AnomalyType.TEMPORAL_ANOMALY,
                severity='medium',
                description=f"Bet has very old timestamp: {row['created_at']}"
            ))

        return anomalies

    def detect_duplicates(self, df: pd.DataFrame) -> List[Anomaly]:
        """Detect duplicate records"""
        anomalies = []

        # Exact ID duplicates
        duplicates = df[df.duplicated(subset=['id'], keep=False)]
        for bet_id in duplicates['id'].unique():
            count = len(duplicates[duplicates['id'] == bet_id])
            anomalies.append(Anomaly(
                record_id=str(bet_id),
                anomaly_type=AnomalyType.DUPLICATE,
                severity='critical',
                description=f"Duplicate bet ID found ({count} occurrences)"
            ))

        return anomalies

    def detect_missing_required(self, df: pd.DataFrame) -> List[Anomaly]:
        """Detect records missing required fields"""
        anomalies = []
        required_fields = ['id', 'sport', 'american_odds']

        for field in required_fields:
            if field in df.columns:
                missing = df[df[field].isna()]
                for idx, row in missing.iterrows():
                    anomalies.append(Anomaly(
                        record_id=str(row.get('id', f'row_{idx}')),
                        anomaly_type=AnomalyType.MISSING_REQUIRED,
                        severity='high',
                        description=f"Missing required field: {field}"
                    ))

        return anomalies

    def run_all_checks(self, df: pd.DataFrame) -> Dict[str, List[Anomaly]]:
        """Run all anomaly detection checks"""
        return {
            'odds': self.detect_odds_anomalies(df),
            'clv': self.detect_clv_anomalies(df),
            'amount': self.detect_amount_anomalies(df),
            'temporal': self.detect_temporal_anomalies(df),
            'duplicates': self.detect_duplicates(df),
            'missing': self.detect_missing_required(df),
        }

    def generate_report(self, anomalies: Dict[str, List[Anomaly]]) -> Dict:
        """Generate summary report of anomalies"""
        all_anomalies = []
        for check_type, check_anomalies in anomalies.items():
            all_anomalies.extend(check_anomalies)

        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for a in all_anomalies:
            severity_counts[a.severity] += 1

        return {
            'total_anomalies': len(all_anomalies),
            'by_severity': severity_counts,
            'by_type': {
                check_type: len(check_anomalies)
                for check_type, check_anomalies in anomalies.items()
            },
            'critical_records': [a.record_id for a in all_anomalies if a.severity == 'critical'],
            'pass_quality_gate': severity_counts['critical'] == 0,
        }


if __name__ == '__main__':
    # Test anomaly detection
    import pandas as pd

    test_df = pd.DataFrame([
        {'id': 'bet1', 'american_odds': -110, 'clv_percentage': 2.5, 'amount': 100, 'created_at': '2024-01-01'},
        {'id': 'bet2', 'american_odds': 50000, 'clv_percentage': 50, 'amount': 100000, 'created_at': '2024-01-01'},  # Multiple anomalies
        {'id': 'bet3', 'american_odds': -110, 'clv_percentage': None, 'amount': -50, 'created_at': '2030-01-01'},  # Future + negative amount
        {'id': 'bet1', 'american_odds': -110, 'clv_percentage': 2.5, 'amount': 100, 'created_at': '2024-01-01'},  # Duplicate
    ])

    detector = AnomalyDetector()
    anomalies = detector.run_all_checks(test_df)
    report = detector.generate_report(anomalies)

    print(f"Total anomalies: {report['total_anomalies']}")
    print(f"By severity: {report['by_severity']}")
    print(f"By type: {report['by_type']}")
    print(f"Critical records: {report['critical_records']}")
