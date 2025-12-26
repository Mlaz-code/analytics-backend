# Pikkit Data Pipeline Enhancement Design

## Executive Summary

This document outlines a comprehensive data pipeline enhancement for the Pikkit Sports Betting ML system. The design focuses on four key areas:

1. **Data Quality Framework** - Schema validation, anomaly detection, and quality gates
2. **Feature Store** - Offline/online feature management with versioning
3. **Incremental Data Loading** - CDC strategy for efficient data refresh
4. **Storage Architecture** - Raw/Processed/Feature layer design

---

## 1. Data Quality Framework

### 1.1 Schema Validation

The current system lacks formal schema validation for incoming bet data. We propose implementing Pydantic-based validation.

```python
# /root/pikkit/ml/data_pipeline/schemas.py

from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class BetStatus(str, Enum):
    PENDING = "PENDING"
    SETTLED_WIN = "SETTLED_WIN"
    SETTLED_LOSS = "SETTLED_LOSS"
    SETTLED_PUSH = "SETTLED_PUSH"
    SETTLED_VOID = "SETTLED_VOID"
    SETTLED_CASHEDOUT = "SETTLED_CASHEDOUT"

class BetType(str, Enum):
    STRAIGHT = "straight"
    PARLAY = "parlay"
    ROUND_ROBIN_3 = "round_robin_3"
    ROUND_ROBIN_4 = "round_robin_4"
    ROUND_ROBIN_5 = "round_robin_5"

class BetSchema(BaseModel):
    """Schema for validating incoming bet data"""
    id: str = Field(..., min_length=1, max_length=100)
    bet_type: BetType
    status: BetStatus

    # Financial fields
    odds: Optional[float] = Field(None, ge=-10000, le=10000)
    american_odds: Optional[int] = Field(None, ge=-10000, le=10000)
    amount: Optional[float] = Field(None, ge=0, le=1000000)
    profit: Optional[float] = Field(None, ge=-1000000, le=1000000)
    roi: Optional[float] = Field(None, ge=-1000, le=10000)

    # CLV fields
    clv_percentage: Optional[float] = Field(None, ge=-100, le=100)
    clv_ev: Optional[float] = Field(None, ge=-100, le=100)
    clv_current_odds: Optional[float] = None

    # Categorization
    sport: Optional[str] = Field(None, max_length=100)
    league: Optional[str] = Field(None, max_length=100)
    market: Optional[str] = Field(None, max_length=200)
    institution_name: Optional[str] = Field(None, max_length=100)

    # Flags
    is_live: bool = False
    is_settled: Optional[bool] = None
    is_win: Optional[bool] = None

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    time_placed: Optional[datetime] = None

    @validator('roi', always=True)
    def validate_roi_consistency(cls, v, values):
        """Validate ROI is consistent with amount and profit"""
        amount = values.get('amount')
        profit = values.get('profit')

        if amount and profit and amount > 0:
            expected_roi = (profit / amount) * 100
            if v is not None and abs(v - expected_roi) > 1.0:  # 1% tolerance
                raise ValueError(f"ROI mismatch: expected {expected_roi:.2f}, got {v}")
        return v

    @root_validator
    def validate_settled_state(cls, values):
        """Validate consistency between status and is_settled/is_win"""
        status = values.get('status')
        is_settled = values.get('is_settled')
        is_win = values.get('is_win')

        if status:
            status_str = str(status.value) if hasattr(status, 'value') else str(status)
            expected_settled = status_str.startswith('SETTLED_')
            expected_win = status_str == 'SETTLED_WIN'

            if is_settled is not None and is_settled != expected_settled:
                raise ValueError(f"is_settled={is_settled} inconsistent with status={status}")
            if is_win is not None and expected_settled and is_win != expected_win:
                raise ValueError(f"is_win={is_win} inconsistent with status={status}")

        return values

class ValidationResult(BaseModel):
    """Result of schema validation"""
    valid: bool
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    record_count: int = 0
    valid_count: int = 0
    invalid_count: int = 0
```

### 1.2 Anomaly Detection

Implement statistical anomaly detection for odds, CLV values, and financial metrics.

```python
# /root/pikkit/ml/data_pipeline/anomaly_detection.py

import numpy as np
import pandas as pd
from scipy import stats
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
        self._compute_baselines()

    def _compute_baselines(self):
        """Compute baseline statistics from historical data"""
        if self.historical_df is None or len(self.historical_df) == 0:
            self.baselines = {}
            return

        self.baselines = {
            'odds': {
                'mean': self.historical_df['american_odds'].mean(),
                'std': self.historical_df['american_odds'].std(),
                'q1': self.historical_df['american_odds'].quantile(0.25),
                'q3': self.historical_df['american_odds'].quantile(0.75),
            },
            'clv': {
                'mean': self.historical_df['clv_percentage'].dropna().mean(),
                'std': self.historical_df['clv_percentage'].dropna().std(),
            },
            'amount': {
                'mean': self.historical_df['amount'].mean(),
                'std': self.historical_df['amount'].std(),
                'p99': self.historical_df['amount'].quantile(0.99),
            },
        }

    def detect_odds_anomalies(self, df: pd.DataFrame) -> List[Anomaly]:
        """Detect outliers in odds values"""
        anomalies = []

        if 'american_odds' not in df.columns:
            return anomalies

        # Z-score method
        if 'odds' in self.baselines:
            mean = self.baselines['odds']['mean']
            std = self.baselines['odds']['std']

            if std > 0:
                z_scores = (df['american_odds'] - mean) / std
                outliers = df[abs(z_scores) > self.thresholds['odds_z_score']]

                for idx, row in outliers.iterrows():
                    z = z_scores.loc[idx]
                    anomalies.append(Anomaly(
                        record_id=row['id'],
                        anomaly_type=AnomalyType.ODDS_OUTLIER,
                        severity='medium' if abs(z) < 5 else 'high',
                        description=f"Odds {row['american_odds']} is {abs(z):.1f} std from mean",
                        value=row['american_odds'],
                        expected_range=(mean - 3*std, mean + 3*std)
                    ))

        # IQR method for extreme values
        q1, q3 = df['american_odds'].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - self.thresholds['iqr_multiplier'] * iqr
        upper = q3 + self.thresholds['iqr_multiplier'] * iqr

        extreme = df[(df['american_odds'] < lower) | (df['american_odds'] > upper)]
        for idx, row in extreme.iterrows():
            if row['id'] not in [a.record_id for a in anomalies]:
                anomalies.append(Anomaly(
                    record_id=row['id'],
                    anomaly_type=AnomalyType.ODDS_OUTLIER,
                    severity='high',
                    description=f"Odds {row['american_odds']} outside IQR bounds",
                    value=row['american_odds'],
                    expected_range=(lower, upper)
                ))

        return anomalies

    def detect_clv_anomalies(self, df: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies in CLV values"""
        anomalies = []

        if 'clv_percentage' not in df.columns:
            return anomalies

        clv_data = df[df['clv_percentage'].notna()]

        # Extreme CLV values (> 20% is suspicious)
        extreme_clv = clv_data[abs(clv_data['clv_percentage']) > 20]
        for idx, row in extreme_clv.iterrows():
            anomalies.append(Anomaly(
                record_id=row['id'],
                anomaly_type=AnomalyType.CLV_OUTLIER,
                severity='high' if abs(row['clv_percentage']) > 30 else 'medium',
                description=f"CLV {row['clv_percentage']:.1f}% is unusually high",
                value=row['clv_percentage'],
                expected_range=(-20, 20)
            ))

        # Statistical outliers using historical baseline
        if 'clv' in self.baselines and self.baselines['clv']['std'] > 0:
            mean = self.baselines['clv']['mean']
            std = self.baselines['clv']['std']
            z_scores = (clv_data['clv_percentage'] - mean) / std
            outliers = clv_data[abs(z_scores) > self.thresholds['clv_z_score']]

            for idx, row in outliers.iterrows():
                if row['id'] not in [a.record_id for a in anomalies]:
                    anomalies.append(Anomaly(
                        record_id=row['id'],
                        anomaly_type=AnomalyType.CLV_OUTLIER,
                        severity='medium',
                        description=f"CLV {row['clv_percentage']:.1f}% is statistical outlier",
                        value=row['clv_percentage']
                    ))

        return anomalies

    def detect_amount_anomalies(self, df: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies in bet amounts"""
        anomalies = []

        if 'amount' not in df.columns:
            return anomalies

        # Extreme amounts (> $10,000 single bet is suspicious)
        extreme = df[df['amount'] > 10000]
        for idx, row in extreme.iterrows():
            anomalies.append(Anomaly(
                record_id=row['id'],
                anomaly_type=AnomalyType.AMOUNT_OUTLIER,
                severity='high',
                description=f"Bet amount ${row['amount']:.2f} exceeds $10,000 threshold",
                value=row['amount'],
                expected_range=(0, 10000)
            ))

        # Zero or negative amounts
        invalid = df[(df['amount'] <= 0) & df['amount'].notna()]
        for idx, row in invalid.iterrows():
            anomalies.append(Anomaly(
                record_id=row['id'],
                anomaly_type=AnomalyType.AMOUNT_OUTLIER,
                severity='critical',
                description=f"Invalid bet amount: ${row['amount']:.2f}",
                value=row['amount'],
                expected_range=(0.01, float('inf'))
            ))

        return anomalies

    def detect_temporal_anomalies(self, df: pd.DataFrame) -> List[Anomaly]:
        """Detect time-based anomalies"""
        anomalies = []

        if 'created_at' not in df.columns:
            return anomalies

        # Future timestamps
        now = pd.Timestamp.now(tz='UTC')
        future_bets = df[pd.to_datetime(df['created_at']) > now]
        for idx, row in future_bets.iterrows():
            anomalies.append(Anomaly(
                record_id=row['id'],
                anomaly_type=AnomalyType.TEMPORAL_ANOMALY,
                severity='critical',
                description=f"Bet has future timestamp: {row['created_at']}"
            ))

        # Very old timestamps (> 5 years)
        old_threshold = now - pd.Timedelta(days=365*5)
        old_bets = df[pd.to_datetime(df['created_at']) < old_threshold]
        for idx, row in old_bets.iterrows():
            anomalies.append(Anomaly(
                record_id=row['id'],
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
                record_id=bet_id,
                anomaly_type=AnomalyType.DUPLICATE,
                severity='critical',
                description=f"Duplicate bet ID found ({count} occurrences)"
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
```

### 1.3 Data Lineage Tracking

Track data transformations and provenance for auditability.

```python
# /root/pikkit/ml/data_pipeline/lineage.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import hashlib

@dataclass
class TransformationStep:
    """Represents a single data transformation"""
    step_id: str
    operation: str  # 'filter', 'transform', 'aggregate', 'join', 'validate'
    description: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    input_count: int = 0
    output_count: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataLineage:
    """Track data lineage through the pipeline"""
    run_id: str
    source: str  # 'pikkit_api', 'supabase', 'manual'
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    transformations: List[TransformationStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_transformation(self, operation: str, description: str,
                          input_count: int = 0, output_count: int = 0,
                          parameters: Dict = None):
        """Add a transformation step to lineage"""
        step = TransformationStep(
            step_id=f"{self.run_id}_{len(self.transformations)}",
            operation=operation,
            description=description,
            input_count=input_count,
            output_count=output_count,
            parameters=parameters or {}
        )
        self.transformations.append(step)
        return step

    def complete(self):
        """Mark pipeline run as complete"""
        self.completed_at = datetime.utcnow()

    def compute_hash(self, df) -> str:
        """Compute hash of dataframe for verification"""
        return hashlib.md5(
            df.to_json().encode()
        ).hexdigest()

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'run_id': self.run_id,
            'source': self.source,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'transformations': [
                {
                    'step_id': t.step_id,
                    'operation': t.operation,
                    'description': t.description,
                    'timestamp': t.timestamp.isoformat(),
                    'input_count': t.input_count,
                    'output_count': t.output_count,
                    'parameters': t.parameters,
                }
                for t in self.transformations
            ],
            'metadata': self.metadata,
        }

    def save(self, filepath: str):
        """Save lineage to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

class LineageTracker:
    """Manage lineage across pipeline runs"""

    def __init__(self, storage_dir: str = '/root/pikkit/ml/data_pipeline/lineage'):
        self.storage_dir = storage_dir
        import os
        os.makedirs(storage_dir, exist_ok=True)

    def start_run(self, source: str) -> DataLineage:
        """Start a new pipeline run"""
        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        return DataLineage(run_id=run_id, source=source)

    def save_run(self, lineage: DataLineage):
        """Save completed run"""
        filepath = f"{self.storage_dir}/{lineage.run_id}.json"
        lineage.save(filepath)

    def get_recent_runs(self, limit: int = 10) -> List[Dict]:
        """Get recent pipeline runs"""
        import os
        import glob

        files = sorted(
            glob.glob(f"{self.storage_dir}/*.json"),
            key=os.path.getmtime,
            reverse=True
        )[:limit]

        runs = []
        for f in files:
            with open(f) as fp:
                runs.append(json.load(fp))
        return runs
```

### 1.4 Quality Gates

Define quality thresholds that must be met before training.

```python
# /root/pikkit/ml/data_pipeline/quality_gates.py

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
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self.results = {}

    def check_completeness(self, df: pd.DataFrame) -> Dict:
        """Check data completeness metrics"""
        checks = {}

        # Record count
        checks['min_records'] = {
            'value': len(df),
            'threshold': self.thresholds['min_records'].threshold,
            'passed': len(df) >= self.thresholds['min_records'].threshold
        }

        # Settled percentage
        settled_pct = df['is_settled'].sum() / len(df) if len(df) > 0 else 0
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
                null_pct = df[col].isna().sum() / len(df) if len(df) > 0 else 0
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
            max_date = pd.to_datetime(df['created_at']).max()
            age_days = (pd.Timestamp.now(tz='UTC') - max_date).days

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
        if 'is_win' in df.columns:
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
            for name, result in checks.items():
                all_checks.append(result['passed'])

        self.results['summary'] = {
            'total_checks': len(all_checks),
            'passed_checks': sum(all_checks),
            'failed_checks': len(all_checks) - sum(all_checks),
            'overall_passed': all(all_checks),
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
            lines.append(f"\n{category.upper()}")
            lines.append("-" * 40)

            for name, result in checks.items():
                status = "PASS" if result['passed'] else "FAIL"
                lines.append(f"  {name}: {result['value']:.4f} (threshold: {result['threshold']}) [{status}]")

        summary = self.results.get('summary', {})
        lines.append(f"\n{'=' * 60}")
        lines.append(f"OVERALL: {'PASSED' if summary.get('overall_passed') else 'FAILED'}")
        lines.append(f"  Checks: {summary.get('passed_checks', 0)}/{summary.get('total_checks', 0)} passed")
        lines.append("=" * 60)

        return "\n".join(lines)
```

---

## 2. Feature Store Design

### 2.1 Architecture Overview

The feature store separates features into offline (batch) and online (real-time) tiers:

```
Feature Store Architecture
==========================

+----------------+     +------------------+     +----------------+
|   Raw Data     | --> | Feature Pipeline | --> | Feature Store  |
| (Supabase)     |     | (Python/Pandas)  |     | (Supabase)     |
+----------------+     +------------------+     +----------------+
                                                       |
                       +-------------------------------+
                       |                               |
               +-------v-------+               +-------v-------+
               | Offline Store |               | Online Store  |
               | (Historical)  |               | (Real-time)   |
               +---------------+               +---------------+
                       |                               |
               +-------v-------+               +-------v-------+
               |    Training   |               |   Inference   |
               +---------------+               +---------------+
```

### 2.2 Feature Definitions

```python
# /root/pikkit/ml/feature_store/features.py

from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
from enum import Enum
import pandas as pd

class FeatureType(str, Enum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    BOOLEAN = "boolean"
    TEMPORAL = "temporal"
    EMBEDDING = "embedding"

class ComputationMode(str, Enum):
    BATCH = "batch"       # Computed offline during training
    STREAMING = "streaming"  # Computed in real-time
    HYBRID = "hybrid"     # Precomputed base, updated incrementally

@dataclass
class FeatureDefinition:
    """Define a feature with its metadata and computation logic"""
    name: str
    feature_type: FeatureType
    computation_mode: ComputationMode
    description: str
    entity_key: str  # 'bet_id', 'sport_market', 'user', etc.
    dependencies: List[str] = None  # Other features this depends on
    ttl_seconds: Optional[int] = None  # For cached features
    version: str = "1.0.0"

# Feature Registry
FEATURE_REGISTRY = {
    # === BET-LEVEL FEATURES ===
    'implied_prob': FeatureDefinition(
        name='implied_prob',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.STREAMING,
        description='Implied probability from American odds',
        entity_key='bet_id',
    ),
    'clv_percentage': FeatureDefinition(
        name='clv_percentage',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.STREAMING,
        description='Closing Line Value percentage',
        entity_key='bet_id',
    ),
    'clv_ev': FeatureDefinition(
        name='clv_ev',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.STREAMING,
        description='CLV * implied_prob (expected value contribution)',
        entity_key='bet_id',
        dependencies=['clv_percentage', 'implied_prob'],
    ),
    'is_live': FeatureDefinition(
        name='is_live',
        feature_type=FeatureType.BOOLEAN,
        computation_mode=ComputationMode.STREAMING,
        description='Whether bet was placed live',
        entity_key='bet_id',
    ),

    # === AGGREGATED FEATURES (OFFLINE) ===
    'sport_win_rate': FeatureDefinition(
        name='sport_win_rate',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.BATCH,
        description='Historical win rate for this sport',
        entity_key='sport',
        ttl_seconds=86400,  # 24 hours
    ),
    'sport_roi': FeatureDefinition(
        name='sport_roi',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.BATCH,
        description='Historical ROI for this sport',
        entity_key='sport',
        ttl_seconds=86400,
    ),
    'sport_market_win_rate': FeatureDefinition(
        name='sport_market_win_rate',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.BATCH,
        description='Historical win rate for sport+market combination',
        entity_key='sport_market',
        ttl_seconds=86400,
    ),
    'sport_market_roi': FeatureDefinition(
        name='sport_market_roi',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.BATCH,
        description='Historical ROI for sport+market combination',
        entity_key='sport_market',
        ttl_seconds=86400,
    ),
    'institution_win_rate': FeatureDefinition(
        name='institution_win_rate',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.BATCH,
        description='Historical win rate for this sportsbook',
        entity_key='institution_name',
        ttl_seconds=86400,
    ),
    'institution_roi': FeatureDefinition(
        name='institution_roi',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.BATCH,
        description='Historical ROI for this sportsbook',
        entity_key='institution_name',
        ttl_seconds=86400,
    ),

    # === HYBRID FEATURES ===
    'recent_win_rate': FeatureDefinition(
        name='recent_win_rate',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.HYBRID,
        description='Win rate over last 10 bets in same sport+market',
        entity_key='sport_market',
        ttl_seconds=3600,  # 1 hour
    ),
    'sport_market_count': FeatureDefinition(
        name='sport_market_count',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.HYBRID,
        description='Number of historical bets in sport+market',
        entity_key='sport_market',
        ttl_seconds=3600,
    ),

    # === TEMPORAL FEATURES ===
    'day_of_week': FeatureDefinition(
        name='day_of_week',
        feature_type=FeatureType.CATEGORICAL,
        computation_mode=ComputationMode.STREAMING,
        description='Day of week (0=Monday, 6=Sunday)',
        entity_key='bet_id',
    ),
    'hour_of_day': FeatureDefinition(
        name='hour_of_day',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.STREAMING,
        description='Hour of day (0-23)',
        entity_key='bet_id',
    ),

    # === ENCODED FEATURES ===
    'sport_encoded': FeatureDefinition(
        name='sport_encoded',
        feature_type=FeatureType.CATEGORICAL,
        computation_mode=ComputationMode.STREAMING,
        description='Label-encoded sport',
        entity_key='bet_id',
    ),
    'market_encoded': FeatureDefinition(
        name='market_encoded',
        feature_type=FeatureType.CATEGORICAL,
        computation_mode=ComputationMode.STREAMING,
        description='Label-encoded market type',
        entity_key='bet_id',
    ),
    'institution_encoded': FeatureDefinition(
        name='institution_encoded',
        feature_type=FeatureType.CATEGORICAL,
        computation_mode=ComputationMode.STREAMING,
        description='Label-encoded sportsbook',
        entity_key='bet_id',
    ),
}
```

### 2.3 Feature Store Implementation

```python
# /root/pikkit/ml/feature_store/store.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import hashlib
from .features import FEATURE_REGISTRY, FeatureDefinition, ComputationMode

class FeatureStore:
    """
    Feature store for Pikkit ML models
    Manages both offline (batch) and online (real-time) features
    """

    def __init__(self, supabase_client=None, cache_dir: str = '/root/pikkit/ml/feature_store/cache'):
        self.supabase = supabase_client
        self.cache_dir = cache_dir
        self.offline_features: Dict[str, pd.DataFrame] = {}
        self.online_cache: Dict[str, Any] = {}
        self.encoders: Dict[str, Dict] = {}
        self.feature_versions: Dict[str, str] = {}

        import os
        os.makedirs(cache_dir, exist_ok=True)

    def compute_offline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all offline (batch) features from historical data
        Uses expanding window to avoid lookahead bias
        """
        print("Computing offline features...")

        # Sort by date for proper time-series computation
        df = df.sort_values('created_at').reset_index(drop=True)

        # === SPORT-LEVEL FEATURES ===
        for col in ['sport']:
            df[f'{col}_win_rate'] = (
                df.groupby(col)['is_win']
                .transform(lambda x: x.shift(1).expanding().mean())
                .fillna(0.5)
            )
            df[f'{col}_roi'] = (
                df.groupby(col)['roi']
                .transform(lambda x: x.shift(1).expanding().mean())
                .fillna(0)
            )
            df[f'{col}_count'] = df.groupby(col).cumcount()

        # === SPORT + MARKET FEATURES ===
        for group_cols in [['sport', 'market'], ['sport', 'league'], ['sport', 'league', 'market']]:
            group_name = '_'.join(group_cols)

            df[f'{group_name}_win_rate'] = (
                df.groupby(group_cols)['is_win']
                .transform(lambda x: x.shift(1).expanding().mean())
                .fillna(0.5)
            )
            df[f'{group_name}_roi'] = (
                df.groupby(group_cols)['roi']
                .transform(lambda x: x.shift(1).expanding().mean())
                .fillna(0)
            )
            df[f'{group_name}_count'] = df.groupby(group_cols).cumcount()

        # === INSTITUTION FEATURES ===
        df['institution_name_win_rate'] = (
            df.groupby('institution_name')['is_win']
            .transform(lambda x: x.shift(1).expanding().mean())
            .fillna(0.5)
        )
        df['institution_name_roi'] = (
            df.groupby('institution_name')['roi']
            .transform(lambda x: x.shift(1).expanding().mean())
            .fillna(0)
        )
        df['institution_name_count'] = df.groupby('institution_name').cumcount()

        # === RECENT TRENDS (Rolling window) ===
        df['recent_win_rate'] = (
            df.groupby(['sport', 'market'])['is_win']
            .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
            .fillna(0.5)
        )

        print(f"  Computed {len([c for c in df.columns if 'win_rate' in c or 'roi' in c or '_count' in c])} aggregate features")

        return df

    def compute_online_features(self, bet_data: Dict) -> Dict[str, Any]:
        """
        Compute real-time features for a single bet
        Used during inference
        """
        features = {}

        # === DERIVED FEATURES ===
        odds = bet_data.get('american_odds', bet_data.get('odds', -110))
        if odds < 0:
            features['implied_prob'] = abs(odds) / (abs(odds) + 100)
        else:
            features['implied_prob'] = 100 / (odds + 100)

        # CLV features
        clv = bet_data.get('clv_percentage', 0) or 0
        features['clv_percentage'] = clv
        features['clv_ev'] = clv * features['implied_prob']
        features['has_clv'] = int(clv != 0)

        # Boolean features
        features['is_live'] = int(bet_data.get('is_live', False))

        # Temporal features
        now = datetime.now()
        features['day_of_week'] = now.weekday()
        features['hour_of_day'] = now.hour

        # Categorical encoding
        for col in ['sport', 'league', 'market', 'institution_name', 'bet_type']:
            value = bet_data.get(col, '')
            features[f'{col}_encoded'] = self.encode_categorical(col, value)

        return features

    def encode_categorical(self, column: str, value: str) -> int:
        """Encode categorical value to integer"""
        if column not in self.encoders:
            return 0

        encoder = self.encoders[column]
        if value in encoder:
            return encoder[value]
        return 0  # Unknown category

    def get_historical_features(self, entity_key: str, entity_value: str) -> Dict[str, float]:
        """
        Retrieve pre-computed historical features for an entity
        Used during real-time inference
        """
        cache_key = f"{entity_key}:{entity_value}"

        # Check online cache first
        if cache_key in self.online_cache:
            cached = self.online_cache[cache_key]
            if cached['expires_at'] > datetime.now():
                return cached['features']

        # Compute from offline store or return defaults
        defaults = {
            'sport_win_rate': 0.5,
            'sport_roi': 0.0,
            'sport_market_win_rate': 0.5,
            'sport_market_roi': 0.0,
            'institution_name_win_rate': 0.5,
            'institution_name_roi': 0.0,
            'sport_market_count': 100,
            'institution_name_count': 100,
            'recent_win_rate': 0.5,
        }

        return defaults

    def save_encoders(self, encoders: Dict[str, Dict]):
        """Save label encoders for inference"""
        self.encoders = encoders
        with open(f"{self.cache_dir}/encoders.json", 'w') as f:
            json.dump(encoders, f, indent=2)

    def load_encoders(self):
        """Load label encoders"""
        try:
            with open(f"{self.cache_dir}/encoders.json", 'r') as f:
                self.encoders = json.load(f)
        except FileNotFoundError:
            self.encoders = {}

    def compute_feature_version(self, df: pd.DataFrame, feature_name: str) -> str:
        """Compute version hash for a feature based on its values"""
        if feature_name in df.columns:
            data = df[feature_name].values.tobytes()
            return hashlib.md5(data).hexdigest()[:8]
        return "unknown"

    def get_feature_set(self, df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        """Extract specified features from dataframe"""
        available = [f for f in feature_names if f in df.columns]
        missing = [f for f in feature_names if f not in df.columns]

        if missing:
            print(f"Warning: Missing features: {missing}")

        return df[available]
```

### 2.4 Feature Store Tables (Supabase)

```sql
-- Feature store tables for Supabase
-- Run via: mcp__supabase__apply_migration

-- Aggregated features by sport
CREATE TABLE IF NOT EXISTS feature_sport_stats (
    sport TEXT PRIMARY KEY,
    win_rate REAL NOT NULL DEFAULT 0.5,
    roi REAL NOT NULL DEFAULT 0,
    total_bets INTEGER NOT NULL DEFAULT 0,
    total_amount NUMERIC(12,2) DEFAULT 0,
    total_profit NUMERIC(12,2) DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Aggregated features by sport + market
CREATE TABLE IF NOT EXISTS feature_sport_market_stats (
    sport TEXT NOT NULL,
    market TEXT NOT NULL,
    win_rate REAL NOT NULL DEFAULT 0.5,
    roi REAL NOT NULL DEFAULT 0,
    total_bets INTEGER NOT NULL DEFAULT 0,
    recent_win_rate REAL DEFAULT 0.5,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (sport, market)
);

-- Aggregated features by institution
CREATE TABLE IF NOT EXISTS feature_institution_stats (
    institution_name TEXT PRIMARY KEY,
    win_rate REAL NOT NULL DEFAULT 0.5,
    roi REAL NOT NULL DEFAULT 0,
    total_bets INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Feature metadata for versioning
CREATE TABLE IF NOT EXISTS feature_metadata (
    feature_name TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    computation_mode TEXT NOT NULL,
    last_computed_at TIMESTAMPTZ DEFAULT NOW(),
    record_count INTEGER DEFAULT 0,
    checksum TEXT
);

-- Create indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_sport_market_stats_lookup
    ON feature_sport_market_stats(sport, market);
```

---

## 3. Incremental Data Loading

### 3.1 CDC Strategy

Implement Change Data Capture using Supabase's `updated_at` timestamps.

```python
# /root/pikkit/ml/data_pipeline/incremental.py

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import json

class IncrementalLoader:
    """
    Incremental data loading using CDC (Change Data Capture)
    Tracks last sync timestamp to only fetch changed records
    """

    def __init__(self, supabase_client, state_file: str = '/root/pikkit/ml/data_pipeline/sync_state.json'):
        self.supabase = supabase_client
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load sync state from file"""
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'last_sync_timestamp': None,
                'last_sync_count': 0,
                'total_synced': 0,
                'partitions': {}
            }

    def _save_state(self):
        """Persist sync state to file"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)

    def get_changes_since(self, last_timestamp: str = None,
                          table: str = 'bets',
                          batch_size: int = 1000) -> pd.DataFrame:
        """
        Fetch records changed since last timestamp
        Uses updated_at for CDC
        """
        if last_timestamp is None:
            last_timestamp = self.state.get('last_sync_timestamp')

        all_records = []
        offset = 0

        while True:
            query = self.supabase.table(table).select('*')

            if last_timestamp:
                query = query.gt('updated_at', last_timestamp)

            query = query.order('updated_at').range(offset, offset + batch_size - 1)

            response = query.execute()

            if not response.data:
                break

            all_records.extend(response.data)

            if len(response.data) < batch_size:
                break

            offset += batch_size

        df = pd.DataFrame(all_records)

        if len(df) > 0:
            # Update state
            max_timestamp = df['updated_at'].max()
            self.state['last_sync_timestamp'] = max_timestamp
            self.state['last_sync_count'] = len(df)
            self.state['total_synced'] = self.state.get('total_synced', 0) + len(df)
            self._save_state()

        return df

    def get_partition(self, sport: str = None,
                      date_start: str = None,
                      date_end: str = None) -> pd.DataFrame:
        """
        Fetch data by partition (sport/date range)
        Efficient for targeted queries
        """
        query = self.supabase.table('bets').select('*')

        if sport:
            query = query.eq('sport', sport)

        if date_start:
            query = query.gte('created_at', date_start)

        if date_end:
            query = query.lte('created_at', date_end)

        response = query.execute()
        return pd.DataFrame(response.data)

    def sync_partition(self, sport: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Sync a specific sport partition
        Returns new/updated records and sync stats
        """
        partition_key = f"sport:{sport}"
        last_sync = self.state.get('partitions', {}).get(partition_key)

        query = self.supabase.table('bets').select('*').eq('sport', sport)

        if last_sync:
            query = query.gt('updated_at', last_sync)

        query = query.order('updated_at')
        response = query.execute()
        df = pd.DataFrame(response.data)

        stats = {
            'sport': sport,
            'records_synced': len(df),
            'last_sync': last_sync,
            'new_sync': df['updated_at'].max() if len(df) > 0 else last_sync
        }

        # Update partition state
        if len(df) > 0:
            if 'partitions' not in self.state:
                self.state['partitions'] = {}
            self.state['partitions'][partition_key] = df['updated_at'].max()
            self._save_state()

        return df, stats

    def full_refresh(self, table: str = 'bets') -> pd.DataFrame:
        """
        Full table refresh (use sparingly)
        Resets sync state
        """
        print(f"Performing full refresh of {table}...")

        all_records = []
        offset = 0
        batch_size = 1000

        while True:
            response = (self.supabase.table(table)
                       .select('*')
                       .range(offset, offset + batch_size - 1)
                       .execute())

            if not response.data:
                break

            all_records.extend(response.data)
            print(f"  Fetched {len(all_records)} records...")

            if len(response.data) < batch_size:
                break

            offset += batch_size

        df = pd.DataFrame(all_records)

        # Reset state
        self.state = {
            'last_sync_timestamp': df['updated_at'].max() if len(df) > 0 else None,
            'last_sync_count': len(df),
            'total_synced': len(df),
            'partitions': {}
        }
        self._save_state()

        print(f"Full refresh complete: {len(df)} records")
        return df


class PartitionManager:
    """
    Manage data partitions by sport and date
    Enables efficient incremental processing
    """

    SPORTS = [
        'American Football', 'Basketball', 'Baseball', 'Ice Hockey',
        'Soccer', 'Tennis', 'MMA', 'Golf', 'Cricket'
    ]

    def __init__(self, data_dir: str = '/root/pikkit/ml/data/partitions'):
        self.data_dir = data_dir
        import os
        os.makedirs(data_dir, exist_ok=True)

    def partition_by_sport(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split dataframe by sport"""
        partitions = {}
        for sport in df['sport'].unique():
            partitions[sport] = df[df['sport'] == sport].copy()
        return partitions

    def partition_by_date(self, df: pd.DataFrame,
                          freq: str = 'M') -> Dict[str, pd.DataFrame]:
        """
        Split dataframe by date period
        freq: 'D' (daily), 'W' (weekly), 'M' (monthly)
        """
        df['partition_date'] = pd.to_datetime(df['created_at']).dt.to_period(freq)

        partitions = {}
        for period in df['partition_date'].unique():
            key = str(period)
            partitions[key] = df[df['partition_date'] == period].copy()

        return partitions

    def save_partition(self, df: pd.DataFrame, partition_key: str):
        """Save partition to parquet file"""
        filepath = f"{self.data_dir}/{partition_key.replace('/', '_')}.parquet"
        df.to_parquet(filepath, index=False)

    def load_partition(self, partition_key: str) -> Optional[pd.DataFrame]:
        """Load partition from parquet file"""
        filepath = f"{self.data_dir}/{partition_key.replace('/', '_')}.parquet"
        try:
            return pd.read_parquet(filepath)
        except FileNotFoundError:
            return None

    def get_partition_stats(self) -> List[Dict]:
        """Get statistics for all partitions"""
        import os
        import glob

        stats = []
        for filepath in glob.glob(f"{self.data_dir}/*.parquet"):
            df = pd.read_parquet(filepath)
            filename = os.path.basename(filepath)
            stats.append({
                'partition': filename.replace('.parquet', ''),
                'records': len(df),
                'size_mb': os.path.getsize(filepath) / (1024 * 1024),
                'date_range': f"{df['created_at'].min()} to {df['created_at'].max()}"
            })

        return stats
```

---

## 4. Storage Architecture

### 4.1 Three-Layer Design

```
Storage Architecture
====================

Layer 1: RAW (Supabase)
-----------------------
  - bets table (27,222 records)
  - Original data from Pikkit API
  - Minimal transformations
  - Full audit trail in raw_json/detail_json

Layer 2: PROCESSED (Supabase + Local Parquet)
---------------------------------------------
  - Cleaned, validated data
  - Standardized column names
  - Computed fields (is_win, is_settled, roi)
  - Partitioned by sport/date

Layer 3: FEATURE (Supabase + Local Cache)
-----------------------------------------
  - Aggregated statistics
  - Pre-computed features
  - Ready for ML training/inference
  - Versioned with checksums
```

### 4.2 Implementation

```python
# /root/pikkit/ml/data_pipeline/storage.py

import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import os
import json

class StorageLayer:
    """Base class for storage layers"""

    def __init__(self, name: str, base_path: str):
        self.name = name
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save(self, df: pd.DataFrame, key: str):
        raise NotImplementedError

    def load(self, key: str) -> pd.DataFrame:
        raise NotImplementedError


class RawLayer(StorageLayer):
    """
    Raw data layer - mirrors Supabase bets table
    Stores snapshots for audit/debugging
    """

    def __init__(self, supabase_client, base_path: str = '/root/pikkit/ml/data/raw'):
        super().__init__('raw', base_path)
        self.supabase = supabase_client

    def fetch_from_source(self, limit: int = None) -> pd.DataFrame:
        """Fetch raw data from Supabase"""
        query = self.supabase.table('bets').select('*')

        if limit:
            query = query.limit(limit)

        response = query.execute()
        return pd.DataFrame(response.data)

    def save_snapshot(self, df: pd.DataFrame, tag: str = None):
        """Save a snapshot of raw data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        key = f"snapshot_{timestamp}" if not tag else f"snapshot_{tag}_{timestamp}"

        filepath = f"{self.base_path}/{key}.parquet"
        df.to_parquet(filepath, index=False)

        # Save metadata
        meta = {
            'timestamp': timestamp,
            'records': len(df),
            'columns': list(df.columns),
            'sports': df['sport'].value_counts().to_dict() if 'sport' in df.columns else {}
        }
        with open(f"{self.base_path}/{key}_meta.json", 'w') as f:
            json.dump(meta, f, indent=2)

        return filepath


class ProcessedLayer(StorageLayer):
    """
    Processed data layer - cleaned and validated
    Partitioned by sport for efficient access
    """

    def __init__(self, base_path: str = '/root/pikkit/ml/data/processed'):
        super().__init__('processed', base_path)

    def process_raw(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data to processed format"""
        df = raw_df.copy()

        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        # Parse timestamps
        for col in ['created_at', 'updated_at', 'time_placed']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Ensure numeric types
        for col in ['odds', 'american_odds', 'amount', 'profit', 'roi', 'clv_percentage']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Compute derived fields if missing
        if 'is_settled' not in df.columns and 'status' in df.columns:
            df['is_settled'] = df['status'].str.startswith('SETTLED_')

        if 'is_win' not in df.columns and 'status' in df.columns:
            df['is_win'] = df['status'] == 'SETTLED_WIN'

        if 'roi' not in df.columns and 'profit' in df.columns and 'amount' in df.columns:
            df['roi'] = (df['profit'] / df['amount'] * 100).where(df['amount'] > 0, 0)

        return df

    def save(self, df: pd.DataFrame, partition_key: str = 'all'):
        """Save processed data"""
        filepath = f"{self.base_path}/{partition_key}.parquet"
        df.to_parquet(filepath, index=False)
        return filepath

    def load(self, partition_key: str = 'all') -> Optional[pd.DataFrame]:
        """Load processed data"""
        filepath = f"{self.base_path}/{partition_key}.parquet"
        if os.path.exists(filepath):
            return pd.read_parquet(filepath)
        return None

    def save_by_sport(self, df: pd.DataFrame):
        """Partition and save by sport"""
        for sport in df['sport'].unique():
            sport_df = df[df['sport'] == sport]
            key = sport.lower().replace(' ', '_')
            self.save(sport_df, f"sport_{key}")


class FeatureLayer(StorageLayer):
    """
    Feature layer - aggregated stats ready for ML
    Includes both Supabase tables and local cache
    """

    def __init__(self, supabase_client = None,
                 base_path: str = '/root/pikkit/ml/data/features'):
        super().__init__('feature', base_path)
        self.supabase = supabase_client

    def compute_sport_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute sport-level aggregated features"""
        settled = df[df['is_settled'] == True]

        features = settled.groupby('sport').agg({
            'is_win': ['sum', 'count'],
            'amount': 'sum',
            'profit': 'sum',
            'roi': 'mean'
        }).reset_index()

        features.columns = ['sport', 'wins', 'total_bets', 'total_amount',
                           'total_profit', 'avg_roi']
        features['win_rate'] = features['wins'] / features['total_bets']
        features['roi'] = (features['total_profit'] / features['total_amount'] * 100
                          ).where(features['total_amount'] > 0, 0)

        return features

    def compute_sport_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute sport+market aggregated features"""
        settled = df[df['is_settled'] == True]

        features = settled.groupby(['sport', 'market']).agg({
            'is_win': ['sum', 'count'],
            'amount': 'sum',
            'profit': 'sum',
        }).reset_index()

        features.columns = ['sport', 'market', 'wins', 'total_bets',
                           'total_amount', 'total_profit']
        features['win_rate'] = features['wins'] / features['total_bets']
        features['roi'] = (features['total_profit'] / features['total_amount'] * 100
                          ).where(features['total_amount'] > 0, 0)

        # Recent win rate (last 50 bets per sport/market)
        recent = settled.groupby(['sport', 'market']).tail(50)
        recent_wr = recent.groupby(['sport', 'market'])['is_win'].mean().reset_index()
        recent_wr.columns = ['sport', 'market', 'recent_win_rate']

        features = features.merge(recent_wr, on=['sport', 'market'], how='left')
        features['recent_win_rate'] = features['recent_win_rate'].fillna(0.5)

        return features

    def compute_institution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute institution (sportsbook) aggregated features"""
        settled = df[df['is_settled'] == True]

        features = settled.groupby('institution_name').agg({
            'is_win': ['sum', 'count'],
            'amount': 'sum',
            'profit': 'sum',
        }).reset_index()

        features.columns = ['institution_name', 'wins', 'total_bets',
                           'total_amount', 'total_profit']
        features['win_rate'] = features['wins'] / features['total_bets']
        features['roi'] = (features['total_profit'] / features['total_amount'] * 100
                          ).where(features['total_amount'] > 0, 0)

        return features

    def save_to_supabase(self, df: pd.DataFrame, table: str):
        """Upsert feature data to Supabase"""
        if self.supabase is None:
            print(f"Warning: No Supabase client, saving to local only")
            return

        records = df.to_dict('records')

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            self.supabase.table(table).upsert(batch).execute()

    def refresh_all_features(self, processed_df: pd.DataFrame):
        """Refresh all feature tables"""
        print("Refreshing feature layer...")

        # Sport features
        sport_features = self.compute_sport_features(processed_df)
        sport_features.to_parquet(f"{self.base_path}/sport_stats.parquet")

        # Sport+market features
        sport_market_features = self.compute_sport_market_features(processed_df)
        sport_market_features.to_parquet(f"{self.base_path}/sport_market_stats.parquet")

        # Institution features
        institution_features = self.compute_institution_features(processed_df)
        institution_features.to_parquet(f"{self.base_path}/institution_stats.parquet")

        # Save to Supabase if available
        if self.supabase:
            self.save_to_supabase(sport_features, 'feature_sport_stats')
            self.save_to_supabase(sport_market_features, 'feature_sport_market_stats')
            self.save_to_supabase(institution_features, 'feature_institution_stats')

        print(f"  Sport features: {len(sport_features)} rows")
        print(f"  Sport+Market features: {len(sport_market_features)} rows")
        print(f"  Institution features: {len(institution_features)} rows")


class DataPipelineOrchestrator:
    """
    Orchestrate the full data pipeline across all layers
    """

    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.raw = RawLayer(supabase_client)
        self.processed = ProcessedLayer()
        self.features = FeatureLayer(supabase_client)

    def run_full_pipeline(self, save_snapshot: bool = True) -> Dict:
        """Run complete ETL pipeline"""
        from datetime import datetime

        stats = {
            'started_at': datetime.now().isoformat(),
            'layers': {}
        }

        # Layer 1: Raw
        print("=" * 60)
        print("LAYER 1: RAW")
        print("=" * 60)
        raw_df = self.raw.fetch_from_source()
        stats['layers']['raw'] = {'records': len(raw_df)}

        if save_snapshot:
            snapshot_path = self.raw.save_snapshot(raw_df)
            stats['layers']['raw']['snapshot'] = snapshot_path

        # Layer 2: Processed
        print("\n" + "=" * 60)
        print("LAYER 2: PROCESSED")
        print("=" * 60)
        processed_df = self.processed.process_raw(raw_df)
        self.processed.save(processed_df, 'all')
        self.processed.save_by_sport(processed_df)
        stats['layers']['processed'] = {
            'records': len(processed_df),
            'settled': int(processed_df['is_settled'].sum()),
            'sports': int(processed_df['sport'].nunique())
        }

        # Layer 3: Features
        print("\n" + "=" * 60)
        print("LAYER 3: FEATURES")
        print("=" * 60)
        self.features.refresh_all_features(processed_df)
        stats['layers']['features'] = {
            'sport_stats': len(self.features.compute_sport_features(processed_df)),
            'sport_market_stats': len(self.features.compute_sport_market_features(processed_df)),
        }

        stats['completed_at'] = datetime.now().isoformat()

        # Save pipeline stats
        with open('/root/pikkit/ml/data_pipeline/last_run.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Raw records: {stats['layers']['raw']['records']}")
        print(f"Processed records: {stats['layers']['processed']['records']}")
        print(f"Sports: {stats['layers']['processed']['sports']}")

        return stats
```

---

## 5. Integration with Training Pipeline

### 5.1 Updated Training Script

```python
# Key modifications to train_market_profitability_model.py

def main_with_quality_gates():
    """Enhanced training pipeline with data quality checks"""

    from data_pipeline.quality_gates import QualityGate
    from data_pipeline.anomaly_detection import AnomalyDetector
    from data_pipeline.lineage import LineageTracker
    from data_pipeline.storage import DataPipelineOrchestrator

    # Initialize components
    lineage = LineageTracker().start_run(source='supabase')

    # 1. Fetch and validate data
    print("Step 1: Fetching data...")
    orchestrator = DataPipelineOrchestrator(supabase_client)
    stats = orchestrator.run_full_pipeline(save_snapshot=True)
    lineage.add_transformation('fetch', 'Fetched data from Supabase',
                               output_count=stats['layers']['raw']['records'])

    # 2. Run anomaly detection
    print("\nStep 2: Anomaly detection...")
    df = orchestrator.processed.load('all')
    detector = AnomalyDetector(historical_df=df)
    anomalies = detector.run_all_checks(df)
    anomaly_report = detector.generate_report(anomalies)
    lineage.add_transformation('anomaly_check', 'Ran anomaly detection',
                               parameters=anomaly_report)

    # 3. Quality gates
    print("\nStep 3: Quality gates...")
    gate = QualityGate()
    gate_results = gate.run_all_checks(df, anomaly_report)
    print(gate.generate_report())

    if not gate_results['summary']['overall_passed']:
        print("\n*** QUALITY GATE FAILED - Training aborted ***")
        lineage.metadata['quality_gate'] = 'FAILED'
        lineage.complete()
        return None

    lineage.add_transformation('quality_gate', 'Quality gate passed')

    # 4. Feature engineering
    print("\nStep 4: Feature engineering...")
    df, encoders = prepare_features(df)
    lineage.add_transformation('feature_engineering',
                               f'Computed {len(df.columns)} features',
                               input_count=len(df), output_count=len(df))

    # 5. Train models (existing logic)
    print("\nStep 5: Training models...")
    # ... existing training code ...

    # 6. Complete lineage
    lineage.metadata['quality_gate'] = 'PASSED'
    lineage.metadata['model_metrics'] = {
        'win_auc': val_auc,
        'roi_mae': val_mae
    }
    lineage.complete()
    LineageTracker().save_run(lineage)

    return models
```

---

## 6. Implementation Roadmap

### Phase 1: Foundation (Week 1)
- [ ] Create `/root/pikkit/ml/data_pipeline/` directory structure
- [ ] Implement schema validation (`schemas.py`)
- [ ] Implement anomaly detection (`anomaly_detection.py`)
- [ ] Add quality gates (`quality_gates.py`)

### Phase 2: Feature Store (Week 2)
- [ ] Create feature definitions (`features.py`)
- [ ] Implement feature store (`store.py`)
- [ ] Create Supabase feature tables
- [ ] Add feature versioning

### Phase 3: Incremental Loading (Week 3)
- [ ] Implement CDC loader (`incremental.py`)
- [ ] Create partition manager
- [ ] Set up sync state tracking
- [ ] Test incremental refresh

### Phase 4: Storage Architecture (Week 4)
- [ ] Implement storage layers (`storage.py`)
- [ ] Set up raw/processed/feature directories
- [ ] Create orchestrator
- [ ] Integrate with training pipeline

### Phase 5: Integration & Testing (Week 5)
- [ ] Update weekly_retrain.sh
- [ ] Add monitoring/alerting
- [ ] Performance testing
- [ ] Documentation

---

## 7. Directory Structure

```
/root/pikkit/ml/
├── data_pipeline/
│   ├── __init__.py
│   ├── schemas.py           # Pydantic schemas
│   ├── anomaly_detection.py # Anomaly detector
│   ├── quality_gates.py     # Quality thresholds
│   ├── lineage.py           # Data lineage tracking
│   ├── incremental.py       # CDC/incremental loading
│   ├── storage.py           # Storage layer classes
│   ├── sync_state.json      # CDC state
│   └── lineage/             # Lineage records
│       └── run_*.json
├── feature_store/
│   ├── __init__.py
│   ├── features.py          # Feature definitions
│   ├── store.py             # Feature store class
│   └── cache/
│       └── encoders.json
├── data/
│   ├── raw/                  # Raw snapshots
│   │   └── snapshot_*.parquet
│   ├── processed/            # Cleaned data
│   │   ├── all.parquet
│   │   └── sport_*.parquet
│   ├── features/             # Feature aggregates
│   │   ├── sport_stats.parquet
│   │   ├── sport_market_stats.parquet
│   │   └── institution_stats.parquet
│   └── partitions/           # Date partitions
│       └── *.parquet
├── models/                   # Existing model storage
├── predictions/              # Existing predictions
└── scripts/
    ├── train_market_profitability_model.py
    └── weekly_retrain.sh
```

---

## 8. Key Metrics & Monitoring

### Quality Metrics
- Schema validation pass rate (target: >99%)
- Anomaly rate by severity (target: <1% high, 0 critical)
- Data freshness (target: <24 hours)
- Feature completeness (target: >95%)

### Performance Metrics
- Incremental sync time (target: <30s)
- Full pipeline time (target: <10 min)
- Feature computation time (target: <5 min)
- Storage usage (monitor growth)

### Alerting Thresholds
- Quality gate failure: Telegram notification
- >10 critical anomalies: Immediate alert
- Sync failure: Retry + alert after 3 failures
- Data age >48 hours: Warning notification
