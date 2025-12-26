#!/usr/bin/env python3
"""
Pikkit Data and Model Drift Detection
Statistical monitoring for feature distributions and model performance
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of a drift detection test"""
    feature: str
    drift_score: float
    is_drifted: bool
    method: str
    threshold: float
    details: Dict


class DriftDetector:
    """
    Statistical drift detection for ML features.

    Supports multiple methods:
    - PSI (Population Stability Index) for categorical/binned features
    - KS Test (Kolmogorov-Smirnov) for continuous features
    - Chi-squared test for categorical features
    """

    # PSI thresholds (industry standard)
    PSI_THRESHOLDS = {
        'no_drift': 0.1,      # PSI < 0.1: No significant drift
        'moderate': 0.2,      # 0.1 <= PSI < 0.2: Moderate drift
        'significant': 0.25   # PSI >= 0.25: Significant drift, retrain
    }

    # KS test p-value threshold
    KS_PVALUE_THRESHOLD = 0.05

    def __init__(
        self,
        reference_data: pd.DataFrame,
        categorical_features: List[str] = None,
        numerical_features: List[str] = None
    ):
        """
        Initialize drift detector with reference distribution.

        Args:
            reference_data: Historical "good" data to compare against
            categorical_features: List of categorical column names
            numerical_features: List of numerical column names
        """
        self.reference = reference_data
        self.categorical = categorical_features or []
        self.numerical = numerical_features or []

        # Compute reference statistics
        self._reference_stats = {}
        self._compute_reference_stats()

    def _compute_reference_stats(self) -> None:
        """Pre-compute reference distribution statistics"""
        for col in self.categorical:
            if col in self.reference.columns:
                self._reference_stats[col] = {
                    'type': 'categorical',
                    'distribution': self.reference[col].value_counts(normalize=True).to_dict()
                }

        for col in self.numerical:
            if col in self.reference.columns:
                values = self.reference[col].dropna()
                self._reference_stats[col] = {
                    'type': 'numerical',
                    'mean': values.mean(),
                    'std': values.std(),
                    'quantiles': values.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
                }

    def calculate_psi(
        self,
        reference_dist: np.ndarray,
        current_dist: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index.

        PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
        """
        # Ensure both arrays sum to 1
        ref = reference_dist / reference_dist.sum()
        curr = current_dist / current_dist.sum()

        # Avoid division by zero
        ref = np.clip(ref, 1e-10, 1)
        curr = np.clip(curr, 1e-10, 1)

        psi = np.sum((curr - ref) * np.log(curr / ref))
        return psi

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        features: List[str] = None
    ) -> Dict[str, DriftResult]:
        """
        Detect drift across features.

        Args:
            current_data: New data to compare against reference
            features: Specific features to check (default: all)

        Returns:
            Dictionary of feature -> DriftResult
        """
        if features is None:
            features = self.categorical + self.numerical

        results = {}

        for feature in features:
            if feature not in current_data.columns:
                continue

            if feature in self.categorical:
                results[feature] = self._detect_categorical_drift(feature, current_data)
            elif feature in self.numerical:
                results[feature] = self._detect_numerical_drift(feature, current_data)

        return results

    def _detect_categorical_drift(
        self,
        feature: str,
        current_data: pd.DataFrame
    ) -> DriftResult:
        """Detect drift in categorical feature using PSI"""
        ref_dist = self._reference_stats.get(feature, {}).get('distribution', {})
        curr_dist = current_data[feature].value_counts(normalize=True).to_dict()

        # Align categories
        all_categories = set(ref_dist.keys()) | set(curr_dist.keys())
        ref_arr = np.array([ref_dist.get(c, 1e-10) for c in all_categories])
        curr_arr = np.array([curr_dist.get(c, 1e-10) for c in all_categories])

        psi = self.calculate_psi(ref_arr, curr_arr)

        return DriftResult(
            feature=feature,
            drift_score=psi,
            is_drifted=psi >= self.PSI_THRESHOLDS['significant'],
            method='PSI',
            threshold=self.PSI_THRESHOLDS['significant'],
            details={
                'psi': psi,
                'severity': 'high' if psi >= 0.25 else 'moderate' if psi >= 0.1 else 'low',
                'ref_categories': len(ref_dist),
                'curr_categories': len(curr_dist)
            }
        )

    def _detect_numerical_drift(
        self,
        feature: str,
        current_data: pd.DataFrame
    ) -> DriftResult:
        """Detect drift in numerical feature using KS test"""
        ref_values = self.reference[feature].dropna()
        curr_values = current_data[feature].dropna()

        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(ref_values, curr_values)

        is_drifted = p_value < self.KS_PVALUE_THRESHOLD

        return DriftResult(
            feature=feature,
            drift_score=ks_stat,
            is_drifted=is_drifted,
            method='KS Test',
            threshold=self.KS_PVALUE_THRESHOLD,
            details={
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'ref_mean': ref_values.mean(),
                'curr_mean': curr_values.mean(),
                'mean_shift': (curr_values.mean() - ref_values.mean()) / ref_values.std()
                              if ref_values.std() > 0 else 0
            }
        )

    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Handle NaN and Inf values
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, float):
            # Handle Python float NaN/Inf
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        else:
            return obj

    def generate_drift_report(
        self,
        results: Dict[str, DriftResult]
    ) -> Dict:
        """Generate comprehensive drift report"""
        drifted_features = [r.feature for r in results.values() if r.is_drifted]

        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_features_checked': len(results),
            'features_with_drift': len(drifted_features),
            'drifted_features': drifted_features,
            'requires_retraining': bool(len(drifted_features) >= 3 or
                                   any(r.details.get('severity') == 'high'
                                       for r in results.values())),
            'feature_details': {
                name: self._convert_numpy_types({
                    'drift_score': r.drift_score,
                    'is_drifted': r.is_drifted,
                    'method': r.method,
                    **r.details
                })
                for name, r in results.items()
            }
        }

        return report


class ModelDriftDetector:
    """
    Detect model performance degradation over time.
    """

    def __init__(
        self,
        baseline_metrics: Dict[str, float],
        degradation_thresholds: Dict[str, float] = None
    ):
        """
        Args:
            baseline_metrics: Reference model performance metrics
            degradation_thresholds: Acceptable performance drop per metric
        """
        self.baseline = baseline_metrics
        self.thresholds = degradation_thresholds or {
            'accuracy': 0.05,    # 5% drop
            'auc': 0.03,         # 3% AUC drop
            'precision': 0.05,
            'recall': 0.05,
            'calibration_error': 0.02  # 2% increase in cal error
        }

        self.history = []

    def check_performance(
        self,
        current_metrics: Dict[str, float],
        timestamp: datetime = None
    ) -> Dict:
        """
        Check if current performance indicates model drift.
        """
        timestamp = timestamp or datetime.utcnow()

        degraded_metrics = {}
        for metric, current_val in current_metrics.items():
            if metric not in self.baseline:
                continue

            baseline_val = self.baseline[metric]
            threshold = self.thresholds.get(metric, 0.05)

            # For calibration error, higher is worse
            if 'error' in metric.lower():
                degradation = current_val - baseline_val
                is_degraded = degradation > threshold
            else:
                degradation = baseline_val - current_val
                is_degraded = degradation > threshold

            if is_degraded:
                degraded_metrics[metric] = {
                    'baseline': baseline_val,
                    'current': current_val,
                    'degradation': degradation,
                    'threshold': threshold
                }

        result = {
            'timestamp': timestamp.isoformat(),
            'current_metrics': current_metrics,
            'baseline_metrics': self.baseline,
            'degraded_metrics': degraded_metrics,
            'is_degraded': len(degraded_metrics) > 0,
            'requires_retraining': len(degraded_metrics) >= 2
        }

        self.history.append(result)
        return result


if __name__ == '__main__':
    # Test drift detector
    import pandas as pd
    import numpy as np

    # Create reference data
    np.random.seed(42)
    n = 1000
    ref_data = pd.DataFrame({
        'sport': np.random.choice(['Basketball', 'Football', 'Baseball'], n),
        'implied_prob': np.random.beta(2, 2, n),
        'clv_percentage': np.random.normal(0, 5, n),
    })

    # Create drifted data
    curr_data = pd.DataFrame({
        'sport': np.random.choice(['Basketball', 'Football', 'Hockey'], n),  # Hockey is new
        'implied_prob': np.random.beta(3, 2, n),  # Distribution shifted
        'clv_percentage': np.random.normal(2, 6, n),  # Mean and std changed
    })

    # Initialize detector
    detector = DriftDetector(
        reference_data=ref_data,
        categorical_features=['sport'],
        numerical_features=['implied_prob', 'clv_percentage']
    )

    # Detect drift
    results = detector.detect_drift(curr_data)
    report = detector.generate_drift_report(results)

    print("=" * 60)
    print("DRIFT DETECTION REPORT")
    print("=" * 60)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Features checked: {report['total_features_checked']}")
    print(f"Features with drift: {report['features_with_drift']}")
    print(f"Requires retraining: {report['requires_retraining']}")
    print(f"\nDrifted features: {report['drifted_features']}")
    print("\nDetails:")
    for feature, details in report['feature_details'].items():
        if details['is_drifted']:
            print(f"\n  {feature}:")
            print(f"    Method: {details['method']}")
            print(f"    Drift score: {details['drift_score']:.4f}")
            if 'psi' in details:
                print(f"    PSI: {details['psi']:.4f}")
                print(f"    Severity: {details['severity']}")
            if 'p_value' in details:
                print(f"    p-value: {details['p_value']:.4f}")
                print(f"    Mean shift: {details['mean_shift']:.2f} std devs")
