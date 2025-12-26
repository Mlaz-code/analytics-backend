#!/usr/bin/env python3
"""
Pikkit Model Calibration Module
================================
Probability calibration and evaluation metrics for sports betting models.

Features:
1. Platt Scaling (sigmoid calibration)
2. Isotonic Regression calibration
3. Temperature Scaling
4. Calibration curve analysis
5. Betting-specific metrics (Brier score, CLV correlation, ROI-weighted accuracy)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss, log_loss, accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CalibrationResult:
    """Container for calibration results."""
    method: str
    calibrated_probs: np.ndarray
    brier_before: float
    brier_after: float
    calibration_error_before: float
    calibration_error_after: float
    calibrator: object = None


@dataclass
class EvaluationMetrics:
    """Container for model evaluation metrics."""
    # Classification metrics
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float

    # Calibration metrics
    brier_score: float
    log_loss: float
    expected_calibration_error: float
    max_calibration_error: float

    # Betting-specific metrics
    roi_weighted_accuracy: float
    clv_correlation: float
    profitable_bet_rate: float
    kelly_sharpe: float

    # Confidence metrics
    confidence_auc: float
    high_confidence_accuracy: float
    low_confidence_accuracy: float


class ProbabilityCalibrator:
    """
    Multi-method probability calibration for betting models.

    Supports:
    - Platt Scaling (logistic regression)
    - Isotonic Regression
    - Temperature Scaling
    - Beta Calibration
    """

    def __init__(self, method: str = 'isotonic'):
        """
        Initialize calibrator.

        Args:
            method: Calibration method ('platt', 'isotonic', 'temperature', 'beta')
        """
        self.method = method
        self.calibrator = None
        self.fitted = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> 'ProbabilityCalibrator':
        """
        Fit calibrator on validation data.

        Args:
            y_prob: Predicted probabilities
            y_true: True binary labels

        Returns:
            Self
        """
        y_prob = np.asarray(y_prob).flatten()
        y_true = np.asarray(y_true).flatten()

        if self.method == 'platt':
            self.calibrator = self._fit_platt(y_prob, y_true)
        elif self.method == 'isotonic':
            self.calibrator = self._fit_isotonic(y_prob, y_true)
        elif self.method == 'temperature':
            self.calibrator = self._fit_temperature(y_prob, y_true)
        elif self.method == 'beta':
            self.calibrator = self._fit_beta(y_prob, y_true)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.fitted = True
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply calibration to probabilities.

        Args:
            y_prob: Predicted probabilities

        Returns:
            Calibrated probabilities
        """
        if not self.fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        y_prob = np.asarray(y_prob).flatten()

        if self.method == 'platt':
            return self._transform_platt(y_prob)
        elif self.method == 'isotonic':
            return self._transform_isotonic(y_prob)
        elif self.method == 'temperature':
            return self._transform_temperature(y_prob)
        elif self.method == 'beta':
            return self._transform_beta(y_prob)

    def fit_transform(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(y_prob, y_true)
        return self.transform(y_prob)

    def _fit_platt(self, y_prob: np.ndarray, y_true: np.ndarray) -> LogisticRegression:
        """Fit Platt scaling (sigmoid calibration)."""
        # Avoid log(0) issues
        y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)

        # Transform to logit space
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped)).reshape(-1, 1)

        # Fit logistic regression
        lr = LogisticRegression(solver='lbfgs')
        lr.fit(logits, y_true)

        return lr

    def _transform_platt(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply Platt scaling."""
        y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped)).reshape(-1, 1)
        return self.calibrator.predict_proba(logits)[:, 1]

    def _fit_isotonic(self, y_prob: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
        """Fit isotonic regression calibration."""
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        ir.fit(y_prob, y_true)
        return ir

    def _transform_isotonic(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply isotonic regression."""
        return self.calibrator.predict(y_prob)

    def _fit_temperature(self, y_prob: np.ndarray, y_true: np.ndarray) -> float:
        """
        Fit temperature scaling.
        Find optimal temperature T that minimizes cross-entropy loss.
        """
        y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped))

        def nll_loss(T):
            scaled_probs = 1 / (1 + np.exp(-logits / T))
            return log_loss(y_true, scaled_probs)

        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        return result.x

    def _transform_temperature(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply temperature scaling."""
        y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
        T = self.calibrator
        return 1 / (1 + np.exp(-logits / T))

    def _fit_beta(self, y_prob: np.ndarray, y_true: np.ndarray) -> Tuple[float, float, float]:
        """
        Fit beta calibration.
        p_calibrated = 1 / (1 + 1/(exp(c) * (p/(1-p))^a * (1-p)^(b-1)))
        Simplified to 2-parameter version: a and b.
        """
        y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)

        def nll_loss(params):
            a, b = params
            try:
                # Beta calibration formula
                calibrated = 1 / (1 + 1 / (np.power(y_prob_clipped / (1 - y_prob_clipped), a) *
                                           np.power(1 - y_prob_clipped, b - 1)))
                calibrated = np.clip(calibrated, 1e-10, 1 - 1e-10)
                return log_loss(y_true, calibrated)
            except:
                return 1e10

        from scipy.optimize import minimize
        result = minimize(nll_loss, x0=[1.0, 1.0], method='Nelder-Mead')
        return tuple(result.x)

    def _transform_beta(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply beta calibration."""
        y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)
        a, b = self.calibrator
        calibrated = 1 / (1 + 1 / (np.power(y_prob_clipped / (1 - y_prob_clipped), a) *
                                   np.power(1 - y_prob_clipped, b - 1)))
        return np.clip(calibrated, 0, 1)


class BettingMetricsEvaluator:
    """
    Comprehensive evaluation metrics for sports betting models.
    Includes both standard ML metrics and betting-specific metrics.
    """

    def __init__(self, n_bins: int = 10):
        """
        Initialize evaluator.

        Args:
            n_bins: Number of bins for calibration analysis
        """
        self.n_bins = n_bins

    def evaluate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        odds: np.ndarray = None,
        clv: np.ndarray = None,
        stake: np.ndarray = None
    ) -> EvaluationMetrics:
        """
        Compute comprehensive evaluation metrics.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            odds: American odds for each bet
            clv: Closing line value percentages
            stake: Stake amounts

        Returns:
            EvaluationMetrics dataclass
        """
        y_true = np.asarray(y_true).flatten()
        y_prob = np.asarray(y_prob).flatten()
        y_pred = (y_prob > 0.5).astype(int)

        # Classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_true, y_prob)

        # Calibration metrics
        brier = brier_score_loss(y_true, y_prob)
        logloss = log_loss(y_true, y_prob)
        ece, mce = self._calibration_error(y_true, y_prob)

        # Betting-specific metrics
        roi_weighted_acc = self._roi_weighted_accuracy(y_true, y_pred, odds, stake)
        clv_corr = self._clv_correlation(y_true, y_prob, clv)
        profitable_rate = self._profitable_bet_rate(y_true, odds)
        kelly_sharpe = self._kelly_sharpe_ratio(y_true, y_prob, odds)

        # Confidence metrics
        conf_auc = self._confidence_auc(y_true, y_prob)
        high_conf_acc, low_conf_acc = self._confidence_accuracy(y_true, y_prob)

        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc_roc=auc_roc,
            brier_score=brier,
            log_loss=logloss,
            expected_calibration_error=ece,
            max_calibration_error=mce,
            roi_weighted_accuracy=roi_weighted_acc,
            clv_correlation=clv_corr,
            profitable_bet_rate=profitable_rate,
            kelly_sharpe=kelly_sharpe,
            confidence_auc=conf_auc,
            high_confidence_accuracy=high_conf_acc,
            low_confidence_accuracy=low_conf_acc
        )

    def _calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
        """
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=self.n_bins, strategy='uniform')

        # ECE: weighted average of |accuracy - confidence|
        # Need bin counts for weighting
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        bin_counts = np.bincount(bin_indices, minlength=self.n_bins)
        total_samples = len(y_prob)

        ece = 0.0
        mce = 0.0

        for i in range(len(prob_true)):
            gap = abs(prob_true[i] - prob_pred[i])
            if i < len(bin_counts):
                weight = bin_counts[i] / total_samples
                ece += weight * gap
            mce = max(mce, gap)

        return ece, mce

    def _roi_weighted_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        odds: np.ndarray = None,
        stake: np.ndarray = None
    ) -> float:
        """
        Calculate ROI-weighted accuracy.
        Correct predictions on high-odds bets are weighted more.
        """
        if odds is None:
            return accuracy_score(y_true, y_pred)

        odds = np.asarray(odds).flatten()

        # Convert to decimal odds for weighting
        def american_to_decimal(o):
            if o < 0:
                return 1 + (100 / abs(o))
            else:
                return 1 + (o / 100)

        decimal_odds = np.array([american_to_decimal(o) for o in odds])

        # Weight by potential profit
        weights = decimal_odds - 1

        if stake is not None:
            weights = weights * np.asarray(stake).flatten()

        correct = (y_true == y_pred).astype(float)
        weighted_acc = np.sum(correct * weights) / np.sum(weights)

        return weighted_acc

    def _clv_correlation(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        clv: np.ndarray = None
    ) -> float:
        """
        Calculate correlation between predicted probability edge and CLV.
        High correlation indicates model captures line movement value.
        """
        if clv is None:
            return 0.0

        clv = np.asarray(clv).flatten()

        # Remove NaN values
        mask = ~np.isnan(clv)
        if mask.sum() < 10:  # Need minimum samples
            return 0.0

        # Calculate edge (predicted prob - implied prob)
        # For simplicity, use deviation from 0.5 as proxy
        edge = y_prob[mask] - 0.5

        correlation, _ = stats.pearsonr(edge, clv[mask])
        return correlation if not np.isnan(correlation) else 0.0

    def _profitable_bet_rate(self, y_true: np.ndarray, odds: np.ndarray = None) -> float:
        """
        Calculate rate of profitable bets.
        A bet is profitable if: won and (decimal_odds > 1)
        """
        if odds is None:
            return np.mean(y_true)

        odds = np.asarray(odds).flatten()

        # Calculate profit for each bet
        def calc_profit(won, o):
            if o < 0:
                decimal = 1 + (100 / abs(o))
            else:
                decimal = 1 + (o / 100)
            return (decimal - 1) if won else -1

        profits = np.array([calc_profit(w, o) for w, o in zip(y_true, odds)])
        return np.mean(profits > 0)

    def _kelly_sharpe_ratio(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        odds: np.ndarray = None
    ) -> float:
        """
        Calculate Sharpe ratio of Kelly criterion returns.
        Measures risk-adjusted performance of the model.
        """
        if odds is None:
            return 0.0

        odds = np.asarray(odds).flatten()

        def american_to_decimal(o):
            if o < 0:
                return 1 + (100 / abs(o))
            else:
                return 1 + (o / 100)

        decimal_odds = np.array([american_to_decimal(o) for o in odds])

        # Calculate Kelly fractions
        b = decimal_odds - 1
        p = y_prob
        q = 1 - p
        kelly = np.clip((p * b - q) / b, 0, 0.25)  # Cap at quarter Kelly

        # Calculate returns
        returns = np.where(
            y_true == 1,
            kelly * (decimal_odds - 1),  # Win
            -kelly  # Loss
        )

        # Sharpe ratio
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe = 0.0

        return sharpe

    def _confidence_auc(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Calculate AUC for confidence-correctness relationship.
        High confidence predictions should be more accurate.
        """
        # Confidence is distance from 0.5
        confidence = np.abs(y_prob - 0.5) * 2  # Scale to [0, 1]
        correct = (y_true == (y_prob > 0.5).astype(int)).astype(int)

        try:
            return roc_auc_score(correct, confidence)
        except:
            return 0.5

    def _confidence_accuracy(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        high_threshold: float = 0.7,
        low_threshold: float = 0.55
    ) -> Tuple[float, float]:
        """
        Calculate accuracy for high and low confidence predictions.
        """
        confidence = np.abs(y_prob - 0.5) * 2
        y_pred = (y_prob > 0.5).astype(int)

        high_conf_mask = confidence > high_threshold
        low_conf_mask = confidence < low_threshold

        if high_conf_mask.sum() > 0:
            high_acc = accuracy_score(y_true[high_conf_mask], y_pred[high_conf_mask])
        else:
            high_acc = 0.0

        if low_conf_mask.sum() > 0:
            low_acc = accuracy_score(y_true[low_conf_mask], y_pred[low_conf_mask])
        else:
            low_acc = 0.5

        return high_acc, low_acc

    def calibration_curve_data(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get calibration curve data for plotting.

        Returns:
            Tuple of (mean_predicted, fraction_positive, bin_counts)
        """
        if n_bins is None:
            n_bins = self.n_bins

        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )

        # Get bin counts
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        bin_counts = np.bincount(bin_indices, minlength=n_bins)

        return prob_pred, prob_true, bin_counts

    def generate_report(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        odds: np.ndarray = None,
        clv: np.ndarray = None,
        model_name: str = "Model"
    ) -> str:
        """
        Generate comprehensive evaluation report.
        """
        metrics = self.evaluate(y_true, y_prob, odds, clv)

        report = []
        report.append("=" * 70)
        report.append(f"MODEL EVALUATION REPORT: {model_name}")
        report.append("=" * 70)

        report.append("\n[CLASSIFICATION METRICS]")
        report.append(f"  Accuracy:    {metrics.accuracy:.4f}")
        report.append(f"  Precision:   {metrics.precision:.4f}")
        report.append(f"  Recall:      {metrics.recall:.4f}")
        report.append(f"  F1 Score:    {metrics.f1:.4f}")
        report.append(f"  AUC-ROC:     {metrics.auc_roc:.4f}")

        report.append("\n[CALIBRATION METRICS]")
        report.append(f"  Brier Score: {metrics.brier_score:.4f}")
        report.append(f"  Log Loss:    {metrics.log_loss:.4f}")
        report.append(f"  ECE:         {metrics.expected_calibration_error:.4f}")
        report.append(f"  MCE:         {metrics.max_calibration_error:.4f}")

        report.append("\n[BETTING METRICS]")
        report.append(f"  ROI-Weighted Accuracy: {metrics.roi_weighted_accuracy:.4f}")
        report.append(f"  CLV Correlation:       {metrics.clv_correlation:.4f}")
        report.append(f"  Profitable Bet Rate:   {metrics.profitable_bet_rate:.4f}")
        report.append(f"  Kelly Sharpe Ratio:    {metrics.kelly_sharpe:.4f}")

        report.append("\n[CONFIDENCE METRICS]")
        report.append(f"  Confidence AUC:         {metrics.confidence_auc:.4f}")
        report.append(f"  High Confidence Acc:    {metrics.high_confidence_accuracy:.4f}")
        report.append(f"  Low Confidence Acc:     {metrics.low_confidence_accuracy:.4f}")

        report.append("\n" + "=" * 70)

        return "\n".join(report)


def calibrate_and_evaluate(
    y_true: np.ndarray,
    y_prob_train: np.ndarray,
    y_prob_test: np.ndarray,
    y_true_train: np.ndarray = None,
    methods: List[str] = None,
    odds: np.ndarray = None,
    clv: np.ndarray = None
) -> Dict[str, CalibrationResult]:
    """
    Calibrate probabilities using multiple methods and evaluate.

    Args:
        y_true: True labels for test set
        y_prob_train: Predicted probabilities for training/calibration set
        y_prob_test: Predicted probabilities for test set
        y_true_train: True labels for training/calibration set
        methods: List of calibration methods to try
        odds: American odds for betting metrics
        clv: CLV values for betting metrics

    Returns:
        Dictionary of calibration results by method
    """
    if methods is None:
        methods = ['platt', 'isotonic', 'temperature']

    if y_true_train is None:
        y_true_train = y_true

    evaluator = BettingMetricsEvaluator()
    results = {}

    # Evaluate uncalibrated
    metrics_before = evaluator.evaluate(y_true, y_prob_test, odds, clv)
    ece_before, _ = evaluator._calibration_error(y_true, y_prob_test)

    for method in methods:
        try:
            calibrator = ProbabilityCalibrator(method=method)
            calibrator.fit(y_prob_train, y_true_train)
            calibrated_probs = calibrator.transform(y_prob_test)

            # Evaluate calibrated
            metrics_after = evaluator.evaluate(y_true, calibrated_probs, odds, clv)
            ece_after, _ = evaluator._calibration_error(y_true, calibrated_probs)

            results[method] = CalibrationResult(
                method=method,
                calibrated_probs=calibrated_probs,
                brier_before=metrics_before.brier_score,
                brier_after=metrics_after.brier_score,
                calibration_error_before=ece_before,
                calibration_error_after=ece_after,
                calibrator=calibrator
            )
        except Exception as e:
            print(f"Warning: Calibration method '{method}' failed: {e}")
            continue

    return results


# Example usage and testing
if __name__ == '__main__':
    print("=" * 70)
    print("PIKKIT MODEL CALIBRATION MODULE")
    print("=" * 70)

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000

    # Simulated model predictions (slightly miscalibrated)
    y_true = np.random.binomial(1, 0.5, n_samples)
    # Model is overconfident
    y_prob = np.clip(y_true * 0.7 + (1 - y_true) * 0.3 + np.random.normal(0, 0.15, n_samples), 0.01, 0.99)

    # Simulated odds and CLV
    odds = np.random.choice([-110, -115, -105, 100, 105, 110, 120, 150], n_samples)
    clv = np.where(np.random.random(n_samples) < 0.3, np.random.normal(0, 3, n_samples), np.nan)

    # Split for calibration
    split = int(0.7 * n_samples)
    y_true_cal, y_true_test = y_true[:split], y_true[split:]
    y_prob_cal, y_prob_test = y_prob[:split], y_prob[split:]
    odds_test = odds[split:]
    clv_test = clv[split:]

    print(f"\nData: {n_samples} samples (70% calibration, 30% test)")

    # Evaluate uncalibrated
    print("\n" + "-" * 70)
    print("UNCALIBRATED MODEL")
    evaluator = BettingMetricsEvaluator()
    print(evaluator.generate_report(y_true_test, y_prob_test, odds_test, clv_test, "Uncalibrated"))

    # Calibrate and evaluate
    print("\n" + "-" * 70)
    print("CALIBRATION RESULTS")

    results = calibrate_and_evaluate(
        y_true=y_true_test,
        y_prob_train=y_prob_cal,
        y_prob_test=y_prob_test,
        y_true_train=y_true_cal,
        methods=['platt', 'isotonic', 'temperature'],
        odds=odds_test,
        clv=clv_test
    )

    for method, result in results.items():
        print(f"\n{method.upper()} Calibration:")
        print(f"  Brier Score: {result.brier_before:.4f} -> {result.brier_after:.4f}")
        print(f"  ECE:         {result.calibration_error_before:.4f} -> {result.calibration_error_after:.4f}")

    # Find best method
    best_method = min(results.keys(), key=lambda m: results[m].brier_after)
    print(f"\nBest calibration method: {best_method}")

    print("\n" + "=" * 70)
    print("Calibration complete!")
    print("=" * 70)
