#!/usr/bin/env python3
"""
Pikkit Backtesting and A/B Testing Framework
==============================================
Comprehensive backtesting and model comparison for betting models.

Features:
1. Time-series walk-forward backtesting
2. A/B testing framework with statistical significance
3. Monte Carlo simulation for robustness
4. Bankroll simulation with Kelly criterion
5. Model comparison dashboards
6. Statistical significance testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy import stats
import json
import os

from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    accuracy_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestResult:
    """Container for backtest results."""
    model_name: str
    total_bets: int
    win_rate: float
    roi: float
    cumulative_profit: float
    sharpe_ratio: float
    max_drawdown: float
    brier_score: float
    auc_roc: float
    calibration_error: float
    predictions: np.ndarray
    actual_results: np.ndarray
    profits_per_bet: np.ndarray
    timestamps: np.ndarray
    period_start: datetime
    period_end: datetime


@dataclass
class ABTestResult:
    """Container for A/B test results."""
    model_a_name: str
    model_b_name: str
    metric: str
    model_a_value: float
    model_b_value: float
    difference: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    sample_size: int
    recommendation: str


@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation results."""
    mean_final_bankroll: float
    std_final_bankroll: float
    median_final_bankroll: float
    percentile_5: float
    percentile_95: float
    probability_profit: float
    probability_ruin: float
    max_drawdown_mean: float
    max_drawdown_95: float
    sharpe_mean: float
    all_final_bankrolls: np.ndarray


class WalkForwardBacktester:
    """
    Walk-forward backtesting for betting models.
    Simulates real-world model deployment with periodic retraining.
    """

    def __init__(
        self,
        initial_train_size: float = 0.5,
        test_size: int = 100,
        retrain_frequency: int = 200,
        initial_bankroll: float = 10000,
        kelly_fraction: float = 0.25,
        min_bet_confidence: float = 0.55
    ):
        """
        Initialize backtester.

        Args:
            initial_train_size: Fraction of data for initial training
            test_size: Number of bets per test period
            retrain_frequency: How often to retrain (number of bets)
            initial_bankroll: Starting bankroll
            kelly_fraction: Fraction of Kelly to use (e.g., 0.25 for quarter Kelly)
            min_bet_confidence: Minimum confidence to place bet
        """
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.retrain_frequency = retrain_frequency
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.min_bet_confidence = min_bet_confidence

    def backtest(
        self,
        model_factory: Callable,
        X: np.ndarray,
        y: np.ndarray,
        odds: np.ndarray,
        timestamps: np.ndarray = None,
        feature_names: List[str] = None,
        df: pd.DataFrame = None,
        model_name: str = "Model"
    ) -> BacktestResult:
        """
        Run walk-forward backtest.

        Args:
            model_factory: Callable that returns a new model instance
            X: Features
            y: Labels
            odds: American odds
            timestamps: Timestamps for each bet
            feature_names: Feature names
            df: DataFrame with metadata
            model_name: Name for this model

        Returns:
            BacktestResult
        """
        n_samples = len(X)
        initial_train_end = int(n_samples * self.initial_train_size)

        if timestamps is None:
            timestamps = np.arange(n_samples)

        # Initialize tracking
        all_predictions = []
        all_actuals = []
        all_profits = []
        all_timestamps = []
        bankroll = self.initial_bankroll
        bankroll_history = [bankroll]

        # Initial training
        train_end = initial_train_end
        model = model_factory()

        print(f"\nBacktesting {model_name}...")
        print(f"  Initial training: {initial_train_end} samples")

        # Fit initial model
        X_train = X[:train_end]
        y_train = y[:train_end]

        # Handle models that need eval_set
        try:
            val_split = int(0.9 * len(X_train))
            model.fit(
                X_train[:val_split], y_train[:val_split],
                eval_set=[(X_train[val_split:], y_train[val_split:])],
                verbose=False
            )
        except:
            model.fit(X_train, y_train)

        # Walk forward
        test_idx = train_end
        bets_since_retrain = 0

        while test_idx < n_samples:
            # Get next batch
            batch_end = min(test_idx + self.test_size, n_samples)
            X_batch = X[test_idx:batch_end]
            y_batch = y[test_idx:batch_end]
            odds_batch = odds[test_idx:batch_end]
            ts_batch = timestamps[test_idx:batch_end]

            # Get predictions
            y_prob = model.predict_proba(X_batch)[:, 1]

            # Simulate betting
            for i in range(len(X_batch)):
                confidence = abs(y_prob[i] - 0.5) * 2

                if confidence < self.min_bet_confidence:
                    continue

                # Calculate bet size using Kelly
                bet_size = self._calculate_kelly_bet(
                    y_prob[i], odds_batch[i], bankroll
                )

                if bet_size <= 0:
                    continue

                # Calculate profit/loss
                profit = self._calculate_profit(
                    y_batch[i], odds_batch[i], bet_size
                )

                bankroll += profit
                all_predictions.append(y_prob[i])
                all_actuals.append(y_batch[i])
                all_profits.append(profit)
                all_timestamps.append(ts_batch[i])
                bankroll_history.append(bankroll)

            # Check if need to retrain
            bets_since_retrain += len(X_batch)

            if bets_since_retrain >= self.retrain_frequency:
                # Retrain model
                train_end = batch_end
                X_train = X[:train_end]
                y_train = y[:train_end]

                model = model_factory()
                try:
                    val_split = int(0.9 * len(X_train))
                    model.fit(
                        X_train[:val_split], y_train[:val_split],
                        eval_set=[(X_train[val_split:], y_train[val_split:])],
                        verbose=False
                    )
                except:
                    model.fit(X_train, y_train)

                bets_since_retrain = 0

            test_idx = batch_end

        # Convert to arrays
        predictions = np.array(all_predictions)
        actuals = np.array(all_actuals)
        profits = np.array(all_profits)
        ts_array = np.array(all_timestamps)
        bankroll_array = np.array(bankroll_history)

        # Calculate metrics
        if len(predictions) > 0:
            win_rate = np.mean(actuals)
            roi = (np.sum(profits) / (self.initial_bankroll * len(profits))) * 100
            cumulative_profit = np.sum(profits)
            sharpe = self._calculate_sharpe(profits)
            max_dd = self._calculate_max_drawdown(bankroll_array)

            # ML metrics
            brier = brier_score_loss(actuals, predictions)
            try:
                auc = roc_auc_score(actuals, predictions)
            except:
                auc = 0.5
            cal_error = self._calculate_calibration_error(actuals, predictions)
        else:
            win_rate = roi = cumulative_profit = sharpe = 0
            max_dd = brier = auc = cal_error = 0

        return BacktestResult(
            model_name=model_name,
            total_bets=len(predictions),
            win_rate=win_rate,
            roi=roi,
            cumulative_profit=cumulative_profit,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            brier_score=brier,
            auc_roc=auc,
            calibration_error=cal_error,
            predictions=predictions,
            actual_results=actuals,
            profits_per_bet=profits,
            timestamps=ts_array,
            period_start=datetime.fromtimestamp(ts_array[0]) if len(ts_array) > 0 else datetime.now(),
            period_end=datetime.fromtimestamp(ts_array[-1]) if len(ts_array) > 0 else datetime.now()
        )

    def _calculate_kelly_bet(
        self,
        predicted_prob: float,
        american_odds: float,
        bankroll: float
    ) -> float:
        """Calculate bet size using Kelly criterion."""
        # Convert to decimal odds
        if american_odds < 0:
            decimal_odds = 1 + (100 / abs(american_odds))
        else:
            decimal_odds = 1 + (american_odds / 100)

        b = decimal_odds - 1
        p = predicted_prob
        q = 1 - p

        # Kelly formula
        kelly = (p * b - q) / b

        if kelly <= 0:
            return 0

        # Apply fraction and cap
        bet_size = bankroll * kelly * self.kelly_fraction
        bet_size = min(bet_size, bankroll * 0.1)  # Max 10% of bankroll

        return bet_size

    def _calculate_profit(
        self,
        actual: int,
        american_odds: float,
        bet_size: float
    ) -> float:
        """Calculate profit/loss from a bet."""
        if actual == 1:  # Win
            if american_odds < 0:
                return bet_size * (100 / abs(american_odds))
            else:
                return bet_size * (american_odds / 100)
        else:  # Loss
            return -bet_size

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)

    def _calculate_max_drawdown(self, bankroll_history: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = bankroll_history[0]
        max_dd = 0

        for value in bankroll_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)

        return max_dd

    def _calculate_calibration_error(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Calculate expected calibration error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0

        for i in range(n_bins):
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_accuracy = y_true[mask].mean()
                bin_confidence = y_prob[mask].mean()
                ece += mask.sum() / len(y_prob) * abs(bin_accuracy - bin_confidence)

        return ece


class ABTester:
    """
    A/B testing framework for comparing betting models.
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize A/B tester.

        Args:
            significance_level: Alpha for significance testing
        """
        self.significance_level = significance_level

    def compare_models(
        self,
        result_a: BacktestResult,
        result_b: BacktestResult,
        metrics: List[str] = None
    ) -> List[ABTestResult]:
        """
        Compare two models across multiple metrics.

        Args:
            result_a: Backtest results for model A
            result_b: Backtest results for model B
            metrics: List of metrics to compare

        Returns:
            List of ABTestResult for each metric
        """
        if metrics is None:
            metrics = ['roi', 'win_rate', 'brier_score', 'sharpe_ratio']

        results = []

        for metric in metrics:
            if metric == 'roi':
                test_result = self._compare_roi(result_a, result_b)
            elif metric == 'win_rate':
                test_result = self._compare_proportions(
                    result_a, result_b, 'win_rate'
                )
            elif metric == 'brier_score':
                test_result = self._compare_brier(result_a, result_b)
            elif metric == 'sharpe_ratio':
                test_result = self._compare_sharpe(result_a, result_b)
            elif metric == 'auc_roc':
                test_result = self._compare_auc(result_a, result_b)
            else:
                continue

            results.append(test_result)

        return results

    def _compare_roi(
        self,
        result_a: BacktestResult,
        result_b: BacktestResult
    ) -> ABTestResult:
        """Compare ROI using bootstrap."""
        profits_a = result_a.profits_per_bet
        profits_b = result_b.profits_per_bet

        # Bootstrap confidence interval
        n_bootstrap = 10000
        diff_samples = []

        for _ in range(n_bootstrap):
            sample_a = np.random.choice(profits_a, size=len(profits_a), replace=True)
            sample_b = np.random.choice(profits_b, size=len(profits_b), replace=True)
            diff_samples.append(sample_a.mean() - sample_b.mean())

        diff_samples = np.array(diff_samples)
        ci_lower = np.percentile(diff_samples, 2.5)
        ci_upper = np.percentile(diff_samples, 97.5)

        # P-value for difference != 0
        p_value = 2 * min(
            np.mean(diff_samples <= 0),
            np.mean(diff_samples >= 0)
        )

        difference = result_a.roi - result_b.roi
        is_significant = p_value < self.significance_level

        return ABTestResult(
            model_a_name=result_a.model_name,
            model_b_name=result_b.model_name,
            metric='roi',
            model_a_value=result_a.roi,
            model_b_value=result_b.roi,
            difference=difference,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            sample_size=min(len(profits_a), len(profits_b)),
            recommendation=self._get_recommendation('roi', difference, is_significant)
        )

    def _compare_proportions(
        self,
        result_a: BacktestResult,
        result_b: BacktestResult,
        metric_name: str
    ) -> ABTestResult:
        """Compare two proportions using chi-square test."""
        n_a = result_a.total_bets
        n_b = result_b.total_bets
        p_a = result_a.win_rate
        p_b = result_b.win_rate

        # Chi-square test
        successes_a = int(p_a * n_a)
        successes_b = int(p_b * n_b)

        # Contingency table
        table = [
            [successes_a, n_a - successes_a],
            [successes_b, n_b - successes_b]
        ]

        try:
            _, p_value, _, _ = stats.chi2_contingency(table)
        except:
            p_value = 1.0

        # Wilson confidence interval for difference
        from statsmodels.stats.proportion import proportion_confint

        ci_a = proportion_confint(successes_a, n_a, method='wilson')
        ci_b = proportion_confint(successes_b, n_b, method='wilson')

        difference = p_a - p_b
        ci_lower = (ci_a[0] - ci_b[1])
        ci_upper = (ci_a[1] - ci_b[0])

        is_significant = p_value < self.significance_level

        return ABTestResult(
            model_a_name=result_a.model_name,
            model_b_name=result_b.model_name,
            metric=metric_name,
            model_a_value=p_a,
            model_b_value=p_b,
            difference=difference,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            sample_size=min(n_a, n_b),
            recommendation=self._get_recommendation(metric_name, difference, is_significant)
        )

    def _compare_brier(
        self,
        result_a: BacktestResult,
        result_b: BacktestResult
    ) -> ABTestResult:
        """Compare Brier scores using DeLong test approximation."""
        brier_a = result_a.brier_score
        brier_b = result_b.brier_score

        # Bootstrap for confidence interval
        preds_a = result_a.predictions
        actual_a = result_a.actual_results
        preds_b = result_b.predictions
        actual_b = result_b.actual_results

        n_bootstrap = 5000
        diff_samples = []

        min_len = min(len(preds_a), len(preds_b))

        for _ in range(n_bootstrap):
            idx_a = np.random.choice(len(preds_a), size=min_len, replace=True)
            idx_b = np.random.choice(len(preds_b), size=min_len, replace=True)

            bs_a = brier_score_loss(actual_a[idx_a], preds_a[idx_a])
            bs_b = brier_score_loss(actual_b[idx_b], preds_b[idx_b])
            diff_samples.append(bs_a - bs_b)

        diff_samples = np.array(diff_samples)
        p_value = 2 * min(np.mean(diff_samples <= 0), np.mean(diff_samples >= 0))

        ci_lower = np.percentile(diff_samples, 2.5)
        ci_upper = np.percentile(diff_samples, 97.5)

        difference = brier_a - brier_b
        is_significant = p_value < self.significance_level

        return ABTestResult(
            model_a_name=result_a.model_name,
            model_b_name=result_b.model_name,
            metric='brier_score',
            model_a_value=brier_a,
            model_b_value=brier_b,
            difference=difference,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            sample_size=min(len(preds_a), len(preds_b)),
            recommendation=self._get_recommendation('brier_score', -difference, is_significant)
        )

    def _compare_sharpe(
        self,
        result_a: BacktestResult,
        result_b: BacktestResult
    ) -> ABTestResult:
        """Compare Sharpe ratios using bootstrap."""
        profits_a = result_a.profits_per_bet
        profits_b = result_b.profits_per_bet

        def calc_sharpe(p):
            if len(p) < 2 or np.std(p) == 0:
                return 0
            return np.mean(p) / np.std(p) * np.sqrt(252)

        sharpe_a = result_a.sharpe_ratio
        sharpe_b = result_b.sharpe_ratio

        # Bootstrap
        n_bootstrap = 5000
        diff_samples = []

        for _ in range(n_bootstrap):
            sample_a = np.random.choice(profits_a, size=len(profits_a), replace=True)
            sample_b = np.random.choice(profits_b, size=len(profits_b), replace=True)
            diff_samples.append(calc_sharpe(sample_a) - calc_sharpe(sample_b))

        diff_samples = np.array(diff_samples)
        p_value = 2 * min(np.mean(diff_samples <= 0), np.mean(diff_samples >= 0))

        ci_lower = np.percentile(diff_samples, 2.5)
        ci_upper = np.percentile(diff_samples, 97.5)

        difference = sharpe_a - sharpe_b
        is_significant = p_value < self.significance_level

        return ABTestResult(
            model_a_name=result_a.model_name,
            model_b_name=result_b.model_name,
            metric='sharpe_ratio',
            model_a_value=sharpe_a,
            model_b_value=sharpe_b,
            difference=difference,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            sample_size=min(len(profits_a), len(profits_b)),
            recommendation=self._get_recommendation('sharpe_ratio', difference, is_significant)
        )

    def _compare_auc(
        self,
        result_a: BacktestResult,
        result_b: BacktestResult
    ) -> ABTestResult:
        """Compare AUC-ROC scores."""
        auc_a = result_a.auc_roc
        auc_b = result_b.auc_roc

        # DeLong test approximation using bootstrap
        preds_a = result_a.predictions
        actual_a = result_a.actual_results
        preds_b = result_b.predictions
        actual_b = result_b.actual_results

        n_bootstrap = 5000
        diff_samples = []

        for _ in range(n_bootstrap):
            idx_a = np.random.choice(len(preds_a), size=len(preds_a), replace=True)
            idx_b = np.random.choice(len(preds_b), size=len(preds_b), replace=True)

            try:
                auc_sample_a = roc_auc_score(actual_a[idx_a], preds_a[idx_a])
                auc_sample_b = roc_auc_score(actual_b[idx_b], preds_b[idx_b])
                diff_samples.append(auc_sample_a - auc_sample_b)
            except:
                continue

        if len(diff_samples) < 100:
            p_value = 1.0
            ci_lower = ci_upper = 0
        else:
            diff_samples = np.array(diff_samples)
            p_value = 2 * min(np.mean(diff_samples <= 0), np.mean(diff_samples >= 0))
            ci_lower = np.percentile(diff_samples, 2.5)
            ci_upper = np.percentile(diff_samples, 97.5)

        difference = auc_a - auc_b
        is_significant = p_value < self.significance_level

        return ABTestResult(
            model_a_name=result_a.model_name,
            model_b_name=result_b.model_name,
            metric='auc_roc',
            model_a_value=auc_a,
            model_b_value=auc_b,
            difference=difference,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            sample_size=min(len(preds_a), len(preds_b)),
            recommendation=self._get_recommendation('auc_roc', difference, is_significant)
        )

    def _get_recommendation(
        self,
        metric: str,
        difference: float,
        is_significant: bool
    ) -> str:
        """Generate recommendation based on comparison."""
        if not is_significant:
            return "No significant difference. Need more data or models perform similarly."

        # For metrics where lower is better
        if metric in ['brier_score']:
            winner = "A" if difference < 0 else "B"
        # For metrics where higher is better
        else:
            winner = "A" if difference > 0 else "B"

        return f"Model {winner} is significantly better for {metric}."


class MonteCarloSimulator:
    """
    Monte Carlo simulation for bankroll risk analysis.
    """

    def __init__(
        self,
        n_simulations: int = 10000,
        n_bets_per_sim: int = 1000,
        initial_bankroll: float = 10000,
        kelly_fraction: float = 0.25,
        ruin_threshold: float = 0.1  # 10% of initial bankroll
    ):
        """
        Initialize simulator.

        Args:
            n_simulations: Number of Monte Carlo simulations
            n_bets_per_sim: Number of bets per simulation
            initial_bankroll: Starting bankroll
            kelly_fraction: Kelly fraction to use
            ruin_threshold: Fraction of bankroll considered ruin
        """
        self.n_simulations = n_simulations
        self.n_bets_per_sim = n_bets_per_sim
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.ruin_threshold = ruin_threshold

    def simulate(
        self,
        win_rate: float,
        avg_odds: float,
        odds_std: float = 20,
        calibration_error: float = 0.02
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation.

        Args:
            win_rate: Expected win rate
            avg_odds: Average American odds
            odds_std: Standard deviation of odds
            calibration_error: Model calibration error

        Returns:
            SimulationResult with statistics
        """
        final_bankrolls = []
        max_drawdowns = []
        sharpe_ratios = []

        for _ in range(self.n_simulations):
            bankroll = self.initial_bankroll
            bankroll_history = [bankroll]
            returns = []

            for _ in range(self.n_bets_per_sim):
                # Sample odds
                odds = np.random.normal(avg_odds, odds_std)

                # Calculate implied probability
                if odds < 0:
                    implied_prob = abs(odds) / (abs(odds) + 100)
                    decimal_odds = 1 + (100 / abs(odds))
                else:
                    implied_prob = 100 / (odds + 100)
                    decimal_odds = 1 + (odds / 100)

                # Model's edge (with calibration error)
                model_prob = win_rate + np.random.normal(0, calibration_error)
                model_prob = np.clip(model_prob, 0.01, 0.99)

                # Calculate Kelly bet
                b = decimal_odds - 1
                p = model_prob
                q = 1 - p
                kelly = max(0, (p * b - q) / b)
                bet_size = bankroll * kelly * self.kelly_fraction
                bet_size = min(bet_size, bankroll * 0.1)

                if bet_size <= 0:
                    continue

                # Simulate outcome
                won = np.random.random() < win_rate

                if won:
                    profit = bet_size * (decimal_odds - 1)
                else:
                    profit = -bet_size

                bankroll += profit
                bankroll_history.append(bankroll)
                returns.append(profit / self.initial_bankroll)

                # Check for ruin
                if bankroll < self.initial_bankroll * self.ruin_threshold:
                    break

            final_bankrolls.append(bankroll)

            # Calculate max drawdown
            peak = bankroll_history[0]
            max_dd = 0
            for value in bankroll_history:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)
            max_drawdowns.append(max_dd)

            # Calculate Sharpe
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_ratios.append(
                    np.mean(returns) / np.std(returns) * np.sqrt(252)
                )
            else:
                sharpe_ratios.append(0)

        final_bankrolls = np.array(final_bankrolls)
        max_drawdowns = np.array(max_drawdowns)
        sharpe_ratios = np.array(sharpe_ratios)

        return SimulationResult(
            mean_final_bankroll=np.mean(final_bankrolls),
            std_final_bankroll=np.std(final_bankrolls),
            median_final_bankroll=np.median(final_bankrolls),
            percentile_5=np.percentile(final_bankrolls, 5),
            percentile_95=np.percentile(final_bankrolls, 95),
            probability_profit=np.mean(final_bankrolls > self.initial_bankroll),
            probability_ruin=np.mean(final_bankrolls < self.initial_bankroll * self.ruin_threshold),
            max_drawdown_mean=np.mean(max_drawdowns),
            max_drawdown_95=np.percentile(max_drawdowns, 95),
            sharpe_mean=np.mean(sharpe_ratios),
            all_final_bankrolls=final_bankrolls
        )


def generate_backtest_report(
    results: List[BacktestResult],
    ab_tests: List[ABTestResult] = None
) -> str:
    """
    Generate comprehensive backtest report.

    Args:
        results: List of BacktestResult for each model
        ab_tests: Optional A/B test results

    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append("BACKTEST REPORT")
    report.append("=" * 80)

    # Model comparison table
    report.append("\n[MODEL COMPARISON]")
    report.append("-" * 80)
    headers = ["Model", "Bets", "Win Rate", "ROI", "Sharpe", "Max DD", "Brier", "AUC"]
    report.append(f"{headers[0]:<20} {headers[1]:>8} {headers[2]:>10} {headers[3]:>10} "
                  f"{headers[4]:>8} {headers[5]:>8} {headers[6]:>8} {headers[7]:>8}")
    report.append("-" * 80)

    for r in results:
        report.append(
            f"{r.model_name:<20} {r.total_bets:>8} {r.win_rate:>10.2%} "
            f"{r.roi:>9.2f}% {r.sharpe_ratio:>8.2f} {r.max_drawdown:>7.2%} "
            f"{r.brier_score:>8.4f} {r.auc_roc:>8.4f}"
        )

    # A/B test results
    if ab_tests:
        report.append("\n" + "=" * 80)
        report.append("[A/B TEST RESULTS]")
        report.append("-" * 80)

        for test in ab_tests:
            report.append(f"\nMetric: {test.metric}")
            report.append(f"  {test.model_a_name}: {test.model_a_value:.4f}")
            report.append(f"  {test.model_b_name}: {test.model_b_value:.4f}")
            report.append(f"  Difference: {test.difference:+.4f}")
            report.append(f"  P-value: {test.p_value:.4f}")
            report.append(f"  95% CI: ({test.confidence_interval[0]:.4f}, {test.confidence_interval[1]:.4f})")
            sig_marker = "[SIGNIFICANT]" if test.is_significant else "[NOT SIGNIFICANT]"
            report.append(f"  {sig_marker}")
            report.append(f"  Recommendation: {test.recommendation}")

    report.append("\n" + "=" * 80)

    return "\n".join(report)


# Example usage and testing
if __name__ == '__main__':
    import xgboost as xgb

    print("=" * 70)
    print("PIKKIT BACKTESTING FRAMEWORK")
    print("=" * 70)

    # Generate sample data
    np.random.seed(42)
    n_samples = 2000

    X = np.random.randn(n_samples, 15)
    y = (X[:, 0] * 0.3 + X[:, 1] * 0.2 + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    odds = np.random.choice([-110, -115, -105, 100, 105, 110, 120], n_samples)
    timestamps = np.arange(n_samples)

    print(f"\nData: {n_samples} samples")

    # Define model factories
    def model_factory_v1():
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            early_stopping_rounds=20,
            eval_metric='logloss',
            random_state=42
        )

    def model_factory_v2():
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            early_stopping_rounds=20,
            eval_metric='logloss',
            random_state=42
        )

    # Run backtests
    print("\n" + "-" * 70)
    backtester = WalkForwardBacktester(
        initial_train_size=0.4,
        test_size=50,
        retrain_frequency=100,
        kelly_fraction=0.25
    )

    result_v1 = backtester.backtest(
        model_factory_v1, X, y, odds, timestamps,
        model_name="Model_v1"
    )

    result_v2 = backtester.backtest(
        model_factory_v2, X, y, odds, timestamps,
        model_name="Model_v2"
    )

    # Run A/B tests
    print("\n" + "-" * 70)
    print("Running A/B tests...")

    ab_tester = ABTester(significance_level=0.05)
    ab_results = ab_tester.compare_models(result_v1, result_v2)

    # Generate report
    print("\n" + generate_backtest_report([result_v1, result_v2], ab_results))

    # Monte Carlo simulation
    print("\n" + "-" * 70)
    print("Running Monte Carlo simulation...")

    simulator = MonteCarloSimulator(
        n_simulations=1000,
        n_bets_per_sim=500
    )

    sim_result = simulator.simulate(
        win_rate=0.52,
        avg_odds=-110,
        odds_std=20
    )

    print(f"\nMonte Carlo Results (1000 sims, 500 bets each):")
    print(f"  Mean final bankroll:   ${sim_result.mean_final_bankroll:,.0f}")
    print(f"  Median final bankroll: ${sim_result.median_final_bankroll:,.0f}")
    print(f"  5th percentile:        ${sim_result.percentile_5:,.0f}")
    print(f"  95th percentile:       ${sim_result.percentile_95:,.0f}")
    print(f"  Probability of profit: {sim_result.probability_profit:.1%}")
    print(f"  Probability of ruin:   {sim_result.probability_ruin:.1%}")
    print(f"  Mean max drawdown:     {sim_result.max_drawdown_mean:.1%}")

    print("\n" + "=" * 70)
    print("Backtesting complete!")
    print("=" * 70)
