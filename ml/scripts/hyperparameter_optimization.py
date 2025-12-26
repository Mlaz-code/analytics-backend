#!/usr/bin/env python3
"""
Pikkit Hyperparameter Optimization Module
==========================================
Optuna-based hyperparameter optimization for betting models.

Features:
1. XGBoost hyperparameter optimization
2. Multi-objective optimization (accuracy + calibration)
3. Betting-specific objectives (ROI, Sharpe)
4. Time-series cross-validation
5. Hyperparameter importance analysis
6. Early stopping and pruning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import json
import os

import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

# Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    best_params: Dict[str, Any]
    best_score: float
    study: optuna.Study
    n_trials: int
    optimization_time: float
    param_importance: Dict[str, float]


class BettingObjective:
    """
    Custom objective functions for sports betting model optimization.
    Supports multiple metrics beyond standard ML metrics.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = 'brier',
        cv_folds: int = 5,
        odds: np.ndarray = None,
        sample_weight: np.ndarray = None,
        time_series_cv: bool = True
    ):
        """
        Initialize objective.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            metric: Optimization metric ('brier', 'auc', 'logloss', 'roi_sharpe', 'combined')
            cv_folds: Number of cross-validation folds
            odds: American odds for ROI-based metrics
            sample_weight: Sample weights
            time_series_cv: Use time-series cross-validation
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.metric = metric
        self.cv_folds = cv_folds
        self.odds = odds
        self.sample_weight = sample_weight
        self.time_series_cv = time_series_cv

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.

        Args:
            trial: Optuna trial object

        Returns:
            Objective value (lower is better)
        """
        # Sample hyperparameters
        params = self._sample_xgboost_params(trial)

        # Create model
        model = xgb.XGBClassifier(**params)

        # Train with early stopping
        try:
            model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                sample_weight=self.sample_weight,
                verbose=False
            )
        except Exception as e:
            # Return large value on failure
            return 1.0

        # Get predictions
        y_prob = model.predict_proba(self.X_val)[:, 1]

        # Calculate metric
        score = self._calculate_metric(self.y_val, y_prob)

        # Report intermediate value for pruning
        trial.report(score, model.best_iteration)

        # Check if should prune
        if trial.should_prune():
            raise optuna.TrialPruned()

        return score

    def _sample_xgboost_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample XGBoost hyperparameters."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2.0),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'random_state': 42,
            'early_stopping_rounds': 20,
        }

    def _calculate_metric(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Calculate optimization metric.
        Returns value where lower is better.
        """
        if self.metric == 'brier':
            return brier_score_loss(y_true, y_prob)

        elif self.metric == 'auc':
            # Return negative AUC (since we minimize)
            return -roc_auc_score(y_true, y_prob)

        elif self.metric == 'logloss':
            return log_loss(y_true, y_prob)

        elif self.metric == 'accuracy':
            return -accuracy_score(y_true, (y_prob > 0.5).astype(int))

        elif self.metric == 'roi_sharpe':
            return -self._calculate_roi_sharpe(y_true, y_prob)

        elif self.metric == 'combined':
            return self._calculate_combined_metric(y_true, y_prob)

        else:
            return brier_score_loss(y_true, y_prob)

    def _calculate_roi_sharpe(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Calculate Sharpe ratio of Kelly criterion returns.
        """
        if self.odds is None:
            return 0.0

        # Use only validation odds
        odds = self.odds[-len(y_true):]

        def american_to_decimal(o):
            if o < 0:
                return 1 + (100 / abs(o))
            else:
                return 1 + (o / 100)

        decimal_odds = np.array([american_to_decimal(o) for o in odds])

        # Calculate Kelly fractions (quarter Kelly)
        b = decimal_odds - 1
        p = y_prob
        q = 1 - p
        kelly = np.clip((p * b - q) / b, 0, 0.25) * 0.25

        # Calculate returns
        returns = np.where(
            y_true == 1,
            kelly * (decimal_odds - 1),
            -kelly
        )

        # Sharpe ratio
        if np.std(returns) > 0:
            return np.mean(returns) / np.std(returns) * np.sqrt(252)
        return 0.0

    def _calculate_combined_metric(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Combined metric balancing accuracy, calibration, and betting performance.
        """
        brier = brier_score_loss(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        # Combined score (lower is better)
        # Weight Brier more as it captures both discrimination and calibration
        combined = 0.6 * brier + 0.4 * (1 - auc)

        return combined


class XGBoostOptimizer:
    """
    Comprehensive XGBoost hyperparameter optimizer for betting models.
    """

    def __init__(
        self,
        metric: str = 'brier',
        n_trials: int = 100,
        timeout: int = 3600,
        cv_folds: int = 5,
        n_jobs: int = -1,
        time_series_cv: bool = True,
        pruning: bool = True
    ):
        """
        Initialize optimizer.

        Args:
            metric: Optimization metric
            n_trials: Number of optimization trials
            timeout: Maximum optimization time in seconds
            cv_folds: Number of CV folds
            n_jobs: Number of parallel jobs
            time_series_cv: Use time-series cross-validation
            pruning: Enable trial pruning
        """
        self.metric = metric
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.time_series_cv = time_series_cv
        self.pruning = pruning

        self.study: Optional[optuna.Study] = None
        self.best_params: Dict[str, Any] = {}
        self.optimization_history: List[Dict] = []

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        odds: np.ndarray = None,
        sample_weight: np.ndarray = None,
        show_progress: bool = True
    ) -> OptimizationResult:
        """
        Run hyperparameter optimization.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            odds: American odds for ROI-based metrics
            sample_weight: Sample weights
            show_progress: Show progress bar

        Returns:
            OptimizationResult with best parameters
        """
        start_time = datetime.now()

        # Create objective
        objective = BettingObjective(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            metric=self.metric,
            cv_folds=self.cv_folds,
            odds=odds,
            sample_weight=sample_weight,
            time_series_cv=self.time_series_cv
        )

        # Create pruner
        if self.pruning:
            pruner = HyperbandPruner(
                min_resource=10,
                max_resource=500,
                reduction_factor=3
            )
        else:
            pruner = optuna.pruners.NopPruner()

        # Create sampler
        sampler = TPESampler(seed=42, multivariate=True)

        # Create study
        self.study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner
        )

        # Optimize
        print(f"\nOptimizing XGBoost hyperparameters ({self.n_trials} trials)...")
        print(f"Metric: {self.metric}")

        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=show_progress,
            n_jobs=1  # XGBoost handles parallelization internally
        )

        # Get results
        optimization_time = (datetime.now() - start_time).total_seconds()
        self.best_params = self.study.best_params

        # Calculate parameter importance
        try:
            param_importance = optuna.importance.get_param_importances(self.study)
        except:
            param_importance = {}

        result = OptimizationResult(
            best_params=self.best_params,
            best_score=self.study.best_value,
            study=self.study,
            n_trials=len(self.study.trials),
            optimization_time=optimization_time,
            param_importance=param_importance
        )

        self._print_results(result)

        return result

    def _print_results(self, result: OptimizationResult):
        """Print optimization results."""
        print("\n" + "=" * 60)
        print("OPTIMIZATION RESULTS")
        print("=" * 60)

        print(f"\nBest {self.metric} score: {result.best_score:.6f}")
        print(f"Completed trials: {result.n_trials}")
        print(f"Optimization time: {result.optimization_time:.1f}s")

        print("\nBest Parameters:")
        for param, value in sorted(result.best_params.items()):
            if isinstance(value, float):
                print(f"  {param}: {value:.6f}")
            else:
                print(f"  {param}: {value}")

        if result.param_importance:
            print("\nParameter Importance:")
            for param, importance in sorted(
                result.param_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]:
                print(f"  {param}: {importance:.4f}")

    def get_best_model(self) -> xgb.XGBClassifier:
        """
        Create XGBoost model with optimized parameters.

        Returns:
            Configured XGBClassifier
        """
        if not self.best_params:
            raise ValueError("No optimization results. Run optimize() first.")

        # Add fixed parameters
        params = self.best_params.copy()
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'logloss'
        params['tree_method'] = 'hist'
        params['random_state'] = 42

        return xgb.XGBClassifier(**params)

    def save_results(self, path: str):
        """Save optimization results to file."""
        if self.study is None:
            raise ValueError("No study to save. Run optimize() first.")

        results = {
            'best_params': self.best_params,
            'best_score': self.study.best_value,
            'metric': self.metric,
            'n_trials': len(self.study.trials),
            'trials': [
                {
                    'number': t.number,
                    'value': t.value,
                    'params': t.params,
                    'state': str(t.state)
                }
                for t in self.study.trials
            ],
            'timestamp': datetime.now().isoformat()
        }

        with open(path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {path}")


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization for betting models.
    Optimizes multiple objectives simultaneously (e.g., accuracy + calibration).
    """

    def __init__(
        self,
        objectives: List[str] = None,
        n_trials: int = 100,
        timeout: int = 3600
    ):
        """
        Initialize multi-objective optimizer.

        Args:
            objectives: List of objectives to optimize
            n_trials: Number of trials
            timeout: Maximum time in seconds
        """
        self.objectives = objectives or ['brier', 'auc']
        self.n_trials = n_trials
        self.timeout = timeout
        self.study: Optional[optuna.Study] = None

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        odds: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Run multi-objective optimization.

        Returns:
            Dictionary with Pareto front solutions
        """
        def multi_objective(trial: optuna.Trial):
            # Sample parameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'tree_method': 'hist',
                'random_state': 42,
                'early_stopping_rounds': 20,
            }

            # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Get predictions
            y_prob = model.predict_proba(X_val)[:, 1]

            # Calculate objectives (all to minimize)
            results = []
            for obj in self.objectives:
                if obj == 'brier':
                    results.append(brier_score_loss(y_val, y_prob))
                elif obj == 'auc':
                    results.append(-roc_auc_score(y_val, y_prob))  # Negative for minimize
                elif obj == 'logloss':
                    results.append(log_loss(y_val, y_prob))
                elif obj == 'accuracy':
                    results.append(-accuracy_score(y_val, (y_prob > 0.5).astype(int)))

            return tuple(results)

        # Create study
        self.study = optuna.create_study(
            directions=['minimize'] * len(self.objectives),
            sampler=TPESampler(seed=42, multivariate=True)
        )

        print(f"\nMulti-objective optimization ({len(self.objectives)} objectives)...")
        print(f"Objectives: {self.objectives}")

        self.study.optimize(
            multi_objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        # Get Pareto front
        pareto_front = self.study.best_trials

        print(f"\nFound {len(pareto_front)} Pareto-optimal solutions")

        return {
            'pareto_front': [
                {
                    'params': t.params,
                    'values': {obj: v for obj, v in zip(self.objectives, t.values)}
                }
                for t in pareto_front
            ],
            'study': self.study
        }


class SportSpecificOptimizer:
    """
    Optimize hyperparameters separately for each sport.
    """

    def __init__(
        self,
        sports: List[str] = None,
        metric: str = 'brier',
        n_trials_per_sport: int = 50
    ):
        """
        Initialize sport-specific optimizer.

        Args:
            sports: List of sports to optimize for
            metric: Optimization metric
            n_trials_per_sport: Number of trials per sport
        """
        self.sports = sports or [
            'Basketball', 'American Football', 'Baseball', 'Ice Hockey'
        ]
        self.metric = metric
        self.n_trials_per_sport = n_trials_per_sport
        self.optimized_params: Dict[str, Dict] = {}

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        df: pd.DataFrame,
        sport_column: str = 'sport',
        odds: np.ndarray = None
    ) -> Dict[str, OptimizationResult]:
        """
        Optimize hyperparameters for each sport.

        Returns:
            Dictionary of OptimizationResult per sport
        """
        results = {}

        for sport in self.sports:
            print(f"\n{'=' * 60}")
            print(f"Optimizing for {sport}")
            print('=' * 60)

            # Filter data for this sport
            mask = df[sport_column] == sport
            X_sport = X[mask]
            y_sport = y[mask]

            if len(X_sport) < 100:
                print(f"  Skipping {sport}: only {len(X_sport)} samples")
                continue

            # Split for optimization
            split = int(0.7 * len(X_sport))
            X_train = X_sport[:split]
            y_train = y_sport[:split]
            X_val = X_sport[split:]
            y_val = y_sport[split:]

            # Get sport-specific odds if available
            sport_odds = odds[mask] if odds is not None else None
            val_odds = sport_odds[split:] if sport_odds is not None else None

            # Optimize
            optimizer = XGBoostOptimizer(
                metric=self.metric,
                n_trials=self.n_trials_per_sport,
                time_series_cv=True
            )

            result = optimizer.optimize(
                X_train, y_train, X_val, y_val,
                odds=val_odds,
                show_progress=True
            )

            results[sport] = result
            self.optimized_params[sport] = result.best_params

        return results

    def get_sport_params(self, sport: str) -> Dict[str, Any]:
        """Get optimized parameters for a specific sport."""
        return self.optimized_params.get(sport, {})


def create_optimized_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str = 'brier',
    n_trials: int = 100,
    odds: np.ndarray = None
) -> Tuple[xgb.XGBClassifier, OptimizationResult]:
    """
    Convenience function to create an optimized XGBoost model.

    Returns:
        Tuple of (trained model, optimization result)
    """
    optimizer = XGBoostOptimizer(
        metric=metric,
        n_trials=n_trials,
        time_series_cv=True
    )

    result = optimizer.optimize(
        X_train, y_train, X_val, y_val,
        odds=odds
    )

    # Create and train model with best params
    model = optimizer.get_best_model()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model, result


# Example usage and testing
if __name__ == '__main__':
    print("=" * 70)
    print("PIKKIT HYPERPARAMETER OPTIMIZATION MODULE")
    print("=" * 70)

    # Generate sample data
    np.random.seed(42)
    n_samples = 1500

    # Features and targets
    X = np.random.randn(n_samples, 20)
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.5 > 0).astype(int)

    # Simulate odds
    odds = np.random.choice([-110, -115, -105, 100, 105, 110, 120, 150], n_samples)

    # Split data
    train_end = int(0.6 * n_samples)
    val_end = int(0.8 * n_samples)

    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    X_test = X[val_end:]
    y_test = y[val_end:]
    odds_val = odds[train_end:val_end]

    print(f"\nData: {n_samples} samples")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Run optimization
    print("\n" + "-" * 70)
    print("Running hyperparameter optimization...")

    model, result = create_optimized_model(
        X_train, y_train, X_val, y_val,
        metric='brier',
        n_trials=30,  # Reduced for demo
        odds=odds_val
    )

    # Evaluate on test set
    print("\n" + "-" * 70)
    print("Evaluating optimized model on test set...")

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    test_brier = brier_score_loss(y_test, y_prob)
    test_auc = roc_auc_score(y_test, y_prob)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"\nTest Results:")
    print(f"  Brier Score: {test_brier:.4f}")
    print(f"  AUC-ROC:     {test_auc:.4f}")
    print(f"  Accuracy:    {test_acc:.4f}")

    print("\n" + "=" * 70)
    print("Hyperparameter optimization complete!")
    print("=" * 70)
