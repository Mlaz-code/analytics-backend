#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization for Pikkit ML Pipeline

Provides automated hyperparameter tuning with:
- Configurable search spaces
- Early stopping and pruning
- Best parameters persistence
- Integration with experiment tracking
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, roc_auc_score, mean_absolute_error,
    log_loss, mean_squared_error
)
from sklearn.model_selection import TimeSeriesSplit

try:
    import optuna
    from optuna.integration import XGBoostPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from .config import PipelineConfig, OptimizationConfig

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Optuna-based hyperparameter optimization for XGBoost models.

    Features:
    - Flexible search space definition
    - Early stopping with pruning
    - Cross-validation integration
    - Best parameters persistence
    """

    DEFAULT_SEARCH_SPACE = {
        'n_estimators': [100, 500],
        'max_depth': [3, 10],
        'learning_rate': [0.01, 0.3],
        'subsample': [0.6, 1.0],
        'colsample_bytree': [0.6, 1.0],
        'min_child_weight': [1, 10],
        'gamma': [0, 5],
        'reg_alpha': [0, 10],
        'reg_lambda': [0, 10],
    }

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize optimizer.

        Args:
            config: Pipeline configuration
        """
        if config is None:
            config = PipelineConfig()

        self.config = config.optimization
        self.paths = config.paths
        self.enabled = self.config.enabled and OPTUNA_AVAILABLE

        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not installed. Hyperparameter optimization disabled.")

        # Storage for best parameters
        self._best_params = {}
        self._study = None

    def optimize_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        search_space: Optional[Dict] = None,
        n_trials: Optional[int] = None,
        metric: str = "auc"
    ) -> Tuple[Dict[str, Any], float]:
        """
        Optimize XGBoost classifier hyperparameters.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            sample_weights: Optional sample weights
            search_space: Custom search space (uses config if not provided)
            n_trials: Number of optimization trials
            metric: Optimization metric (auc, accuracy, logloss)

        Returns:
            Tuple of (best_params, best_value)
        """
        if not self.enabled:
            logger.warning("Optimization disabled, returning default params")
            return self._get_default_params('classifier'), 0.0

        # Set up search space
        if search_space is None:
            search_space = self.config.win_classifier_search_space or self.DEFAULT_SEARCH_SPACE

        n_trials = n_trials or self.config.n_trials

        logger.info(f"Starting classifier optimization with {n_trials} trials")

        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            params = self._sample_params(trial, search_space)
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'logloss'
            params['tree_method'] = 'hist'
            params['random_state'] = 42

            # Create model with pruning callback
            callbacks = []
            if self.config.pruning.get('enabled', True):
                callbacks.append(
                    XGBoostPruningCallback(trial, f"validation_0-logloss")
                )

            model = xgb.XGBClassifier(**params)

            # Train with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                sample_weight=sample_weights,
                callbacks=callbacks,
                verbose=False
            )

            # Evaluate
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            if metric == "auc":
                return roc_auc_score(y_val, y_pred_proba)
            elif metric == "accuracy":
                return accuracy_score(y_val, model.predict(X_val))
            elif metric == "logloss":
                return -log_loss(y_val, y_pred_proba)  # Negative because we maximize
            else:
                raise ValueError(f"Unknown metric: {metric}")

        # Create study
        study_name = f"classifier_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            ) if self.config.pruning.get('enabled', True) else None
        )

        # Run optimization
        self._study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.config.timeout_per_trial * n_trials,
            show_progress_bar=True,
            n_jobs=1  # XGBoost handles parallelism internally
        )

        best_params = self._study.best_params
        best_value = self._study.best_value

        # Add fixed parameters
        best_params['objective'] = 'binary:logistic'
        best_params['eval_metric'] = 'logloss'
        best_params['tree_method'] = 'hist'
        best_params['random_state'] = 42

        self._best_params['classifier'] = best_params

        logger.info(f"Best classifier {metric}: {best_value:.4f}")
        logger.info(f"Best params: {best_params}")

        return best_params, best_value

    def optimize_regressor(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        search_space: Optional[Dict] = None,
        n_trials: Optional[int] = None,
        metric: str = "mae"
    ) -> Tuple[Dict[str, Any], float]:
        """
        Optimize XGBoost regressor hyperparameters.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            sample_weights: Optional sample weights
            search_space: Custom search space
            n_trials: Number of optimization trials
            metric: Optimization metric (mae, rmse, mse)

        Returns:
            Tuple of (best_params, best_value)
        """
        if not self.enabled:
            logger.warning("Optimization disabled, returning default params")
            return self._get_default_params('regressor'), 0.0

        # Set up search space
        if search_space is None:
            search_space = self.config.roi_regressor_search_space or self.DEFAULT_SEARCH_SPACE

        n_trials = n_trials or self.config.n_trials

        logger.info(f"Starting regressor optimization with {n_trials} trials")

        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            params = self._sample_params(trial, search_space)
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'mae'
            params['tree_method'] = 'hist'
            params['random_state'] = 42

            # Create model with pruning callback
            callbacks = []
            if self.config.pruning.get('enabled', True):
                callbacks.append(
                    XGBoostPruningCallback(trial, f"validation_0-mae")
                )

            model = xgb.XGBRegressor(**params)

            # Train with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                sample_weight=sample_weights,
                callbacks=callbacks,
                verbose=False
            )

            # Evaluate
            y_pred = model.predict(X_val)

            if metric == "mae":
                return -mean_absolute_error(y_val, y_pred)  # Negative to maximize
            elif metric == "rmse":
                return -np.sqrt(mean_squared_error(y_val, y_pred))
            elif metric == "mse":
                return -mean_squared_error(y_val, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")

        # Create study
        study_name = f"regressor_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._study = optuna.create_study(
            study_name=study_name,
            direction="maximize",  # We negate loss so higher is better
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            ) if self.config.pruning.get('enabled', True) else None
        )

        # Run optimization
        self._study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.config.timeout_per_trial * n_trials,
            show_progress_bar=True,
            n_jobs=1
        )

        best_params = self._study.best_params
        best_value = -self._study.best_value  # Convert back to positive

        # Add fixed parameters
        best_params['objective'] = 'reg:squarederror'
        best_params['eval_metric'] = 'mae'
        best_params['tree_method'] = 'hist'
        best_params['random_state'] = 42

        self._best_params['regressor'] = best_params

        logger.info(f"Best regressor {metric}: {best_value:.4f}")
        logger.info(f"Best params: {best_params}")

        return best_params, best_value

    def optimize_with_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "classifier",
        n_splits: int = 5,
        search_space: Optional[Dict] = None,
        n_trials: Optional[int] = None,
        metric: str = "auc"
    ) -> Tuple[Dict[str, Any], float]:
        """
        Optimize using time-series cross-validation.

        Args:
            X: Features
            y: Targets
            model_type: 'classifier' or 'regressor'
            n_splits: Number of CV splits
            search_space: Custom search space
            n_trials: Number of optimization trials
            metric: Optimization metric

        Returns:
            Tuple of (best_params, best_cv_score)
        """
        if not self.enabled:
            return self._get_default_params(model_type), 0.0

        tscv = TimeSeriesSplit(n_splits=n_splits)
        n_trials = n_trials or self.config.n_trials

        # Determine search space
        if search_space is None:
            if model_type == "classifier":
                search_space = self.config.win_classifier_search_space or self.DEFAULT_SEARCH_SPACE
            else:
                search_space = self.config.roi_regressor_search_space or self.DEFAULT_SEARCH_SPACE

        logger.info(f"Starting {model_type} CV optimization with {n_trials} trials, {n_splits} folds")

        def objective(trial: optuna.Trial) -> float:
            params = self._sample_params(trial, search_space)
            params['tree_method'] = 'hist'
            params['random_state'] = 42

            if model_type == "classifier":
                params['objective'] = 'binary:logistic'
                params['eval_metric'] = 'logloss'
            else:
                params['objective'] = 'reg:squarederror'
                params['eval_metric'] = 'mae'

            scores = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                if model_type == "classifier":
                    model = xgb.XGBClassifier(**params, early_stopping_rounds=20)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                    y_pred = model.predict_proba(X_val)[:, 1]

                    if metric == "auc":
                        score = roc_auc_score(y_val, y_pred)
                    elif metric == "accuracy":
                        score = accuracy_score(y_val, model.predict(X_val))
                    else:
                        score = -log_loss(y_val, y_pred)
                else:
                    model = xgb.XGBRegressor(**params, early_stopping_rounds=20)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                    y_pred = model.predict(X_val)

                    if metric == "mae":
                        score = -mean_absolute_error(y_val, y_pred)
                    else:
                        score = -np.sqrt(mean_squared_error(y_val, y_pred))

                scores.append(score)

                # Early pruning based on intermediate CV results
                trial.report(np.mean(scores), fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)

        # Create and run study
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3)
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.config.timeout_per_trial * n_trials,
            show_progress_bar=True
        )

        best_params = study.best_params
        best_value = study.best_value if model_type == "classifier" else -study.best_value

        # Add fixed params
        if model_type == "classifier":
            best_params['objective'] = 'binary:logistic'
            best_params['eval_metric'] = 'logloss'
        else:
            best_params['objective'] = 'reg:squarederror'
            best_params['eval_metric'] = 'mae'

        best_params['tree_method'] = 'hist'
        best_params['random_state'] = 42

        self._best_params[model_type] = best_params
        self._study = study

        logger.info(f"Best CV {metric}: {best_value:.4f}")

        return best_params, best_value

    def _sample_params(
        self,
        trial: 'optuna.Trial',
        search_space: Dict[str, List]
    ) -> Dict[str, Any]:
        """
        Sample hyperparameters from search space.

        Args:
            trial: Optuna trial
            search_space: Dictionary mapping param names to [min, max] ranges

        Returns:
            Dictionary of sampled parameters
        """
        params = {}

        for param_name, bounds in search_space.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                continue

            low, high = bounds

            # Determine parameter type
            if param_name == 'n_estimators':
                params[param_name] = trial.suggest_int(param_name, low, high)
            elif param_name == 'max_depth':
                params[param_name] = trial.suggest_int(param_name, low, high)
            elif param_name == 'min_child_weight':
                params[param_name] = trial.suggest_int(param_name, low, high)
            elif param_name == 'learning_rate':
                params[param_name] = trial.suggest_float(param_name, low, high, log=True)
            elif param_name in ['subsample', 'colsample_bytree']:
                params[param_name] = trial.suggest_float(param_name, low, high)
            elif param_name in ['gamma', 'reg_alpha', 'reg_lambda']:
                params[param_name] = trial.suggest_float(param_name, low, high)
            else:
                # Default to float
                params[param_name] = trial.suggest_float(param_name, low, high)

        return params

    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters when optimization is disabled"""
        if model_type == "classifier":
            return {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'tree_method': 'hist',
                'random_state': 42,
            }
        else:
            return {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'tree_method': 'hist',
                'random_state': 42,
            }

    def get_optimization_history(self) -> Optional[pd.DataFrame]:
        """
        Get optimization history from last study.

        Returns:
            DataFrame with trial history
        """
        if self._study is None:
            return None

        trials = self._study.trials
        history = []

        for trial in trials:
            record = {
                'trial_number': trial.number,
                'value': trial.value,
                'state': trial.state.name,
                'datetime_start': trial.datetime_start,
                'datetime_complete': trial.datetime_complete,
            }
            record.update(trial.params)
            history.append(record)

        return pd.DataFrame(history)

    def save_best_params(self, filepath: Optional[str] = None) -> str:
        """
        Save best parameters to JSON file.

        Args:
            filepath: Path to save parameters (auto-generated if not provided)

        Returns:
            Path where parameters were saved
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"{self.paths.models}/best_params_{timestamp}.json"

        with open(filepath, 'w') as f:
            json.dump(self._best_params, f, indent=2, default=str)

        logger.info(f"Saved best parameters to {filepath}")
        return filepath

    def load_best_params(self, filepath: str) -> Dict[str, Dict]:
        """
        Load best parameters from JSON file.

        Args:
            filepath: Path to load parameters from

        Returns:
            Dictionary of best parameters
        """
        with open(filepath, 'r') as f:
            self._best_params = json.load(f)

        logger.info(f"Loaded best parameters from {filepath}")
        return self._best_params

    def get_importance_of_hyperparameters(self) -> Optional[Dict[str, float]]:
        """
        Get importance of hyperparameters from last study.

        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if self._study is None or not OPTUNA_AVAILABLE:
            return None

        try:
            importances = optuna.importance.get_param_importances(self._study)
            return dict(importances)
        except Exception as e:
            logger.warning(f"Could not compute hyperparameter importance: {e}")
            return None
