#!/usr/bin/env python3
"""
Pikkit Ensemble Models Module
==============================
Sport-specific sub-models and ensemble strategies for betting predictions.

Features:
1. Sport-specific XGBoost models
2. Market-type specific models (props, spreads, totals, moneylines)
3. Meta-learner ensemble (stacking)
4. Weighted averaging ensemble
5. Dynamic ensemble selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime
import pickle
import json
import os

import xgboost as xgb
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelConfig:
    """Configuration for a sub-model."""
    name: str
    model_type: str  # 'xgboost', 'rf', 'gbm', 'mlp'
    filter_column: Optional[str] = None
    filter_values: Optional[List[str]] = None
    hyperparameters: Optional[Dict] = None
    weight: float = 1.0


@dataclass
class EnsembleResult:
    """Container for ensemble prediction results."""
    prediction: np.ndarray
    model_weights: Dict[str, float]
    model_predictions: Dict[str, np.ndarray]
    confidence: np.ndarray


class SportSpecificModelFactory:
    """Factory for creating sport-specific models."""

    @staticmethod
    def create_xgboost(
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        **kwargs
    ) -> xgb.XGBClassifier:
        """Create XGBoost classifier with betting-optimized defaults."""
        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            tree_method='hist',
            early_stopping_rounds=20,
            **kwargs
        )

    @staticmethod
    def create_random_forest(
        n_estimators: int = 200,
        max_depth: int = 10,
        **kwargs
    ) -> RandomForestClassifier:
        """Create Random Forest classifier."""
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            **kwargs
        )

    @staticmethod
    def create_gbm(
        n_estimators: int = 150,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        **kwargs
    ) -> GradientBoostingClassifier:
        """Create Gradient Boosting classifier."""
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=42,
            **kwargs
        )

    @staticmethod
    def create_mlp(
        hidden_layers: Tuple[int, ...] = (128, 64, 32),
        **kwargs
    ) -> MLPClassifier:
        """Create MLP classifier."""
        return MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            early_stopping=True,
            random_state=42,
            **kwargs
        )


class SubModel:
    """
    Wrapper for a sport/market-specific sub-model.
    Handles filtering, training, and prediction for specific data subsets.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize sub-model.

        Args:
            config: ModelConfig with model specifications
        """
        self.config = config
        self.model = None
        self.fitted = False
        self.training_samples = 0
        self.feature_names: List[str] = []

    def _create_model(self) -> Any:
        """Create the underlying model based on config."""
        factory = SportSpecificModelFactory()
        params = self.config.hyperparameters or {}

        if self.config.model_type == 'xgboost':
            return factory.create_xgboost(**params)
        elif self.config.model_type == 'rf':
            return factory.create_random_forest(**params)
        elif self.config.model_type == 'gbm':
            return factory.create_gbm(**params)
        elif self.config.model_type == 'mlp':
            return factory.create_mlp(**params)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def _filter_data(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        df: pd.DataFrame = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Filter data based on config criteria.

        Returns:
            Tuple of (filtered_X, filtered_y, mask)
        """
        if self.config.filter_column is None or df is None:
            mask = np.ones(len(X), dtype=bool)
        else:
            if self.config.filter_column in df.columns:
                mask = df[self.config.filter_column].isin(self.config.filter_values).values
            else:
                mask = np.ones(len(X), dtype=bool)

        filtered_X = X[mask]
        filtered_y = y[mask] if y is not None else None

        return filtered_X, filtered_y, mask

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        df: pd.DataFrame = None,
        feature_names: List[str] = None,
        eval_set: Tuple[np.ndarray, np.ndarray] = None,
        sample_weight: np.ndarray = None
    ) -> 'SubModel':
        """
        Fit the sub-model.

        Args:
            X: Training features
            y: Training labels
            df: DataFrame for filtering (optional)
            feature_names: Feature names for importance analysis
            eval_set: Validation set for early stopping
            sample_weight: Sample weights

        Returns:
            Self
        """
        # Filter data
        X_filtered, y_filtered, mask = self._filter_data(X, y, df)

        if len(X_filtered) < 50:  # Minimum samples threshold
            print(f"  Warning: {self.config.name} has only {len(X_filtered)} samples. "
                  f"Using full dataset instead.")
            X_filtered, y_filtered = X, y
            mask = np.ones(len(X), dtype=bool)

        # Filter weights if provided
        weights_filtered = sample_weight[mask] if sample_weight is not None else None

        # Create and fit model
        self.model = self._create_model()
        self.training_samples = len(X_filtered)
        self.feature_names = feature_names or []

        # Handle eval_set for models that support it
        if eval_set is not None and self.config.model_type == 'xgboost':
            X_val, y_val = eval_set
            # Filter validation set if applicable
            if df is not None and self.config.filter_column is not None:
                # For validation, we still want to predict on all data
                # but train on filtered data
                pass
            self.model.fit(
                X_filtered, y_filtered,
                eval_set=[(X_val, y_val)],
                sample_weight=weights_filtered,
                verbose=False
            )
        else:
            self.model.fit(X_filtered, y_filtered, sample_weight=weights_filtered)

        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.

        Args:
            X: Features to predict

        Returns:
            Predicted probabilities for positive class
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        proba = self.model.predict_proba(X)
        return proba[:, 1] if len(proba.shape) > 1 else proba

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model."""
        if not self.fitted or not self.feature_names:
            return pd.DataFrame()

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            return pd.DataFrame()

        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


class BettingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble model combining multiple sport/market-specific sub-models.

    Supports:
    - Weighted averaging
    - Stacking (meta-learner)
    - Dynamic weighting based on confidence
    - Sport/market routing
    """

    def __init__(
        self,
        sub_model_configs: List[ModelConfig] = None,
        ensemble_method: str = 'weighted_average',
        meta_learner: str = 'logistic',
        calibrate: bool = True
    ):
        """
        Initialize ensemble.

        Args:
            sub_model_configs: List of ModelConfig for sub-models
            ensemble_method: 'weighted_average', 'stacking', 'dynamic'
            meta_learner: Type of meta-learner for stacking
            calibrate: Whether to calibrate final probabilities
        """
        self.sub_model_configs = sub_model_configs or self._default_configs()
        self.ensemble_method = ensemble_method
        self.meta_learner_type = meta_learner
        self.calibrate = calibrate

        self.sub_models: Dict[str, SubModel] = {}
        self.meta_learner = None
        self.calibrator = None
        self.fitted = False
        self.feature_names: List[str] = []

    def _default_configs(self) -> List[ModelConfig]:
        """Default configuration for sport-specific models."""
        return [
            # Sport-specific models
            ModelConfig(
                name='nba_model',
                model_type='xgboost',
                filter_column='sport',
                filter_values=['Basketball'],
                hyperparameters={'max_depth': 6, 'learning_rate': 0.05},
                weight=1.0
            ),
            ModelConfig(
                name='nfl_model',
                model_type='xgboost',
                filter_column='sport',
                filter_values=['American Football'],
                hyperparameters={'max_depth': 7, 'learning_rate': 0.03},
                weight=1.0
            ),
            ModelConfig(
                name='mlb_model',
                model_type='xgboost',
                filter_column='sport',
                filter_values=['Baseball'],
                hyperparameters={'max_depth': 5, 'learning_rate': 0.05},
                weight=1.0
            ),
            ModelConfig(
                name='hockey_model',
                model_type='xgboost',
                filter_column='sport',
                filter_values=['Ice Hockey'],
                hyperparameters={'max_depth': 5, 'learning_rate': 0.05},
                weight=1.0
            ),
            # Market-type models
            ModelConfig(
                name='props_model',
                model_type='xgboost',
                filter_column='is_player_prop',
                filter_values=[1],
                hyperparameters={'max_depth': 8, 'learning_rate': 0.03},
                weight=0.8
            ),
            # General fallback model
            ModelConfig(
                name='general_model',
                model_type='xgboost',
                filter_column=None,
                hyperparameters={'max_depth': 6, 'learning_rate': 0.05},
                weight=0.5
            ),
            # Alternative model types for diversity
            ModelConfig(
                name='rf_model',
                model_type='rf',
                hyperparameters={'n_estimators': 200, 'max_depth': 10},
                weight=0.3
            ),
        ]

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        df: pd.DataFrame = None,
        feature_names: List[str] = None,
        eval_set: Tuple[np.ndarray, np.ndarray] = None,
        sample_weight: np.ndarray = None
    ) -> 'BettingEnsemble':
        """
        Fit the ensemble.

        Args:
            X: Training features
            y: Training labels
            df: DataFrame with metadata for filtering
            feature_names: Feature names
            eval_set: Validation set
            sample_weight: Sample weights

        Returns:
            Self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = feature_names or []

        print(f"\nTraining {len(self.sub_model_configs)} sub-models...")

        # Train sub-models
        for config in self.sub_model_configs:
            print(f"  Training {config.name}...")
            sub_model = SubModel(config)
            sub_model.fit(
                X, y, df=df,
                feature_names=self.feature_names,
                eval_set=eval_set,
                sample_weight=sample_weight
            )
            self.sub_models[config.name] = sub_model
            print(f"    -> Trained on {sub_model.training_samples} samples")

        # Train meta-learner if using stacking
        if self.ensemble_method == 'stacking':
            self._fit_meta_learner(X, y)

        # Fit calibrator if requested
        if self.calibrate and eval_set is not None:
            self._fit_calibrator(eval_set[0], eval_set[1])

        self.fitted = True
        return self

    def _fit_meta_learner(self, X: np.ndarray, y: np.ndarray):
        """Fit meta-learner for stacking ensemble."""
        # Generate sub-model predictions as meta-features
        meta_features = self._generate_meta_features(X)

        if self.meta_learner_type == 'logistic':
            self.meta_learner = LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )
        elif self.meta_learner_type == 'xgboost':
            self.meta_learner = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        else:
            self.meta_learner = LogisticRegression(solver='lbfgs', max_iter=1000)

        self.meta_learner.fit(meta_features, y)

    def _fit_calibrator(self, X_val: np.ndarray, y_val: np.ndarray):
        """Fit probability calibrator."""
        from model_calibration import ProbabilityCalibrator

        # Get ensemble predictions on validation set
        y_prob = self._predict_proba_raw(X_val)

        self.calibrator = ProbabilityCalibrator(method='isotonic')
        self.calibrator.fit(y_prob, y_val)

    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features from sub-model predictions."""
        predictions = []
        for name, sub_model in self.sub_models.items():
            pred = sub_model.predict_proba(X)
            predictions.append(pred)

        return np.column_stack(predictions)

    def _predict_proba_raw(self, X: np.ndarray, df: pd.DataFrame = None) -> np.ndarray:
        """
        Get raw ensemble predictions (before calibration).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.ensemble_method == 'weighted_average':
            return self._weighted_average_predict(X, df)
        elif self.ensemble_method == 'stacking':
            return self._stacking_predict(X)
        elif self.ensemble_method == 'dynamic':
            return self._dynamic_predict(X, df)
        else:
            return self._weighted_average_predict(X, df)

    def _weighted_average_predict(
        self,
        X: np.ndarray,
        df: pd.DataFrame = None
    ) -> np.ndarray:
        """
        Weighted average of sub-model predictions.
        Weights are adjusted based on relevance to each sample.
        """
        predictions = []
        weights = []

        for name, sub_model in self.sub_models.items():
            config = sub_model.config
            pred = sub_model.predict_proba(X)
            predictions.append(pred)

            # Compute weight based on relevance
            if config.filter_column is not None and df is not None:
                if config.filter_column in df.columns:
                    relevance = df[config.filter_column].isin(config.filter_values).astype(float).values
                else:
                    relevance = np.ones(len(X))
            else:
                relevance = np.ones(len(X))

            # Weight = base_weight * relevance * sample_size_factor
            sample_factor = min(1.0, sub_model.training_samples / 500)
            weight = config.weight * relevance * sample_factor
            weights.append(weight)

        predictions = np.array(predictions)  # (n_models, n_samples)
        weights = np.array(weights)  # (n_models, n_samples)

        # Normalize weights
        weights_sum = weights.sum(axis=0, keepdims=True)
        weights_sum[weights_sum == 0] = 1  # Avoid division by zero
        normalized_weights = weights / weights_sum

        # Weighted average
        weighted_pred = (predictions * normalized_weights).sum(axis=0)

        return weighted_pred

    def _stacking_predict(self, X: np.ndarray) -> np.ndarray:
        """Stacking ensemble prediction using meta-learner."""
        meta_features = self._generate_meta_features(X)
        return self.meta_learner.predict_proba(meta_features)[:, 1]

    def _dynamic_predict(
        self,
        X: np.ndarray,
        df: pd.DataFrame = None
    ) -> np.ndarray:
        """
        Dynamic ensemble: select models based on prediction confidence.
        Uses only high-confidence predictions from relevant models.
        """
        all_predictions = {}
        all_confidences = {}

        for name, sub_model in self.sub_models.items():
            pred = sub_model.predict_proba(X)
            conf = np.abs(pred - 0.5) * 2  # Distance from 0.5 as confidence
            all_predictions[name] = pred
            all_confidences[name] = conf

        # For each sample, use top-K most confident relevant models
        k = 3
        final_predictions = np.zeros(len(X))

        for i in range(len(X)):
            # Get predictions and confidences for this sample
            sample_preds = {name: all_predictions[name][i] for name in self.sub_models}
            sample_confs = {name: all_confidences[name][i] for name in self.sub_models}

            # Select top-K by confidence
            top_models = sorted(sample_confs.keys(), key=lambda n: sample_confs[n], reverse=True)[:k]

            # Weighted average of top models
            total_conf = sum(sample_confs[m] for m in top_models)
            if total_conf > 0:
                final_predictions[i] = sum(
                    sample_preds[m] * sample_confs[m] / total_conf
                    for m in top_models
                )
            else:
                final_predictions[i] = np.mean([sample_preds[m] for m in top_models])

        return final_predictions

    def predict_proba(self, X: np.ndarray, df: pd.DataFrame = None) -> np.ndarray:
        """
        Predict probabilities.

        Args:
            X: Features
            df: DataFrame with metadata

        Returns:
            Calibrated probabilities
        """
        if not self.fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        proba = self._predict_proba_raw(X, df)

        if self.calibrate and self.calibrator is not None:
            proba = self.calibrator.transform(proba)

        return proba

    def predict(self, X: np.ndarray, df: pd.DataFrame = None) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X, df)
        return (proba > 0.5).astype(int)

    def predict_with_details(
        self,
        X: np.ndarray,
        df: pd.DataFrame = None
    ) -> EnsembleResult:
        """
        Predict with detailed information about each sub-model's contribution.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        model_predictions = {}
        model_weights = {}

        for name, sub_model in self.sub_models.items():
            pred = sub_model.predict_proba(X)
            model_predictions[name] = pred
            model_weights[name] = sub_model.config.weight

        final_pred = self.predict_proba(X, df)

        # Calculate confidence as agreement between models
        preds_array = np.array(list(model_predictions.values()))
        confidence = 1 - np.std(preds_array, axis=0)  # Lower variance = higher confidence

        return EnsembleResult(
            prediction=final_pred,
            model_weights=model_weights,
            model_predictions=model_predictions,
            confidence=confidence
        )

    def get_feature_importance(self, aggregate: str = 'mean') -> pd.DataFrame:
        """
        Get aggregated feature importance across all sub-models.

        Args:
            aggregate: Aggregation method ('mean', 'max', 'weighted')
        """
        all_importances = []

        for name, sub_model in self.sub_models.items():
            imp_df = sub_model.get_feature_importance()
            if len(imp_df) > 0:
                imp_df['model'] = name
                imp_df['weight'] = sub_model.config.weight
                all_importances.append(imp_df)

        if not all_importances:
            return pd.DataFrame()

        combined = pd.concat(all_importances, ignore_index=True)

        if aggregate == 'mean':
            result = combined.groupby('feature')['importance'].mean().reset_index()
        elif aggregate == 'max':
            result = combined.groupby('feature')['importance'].max().reset_index()
        elif aggregate == 'weighted':
            combined['weighted_imp'] = combined['importance'] * combined['weight']
            total_weight = combined.groupby('feature')['weight'].sum()
            weighted_sum = combined.groupby('feature')['weighted_imp'].sum()
            result = pd.DataFrame({
                'feature': total_weight.index,
                'importance': (weighted_sum / total_weight).values
            })
        else:
            result = combined.groupby('feature')['importance'].mean().reset_index()

        return result.sort_values('importance', ascending=False)

    def save(self, path: str):
        """Save ensemble to disk."""
        os.makedirs(path, exist_ok=True)

        # Save sub-models
        for name, sub_model in self.sub_models.items():
            model_path = os.path.join(path, f"{name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(sub_model, f)

        # Save meta-learner if exists
        if self.meta_learner is not None:
            with open(os.path.join(path, 'meta_learner.pkl'), 'wb') as f:
                pickle.dump(self.meta_learner, f)

        # Save calibrator if exists
        if self.calibrator is not None:
            with open(os.path.join(path, 'calibrator.pkl'), 'wb') as f:
                pickle.dump(self.calibrator, f)

        # Save metadata
        metadata = {
            'ensemble_method': self.ensemble_method,
            'meta_learner_type': self.meta_learner_type,
            'calibrate': self.calibrate,
            'feature_names': self.feature_names,
            'sub_model_names': list(self.sub_models.keys()),
            'timestamp': datetime.now().isoformat()
        }
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'BettingEnsemble':
        """Load ensemble from disk."""
        # Load metadata
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        # Create ensemble
        ensemble = cls(
            sub_model_configs=[],  # Will be loaded from pickles
            ensemble_method=metadata['ensemble_method'],
            meta_learner=metadata['meta_learner_type'],
            calibrate=metadata['calibrate']
        )
        ensemble.feature_names = metadata['feature_names']

        # Load sub-models
        for name in metadata['sub_model_names']:
            model_path = os.path.join(path, f"{name}.pkl")
            with open(model_path, 'rb') as f:
                ensemble.sub_models[name] = pickle.load(f)

        # Load meta-learner if exists
        meta_path = os.path.join(path, 'meta_learner.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                ensemble.meta_learner = pickle.load(f)

        # Load calibrator if exists
        cal_path = os.path.join(path, 'calibrator.pkl')
        if os.path.exists(cal_path):
            with open(cal_path, 'rb') as f:
                ensemble.calibrator = pickle.load(f)

        ensemble.fitted = True
        print(f"Ensemble loaded from {path}")

        return ensemble


def create_sport_specific_ensemble(
    sports: List[str] = None,
    markets: List[str] = None,
    include_general: bool = True
) -> BettingEnsemble:
    """
    Create a sport-specific ensemble with custom configuration.

    Args:
        sports: List of sports to create models for
        markets: List of market types to create models for
        include_general: Whether to include general fallback model

    Returns:
        Configured BettingEnsemble
    """
    configs = []

    # Sport-specific configs
    sport_configs = {
        'Basketball': {'max_depth': 6, 'learning_rate': 0.05},
        'American Football': {'max_depth': 7, 'learning_rate': 0.03},
        'Baseball': {'max_depth': 5, 'learning_rate': 0.05},
        'Ice Hockey': {'max_depth': 5, 'learning_rate': 0.05},
        'Soccer': {'max_depth': 5, 'learning_rate': 0.04},
        'Tennis': {'max_depth': 4, 'learning_rate': 0.05},
    }

    if sports is None:
        sports = list(sport_configs.keys())

    for sport in sports:
        params = sport_configs.get(sport, {'max_depth': 6, 'learning_rate': 0.05})
        configs.append(ModelConfig(
            name=f"{sport.lower().replace(' ', '_')}_model",
            model_type='xgboost',
            filter_column='sport',
            filter_values=[sport],
            hyperparameters=params,
            weight=1.0
        ))

    # Market-type configs
    if markets:
        for market in markets:
            configs.append(ModelConfig(
                name=f"{market.lower().replace(' ', '_')}_model",
                model_type='xgboost',
                filter_column='market',
                filter_values=[market],
                hyperparameters={'max_depth': 6, 'learning_rate': 0.05},
                weight=0.8
            ))

    # General fallback
    if include_general:
        configs.append(ModelConfig(
            name='general_model',
            model_type='xgboost',
            filter_column=None,
            hyperparameters={'max_depth': 6, 'learning_rate': 0.05},
            weight=0.5
        ))

    return BettingEnsemble(
        sub_model_configs=configs,
        ensemble_method='weighted_average',
        calibrate=True
    )


# Example usage and testing
if __name__ == '__main__':
    print("=" * 70)
    print("PIKKIT ENSEMBLE MODELS MODULE")
    print("=" * 70)

    # Generate sample data
    np.random.seed(42)
    n_samples = 2000

    sports = ['Basketball', 'American Football', 'Baseball', 'Ice Hockey']
    markets = ['Spread', 'Moneyline', 'Total', 'Player Points']

    # Create sample DataFrame
    df = pd.DataFrame({
        'sport': np.random.choice(sports, n_samples),
        'market': np.random.choice(markets, n_samples),
        'is_player_prop': (np.random.choice(markets, n_samples) == 'Player Points').astype(int),
    })

    # Generate features
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    y = np.random.binomial(1, 0.5, n_samples)

    # Split data
    train_idx = int(0.7 * n_samples)
    val_idx = int(0.85 * n_samples)

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]

    df_train = df.iloc[:train_idx]
    df_test = df.iloc[val_idx:]

    print(f"\nData: {n_samples} samples")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create and train ensemble
    print("\n" + "-" * 70)
    print("Creating sport-specific ensemble...")

    ensemble = create_sport_specific_ensemble(
        sports=['Basketball', 'American Football', 'Baseball'],
        markets=['Player Points'],
        include_general=True
    )

    ensemble.fit(
        X_train, y_train,
        df=df_train,
        feature_names=[f'feature_{i}' for i in range(n_features)],
        eval_set=(X_val, y_val)
    )

    # Make predictions
    print("\n" + "-" * 70)
    print("Making predictions...")

    result = ensemble.predict_with_details(X_test, df_test)

    # Evaluate
    from sklearn.metrics import accuracy_score, roc_auc_score

    acc = accuracy_score(y_test, (result.prediction > 0.5).astype(int))
    auc = roc_auc_score(y_test, result.prediction)

    print(f"\nTest Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC-ROC:  {auc:.4f}")
    print(f"  Mean Confidence: {result.confidence.mean():.4f}")

    # Show feature importance
    print("\n" + "-" * 70)
    print("Top 10 Features (Aggregated Importance):")
    importance = ensemble.get_feature_importance(aggregate='weighted')
    print(importance.head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print("Ensemble training complete!")
    print("=" * 70)
