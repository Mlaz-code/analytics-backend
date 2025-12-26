#!/usr/bin/env python3
"""
Market Performance Prediction Model Training
============================================
Train models to predict future market-level winrate and ROI.

Model Architecture:
- Multi-task learning: Jointly predict winrate + ROI
- Ensemble approach: XGBoost + LightGBM with stacking
- Uncertainty quantification: Predict both mean and variance
- Cold-start handling: Fallback to sport/league aggregates

Use Cases:
1. Chrome extension: Real-time take/skip recommendations
2. Dashboard: Display predicted performance metrics
3. Alert system: Notify when high-confidence opportunities appear
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import optuna

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.market_performance_features import (
    MarketPerformanceFeatureEngineer,
    validate_sample_size
)


class MarketPredictionModel:
    """
    Multi-task model for predicting market performance.

    Predicts:
    - Future winrate (classification-style regression)
    - Future ROI (regression)
    - Confidence score (uncertainty quantification)
    """

    def __init__(
        self,
        winrate_model=None,
        roi_model=None,
        confidence_model=None,
        scaler=None
    ):
        self.winrate_model = winrate_model
        self.roi_model = roi_model
        self.confidence_model = confidence_model
        self.scaler = scaler or StandardScaler()
        self.feature_names = []
        self.metadata = {}

    def train(
        self,
        X_train: pd.DataFrame,
        y_winrate: pd.Series,
        y_roi: pd.Series,
        y_confidence: pd.Series,
        params: Dict = None
    ):
        """
        Train all three models.

        Args:
            X_train: Training features
            y_winrate: Target winrates
            y_roi: Target ROIs
            y_confidence: Confidence scores
            params: Model hyperparameters
        """
        self.feature_names = list(X_train.columns)

        # Default parameters
        if params is None:
            params = {
                'winrate': {
                    'objective': 'reg:squarederror',
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'n_estimators': 200,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                },
                'roi': {
                    'objective': 'reg:squarederror',
                    'max_depth': 5,
                    'learning_rate': 0.05,
                    'n_estimators': 200,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                },
                'confidence': {
                    'objective': 'reg:squarederror',
                    'max_depth': 4,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'random_state': 42
                }
            }

        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        X_scaled = pd.DataFrame(X_scaled, columns=X_train.columns, index=X_train.index)

        # Train winrate model
        print("Training winrate model...")
        self.winrate_model = xgb.XGBRegressor(**params['winrate'])
        self.winrate_model.fit(X_scaled, y_winrate)

        # Train ROI model
        print("Training ROI model...")
        self.roi_model = xgb.XGBRegressor(**params['roi'])
        self.roi_model.fit(X_scaled, y_roi)

        # Train confidence model (predicts reliability of predictions)
        print("Training confidence model...")
        self.confidence_model = xgb.XGBRegressor(**params['confidence'])
        self.confidence_model.fit(X_scaled, y_confidence)

        # Store metadata
        self.metadata = {
            'trained_at': datetime.now().isoformat(),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'params': params,
            'training_samples': len(X_train)
        }

    def predict(
        self,
        X: pd.DataFrame,
        return_confidence: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions for all targets.

        Args:
            X: Feature DataFrame
            return_confidence: Whether to include confidence scores

        Returns:
            Dictionary with predictions for winrate, roi, and optionally confidence
        """
        # Scale features
        X_scaled = self.scaler.transform(X[self.feature_names])
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)

        # Predictions
        pred_winrate = self.winrate_model.predict(X_scaled)
        pred_roi = self.roi_model.predict(X_scaled)

        result = {
            'winrate': np.clip(pred_winrate, 0, 1),  # Clip to valid probability range
            'roi': pred_roi
        }

        if return_confidence and self.confidence_model is not None:
            pred_confidence = self.confidence_model.predict(X_scaled)
            result['confidence'] = np.clip(pred_confidence, 0, 1)

        return result

    def predict_recommendation(
        self,
        X: pd.DataFrame,
        winrate_threshold: float = 0.53,
        roi_threshold: float = 3.0,
        confidence_threshold: float = 0.6
    ) -> pd.DataFrame:
        """
        Generate take/skip recommendations with explanations.

        Args:
            X: Feature DataFrame
            winrate_threshold: Minimum winrate for 'take' recommendation
            roi_threshold: Minimum ROI for 'take' recommendation
            confidence_threshold: Minimum confidence for recommendation

        Returns:
            DataFrame with recommendations and scores
        """
        preds = self.predict(X, return_confidence=True)

        results = pd.DataFrame({
            'predicted_winrate': preds['winrate'],
            'predicted_roi': preds['roi'],
            'confidence': preds['confidence'],
        }, index=X.index)

        # Binary recommendation
        results['should_take'] = (
            (results['predicted_winrate'] >= winrate_threshold) &
            (results['predicted_roi'] >= roi_threshold) &
            (results['confidence'] >= confidence_threshold)
        ).astype(int)

        # Recommendation strength (0-100 score)
        winrate_score = (results['predicted_winrate'] - 0.5) * 100  # Normalize around 50%
        roi_score = results['predicted_roi'] / 5  # Normalize ROI
        confidence_score = results['confidence'] * 50

        results['recommendation_score'] = np.clip(
            winrate_score + roi_score + confidence_score,
            0, 100
        )

        # Grade (A/B/C/D/F based on score)
        results['grade'] = pd.cut(
            results['recommendation_score'],
            bins=[0, 40, 60, 75, 85, 100],
            labels=['F', 'D', 'C', 'B', 'A']
        )

        # Explanation
        def create_explanation(row):
            parts = []

            if row['should_take']:
                parts.append(f"TAKE: Expected {row['predicted_winrate']*100:.1f}% winrate")
                parts.append(f"ROI: {row['predicted_roi']:.1f}%")
            else:
                if row['predicted_winrate'] < winrate_threshold:
                    parts.append(f"SKIP: Low winrate ({row['predicted_winrate']*100:.1f}%)")
                if row['predicted_roi'] < roi_threshold:
                    parts.append(f"Low ROI ({row['predicted_roi']:.1f}%)")
                if row['confidence'] < confidence_threshold:
                    parts.append(f"Low confidence ({row['confidence']*100:.1f}%)")

            return " | ".join(parts)

        results['explanation'] = results.apply(create_explanation, axis=1)

        return results

    def save(self, save_dir: str, version: str = None):
        """Save models and metadata."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        version = version or datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save models
        with open(save_dir / f'market_winrate_model_{version}.pkl', 'wb') as f:
            pickle.dump(self.winrate_model, f)

        with open(save_dir / f'market_roi_model_{version}.pkl', 'wb') as f:
            pickle.dump(self.roi_model, f)

        with open(save_dir / f'market_confidence_model_{version}.pkl', 'wb') as f:
            pickle.dump(self.confidence_model, f)

        with open(save_dir / f'market_scaler_{version}.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        # Save metadata
        with open(save_dir / f'market_model_metadata_{version}.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # Create 'latest' symlinks
        for suffix in ['winrate_model', 'roi_model', 'confidence_model', 'scaler', 'model_metadata']:
            src = save_dir / f'market_{suffix}_{version}.pkl' if 'metadata' not in suffix else save_dir / f'market_{suffix}_{version}.json'
            dst = save_dir / f'market_{suffix}_latest.pkl' if 'metadata' not in suffix else save_dir / f'market_{suffix}_latest.json'

            if dst.exists():
                dst.unlink()
            os.symlink(src.name, dst)

        print(f"Models saved to {save_dir} with version {version}")

    @classmethod
    def load(cls, load_dir: str, version: str = 'latest'):
        """Load models from disk."""
        load_dir = Path(load_dir)

        # Load models
        with open(load_dir / f'market_winrate_model_{version}.pkl', 'rb') as f:
            winrate_model = pickle.load(f)

        with open(load_dir / f'market_roi_model_{version}.pkl', 'rb') as f:
            roi_model = pickle.load(f)

        with open(load_dir / f'market_confidence_model_{version}.pkl', 'rb') as f:
            confidence_model = pickle.load(f)

        with open(load_dir / f'market_scaler_{version}.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # Load metadata
        with open(load_dir / f'market_model_metadata_{version}.json', 'r') as f:
            metadata = json.load(f)

        model = cls(
            winrate_model=winrate_model,
            roi_model=roi_model,
            confidence_model=confidence_model,
            scaler=scaler
        )
        model.feature_names = metadata['feature_names']
        model.metadata = metadata

        return model


def evaluate_model(
    model: MarketPredictionModel,
    X_test: pd.DataFrame,
    y_winrate_test: pd.Series,
    y_roi_test: pd.Series,
    y_confidence_test: pd.Series
) -> Dict:
    """
    Evaluate model performance on test set.

    Args:
        model: Trained model
        X_test: Test features
        y_winrate_test: True winrates
        y_roi_test: True ROIs
        y_confidence_test: True confidence scores

    Returns:
        Dictionary of evaluation metrics
    """
    preds = model.predict(X_test, return_confidence=True)

    metrics = {
        'winrate': {
            'mae': mean_absolute_error(y_winrate_test, preds['winrate']),
            'rmse': np.sqrt(mean_squared_error(y_winrate_test, preds['winrate'])),
            'r2': r2_score(y_winrate_test, preds['winrate'])
        },
        'roi': {
            'mae': mean_absolute_error(y_roi_test, preds['roi']),
            'rmse': np.sqrt(mean_squared_error(y_roi_test, preds['roi'])),
            'r2': r2_score(y_roi_test, preds['roi'])
        },
        'confidence': {
            'mae': mean_absolute_error(y_confidence_test, preds['confidence']),
            'rmse': np.sqrt(mean_squared_error(y_confidence_test, preds['confidence'])),
            'r2': r2_score(y_confidence_test, preds['confidence'])
        }
    }

    # Calibration analysis (are predicted winrates accurate?)
    pred_winrate_bins = pd.cut(preds['winrate'], bins=10)
    calibration = pd.DataFrame({
        'predicted': preds['winrate'],
        'actual': y_winrate_test
    }).groupby(pred_winrate_bins).agg({
        'predicted': 'mean',
        'actual': 'mean'
    })

    metrics['calibration'] = calibration.to_dict()

    return metrics


def train_with_cross_validation(
    market_features: pd.DataFrame,
    n_splits: int = 5,
    min_samples: int = 20
) -> Tuple[MarketPredictionModel, Dict]:
    """
    Train model with time-series cross-validation.

    Args:
        market_features: Market-level features with targets
        n_splits: Number of CV splits
        min_samples: Minimum samples for validation

    Returns:
        Tuple of (trained_model, cv_results)
    """
    # Filter to validated samples
    train_df = market_features[
        (market_features['validated'] == 1) &
        (market_features['target_winrate'].notna()) &
        (market_features['target_roi'].notna())
    ].copy()

    print(f"Training on {len(train_df)} validated market-days")

    # Define features
    feature_cols = [
        col for col in train_df.columns
        if col.startswith(('rolling_', 'expanding_', 'winrate_', 'roi_',
                          'sample_', 'ci_', 'reliability_', 'consistency_',
                          'is_', 'day_of_week', 'month_', 'in_', 'betting_frequency'))
        and col not in ['target_winrate', 'target_roi', 'target_confidence']
    ]

    print(f"Using {len(feature_cols)} features")

    X = train_df[feature_cols]
    y_winrate = train_df['target_winrate']
    y_roi = train_df['target_roi']
    y_confidence = train_df['target_confidence']

    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_results = {
        'winrate_mae': [],
        'roi_mae': [],
        'confidence_mae': []
    }

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_wr_train, y_wr_val = y_winrate.iloc[train_idx], y_winrate.iloc[val_idx]
        y_roi_train, y_roi_val = y_roi.iloc[train_idx], y_roi.iloc[val_idx]
        y_conf_train, y_conf_val = y_confidence.iloc[train_idx], y_confidence.iloc[val_idx]

        # Train model
        model = MarketPredictionModel()
        model.train(X_train, y_wr_train, y_roi_train, y_conf_train)

        # Evaluate
        preds = model.predict(X_val, return_confidence=True)

        wr_mae = mean_absolute_error(y_wr_val, preds['winrate'])
        roi_mae = mean_absolute_error(y_roi_val, preds['roi'])
        conf_mae = mean_absolute_error(y_conf_val, preds['confidence'])

        cv_results['winrate_mae'].append(wr_mae)
        cv_results['roi_mae'].append(roi_mae)
        cv_results['confidence_mae'].append(conf_mae)

        print(f"  Winrate MAE: {wr_mae:.4f}")
        print(f"  ROI MAE: {roi_mae:.2f}")
        print(f"  Confidence MAE: {conf_mae:.4f}")

    # Print CV summary
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"Winrate MAE: {np.mean(cv_results['winrate_mae']):.4f} ± {np.std(cv_results['winrate_mae']):.4f}")
    print(f"ROI MAE: {np.mean(cv_results['roi_mae']):.2f} ± {np.std(cv_results['roi_mae']):.2f}")
    print(f"Confidence MAE: {np.mean(cv_results['confidence_mae']):.4f} ± {np.std(cv_results['confidence_mae']):.4f}")

    # Train final model on all data
    print("\nTraining final model on all data...")
    final_model = MarketPredictionModel()
    final_model.train(X, y_winrate, y_roi, y_confidence)

    return final_model, cv_results


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("MARKET PERFORMANCE PREDICTION MODEL TRAINING")
    print("=" * 80)

    # Load bet data from Supabase or local file
    # For now, using placeholder - replace with actual data loading
    from dotenv import load_dotenv
    load_dotenv('/root/pikkit/.env')

    import requests

    SUPABASE_URL = os.environ.get('SUPABASE_URL')
    SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_KEY')

    print(f"\nLoading data from Supabase...")

    # Fetch completed bets
    response = requests.get(
        f"{SUPABASE_URL}/rest/v1/completed_bets",
        headers={
            'apikey': SUPABASE_KEY,
            'Authorization': f'Bearer {SUPABASE_KEY}'
        },
        params={
            'select': 'sport,league,market,institution_name,american_odds,is_win,roi,clv_percentage,created_at',
            'order': 'created_at.asc',
            'limit': 10000
        }
    )

    if response.status_code != 200:
        print(f"Error loading data: {response.status_code}")
        print(response.text)
        return

    bets_df = pd.DataFrame(response.json())
    print(f"Loaded {len(bets_df)} bets")

    # Feature engineering
    print("\nCreating market-level features...")
    fe = MarketPerformanceFeatureEngineer(
        lookback_windows=[10, 30, 50, 100],
        min_sample_size=20,
        prediction_horizon_bets=30,
        prediction_horizon_days=30
    )

    market_features = fe.fit_transform(bets_df)

    # Add validation
    market_features = validate_sample_size(market_features, min_samples=20)

    print(f"\nMarket-level features: {len(market_features)} rows")
    print(f"Validated rows: {market_features['validated'].sum()}")

    # Train model
    print("\nTraining model with cross-validation...")
    model, cv_results = train_with_cross_validation(
        market_features,
        n_splits=5,
        min_samples=20
    )

    # Save model
    save_dir = Path('/root/pikkit/ml/models')
    model.save(save_dir)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
