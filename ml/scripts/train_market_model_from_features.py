#!/usr/bin/env python3
"""
Train market prediction model from prepared features
Outputs 0-100 recommendation score like existing XGBoost model
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class MarketPredictor:
    """
    Market performance predictor with 0-100 scoring
    """

    def __init__(self):
        self.winrate_model = None
        self.roi_model = None
        self.confidence_model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.metadata = {}

    def train(self, train_df):
        """Train all three models"""

        # Define features
        self.feature_cols = [
            'hist_bets', 'hist_winrate', 'hist_roi',
            'last10_winrate', 'last10_roi',
            'last20_winrate', 'last20_roi',
            'winrate_momentum', 'roi_momentum', 'roi_std'
        ]

        # Clean data - remove NaN/Inf values
        clean_df = train_df.copy()
        clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
        clean_df = clean_df.dropna(subset=self.feature_cols + ['target_winrate', 'target_roi', 'target_confidence'])

        print(f"Original samples: {len(train_df)}")
        print(f"After cleaning: {len(clean_df)}")

        X = clean_df[self.feature_cols]
        y_winrate = clean_df['target_winrate']
        y_roi = clean_df['target_roi']
        y_confidence = clean_df['target_confidence']

        print(f"Training on {len(X)} samples with {len(self.feature_cols)} features")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train winrate model
        print("Training winrate model...")
        self.winrate_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=6,
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.winrate_model.fit(X_scaled, y_winrate)

        # Train ROI model
        print("Training ROI model...")
        self.roi_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=5,
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.roi_model.fit(X_scaled, y_roi)

        # Train confidence model
        print("Training confidence model...")
        self.confidence_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=4,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42
        )
        self.confidence_model.fit(X_scaled, y_confidence)

        self.metadata = {
            'trained_at': datetime.now().isoformat(),
            'n_features': len(self.feature_cols),
            'feature_names': self.feature_cols,
            'training_samples': len(X)
        }

        print("✅ Training complete!")

    def predict(self, X):
        """Make predictions with 0-100 scoring"""

        X_scaled = self.scaler.transform(X[self.feature_cols])

        # Predictions
        pred_winrate = np.clip(self.winrate_model.predict(X_scaled), 0, 1)
        pred_roi = self.roi_model.predict(X_scaled)
        pred_confidence = np.clip(self.confidence_model.predict(X_scaled), 0, 1)

        # Calculate 0-100 recommendation score (same as existing model)
        winrate_score = (pred_winrate - 0.5) * 100  # Normalize around 50%
        roi_score = pred_roi / 5  # Normalize ROI
        confidence_score = pred_confidence * 50

        recommendation_score = np.clip(
            winrate_score + roi_score + confidence_score,
            0, 100
        )

        # Letter grade
        grades = pd.cut(
            recommendation_score,
            bins=[0, 40, 60, 75, 85, 100],
            labels=['F', 'D', 'C', 'B', 'A']
        )

        # Should take decision
        should_take = (
            (pred_winrate >= 0.53) &
            (pred_roi >= 3.0) &
            (pred_confidence >= 0.6)
        ).astype(int)

        return {
            'predicted_winrate': pred_winrate,
            'predicted_roi': pred_roi,
            'confidence': pred_confidence,
            'recommendation_score': recommendation_score,
            'grade': grades,
            'should_take': should_take
        }

    def save(self, save_dir):
        """Save models"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save models
        with open(save_dir / f'market_winrate_{timestamp}.pkl', 'wb') as f:
            pickle.dump(self.winrate_model, f)

        with open(save_dir / f'market_roi_{timestamp}.pkl', 'wb') as f:
            pickle.dump(self.roi_model, f)

        with open(save_dir / f'market_confidence_{timestamp}.pkl', 'wb') as f:
            pickle.dump(self.confidence_model, f)

        with open(save_dir / f'market_scaler_{timestamp}.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(save_dir / f'market_metadata_{timestamp}.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # Create latest symlinks
        import os
        for name in ['winrate', 'roi', 'confidence', 'scaler']:
            src = f'market_{name}_{timestamp}.pkl'
            dst = save_dir / f'market_{name}_latest.pkl'
            if dst.exists():
                dst.unlink()
            os.symlink(src, dst)

        src = f'market_metadata_{timestamp}.json'
        dst = save_dir / 'market_metadata_latest.json'
        if dst.exists():
            dst.unlink()
        os.symlink(src, dst)

        print(f"\n✅ Models saved to {save_dir}")
        print(f"   Version: {timestamp}")

        return timestamp


def main():
    print("=" * 80)
    print("MARKET PREDICTION MODEL TRAINING")
    print("=" * 80)

    # Load latest features
    data_dir = Path('/root/pikkit/ml/data')
    train_files = sorted(data_dir.glob('market_train_*.parquet'))
    test_files = sorted(data_dir.glob('market_test_*.parquet'))

    if not train_files:
        print("ERROR: No training data found!")
        sys.exit(1)

    train_file = train_files[-1]
    test_file = test_files[-1]

    print(f"\nLoading: {train_file.name}")
    train_df = pd.read_parquet(train_file)
    test_df = pd.read_parquet(test_file)

    print(f"Train: {len(train_df):,} samples")
    print(f"Test: {len(test_df):,} samples")

    # Train model
    print("\nTraining model...")
    model = MarketPredictor()
    model.train(train_df)

    # Evaluate on test set
    print("\nEvaluating on test set...")

    # Clean test data
    test_clean = test_df.replace([np.inf, -np.inf], np.nan)
    test_clean = test_clean.dropna(subset=model.feature_cols + ['target_winrate', 'target_roi', 'target_confidence'])

    print(f"Test samples after cleaning: {len(test_clean)}")

    if len(test_clean) == 0:
        print("WARNING: No clean test samples available. Skipping evaluation.")
        test_preds = None
        wr_mae = roi_mae = conf_mae = 0
    else:
        test_preds = model.predict(test_clean)

        wr_mae = mean_absolute_error(test_clean['target_winrate'], test_preds['predicted_winrate'])
        roi_mae = mean_absolute_error(test_clean['target_roi'], test_preds['predicted_roi'])
        conf_mae = mean_absolute_error(test_clean['target_confidence'], test_preds['confidence'])

    print(f"\n📊 Test Set Performance:")
    print(f"   Winrate MAE: {wr_mae:.4f} ({wr_mae*100:.2f}%)")
    print(f"   ROI MAE: {roi_mae:.2f}")
    print(f"   Confidence MAE: {conf_mae:.4f}")

    # Show sample predictions
    print(f"\n📈 Sample Predictions:")
    if test_preds is None or len(test_clean) == 0:
        print("  (No test samples available)")
    else:
        sample_df = test_clean.head(10).copy()
        sample_preds = model.predict(sample_df)

        for i in range(min(10, len(sample_df))):
            print(f"\n  Market: {sample_df.iloc[i]['market_key']}")
            print(f"    Predicted: {sample_preds['predicted_winrate'][i]:.1%} win, {sample_preds['predicted_roi'][i]:+.1f} ROI")
            print(f"    Actual: {sample_df.iloc[i]['target_winrate']:.1%} win, {sample_df.iloc[i]['target_roi']:+.1f} ROI")
            print(f"    Score: {sample_preds['recommendation_score'][i]:.0f}/100 (Grade {sample_preds['grade'][i]})")
            print(f"    Recommendation: {'TAKE' if sample_preds['should_take'][i] else 'SKIP'}")

    # Save model
    save_dir = Path('/root/pikkit/ml/models')
    version = model.save(save_dir)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModel ready for deployment!")
    print(f"To use: load market_*_latest.pkl files from {save_dir}")


if __name__ == '__main__':
    main()
