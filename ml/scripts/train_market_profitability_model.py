#!/usr/bin/env python3
"""
Pikkit Market Profitability XGBoost Model
Multi-task model predicting both win probability and expected ROI
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ML imports
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from sklearn.calibration import calibration_curve

# Supabase configuration
SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://mnnjjvbaxzumfcgibtme.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_KEY', '')

# Model paths
MODEL_DIR = '/root/pikkit/ml/models'
PREDICTIONS_DIR = '/root/pikkit/ml/predictions'
DATA_DIR = '/root/pikkit/ml/data'

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_bets_from_supabase(limit=50000) -> pd.DataFrame:
    """Fetch betting data from Supabase"""
    print("📥 Fetching bet data from Supabase...")

    try:
        from supabase import create_client

        if not SUPABASE_KEY:
            raise ValueError("SUPABASE_KEY environment variable not set")

        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

        # Fetch bets with pagination (only settled bets to reduce load)
        all_bets = []
        offset = 0
        batch_size = 1000

        while True:
            try:
                # Only fetch settled bets with necessary fields
                response = (supabase.table('bets')
                    .select('id,sport,league,market,institution_name,bet_type,'
                           'american_odds,amount,is_win,is_live,is_settled,'
                           'clv_percentage,profit,roi,created_at,updated_at')
                    .eq('is_settled', True)
                    .range(offset, offset + batch_size - 1)
                    .execute())

                if not response.data:
                    break

                all_bets.extend(response.data)
                print(f"  Fetched {len(all_bets)} bets...")

                if len(response.data) < batch_size:
                    break

                offset += batch_size

                if len(all_bets) >= limit:
                    break

            except Exception as e:
                print(f"  ⚠️  Fetch error at offset {offset}: {e}")
                if len(all_bets) > 0:
                    print(f"  ✅ Continuing with {len(all_bets)} bets fetched so far")
                    break
                else:
                    raise

        df = pd.DataFrame(all_bets)
        print(f"✅ Loaded {len(df)} bets from Supabase\n")

        return df

    except ImportError:
        print("❌ Supabase library not installed. Install with: pip install supabase")
        print("   Falling back to sample data generation...")
        return generate_sample_data()
    except Exception as e:
        print(f"❌ Error fetching from Supabase: {e}")
        print("   Falling back to sample data generation...")
        return generate_sample_data()


def generate_sample_data(n_samples=5000) -> pd.DataFrame:
    """Generate realistic sample betting data for testing"""
    print(f"🔧 Generating {n_samples} sample bets for testing...")

    np.random.seed(42)

    sports = ['NBA', 'NFL', 'MLB', 'NHL', 'NCAAF', 'NCAAB']
    markets = ['Moneyline', 'Spread', 'Total', 'Player Props']
    bookmakers = ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars', 'PointsBet']
    bet_types = ['Over', 'Under', 'Home', 'Away']

    data = []
    for i in range(n_samples):
        sport = np.random.choice(sports)
        market = np.random.choice(markets)
        bookmaker = np.random.choice(bookmakers)

        # Generate odds (American format)
        odds = np.random.choice([-110, -105, -115, -120, 100, 105, 110, 120, 130, 140, 150])

        # Calculate implied probability
        if odds < 0:
            implied_prob = abs(odds) / (abs(odds) + 100)
        else:
            implied_prob = 100 / (odds + 100)

        # Generate outcome with some skill (not purely random)
        # Better odds should have better win rates
        true_prob = implied_prob + np.random.normal(0, 0.05)
        won = np.random.random() < true_prob

        # Generate CLV (available for ~20% of bets)
        has_clv = np.random.random() < 0.2
        clv_percentage = np.random.normal(0, 3) if has_clv else None

        # Calculate ROI
        wager = 100
        if odds < 0:
            to_win = wager / (abs(odds) / 100)
        else:
            to_win = wager * (odds / 100)

        profit = to_win if won else -wager
        roi = (profit / wager) * 100

        data.append({
            'id': f'bet_{i}',
            'sport': sport,
            'league': sport,  # Simplified
            'market': market,
            'institution_name': bookmaker,
            'bet_type': np.random.choice(bet_types),
            'odds': odds,
            'american_odds': odds,
            'implied_prob': implied_prob,
            'wager': wager,
            'amount': wager,
            'to_win': to_win,
            'result': 'Won' if won else 'Lost',
            'is_settled': True,
            'settled': True,
            'clv_percentage': clv_percentage,
            'is_live': np.random.choice([True, False]),
            'created_at': (datetime.now() - timedelta(days=np.random.randint(1, 365))).isoformat(),
            'settled_at': (datetime.now() - timedelta(days=np.random.randint(0, 364))).isoformat(),
        })

    df = pd.DataFrame(data)
    print(f"✅ Generated {len(df)} sample bets\n")
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering pipeline"""
    print("🔧 Preparing features...")

    # Convert dates
    df['created_at'] = pd.to_datetime(df['created_at'])
    if 'settled_at' in df.columns:
        df['settled_at'] = pd.to_datetime(df['settled_at'])

    # Target variables
    if 'is_win' in df.columns:
        df['won'] = df['is_win'].fillna(False).astype(int)
    elif 'result' in df.columns:
        df['won'] = (df['result'] == 'Won').astype(int)
    else:
        raise ValueError("No result field found (is_win or result)")

    # Handle profit and ROI
    if 'profit' in df.columns and 'roi' in df.columns:
        # Already calculated in Supabase
        df['profit'] = df['profit'].fillna(0)
        df['roi'] = df['roi'].fillna(0)
    else:
        # Calculate from wager/amount and to_win
        wager_col = 'amount' if 'amount' in df.columns else 'wager'
        df['profit'] = df.apply(lambda row:
            (row.get('to_win', 0) if row['won'] == 1 else -row.get(wager_col, 0)), axis=1)
        df['roi'] = (df['profit'] / df[wager_col]) * 100

    # Sort by date for time-based features
    df = df.sort_values('created_at').reset_index(drop=True)

    # === FEATURE ENGINEERING ===

    # 1. Basic features
    # Calculate implied probability from American odds
    odds_col = 'american_odds' if 'american_odds' in df.columns else 'odds'
    if 'implied_prob' not in df.columns:
        df['implied_prob'] = df[odds_col].apply(lambda x:
            abs(x) / (abs(x) + 100) if x < 0 else 100 / (x + 100) if pd.notna(x) else 0.5)

    # 2. CLV features (with imputation for missing values)
    df['has_clv'] = df['clv_percentage'].notna().astype(int)
    df['clv_percentage'] = df['clv_percentage'].fillna(0)
    df['clv_ev'] = df['clv_percentage'] * df['implied_prob']

    # 3. Categorical encoding
    encoders = {}
    for col in ['sport', 'league', 'market', 'institution_name', 'bet_type']:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # 4. Historical performance features (using expanding window to avoid lookahead)
    for group_cols in [
        ['sport'],
        ['sport', 'market'],
        ['sport', 'league'],
        ['sport', 'league', 'market'],
        ['institution_name'],
    ]:
        group_name = '_'.join(group_cols)

        # Expanding window win rate and ROI
        df[f'{group_name}_win_rate'] = (
            df.groupby(group_cols)['won']
            .transform(lambda x: x.shift(1).expanding().mean())
            .fillna(0.5)  # Prior: 50% win rate
        )

        df[f'{group_name}_roi'] = (
            df.groupby(group_cols)['roi']
            .transform(lambda x: x.shift(1).expanding().mean())
            .fillna(0)  # Prior: 0% ROI
        )

        # Sample size (for confidence weighting)
        df[f'{group_name}_count'] = (
            df.groupby(group_cols).cumcount()
        )

    # 5. Recent performance (last 10 bets)
    df['recent_win_rate'] = (
        df.groupby(['sport', 'market'])['won']
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        .fillna(0.5)
    )

    # 6. Temporal features
    df['day_of_week'] = df['created_at'].dt.dayofweek
    df['hour_of_day'] = df['created_at'].dt.hour
    df['days_since_first_bet'] = (df['created_at'] - df['created_at'].min()).dt.days

    # 7. Is live bet
    if 'is_live' in df.columns:
        df['is_live'] = df['is_live'].fillna(False).astype(int)
    else:
        df['is_live'] = 0

    # 8. Market efficiency features (if multiple books available)
    # Simplified: just use institution as proxy for now

    print(f"✅ Feature engineering complete")
    print(f"   Total features: {len([c for c in df.columns if c not in ['id', 'result', 'created_at', 'settled_at']])}")

    return df, encoders


def select_features(df: pd.DataFrame) -> List[str]:
    """Select features for model training"""

    # Core features
    features = [
        # Categorical
        'sport_encoded', 'league_encoded', 'market_encoded',
        'institution_name_encoded', 'bet_type_encoded',

        # Bet characteristics
        'implied_prob', 'is_live',

        # CLV features
        'clv_percentage', 'clv_ev', 'has_clv',

        # Historical performance
        'sport_win_rate', 'sport_roi',
        'sport_market_win_rate', 'sport_market_roi',
        'sport_league_win_rate', 'sport_league_roi',
        'sport_league_market_win_rate', 'sport_league_market_roi',
        'institution_name_win_rate', 'institution_name_roi',

        # Sample sizes (for weighting)
        'sport_market_count', 'institution_name_count',

        # Recent trends
        'recent_win_rate',

        # Temporal
        'day_of_week', 'hour_of_day', 'days_since_first_bet',
    ]

    # Filter to features that exist in the dataframe
    features = [f for f in features if f in df.columns]

    print(f"\n📊 Selected {len(features)} features:")
    for f in features:
        print(f"  - {f}")
    print()

    return features


def train_multi_task_model(
    X_train, y_win_train, y_roi_train,
    X_val, y_win_val, y_roi_val,
    sample_weights_train=None
) -> Tuple[xgb.XGBClassifier, xgb.XGBRegressor]:
    """Train separate XGBoost models for win probability and ROI prediction"""

    print("🤖 Training multi-task XGBoost model...")
    print("=" * 60)

    # Model 1: Win Probability (Classification)
    print("\n📈 Task 1: Win Probability Classifier")
    print("-" * 60)

    win_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        tree_method='hist',
        early_stopping_rounds=20,
    )

    win_model.fit(
        X_train, y_win_train,
        eval_set=[(X_val, y_win_val)],
        sample_weight=sample_weights_train,
        verbose=False
    )

    # Evaluate win probability model
    y_win_pred_train = win_model.predict_proba(X_train)[:, 1]
    y_win_pred_val = win_model.predict_proba(X_val)[:, 1]

    train_acc = accuracy_score(y_win_train, y_win_pred_train > 0.5)
    val_acc = accuracy_score(y_win_val, y_win_pred_val > 0.5)
    train_auc = roc_auc_score(y_win_train, y_win_pred_train)
    val_auc = roc_auc_score(y_win_val, y_win_pred_val)

    print(f"  Train Accuracy: {train_acc:.3f} | AUC: {train_auc:.3f}")
    print(f"  Val Accuracy:   {val_acc:.3f} | AUC: {val_auc:.3f}")

    # Model 2: ROI Prediction (Regression)
    print("\n💰 Task 2: ROI Regressor")
    print("-" * 60)

    roi_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        eval_metric='mae',
        random_state=42,
        tree_method='hist',
        early_stopping_rounds=20,
    )

    roi_model.fit(
        X_train, y_roi_train,
        eval_set=[(X_val, y_roi_val)],
        sample_weight=sample_weights_train,
        verbose=False
    )

    # Evaluate ROI model
    y_roi_pred_train = roi_model.predict(X_train)
    y_roi_pred_val = roi_model.predict(X_val)

    train_mae = mean_absolute_error(y_roi_train, y_roi_pred_train)
    val_mae = mean_absolute_error(y_roi_val, y_roi_pred_val)

    print(f"  Train MAE: {train_mae:.2f}%")
    print(f"  Val MAE:   {val_mae:.2f}%")

    print("\n" + "=" * 60)
    print("✅ Multi-task training complete!\n")

    return win_model, roi_model


def analyze_feature_importance(win_model, roi_model, feature_names):
    """Analyze and display feature importance"""
    print("\n🔍 Feature Importance Analysis")
    print("=" * 60)

    # Win model importance
    win_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': win_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n📊 Top 15 Features for Win Prediction:")
    for idx, row in win_importance.head(15).iterrows():
        print(f"  {row['feature']:35s} {row['importance']:.4f}")

    # ROI model importance
    roi_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': roi_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n💰 Top 15 Features for ROI Prediction:")
    for idx, row in roi_importance.head(15).iterrows():
        print(f"  {row['feature']:35s} {row['importance']:.4f}")

    return win_importance, roi_importance


def generate_market_predictions(df, win_model, roi_model, features, min_bets=30):
    """Generate predictions for market combinations"""
    print(f"\n🎯 Generating market predictions (min {min_bets} bets)...")

    predictions = []

    # Group by sport/league/market
    for (sport, league, market), group in df.groupby(['sport', 'league', 'market']):
        if len(group) < min_bets:
            continue

        # Get latest features for this market
        latest = group.iloc[-1]

        # Prepare features
        X = pd.DataFrame([latest[features]])

        # Predict
        win_prob = win_model.predict_proba(X)[0, 1]
        expected_roi = roi_model.predict(X)[0]

        # Calculate historical stats
        historical_win_rate = group['won'].mean()
        historical_roi = group['roi'].mean()

        predictions.append({
            'sport': sport,
            'league': league,
            'market': market,
            'sample_size': len(group),
            'predicted_win_prob': float(win_prob),
            'predicted_roi': float(expected_roi),
            'historical_win_rate': float(historical_win_rate),
            'historical_roi': float(historical_roi),
            'confidence': min(1.0, len(group) / 100),  # Higher confidence with more samples
        })

    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df.sort_values('predicted_roi', ascending=False)

    print(f"✅ Generated {len(predictions_df)} market predictions\n")

    return predictions_df


def save_models_and_predictions(win_model, roi_model, predictions_df, encoders, features):
    """Save trained models and predictions"""
    import pickle
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save models using pickle (more reliable for sklearn wrappers)
    win_model_path = f"{MODEL_DIR}/win_probability_model_{timestamp}.pkl"
    roi_model_path = f"{MODEL_DIR}/roi_prediction_model_{timestamp}.pkl"

    with open(win_model_path, 'wb') as f:
        pickle.dump(win_model, f)
    with open(roi_model_path, 'wb') as f:
        pickle.dump(roi_model, f)

    print(f"💾 Saved win model: {win_model_path}")
    print(f"💾 Saved ROI model: {roi_model_path}")

    # Save latest models (overwrite)
    with open(f"{MODEL_DIR}/win_probability_model_latest.pkl", 'wb') as f:
        pickle.dump(win_model, f)
    with open(f"{MODEL_DIR}/roi_prediction_model_latest.pkl", 'wb') as f:
        pickle.dump(roi_model, f)

    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'features': features,
        'encoders': {k: list(v.classes_) for k, v in encoders.items()},
    }

    with open(f"{MODEL_DIR}/model_metadata_{timestamp}.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    with open(f"{MODEL_DIR}/model_metadata_latest.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save predictions
    predictions_path = f"{PREDICTIONS_DIR}/market_predictions_{timestamp}.json"
    predictions_df.to_json(predictions_path, orient='records', indent=2)

    predictions_df.to_json(f"{PREDICTIONS_DIR}/market_predictions_latest.json", orient='records', indent=2)

    print(f"💾 Saved predictions: {predictions_path}")
    print(f"\n✅ All artifacts saved successfully!\n")


def main():
    """Main training pipeline"""
    print("\n" + "=" * 60)
    print("🎰 PIKKIT MARKET PROFITABILITY MODEL TRAINING")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Fetch data
    df = fetch_bets_from_supabase()

    # Filter to settled bets only
    if 'is_settled' in df.columns:
        df = df[df['is_settled'] == True].copy()
        print(f"📊 Filtered to {len(df)} settled bets\n")
    elif 'settled' in df.columns:
        df = df[df['settled'] == True].copy()
        print(f"📊 Filtered to {len(df)} settled bets\n")

    # 2. Feature engineering
    df, encoders = prepare_features(df)

    # 3. Select features
    features = select_features(df)

    # 4. Prepare train/val split (time-based)
    print("🔀 Creating time-based train/validation split...")
    df = df.sort_values('created_at')

    # 80/20 split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    print(f"  Train: {len(train_df)} bets ({train_df['created_at'].min()} to {train_df['created_at'].max()})")
    print(f"  Val:   {len(val_df)} bets ({val_df['created_at'].min()} to {val_df['created_at'].max()})")

    # 5. Prepare X and y
    X_train = train_df[features].values
    y_win_train = train_df['won'].values
    y_roi_train = train_df['roi'].values

    X_val = val_df[features].values
    y_win_val = val_df['won'].values
    y_roi_val = val_df['roi'].values

    # Sample weights (more weight to recent bets)
    train_weights = np.exp(np.linspace(-1, 0, len(train_df)))

    print(f"\n✅ Data preparation complete")
    print(f"   Features: {len(features)}")
    print(f"   Train samples: {len(X_train)}")
    print(f"   Val samples: {len(X_val)}\n")

    # 6. Train models
    win_model, roi_model = train_multi_task_model(
        X_train, y_win_train, y_roi_train,
        X_val, y_win_val, y_roi_val,
        sample_weights_train=train_weights
    )

    # 7. Feature importance
    win_importance, roi_importance = analyze_feature_importance(win_model, roi_model, features)

    # 8. Generate predictions
    predictions_df = generate_market_predictions(df, win_model, roi_model, features)

    # Display top predictions
    print("🏆 Top 10 Most Profitable Markets:")
    print("-" * 60)
    for idx, row in predictions_df.head(10).iterrows():
        print(f"  {row['sport']:6s} | {row['market']:15s} | "
              f"ROI: {row['predicted_roi']:+6.2f}% | "
              f"Win: {row['predicted_win_prob']:.1%} | "
              f"n={row['sample_size']:4d}")

    # 9. Save everything
    save_models_and_predictions(win_model, roi_model, predictions_df, encoders, features)

    print("=" * 60)
    print(f"✅ Training complete! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
