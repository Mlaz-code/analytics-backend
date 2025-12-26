#!/usr/bin/env python3
"""
Materialize Market Performance Features to Feature Store
========================================================
Compute and materialize features for the Feast feature store.

This script:
1. Loads bet data from Supabase
2. Generates market-level features
3. Runs predictions with trained models
4. Writes to parquet files for Feast
5. Materializes to online store (SQLite)

Run schedule: Every 6-12 hours via cron
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import requests
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.market_performance_features import (
    MarketPerformanceFeatureEngineer,
    validate_sample_size
)
from scripts.train_market_prediction_model import MarketPredictionModel

# Load environment
load_dotenv('/root/pikkit/.env')

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_KEY')
FEATURE_STORE_DATA_DIR = Path('/root/pikkit/ml/feature_store/data')
MODEL_DIR = Path('/root/pikkit/ml/models')


def load_bet_data(days_back: int = 180) -> pd.DataFrame:
    """
    Load completed bets from Supabase.

    Args:
        days_back: Number of days of history to load

    Returns:
        DataFrame with bet data
    """
    print(f"Loading bet data from Supabase (last {days_back} days)...")

    cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()

    response = requests.get(
        f"{SUPABASE_URL}/rest/v1/completed_bets",
        headers={
            'apikey': SUPABASE_KEY,
            'Authorization': f'Bearer {SUPABASE_KEY}'
        },
        params={
            'select': 'sport,league,market,institution_name,american_odds,is_win,roi,clv_percentage,created_at',
            'created_at': f'gte.{cutoff_date}',
            'order': 'created_at.asc',
            'limit': 50000
        }
    )

    if response.status_code != 200:
        raise Exception(f"Error loading data: {response.status_code} - {response.text}")

    df = pd.DataFrame(response.json())
    print(f"Loaded {len(df)} bets")

    return df


def generate_market_features(bets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate market-level features from bet data.

    Args:
        bets_df: Bet-level DataFrame

    Returns:
        Market-level features DataFrame
    """
    print("Generating market-level features...")

    fe = MarketPerformanceFeatureEngineer(
        lookback_windows=[10, 30, 50, 100],
        min_sample_size=20,
        prediction_horizon_bets=30,
        prediction_horizon_days=30
    )

    market_features = fe.fit_transform(bets_df)
    market_features = validate_sample_size(market_features, min_samples=20)

    print(f"Generated {len(market_features)} market-date rows")
    print(f"Validated rows: {market_features['validated'].sum()}")

    return market_features


def generate_predictions(market_features: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions using trained model.

    Args:
        market_features: Market-level features

    Returns:
        DataFrame with predictions added
    """
    print("Loading trained model...")

    try:
        model = MarketPredictionModel.load(MODEL_DIR, version='latest')
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Skipping predictions - will use default values")
        return add_default_predictions(market_features)

    print("Generating predictions...")

    # Get latest row per market (most recent features)
    latest_markets = market_features.sort_values('date').groupby('market_key').tail(1)

    # Get feature columns from model
    feature_cols = model.feature_names

    # Filter to available features
    available_features = [col for col in feature_cols if col in latest_markets.columns]
    missing_features = [col for col in feature_cols if col not in latest_markets.columns]

    if missing_features:
        print(f"Warning: Missing {len(missing_features)} features")

    X = latest_markets[available_features]

    # Generate predictions and recommendations
    recommendations = model.predict_recommendation(
        X,
        winrate_threshold=0.53,
        roi_threshold=3.0,
        confidence_threshold=0.6
    )

    # Add metadata
    recommendations['model_version'] = model.metadata.get('trained_at', 'unknown')
    recommendations['prediction_timestamp'] = datetime.now().isoformat()

    # Merge back to market_key
    predictions_df = latest_markets[['market_key', 'date']].copy()
    predictions_df = predictions_df.merge(recommendations, left_index=True, right_index=True)

    print(f"Generated predictions for {len(predictions_df)} markets")

    return predictions_df


def add_default_predictions(market_features: pd.DataFrame) -> pd.DataFrame:
    """Add default predictions when model is unavailable."""
    latest_markets = market_features.sort_values('date').groupby('market_key').tail(1)

    predictions_df = latest_markets[['market_key', 'date']].copy()
    predictions_df['predicted_winrate'] = 0.5
    predictions_df['predicted_roi'] = 0.0
    predictions_df['prediction_confidence'] = 0.0
    predictions_df['should_take'] = 0
    predictions_df['recommendation_score'] = 50.0
    predictions_df['grade'] = 'C'
    predictions_df['model_version'] = 'none'
    predictions_df['prediction_timestamp'] = datetime.now().isoformat()

    return predictions_df


def write_parquet_features(
    market_features: pd.DataFrame,
    predictions_df: pd.DataFrame
):
    """
    Write features to parquet files for Feast.

    Args:
        market_features: Market-level features
        predictions_df: Predictions DataFrame
    """
    FEATURE_STORE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Writing features to parquet files...")

    # Market performance features
    performance_cols = [
        'market_key', 'date',
        'expanding_winrate', 'expanding_roi', 'expanding_bet_count',
        'rolling_10_winrate', 'rolling_10_roi', 'rolling_10_volume', 'rolling_10_clv',
        'rolling_10_winrate_std', 'rolling_10_roi_std',
        'rolling_30_winrate', 'rolling_30_roi', 'rolling_30_volume', 'rolling_30_clv',
        'rolling_30_winrate_std', 'rolling_30_roi_std',
        'rolling_50_winrate', 'rolling_50_roi', 'rolling_50_volume', 'rolling_50_clv',
        'rolling_100_winrate', 'rolling_100_roi', 'rolling_100_volume',
        'winrate_momentum_10_100', 'winrate_momentum_30_exp', 'roi_momentum_10_50',
        'consistency_score', 'sample_adequacy', 'ci_lower', 'ci_upper', 'ci_width',
        'reliability_score', 'is_hot', 'is_cold', 'roi_improving', 'roi_declining',
        'significantly_profitable', 'significantly_unprofitable',
        'market_popularity_norm', 'is_player_prop', 'is_main_market', 'betting_frequency'
    ]

    # Filter to available columns
    available_perf_cols = [col for col in performance_cols if col in market_features.columns]

    performance_df = market_features[available_perf_cols].copy()
    performance_df['event_timestamp'] = pd.to_datetime(market_features['date'])

    performance_df.to_parquet(
        FEATURE_STORE_DATA_DIR / 'market_performance.parquet',
        index=False
    )
    print(f"  Wrote market_performance.parquet ({len(performance_df)} rows)")

    # Predictions features
    prediction_cols = [
        'market_key',
        'predicted_winrate', 'predicted_roi', 'prediction_confidence',
        'should_take', 'recommendation_score', 'grade',
        'model_version', 'prediction_timestamp'
    ]

    # Rename confidence column
    predictions_df['prediction_confidence'] = predictions_df['confidence']

    available_pred_cols = [col for col in prediction_cols if col in predictions_df.columns]

    predictions_output = predictions_df[available_pred_cols].copy()
    predictions_output['event_timestamp'] = pd.to_datetime(predictions_df['date'])

    predictions_output.to_parquet(
        FEATURE_STORE_DATA_DIR / 'market_predictions.parquet',
        index=False
    )
    print(f"  Wrote market_predictions.parquet ({len(predictions_output)} rows)")

    # Sport context features
    context_cols = [
        'market_key', 'date',
        'is_ball_sport', 'is_major_league',
        'in_nfl_season', 'in_nba_season', 'in_mlb_season', 'in_nhl_season',
        'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'is_weekend'
    ]

    available_context_cols = [col for col in context_cols if col in market_features.columns]

    context_df = market_features[available_context_cols].copy()
    context_df['event_timestamp'] = pd.to_datetime(market_features['date'])

    # Get unique market contexts (one per market)
    context_df = context_df.sort_values('date').groupby('market_key').tail(1)

    context_df.to_parquet(
        FEATURE_STORE_DATA_DIR / 'sport_context.parquet',
        index=False
    )
    print(f"  Wrote sport_context.parquet ({len(context_df)} rows)")


def materialize_to_online_store():
    """
    Materialize features to Feast online store.

    Requires Feast to be configured and applied.
    """
    print("\nMaterializing to online store...")

    try:
        import subprocess

        feature_repo_dir = Path('/root/pikkit/ml/feature_store/feature_repo')

        # Change to feature repo directory
        result = subprocess.run(
            ['feast', 'materialize-incremental', datetime.now().isoformat()],
            cwd=feature_repo_dir,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("Successfully materialized to online store")
            print(result.stdout)
        else:
            print(f"Warning: Materialization failed: {result.stderr}")

    except Exception as e:
        print(f"Warning: Could not materialize to online store: {e}")
        print("Run 'cd /root/pikkit/ml/feature_store/feature_repo && feast materialize-incremental <end_date>' manually")


def main():
    """Main materialization pipeline."""
    print("=" * 80)
    print("MATERIALIZING MARKET PERFORMANCE FEATURES")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Load bet data
    bets_df = load_bet_data(days_back=180)

    # Generate features
    market_features = generate_market_features(bets_df)

    # Generate predictions
    predictions_df = generate_predictions(market_features)

    # Write to parquet
    write_parquet_features(market_features, predictions_df)

    # Materialize to online store
    materialize_to_online_store()

    print("\n" + "=" * 80)
    print("MATERIALIZATION COMPLETE")
    print("=" * 80)

    # Summary statistics
    print("\nSummary:")
    print(f"  Total markets: {market_features['market_key'].nunique()}")
    print(f"  Date range: {market_features['date'].min()} to {market_features['date'].max()}")

    if 'should_take' in predictions_df.columns:
        take_count = predictions_df['should_take'].sum()
        print(f"  Markets to TAKE: {take_count} / {len(predictions_df)}")

        if 'grade' in predictions_df.columns:
            grade_counts = predictions_df['grade'].value_counts()
            print("\n  Grade distribution:")
            for grade, count in grade_counts.items():
                print(f"    {grade}: {count}")


if __name__ == '__main__':
    main()
