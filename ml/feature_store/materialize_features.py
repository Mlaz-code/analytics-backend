#!/usr/bin/env python3
"""
Materialize Features from Supabase to Feast Feature Store

This script:
1. Fetches historical betting data from Supabase
2. Computes feature aggregations
3. Writes to offline store (parquet files)
4. Materializes to online store for serving
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from feast import FeatureStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://mnnjjvbaxzumfcgibtme.supabase.co')
SUPABASE_SERVICE_KEY = os.environ.get('SUPABASE_SERVICE_KEY', '')
FEATURE_REPO_PATH = '/root/pikkit/ml/feature_store/feature_repo'
FEATURE_DATA_PATH = '/root/pikkit/ml/feature_store/data'

# Ensure directories exist
Path(FEATURE_DATA_PATH).mkdir(parents=True, exist_ok=True)


def fetch_betting_data(days: int = 90) -> pd.DataFrame:
    """
    Fetch settled betting data from Supabase.

    Args:
        days: Number of days to look back

    Returns:
        DataFrame with betting data
    """
    logger.info(f"Fetching {days} days of betting data from Supabase...")

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        headers = {
            'apikey': SUPABASE_SERVICE_KEY,
            'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
            'Content-Type': 'application/json'
        }

        params = {
            'is_settled': 'eq.true',
            'created_at': f'gte.{start_date.isoformat()}',
            'select': 'id,sport,league,market,institution_name,is_win,roi,american_odds,clv_percentage,created_at',
            'limit': '50000',
            'order': 'created_at.desc'
        }

        url = f"{SUPABASE_URL}/rest/v1/bets"
        response = requests.get(url, headers=headers, params=params, timeout=60)
        response.raise_for_status()

        data = response.json()
        df = pd.DataFrame(data)

        logger.info(f"Fetched {len(df)} settled bets")

        # Parse timestamps
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['event_timestamp'] = df['created_at']

        return df

    except Exception as e:
        logger.error(f"Failed to fetch betting data: {e}")
        return pd.DataFrame()


def compute_historical_performance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sport + market level performance features.
    """
    logger.info("Computing historical performance features...")

    # Create sport_market_key
    df['sport_market_key'] = df['sport'] + '_' + df['market']

    # Group by sport_market_key
    grouped = df.groupby('sport_market_key')

    features = grouped.agg({
        'is_win': ['mean', 'count'],
        'roi': 'mean',
        'clv_percentage': 'mean'
    }).reset_index()

    # Flatten column names
    features.columns = ['_'.join(col).strip('_') for col in features.columns.values]
    features.columns = ['sport_market_key', 'sport_market_win_rate', 'sport_market_count',
                        'sport_market_roi', 'avg_clv']

    # Add sport-level aggregations
    df['sport_win_rate'] = df.groupby('sport')['is_win'].transform('mean')
    df['sport_roi'] = df.groupby('sport')['roi'].transform('mean')

    sport_features = df.groupby('sport_market_key')[['sport_win_rate', 'sport_roi']].first().reset_index()
    features = features.merge(sport_features, on='sport_market_key', how='left')

    # Compute recent performance (last 10 bets per sport_market)
    df_sorted = df.sort_values('created_at')
    recent = df_sorted.groupby('sport_market_key').tail(10)
    recent_agg = recent.groupby('sport_market_key').agg({
        'is_win': 'mean',
        'roi': 'mean'
    }).reset_index()
    recent_agg.columns = ['sport_market_key', 'recent_win_rate_10', 'recent_roi_10']

    features = features.merge(recent_agg, on='sport_market_key', how='left')

    # Add timestamp
    features['event_timestamp'] = datetime.now()

    # Fill NaN values
    features = features.fillna(0)

    logger.info(f"Computed features for {len(features)} sport-market combinations")

    return features


def compute_institution_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sportsbook institution-level features.
    """
    logger.info("Computing institution features...")

    grouped = df.groupby('institution_name')

    features = grouped.agg({
        'is_win': 'mean',
        'roi': 'mean',
        'american_odds': 'mean',
        'clv_percentage': 'mean',
        'id': 'count'
    }).reset_index()

    features.columns = [
        'institution_name',
        'institution_win_rate',
        'institution_roi',
        'institution_avg_odds',
        'institution_avg_clv',
        'institution_count'
    ]

    # Compute "sharp percentage" (bets with positive CLV)
    sharp_pct = df[df['clv_percentage'] > 0].groupby('institution_name').size() / df.groupby('institution_name').size()
    features['institution_sharp_pct'] = features['institution_name'].map(sharp_pct).fillna(0)

    # Add timestamp
    features['event_timestamp'] = datetime.now()

    logger.info(f"Computed features for {len(features)} institutions")

    return features


def compute_league_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute league-level features.
    """
    logger.info("Computing league features...")

    # Create sport_league_key
    df['sport_league_key'] = df['sport'] + '_' + df['league']

    grouped = df.groupby('sport_league_key')

    features = grouped.agg({
        'is_win': 'mean',
        'roi': 'mean',
        'clv_percentage': 'mean',
        'id': 'count'
    }).reset_index()

    features.columns = [
        'sport_league_key',
        'league_win_rate',
        'league_roi',
        'league_avg_edge',
        'league_total_bets'
    ]

    # Recent performance (last 50 bets per league)
    df_sorted = df.sort_values('created_at')
    recent = df_sorted.groupby('sport_league_key').tail(50)
    recent_perf = recent.groupby('sport_league_key')['roi'].mean().reset_index()
    recent_perf.columns = ['sport_league_key', 'league_recent_performance']

    features = features.merge(recent_perf, on='sport_league_key', how='left')

    # Add timestamp
    features['event_timestamp'] = datetime.now()

    logger.info(f"Computed features for {len(features)} leagues")

    return features


def save_to_offline_store(features_df: pd.DataFrame, filename: str) -> None:
    """Save features to offline store (parquet file)"""
    output_path = Path(FEATURE_DATA_PATH) / filename
    features_df.to_parquet(output_path, index=False)
    logger.info(f"Saved features to: {output_path}")


def materialize_to_online_store():
    """Materialize features from offline to online store"""
    logger.info("Materializing features to online store...")

    try:
        # Initialize feature store
        fs = FeatureStore(repo_path=FEATURE_REPO_PATH)

        # Materialize features (last 7 days to online store)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        fs.materialize(start_date=start_date, end_date=end_date)

        logger.info("Features materialized to online store successfully")

    except Exception as e:
        logger.error(f"Failed to materialize features: {e}")
        logger.info("Make sure to run 'feast apply' first in the feature_repo directory")


def main():
    logger.info("=" * 60)
    logger.info("FEATURE MATERIALIZATION STARTED")
    logger.info("=" * 60)

    # Fetch betting data
    df = fetch_betting_data(days=90)

    if df.empty:
        logger.error("No data retrieved, cannot compute features")
        return False

    # Compute features
    hist_perf = compute_historical_performance_features(df)
    inst_features = compute_institution_features(df)
    league_features = compute_league_features(df)

    # Save to offline store
    save_to_offline_store(hist_perf, 'historical_performance.parquet')
    save_to_offline_store(inst_features, 'institution_features.parquet')
    save_to_offline_store(league_features, 'league_features.parquet')

    # Materialize to online store
    materialize_to_online_store()

    logger.info("=" * 60)
    logger.info("FEATURE MATERIALIZATION COMPLETED")
    logger.info("=" * 60)

    return True


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Feature materialization failed: {e}", exc_info=True)
        sys.exit(2)
