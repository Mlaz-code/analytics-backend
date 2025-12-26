#!/usr/bin/env python3
"""
Create Baseline Reference Data for Drift Detection

This script fetches historical betting data and saves it as the reference
baseline for drift detection. Run this once to establish the baseline.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://mnnjjvbaxzumfcgibtme.supabase.co')
SUPABASE_SERVICE_KEY = os.environ.get('SUPABASE_SERVICE_KEY', '')
BASELINE_DATA_PATH = '/root/pikkit/ml/data/baseline_reference.parquet'


def fetch_baseline_data(days: int = 30, min_samples: int = 5000) -> pd.DataFrame:
    """
    Fetch historical betting data to use as baseline reference.

    Args:
        days: Number of days of historical data to fetch
        min_samples: Minimum number of samples required

    Returns:
        DataFrame with historical bet data
    """
    logger.info(f"Fetching {days} days of historical data for baseline")

    try:
        # Calculate date range (ending 7 days ago to exclude recent data)
        end_date = datetime.now() - timedelta(days=7)
        start_date = end_date - timedelta(days=days)

        # Construct Supabase query
        headers = {
            'apikey': SUPABASE_SERVICE_KEY,
            'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
            'Content-Type': 'application/json'
        }

        # Query for settled bets in date range
        url = f"{SUPABASE_URL}/rest/v1/bets"

        # Use only gte filter for now (Supabase filtering limitation)
        params = {
            'is_settled': 'eq.true',
            'created_at': f'gte.{start_date.isoformat()}',
            'select': 'sport,league,market,institution_name,bet_type,american_odds,clv_percentage,clv_ev,is_live,created_at',
            'limit': '50000',
            'order': 'created_at.desc'
        }

        response = requests.get(url, headers=headers, params=params, timeout=60)
        response.raise_for_status()

        data = response.json()
        df = pd.DataFrame(data)

        # Calculate implied probability from american_odds
        if 'american_odds' in df.columns:
            def calc_implied_prob(odds):
                if pd.isna(odds) or odds == 0:
                    return 0.5
                if odds < 0:
                    return abs(odds) / (abs(odds) + 100)
                else:
                    return 100 / (odds + 100)

            df['implied_prob'] = df['american_odds'].apply(calc_implied_prob)

        logger.info(f"Fetched {len(df)} historical bets")

        if len(df) < min_samples:
            logger.warning(f"Insufficient samples ({len(df)}), need at least {min_samples}")
            logger.warning("Consider increasing the days parameter or checking data availability")

        return df

    except Exception as e:
        logger.error(f"Failed to fetch baseline data: {e}")
        return pd.DataFrame()


def save_baseline(df: pd.DataFrame) -> bool:
    """Save baseline data to parquet file"""
    try:
        # Ensure directory exists
        Path(BASELINE_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)

        # Save to parquet
        df.to_parquet(BASELINE_DATA_PATH, index=False)
        logger.info(f"Baseline data saved to: {BASELINE_DATA_PATH}")
        logger.info(f"Total samples: {len(df)}")

        # Print summary statistics
        logger.info("\nBaseline Data Summary:")
        logger.info(f"  Date range: {df['created_at'].min()} to {df['created_at'].max()}")
        logger.info(f"  Sports: {df['sport'].nunique()} unique ({', '.join(df['sport'].value_counts().head(5).index)})")
        logger.info(f"  Institutions: {df['institution_name'].nunique()} unique")
        logger.info(f"  Markets: {df['market'].nunique()} unique")

        return True

    except Exception as e:
        logger.error(f"Failed to save baseline data: {e}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("CREATING BASELINE REFERENCE DATA")
    logger.info("=" * 60)

    # Fetch baseline data (30 days of historical data)
    df = fetch_baseline_data(days=30, min_samples=5000)

    if df.empty:
        logger.error("No data retrieved, cannot create baseline")
        return False

    # Save baseline
    success = save_baseline(df)

    if success:
        logger.info("=" * 60)
        logger.info("BASELINE CREATION SUCCESSFUL")
        logger.info("=" * 60)
        logger.info("You can now run drift detection with:")
        logger.info("  python3 /root/pikkit/ml/scripts/check_drift_and_retrain.py")
    else:
        logger.error("Failed to create baseline reference data")

    return success


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Baseline creation failed: {e}", exc_info=True)
        sys.exit(2)
