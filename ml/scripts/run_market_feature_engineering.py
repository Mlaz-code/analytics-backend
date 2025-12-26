#!/usr/bin/env python3
"""
Run market feature engineering on full dataset
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from market_performance_features import (
    MarketPerformanceFeatureEngineer,
    validate_sample_size
)

def main():
    print("=" * 80)
    print("MARKET PERFORMANCE FEATURE ENGINEERING - FULL DATASET")
    print("=" * 80)

    # Load most recent bet data
    data_dir = Path(__file__).parent.parent / 'data'
    parquet_files = sorted(data_dir.glob('all_bets_*.parquet'))

    if not parquet_files:
        print("ERROR: No bet data found. Run fetch_all_bets.py first.")
        sys.exit(1)

    latest_file = parquet_files[-1]
    print(f"\nLoading: {latest_file.name}")

    df = pd.read_parquet(latest_file)
    print(f"Loaded {len(df):,} bets")
    print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")

    # Create feature engineer
    fe = MarketPerformanceFeatureEngineer(
        lookback_windows=[10, 30, 50, 100],
        min_sample_size=20,
        prediction_horizon_bets=30,
        prediction_horizon_days=30
    )

    print("\nProcessing features...")
    print("  1. Aggregating to market-level...")
    print("  2. Creating target variables...")
    print("  3. Generating historical features...")
    print("  4. Computing trend indicators...")
    print("  5. Adding statistical features...")
    print("  6. Creating context features...")
    print("  7. Adding seasonal features...")

    market_features = fe.fit_transform(df)

    print(f"\n✅ Feature engineering complete!")
    print(f"   Market-level rows: {len(market_features):,}")
    print(f"   Total columns: {len(market_features.columns)}")

    # Add validation
    print("\nValidating sample sizes...")
    market_features = validate_sample_size(market_features, min_samples=20)

    print("\nSample adequacy:")
    print(f"  Sufficient sample (>20 bets): {market_features['sufficient_sample'].sum():,} / {len(market_features):,}")
    print(f"  Adequate statistical power: {market_features['adequate_power'].sum():,} / {len(market_features):,}")
    print(f"  Fully validated: {market_features['validated'].sum():,} / {len(market_features):,}")

    # Filter to validated samples for training
    valid_df = market_features[market_features['validated'] == 1].copy()
    print(f"\n✅ Validated training samples: {len(valid_df):,}")

    # Remove rows with missing targets
    valid_df = valid_df.dropna(subset=['target_winrate', 'target_roi'])
    print(f"   After removing missing targets: {len(valid_df):,}")

    # Time-based train/test split (last 14 days for test)
    max_date = valid_df['date'].max()
    test_cutoff = max_date - pd.Timedelta(days=14)

    train_df = valid_df[valid_df['date'] < test_cutoff].copy()
    test_df = valid_df[valid_df['date'] >= test_cutoff].copy()

    print(f"\n📊 Train/Test Split:")
    print(f"   Train: {len(train_df):,} samples ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"   Test:  {len(test_df):,} samples ({test_df['date'].min()} to {test_df['date'].max()})")

    # Save processed features
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    train_file = data_dir / f'market_features_train_{timestamp}.parquet'
    test_file = data_dir / f'market_features_test_{timestamp}.parquet'
    full_file = data_dir / f'market_features_full_{timestamp}.parquet'

    train_df.to_parquet(train_file, index=False)
    test_df.to_parquet(test_file, index=False)
    market_features.to_parquet(full_file, index=False)

    print(f"\n✅ Features saved:")
    print(f"   Training: {train_file.name} ({train_file.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"   Test: {test_file.name} ({test_file.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"   Full: {full_file.name} ({full_file.stat().st_size / 1024 / 1024:.2f} MB)")

    # Feature summary
    print(f"\n📈 Top Markets by Training Samples:")
    market_counts = train_df['market_key'].value_counts().head(15)
    for market_key, count in market_counts.items():
        sport, league, market = market_key.split('|')
        avg_winrate = train_df[train_df['market_key'] == market_key]['target_winrate'].mean()
        avg_roi = train_df[train_df['market_key'] == market_key]['target_roi'].mean()
        print(f"   {sport:15s} | {league:10s} | {market:20s}: {count:4} ({avg_winrate:.1%} win, {avg_roi:+6.2f} ROI)")

    print("\n" + "=" * 80)
    print("Ready for model training!")
    print("=" * 80)


if __name__ == '__main__':
    main()
