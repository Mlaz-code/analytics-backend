#!/usr/bin/env python3
"""
Quick market feature engineering - optimized for speed
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def create_market_features_fast(df):
    """Fast feature engineering for market predictions"""

    print("Creating market key...")
    df['market_key'] = df['sport'] + '|' + df['league'] + '|' + df['market']
    df = df.sort_values('created_at').reset_index(drop=True)

    all_features = []

    markets = df['market_key'].unique()
    print(f"Processing {len(markets)} markets...")

    for i, mkey in enumerate(markets):
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(markets)}...", end='\r')

        mdf = df[df['market_key'] == mkey].copy()

        if len(mdf) < 30:  # Need at least 30 bets
            continue

        sport, league, market = mkey.split('|')

        # For each bet after the first 20, create features
        for idx in range(20, len(mdf) - 10):  # Leave 10 for target
            current_date = mdf.iloc[idx]['created_at']

            # Historical (before this bet)
            hist = mdf.iloc[:idx]

            # Future (for targets)
            future = mdf.iloc[idx:idx+30]

            if len(future) < 10:
                continue

            features = {
                'market_key': mkey,
                'sport': sport,
                'league': league,
                'market': market,
                'date': current_date,

                # Historical stats
                'hist_bets': len(hist),
                'hist_winrate': hist['is_win'].mean(),
                'hist_roi': hist['roi'].mean(),

                # Last 10 bets
                'last10_winrate': hist.tail(10)['is_win'].mean() if len(hist) >= 10 else hist['is_win'].mean(),
                'last10_roi': hist.tail(10)['roi'].mean() if len(hist) >= 10 else hist['roi'].mean(),

                # Last 20 bets
                'last20_winrate': hist.tail(20)['is_win'].mean() if len(hist) >= 20 else hist['is_win'].mean(),
                'last20_roi': hist.tail(20)['roi'].mean() if len(hist) >= 20 else hist['roi'].mean(),

                # Momentum
                'winrate_momentum': (hist.tail(10)['is_win'].mean() - hist['is_win'].mean()) if len(hist) >= 10 else 0,
                'roi_momentum': (hist.tail(10)['roi'].mean() - hist['roi'].mean()) if len(hist) >= 10 else 0,

                # Variance
                'roi_std': hist['roi'].std() if len(hist) > 1 else 0,

                # Targets (next 30 bets)
                'target_winrate': future.head(30)['is_win'].mean(),
                'target_roi': future.head(30)['roi'].mean(),
                'target_confidence': min(1.0, np.sqrt(len(future.head(30)) / 20))
            }

            all_features.append(features)

    print(f"\nCreated {len(all_features)} samples")
    return pd.DataFrame(all_features)


def main():
    # Load data
    data_dir = Path(__file__).parent.parent / 'data'
    latest = sorted(data_dir.glob('all_bets_*.parquet'))[-1]

    print(f"Loading {latest.name}...")
    df = pd.read_parquet(latest)
    print(f"Loaded {len(df):,} bets\n")

    # Create features
    features_df = create_market_features_fast(df)

    # Train/test split
    cutoff = features_df['date'].max() - timedelta(days=14)
    train = features_df[features_df['date'] < cutoff]
    test = features_df[features_df['date'] >= cutoff]

    print(f"\nTrain: {len(train)} samples")
    print(f"Test: {len(test)} samples")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_file = data_dir / f'market_train_{timestamp}.parquet'
    test_file = data_dir / f'market_test_{timestamp}.parquet'

    train.to_parquet(train_file, index=False)
    test.to_parquet(test_file, index=False)

    print(f"\n✅ Saved:")
    print(f"   {train_file.name}")
    print(f"   {test_file.name}")

    # Stats
    print(f"\n📊 Top markets:")
    for mkey, count in train['market_key'].value_counts().head(10).items():
        sport, league, market = mkey.split('|')
        wr = train[train['market_key'] == mkey]['target_winrate'].mean()
        roi = train[train['market_key'] == mkey]['target_roi'].mean()
        print(f"  {sport:15s} | {league:10s} | {market:20s}: {count:3} ({wr:.1%} win, {roi:+6.2f} ROI)")


if __name__ == '__main__':
    main()
