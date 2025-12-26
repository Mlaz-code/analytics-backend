#!/usr/bin/env python3
"""
Fetch all historical bets from Supabase with pagination
Saves to parquet for efficient analysis
"""

import os
import sys
from pathlib import Path
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / '.env')

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

def fetch_all_bets(page_size=1000):
    """Fetch all settled bets with pagination"""

    headers = {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Prefer': 'count=exact'
    }

    all_bets = []
    offset = 0

    # First request to get total count
    params = {
        'is_settled': 'eq.true',
        'select': 'id,sport,league,market,institution_name,bet_type,american_odds,is_win,roi,clv_percentage,clv_ev,is_live,created_at',
        'limit': '1',
        'order': 'created_at.desc'
    }

    response = requests.get(
        f'{SUPABASE_URL}/rest/v1/bets',
        headers=headers,
        params=params
    )

    # Extract total count from Content-Range header
    content_range = response.headers.get('Content-Range')
    if content_range:
        total_count = int(content_range.split('/')[-1])
        print(f"Total settled bets in database: {total_count:,}")
    else:
        print("Warning: Could not determine total count")
        total_count = None

    # Fetch all pages
    while True:
        params = {
            'is_settled': 'eq.true',
            'select': 'id,sport,league,market,institution_name,bet_type,american_odds,is_win,roi,clv_percentage,clv_ev,is_live,created_at',
            'limit': str(page_size),
            'offset': str(offset),
            'order': 'created_at.desc'
        }

        response = requests.get(
            f'{SUPABASE_URL}/rest/v1/bets',
            headers=headers,
            params=params
        )

        # Accept 200 or 206 (Partial Content)
        if response.status_code not in [200, 206]:
            print(f"\nError {response.status_code}: {response.text[:200]}")
            break

        data = response.json()

        if not data:
            print(f"\nNo more data at offset {offset}")
            break

        all_bets.extend(data)
        offset += page_size

        print(f"Fetched {len(all_bets):,} bets so far...", end='\r')

        # Stop if we got fewer records than page size (last page)
        if len(data) < page_size:
            break

    print(f"\nTotal bets fetched: {len(all_bets):,}")

    # Convert to DataFrame
    if not all_bets:
        print("ERROR: No bets were fetched!")
        sys.exit(1)

    df = pd.DataFrame(all_bets)

    # Convert types
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['is_win'] = df['is_win'].astype(bool)
    df['is_live'] = df['is_live'].fillna(False).astype(bool)

    # Calculate implied probability from odds
    def calc_implied_prob(odds):
        if pd.isna(odds) or odds == 0:
            return 0.5
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)

    df['implied_prob'] = df['american_odds'].apply(calc_implied_prob)

    return df


def analyze_data(df):
    """Print comprehensive data analysis"""

    print("\n" + "="*60)
    print("COMPREHENSIVE BET DATA ANALYSIS")
    print("="*60)

    print(f"\n📊 Dataset Overview:")
    print(f"   Total bets: {len(df):,}")
    print(f"   Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    print(f"   Time span: {(df['created_at'].max() - df['created_at'].min()).days} days")

    print(f"\n🎯 Dimensions:")
    print(f"   Unique sports: {df['sport'].nunique()}")
    print(f"   Unique leagues: {df['league'].nunique()}")
    print(f"   Unique markets: {df['market'].nunique()}")
    print(f"   Unique institutions: {df['institution_name'].nunique()}")

    print(f"\n📈 Overall Performance:")
    print(f"   Win rate: {df['is_win'].mean():.2%}")
    print(f"   Average ROI: {df['roi'].mean():.2f}")
    print(f"   Median ROI: {df['roi'].median():.2f}")
    print(f"   Total profit: {df['roi'].sum():.2f}")

    print(f"\n🏀 Top Sports by Volume:")
    sport_counts = df['sport'].value_counts().head(10)
    for sport, count in sport_counts.items():
        winrate = df[df['sport'] == sport]['is_win'].mean()
        avg_roi = df[df['sport'] == sport]['roi'].mean()
        print(f"   {sport:20s}: {count:5,} bets ({winrate:.1%} win, {avg_roi:+7.2f} ROI)")

    print(f"\n🎲 Top Markets by Volume:")
    market_counts = df['market'].value_counts().head(10)
    for market, count in market_counts.items():
        winrate = df[df['market'] == market]['is_win'].mean()
        avg_roi = df[df['market'] == market]['roi'].mean()
        print(f"   {market:25s}: {count:5,} bets ({winrate:.1%} win, {avg_roi:+7.2f} ROI)")

    print(f"\n🏆 Best Performing Markets (min 50 bets):")
    market_stats = df.groupby('market').agg({
        'is_win': ['count', 'mean'],
        'roi': 'mean'
    })
    market_stats.columns = ['count', 'winrate', 'roi']
    market_stats = market_stats[market_stats['count'] >= 50].sort_values('roi', ascending=False)
    for market, row in market_stats.head(10).iterrows():
        print(f"   {market:25s}: {row['count']:4.0f} bets ({row['winrate']:.1%} win, {row['roi']:+7.2f} ROI)")

    print(f"\n⚡ Market Combinations (Sport+League+Market) with 20+ bets:")
    combo_stats = df.groupby(['sport', 'league', 'market']).agg({
        'is_win': ['count', 'mean'],
        'roi': 'mean'
    })
    combo_stats.columns = ['count', 'winrate', 'roi']
    combo_stats = combo_stats[combo_stats['count'] >= 20].sort_values('count', ascending=False)
    print(f"   Total combinations: {len(combo_stats)}")
    print(f"\n   Top 15 by volume:")
    for idx, row in combo_stats.head(15).iterrows():
        sport, league, market = idx
        print(f"   {sport:15s} | {league:10s} | {market:20s}: {row['count']:4.0f} bets ({row['winrate']:.1%} win, {row['roi']:+7.2f} ROI)")

    return combo_stats


if __name__ == '__main__':
    print("Fetching all bets from Supabase...")

    df = fetch_all_bets(page_size=1000)

    # Analyze
    combo_stats = analyze_data(df)

    # Save to parquet
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'all_bets_{timestamp}.parquet'

    df.to_parquet(output_file, index=False)
    print(f"\n✅ Data saved to: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    # Save market combo stats
    combo_file = output_dir / f'market_combos_{timestamp}.csv'
    combo_stats.to_csv(combo_file)
    print(f"✅ Market combos saved to: {combo_file}")
