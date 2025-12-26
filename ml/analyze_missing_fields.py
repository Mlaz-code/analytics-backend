#!/usr/bin/env python3
"""Analyze the ~4000 bets missing fields"""

import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

load_dotenv('/root/pikkit/.env')

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
supabase = create_client(url, key)

print("Fetching ALL bets to analyze missing fields...")
all_data = []

batch_size = 1000
offset = 0

while True:
    response = supabase.table('bets').select(
        'id,market,pick_name,sport,league,bet_type,picks_count,is_live'
    ).order('id').range(offset, offset + batch_size - 1).execute()

    if not response.data:
        break

    all_data.extend(response.data)

    if len(response.data) < batch_size:
        break

    offset += batch_size

df = pd.DataFrame(all_data)

print(f"\n{'='*60}")
print(f"Total bets: {len(df)}")

# Analyze missing fields
df['has_market'] = df['market'].notna()
df['has_sport'] = df['sport'].notna()
df['has_league'] = df['league'].notna()

complete = df[df['has_market'] & df['has_sport'] & df['has_league']]
incomplete = df[~(df['has_market'] & df['has_sport'] & df['has_league'])]

print(f"Complete bets (all 3 fields): {len(complete)}")
print(f"Incomplete bets: {len(incomplete)}")
print(f"\n{'='*60}")
print("Breakdown of missing fields:")
print(f"{'='*60}")

# Missing market only
missing_market_only = df[~df['has_market'] & df['has_sport'] & df['has_league']]
print(f"Missing MARKET only: {len(missing_market_only)}")

# Missing sport only
missing_sport_only = df[df['has_market'] & ~df['has_sport'] & df['has_league']]
print(f"Missing SPORT only: {len(missing_sport_only)}")

# Missing league only
missing_league_only = df[df['has_market'] & df['has_sport'] & ~df['has_league']]
print(f"Missing LEAGUE only: {len(missing_league_only)}")

# Missing multiple fields
missing_market_sport = df[~df['has_market'] & ~df['has_sport'] & df['has_league']]
print(f"Missing MARKET + SPORT: {len(missing_market_sport)}")

missing_market_league = df[~df['has_market'] & df['has_sport'] & ~df['has_league']]
print(f"Missing MARKET + LEAGUE: {len(missing_market_league)}")

missing_sport_league = df[df['has_market'] & ~df['has_sport'] & ~df['has_league']]
print(f"Missing SPORT + LEAGUE: {len(missing_sport_league)}")

missing_all = df[~df['has_market'] & ~df['has_sport'] & ~df['has_league']]
print(f"Missing ALL 3 fields: {len(missing_all)}")

print(f"\n{'='*60}")
print("Sample of incomplete bets:")
print(f"{'='*60}")
print(incomplete[['id', 'market', 'sport', 'league', 'pick_name']].head(20))
