#!/usr/bin/env python3
"""Check what data exists for bets missing league"""

import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

load_dotenv('/root/pikkit/.env')

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
supabase = create_client(url, key)

print("Fetching bets missing league...")
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
df = df.drop_duplicates(subset=['id'])

# Find bets missing league
missing_league = df[df['league'].isna()].copy()

print(f"\nBets missing league: {len(missing_league)}")
print(f"\nField population for these bets:")
print(f"  Have market: {missing_league['market'].notna().sum()}")
print(f"  Have sport: {missing_league['sport'].notna().sum()}")
print(f"  Have pick_name: {missing_league['pick_name'].notna().sum()}")

# Show breakdown
print(f"\nBreakdown:")
both_fields = missing_league[missing_league['market'].notna() & missing_league['sport'].notna()]
print(f"  Have both market AND sport: {len(both_fields)}")

only_market = missing_league[missing_league['market'].notna() & missing_league['sport'].isna()]
print(f"  Have market but NOT sport: {len(only_market)}")

only_sport = missing_league[missing_league['market'].isna() & missing_league['sport'].notna()]
print(f"  Have sport but NOT market: {len(only_sport)}")

neither = missing_league[missing_league['market'].isna() & missing_league['sport'].isna()]
print(f"  Have NEITHER market nor sport: {len(neither)}")

print(f"\nSample of bets with market AND sport (good for prediction):")
if len(both_fields) > 0:
    print(both_fields[['id', 'market', 'sport', 'pick_name']].head(20))
else:
    print("  None found!")

print(f"\nSample of bets with NEITHER market nor sport (poor for prediction):")
if len(neither) > 0:
    print(neither[['id', 'market', 'sport', 'pick_name']].head(20))
