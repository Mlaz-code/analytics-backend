#!/usr/bin/env python3
"""Find non-parlay bets that are missing league"""

import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

load_dotenv('/root/pikkit/.env')

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
supabase = create_client(url, key)

print("Fetching all bets...")
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

print(f"\nTotal unique bets: {len(df)}")

# Find bets missing league
missing_league = df[df['league'].isna()].copy()
print(f"Bets missing league: {len(missing_league)}")

# Filter to NON-parlay bets only
non_parlay_missing_league = missing_league[missing_league['bet_type'] != 'parlay'].copy()

print(f"\nNon-parlay bets missing league: {len(non_parlay_missing_league)}")

if len(non_parlay_missing_league) > 0:
    print(f"\nBreakdown by bet_type:")
    print(non_parlay_missing_league['bet_type'].value_counts())

    print(f"\nBreakdown by sport:")
    print(non_parlay_missing_league['sport'].value_counts())

    print(f"\nBreakdown by market:")
    print(non_parlay_missing_league['market'].value_counts())

    print(f"\nSample of non-parlay bets needing league prediction:")
    print(non_parlay_missing_league[['id', 'bet_type', 'sport', 'market', 'pick_name']].head(20))

    # Save to CSV for prediction
    output_path = '/root/pikkit/ml/data/non_parlay_missing_league.csv'
    non_parlay_missing_league.to_csv(output_path, index=False)
    print(f"\nSaved {len(non_parlay_missing_league)} bets to: {output_path}")
else:
    print("\nAll non-parlay bets already have league information!")
    print("The only bets missing league are parlays, which correctly have 'Parlay' as the league value.")
