#!/usr/bin/env python3
"""Re-fetch all bets from Supabase with proper ordering"""

import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

load_dotenv('/root/pikkit/.env')

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
supabase = create_client(url, key)

print("Fetching ALL bets from Supabase with ORDER BY...")
all_data = []

batch_size = 1000
offset = 0

while True:
    print(f"Fetching batch {offset // batch_size + 1} (offset: {offset})...")

    response = supabase.table('bets').select(
        'id,market,pick_name,sport,league,bet_type,picks_count,is_live'
    ).order('id').range(offset, offset + batch_size - 1).execute()

    if not response.data:
        break

    all_data.extend(response.data)
    print(f"  Fetched {len(response.data)} bets (total: {len(all_data)})")

    if len(response.data) < batch_size:
        break

    offset += batch_size

df = pd.DataFrame(all_data)

print(f"\n{'='*60}")
print(f"Total fetched: {len(df)}")
print(f"Unique IDs: {df['id'].nunique()}")
print(f"Duplicates: {len(df) - df['id'].nunique()}")

# Deduplicate
df = df.drop_duplicates(subset=['id'])

# Check completeness
df_complete = df[
    df['market'].notna() &
    df['sport'].notna() &
    df['league'].notna()
].copy()

print(f"\nComplete bets (all 3 fields): {len(df_complete)}")
print(f"  Sports: {df_complete['sport'].nunique()} unique")
print(f"  Leagues: {df_complete['league'].nunique()} unique")
print(f"  Markets: {df_complete['market'].nunique()} unique")

# Save to CSV
output_path = '/root/pikkit/ml/data/all_bets_training.csv'
df_complete.to_csv(output_path, index=False)
print(f"\nSaved {len(df_complete)} bets to: {output_path}")
