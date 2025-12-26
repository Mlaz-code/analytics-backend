#!/usr/bin/env python3
"""Generate predictions for bets missing market, sport, or league"""

import os
import pandas as pd
from comprehensive_classifier import ComprehensiveClassifier, get_supabase_client
from dotenv import load_dotenv

load_dotenv('/root/pikkit/.env')

# Load model
classifier = ComprehensiveClassifier()
classifier.load()

# Get Supabase client
supabase = get_supabase_client()

print("Fetching bets missing league information...")
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

# Find bets missing league (have market and sport)
missing_league = df[
    df['market'].notna() &
    df['sport'].notna() &
    df['league'].isna()
].copy()

print(f"Bets missing LEAGUE only: {len(missing_league)}")

if len(missing_league) == 0:
    print("No bets need league prediction!")
    exit(0)

# Generate predictions
print(f"\nGenerating predictions for {len(missing_league)} bets...")
predictions, confidences = classifier.predict(missing_league)

# Create results dataframe
results = pd.DataFrame({
    'id': missing_league['id'].values,
    'original_market': missing_league['market'].values,
    'original_sport': missing_league['sport'].values,
    'pick_name': missing_league['pick_name'].values,
    'predicted_market': predictions['predicted_market'],
    'market_confidence': predictions['market_confidence'],
    'predicted_sport': predictions['predicted_sport'],
    'sport_confidence': predictions['sport_confidence'],
    'predicted_league': predictions['predicted_league'],
    'league_confidence': predictions['league_confidence']
})

# Save to CSV
output_path = '/root/pikkit/ml/data/comprehensive_predictions.csv'
results.to_csv(output_path, index=False)

print(f"\nSaved predictions to: {output_path}")
print(f"\nPrediction Summary:")
print(f"  Mean league confidence: {results['league_confidence'].mean():.2%}")
print(f"  Median league confidence: {results['league_confidence'].median():.2%}")
print(f"  >90% confidence: {(results['league_confidence'] > 0.9).sum()} ({(results['league_confidence'] > 0.9).sum()/len(results)*100:.1f}%)")
print(f"  >80% confidence: {(results['league_confidence'] > 0.8).sum()} ({(results['league_confidence'] > 0.8).sum()/len(results)*100:.1f}%)")

print(f"\nTop predicted leagues:")
print(results['predicted_league'].value_counts().head(10))

print(f"\nSample predictions:")
print(results[['original_sport', 'original_market', 'predicted_league', 'league_confidence']].head(20))
