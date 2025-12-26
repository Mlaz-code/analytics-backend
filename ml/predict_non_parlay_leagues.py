#!/usr/bin/env python3
"""Generate league predictions for non-parlay bets"""

import pandas as pd
from comprehensive_classifier import ComprehensiveClassifier

# Load model
classifier = ComprehensiveClassifier()
classifier.load()

# Load bets needing prediction
df = pd.read_csv('/root/pikkit/ml/data/non_parlay_missing_league.csv')

print(f"Generating predictions for {len(df)} non-parlay bets missing league...")

# Generate predictions
predictions, confidences = classifier.predict(df)

# Create results dataframe
results = pd.DataFrame({
    'id': df['id'].values,
    'bet_type': df['bet_type'].values,
    'original_sport': df['sport'].values,
    'original_market': df['market'].values,
    'pick_name': df['pick_name'].values,
    'predicted_market': predictions['predicted_market'],
    'market_confidence': predictions['market_confidence'],
    'predicted_sport': predictions['predicted_sport'],
    'sport_confidence': predictions['sport_confidence'],
    'predicted_league': predictions['predicted_league'],
    'league_confidence': predictions['league_confidence']
})

# Save to CSV
output_path = '/root/pikkit/ml/data/non_parlay_league_predictions.csv'
results.to_csv(output_path, index=False)

print(f"\nSaved predictions to: {output_path}")
print(f"\nPrediction Summary:")
print(f"  Mean league confidence: {results['league_confidence'].mean():.2%}")
print(f"  Median league confidence: {results['league_confidence'].median():.2%}")
print(f"  >90% confidence: {(results['league_confidence'] > 0.9).sum()} ({(results['league_confidence'] > 0.9).sum()/len(results)*100:.1f}%)")
print(f"  >80% confidence: {(results['league_confidence'] > 0.8).sum()} ({(results['league_confidence'] > 0.8).sum()/len(results)*100:.1f}%)")
print(f"  >70% confidence: {(results['league_confidence'] > 0.7).sum()} ({(results['league_confidence'] > 0.7).sum()/len(results)*100:.1f}%)")

print(f"\nPredicted leagues:")
print(results['predicted_league'].value_counts())

print(f"\nSample predictions (sorted by confidence):")
results_sorted = results.sort_values('league_confidence', ascending=False)
print(results_sorted[['pick_name', 'original_sport', 'predicted_league', 'league_confidence']].head(36).to_string())
