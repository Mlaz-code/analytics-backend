#!/usr/bin/env python3
"""
Generate granular market predictions for unlabeled/misclassified bets
Does NOT update the database - only generates predictions for review
"""

import os
import pandas as pd
from granular_market_classifier import GranularMarketClassifier, get_supabase_client
from dotenv import load_dotenv


def fetch_candidates_for_prediction(supabase):
    """Fetch bets that might need market correction"""

    print("Fetching bets for market prediction...")
    all_data = []

    # Strategy 1: Fetch bets with NULL or empty market
    print("Fetching bets with NULL/empty market...")
    response = supabase.table('bets').select(
        'id,market,pick_name,sport,league,bet_type,picks_count,is_live'
    ).is_('market', 'null').limit(1000).execute()
    all_data.extend(response.data)
    print(f"  Found {len(response.data)} bets with NULL market")

    # Strategy 2: Fetch bets with generic market names that could be more specific
    generic_markets = ['Other', 'Parlay', 'Game Props', 'Player Props', 'Other Player Props']
    for market_name in generic_markets:
        print(f"Fetching '{market_name}' bets...")
        response = supabase.table('bets').select(
            'id,market,pick_name,sport,league,bet_type,picks_count,is_live'
        ).eq('market', market_name).limit(500).execute()
        all_data.extend(response.data)
        print(f"  Found {len(response.data)} '{market_name}' bets")

    df = pd.DataFrame(all_data)

    # Remove duplicates
    if len(df) > 0:
        df = df.drop_duplicates(subset=['id'])

    print(f"\nTotal candidates for prediction: {len(df)}")

    return df


def main():
    """Generate predictions for unlabeled bets"""

    load_dotenv('/root/pikkit/.env')

    # Load trained model
    print("Loading trained classifier...")
    classifier = GranularMarketClassifier()
    classifier.load()

    # Fetch candidates
    supabase = get_supabase_client()
    df = fetch_candidates_for_prediction(supabase)

    if len(df) == 0:
        print("No candidates found for prediction.")
        return

    # Make predictions
    print(f"\nGenerating predictions for {len(df)} bets...")
    predictions, probabilities = classifier.predict(df)

    # Add to dataframe
    df['predicted_market'] = predictions
    df['prediction_confidence'] = probabilities

    # Show statistics
    print("\n" + "="*60)
    print("Prediction Summary")
    print("="*60)

    print(f"\nTotal predictions: {len(df)}")
    print(f"Mean confidence: {probabilities.mean():.2%}")
    print(f"Median confidence: {pd.Series(probabilities).median():.2%}")

    print(f"\nConfidence distribution:")
    print(f"  >90%: {(probabilities > 0.9).sum()} ({(probabilities > 0.9).sum() / len(probabilities) * 100:.1f}%)")
    print(f"  >80%: {(probabilities > 0.8).sum()} ({(probabilities > 0.8).sum() / len(probabilities) * 100:.1f}%)")
    print(f"  >70%: {(probabilities > 0.7).sum()} ({(probabilities > 0.7).sum() / len(probabilities) * 100:.1f}%)")
    print(f"  >60%: {(probabilities > 0.6).sum()} ({(probabilities > 0.6).sum() / len(probabilities) * 100:.1f}%)")

    print(f"\nTop 20 predicted markets:")
    print(df['predicted_market'].value_counts().head(20))

    # Save predictions
    output_path = '/root/pikkit/ml/data/granular_market_predictions.csv'
    df[['id', 'sport', 'market', 'predicted_market', 'prediction_confidence', 'pick_name']].to_csv(
        output_path, index=False
    )
    print(f"\nPredictions saved to: {output_path}")

    # Show sample high-confidence predictions
    print("\n" + "="*60)
    print("Sample High-Confidence Predictions (>90%)")
    print("="*60)

    high_conf = df[df['prediction_confidence'] > 0.9].head(20)
    for idx, row in high_conf.iterrows():
        current = row['market'] if pd.notna(row['market']) else 'NULL'
        print(f"\n{row['sport']:20s} | Confidence: {row['prediction_confidence']:.1%}")
        print(f"  Current:   {current}")
        print(f"  Predicted: {row['predicted_market']}")
        print(f"  Pick:      {row['pick_name'][:80]}")

    print("\n" + "="*60)
    print("Review predictions in: " + output_path)
    print("To apply, run: python apply_granular_predictions.py --confidence=0.8")
    print("="*60)


if __name__ == '__main__':
    main()
