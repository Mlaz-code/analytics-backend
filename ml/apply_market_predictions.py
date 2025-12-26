#!/usr/bin/env python3
"""
Apply market type predictions to Supabase database
Updates bets with predicted market_type values
"""

import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
from pathlib import Path
import sys


def get_supabase_client() -> Client:
    """Create Supabase client from environment"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

    return create_client(url, key)


def apply_predictions(
    predictions_file: str,
    confidence_threshold: float = 0.7,
    dry_run: bool = True
):
    """Apply predictions to database"""

    # Load predictions
    if not Path(predictions_file).exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

    df = pd.read_csv(predictions_file)
    print(f"Loaded {len(df)} predictions from {predictions_file}")

    # Filter by confidence threshold
    high_confidence = df[df['prediction_confidence'] >= confidence_threshold]
    print(f"\n{len(high_confidence)} predictions above {confidence_threshold} confidence threshold")

    if len(high_confidence) == 0:
        print("No high-confidence predictions to apply. Exiting.")
        return

    # Show distribution
    print("\nPredictions to apply:")
    print(high_confidence['predicted_market_type'].value_counts())

    # Confirm (skip in non-interactive mode)
    if not dry_run:
        print(f"\n{'='*60}")
        print("WARNING: This will update the database!")
        print(f"{'='*60}")
        print(f"Applying {len(high_confidence)} updates automatically...")
        print("(To abort, press Ctrl+C within 3 seconds)")
        import time
        try:
            time.sleep(3)
        except KeyboardInterrupt:
            print("\nAborted by user.")
            return

    # Initialize Supabase
    supabase = get_supabase_client()

    # Apply updates
    success_count = 0
    error_count = 0

    print(f"\n{'Dry run' if dry_run else 'Applying'} updates...")

    for idx, row in high_confidence.iterrows():
        bet_id = row['id']
        predicted_type = row['predicted_market_type']
        confidence = row['prediction_confidence']

        if dry_run:
            print(f"Would update bet {bet_id}: market_type = {predicted_type} (confidence: {confidence:.2%})")
        else:
            try:
                supabase.table('bets').update({
                    'market_type': predicted_type
                }).eq('id', bet_id).execute()
                success_count += 1

                if success_count % 100 == 0:
                    print(f"Updated {success_count}/{len(high_confidence)} bets...")

            except Exception as e:
                print(f"Error updating bet {bet_id}: {e}")
                error_count += 1

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    if dry_run:
        print(f"Dry run complete. Would update {len(high_confidence)} bets.")
        print("\nTo apply for real, run:")
        print(f"  python apply_market_predictions.py --apply --confidence {confidence_threshold}")
    else:
        print(f"Successfully updated: {success_count}")
        print(f"Errors: {error_count}")
        print(f"Total: {len(high_confidence)}")


def main():
    """Main entry point"""

    load_dotenv('/root/pikkit/.env')

    # Parse args
    dry_run = '--apply' not in sys.argv
    confidence_threshold = 0.7

    # Check for confidence threshold arg
    for arg in sys.argv:
        if arg.startswith('--confidence='):
            confidence_threshold = float(arg.split('=')[1])

    predictions_file = '/root/pikkit/ml/data/market_predictions.csv'

    # Check for custom predictions file
    for arg in sys.argv:
        if arg.startswith('--file='):
            predictions_file = arg.split('=')[1]

    print("Market Type Prediction Applicator")
    print("="*60)
    print(f"Predictions file: {predictions_file}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Mode: {'DRY RUN' if dry_run else 'APPLY TO DATABASE'}")
    print("="*60)

    apply_predictions(
        predictions_file=predictions_file,
        confidence_threshold=confidence_threshold,
        dry_run=dry_run
    )


if __name__ == '__main__':
    main()
