#!/usr/bin/env python3
"""
Fill NULL league values for bets that have sport but missing league
Uses ML classifier to predict leagues with >70% confidence
"""

import os
import sys
import requests
import pandas as pd
from dotenv import load_dotenv

# Add ML directory to path for classifier import
sys.path.insert(0, '/root/pikkit/ml')
from comprehensive_classifier import ComprehensiveClassifier

load_dotenv('/root/pikkit/.env')

SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://mnnjjvbaxzumfcgibtme.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_KEY') or os.environ.get('SUPABASE_ANON_KEY')

headers = {
    'Authorization': f'Bearer {SUPABASE_KEY}',
    'apikey': SUPABASE_KEY,
    'Content-Type': 'application/json',
}

# Load ML classifier (lazy loading)
_classifier = None

def get_classifier():
    """Lazy load the comprehensive classifier"""
    global _classifier
    if _classifier is None:
        try:
            _classifier = ComprehensiveClassifier()
            _classifier.load()
            print("  ✅ ML classifier loaded successfully")
        except Exception as e:
            print(f"  ⚠️  Failed to load ML classifier: {e}")
            _classifier = False
    return _classifier if _classifier is not False else None

def fetch_null_league_bets(sport=None):
    """Fetch all bets with NULL league"""
    all_bets = []
    offset = 0
    batch_size = 1000

    while True:
        # Build query
        if sport:
            url = f"{SUPABASE_URL}/rest/v1/bets?sport=eq.{sport}&league=is.null&select=id,sport,market,pick_name,bet_type,is_live,picks_count&limit={batch_size}&offset={offset}"
        else:
            url = f"{SUPABASE_URL}/rest/v1/bets?league=is.null&sport=not.is.null&select=id,sport,market,pick_name,bet_type,is_live,picks_count&limit={batch_size}&offset={offset}"

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()
        if not data:
            break

        all_bets.extend(data)

        if len(data) < batch_size:
            break

        offset += batch_size
        if len(all_bets) % 1000 == 0:
            print(f"  Fetched {len(all_bets)} bets so far...")

    return all_bets

def update_bet_league(bet_id, league):
    """Update a single bet's league"""
    url = f"{SUPABASE_URL}/rest/v1/bets?id=eq.{bet_id}"

    update_data = {'league': league}

    response = requests.patch(url, headers=headers, json=update_data)
    return response.ok

def main():
    print("="*80)
    print("FILL NULL LEAGUES USING ML CLASSIFIER")
    print("="*80)
    print()

    # Check if user specified a sport
    sport = sys.argv[1] if len(sys.argv) > 1 else None

    if sport:
        print(f"Fetching {sport} bets with NULL league...")
    else:
        print("Fetching all bets with NULL league (but sport populated)...")

    bets = fetch_null_league_bets(sport)
    print(f"Found {len(bets)} bets with NULL league")
    print()

    if not bets:
        print("✅ No bets need league prediction!")
        return

    # Show distribution by sport
    df = pd.DataFrame(bets)
    sport_counts = df['sport'].value_counts()
    print("Distribution by sport:")
    for sport, count in sport_counts.items():
        print(f"  {sport:30} {count:5} bets")
    print()

    # Filter out parlays (they should have league='Parlay')
    non_parlay_bets = [b for b in bets if b.get('bet_type') != 'parlay']
    parlay_bets = [b for b in bets if b.get('bet_type') == 'parlay']

    print(f"Breakdown:")
    print(f"  Non-parlay bets: {len(non_parlay_bets)}")
    print(f"  Parlay bets: {len(parlay_bets)}")
    print()

    # Use ML classifier for non-parlay bets
    if non_parlay_bets:
        print(f"Using ML classifier to predict leagues for {len(non_parlay_bets)} non-parlay bets...")
        classifier = get_classifier()

        if not classifier:
            print("❌ Failed to load classifier. Aborting.")
            return

        try:
            # Prepare data for batch prediction
            df = pd.DataFrame([{
                'id': b['id'],
                'market': b.get('market'),
                'pick_name': b.get('pick_name'),
                'sport': b.get('sport'),
                'league': None,
                'bet_type': b.get('bet_type', 'straight'),
                'picks_count': b.get('picks_count', 1),
                'is_live': b.get('is_live', False)
            } for b in non_parlay_bets])

            predictions, confidences = classifier.predict(df)

            # Show sample predictions
            print()
            print("Sample predictions:")
            for i in range(min(10, len(df))):
                league_pred = predictions['predicted_league'][i]
                league_conf = predictions['league_confidence'][i]
                pick = (df.iloc[i]['pick_name'] or '')[:60]
                sport = df.iloc[i]['sport']
                will_apply = "✅" if league_conf > 0.7 else "❌"

                print(f"{i+1:2}. {sport:20} → {league_pred:20} ({league_conf*100:5.1f}%) {will_apply}")
                print(f"    {pick}")

            print()

            # Count high-confidence predictions
            high_conf_count = sum(1 for conf in predictions['league_confidence'] if conf > 0.7)
            print(f"High-confidence predictions (>70%): {high_conf_count}/{len(df)}")
            print()

            # Ask for confirmation
            if '--yes' in sys.argv or '-y' in sys.argv:
                print("Auto-confirm enabled. Applying predictions...")
                response = 'yes'
            else:
                response = input(f"Apply {high_conf_count} high-confidence league predictions? (yes/no): ")

            if response.lower() not in ['yes', 'y']:
                print("❌ Aborted. No changes made.")
                return

            # Apply high-confidence predictions
            print()
            print("Applying predictions...")
            success = 0
            failed = 0
            skipped = 0

            for i in range(len(df)):
                league_conf = predictions['league_confidence'][i]

                if league_conf > 0.7:
                    league_pred = predictions['predicted_league'][i]
                    bet_id = df.iloc[i]['id']

                    if update_bet_league(bet_id, league_pred):
                        success += 1
                    else:
                        failed += 1
                        print(f"  ❌ Failed to update: {bet_id}")
                else:
                    skipped += 1

                if (success + failed + skipped) % 10 == 0:
                    print(f"  Progress: {success + failed + skipped}/{len(df)} ({success} applied, {skipped} skipped, {failed} failed)")

            print()
            print(f"✅ Done! Applied {success} predictions, skipped {skipped} low-confidence, {failed} failed")

        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            import traceback
            traceback.print_exc()

    # Handle parlay bets separately
    if parlay_bets:
        print()
        print(f"Found {len(parlay_bets)} parlay bets with NULL league - these should be set to 'Parlay'")
        if '--yes' in sys.argv or '-y' in sys.argv:
            print("Auto-confirm enabled. Setting parlay leagues...")
            response = 'yes'
        else:
            response = input(f"Set {len(parlay_bets)} parlay bets to league='Parlay'? (yes/no): ")

        if response.lower() in ['yes', 'y']:
            success = 0
            failed = 0

            for bet in parlay_bets:
                if update_bet_league(bet['id'], 'Parlay'):
                    success += 1
                else:
                    failed += 1

            print(f"✅ Done! Updated {success} parlay bets, {failed} failed")

    print()
    print("="*80)
    print("COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
