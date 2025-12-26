#!/usr/bin/env python3
"""
Backfill sport and league fields for bets that have null sport
but have sport_id in raw_json.picks[0].linking_context
Uses ML classifier as fallback when ID mapping fails
"""

import os
import sys
import requests
import json
import pandas as pd

# Add ML directory to path for classifier import
sys.path.insert(0, '/root/pikkit/ml')
from comprehensive_classifier import ComprehensiveClassifier

# Load environment variables
env_file = '/root/pikkit/.env'
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://mnnjjvbaxzumfcgibtme.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_KEY') or os.environ.get('SUPABASE_ANON_KEY', '')

# Mappings from sync_to_supabase.py
SPORT_ID_MAP = {
    '60ee607964234495fe117365': 'American Football',
    '60ee607964234495fe117366': 'Baseball',
    '60ee607964234495fe117367': 'Basketball',
    '60ee607964234495fe117368': 'Soccer',
    '60ee607964234495fe117369': 'Ice Hockey',
    '60ee607964234495fe117375': 'MMA',
    '62a2647740035a9fbc3830d3': 'Tennis',
}

LEAGUE_ID_MAP = {
    '5fc19b66542917b3cedce72f': 'NFL',
    '60ee607964234495fe11737a': 'NCAAFB',
    '5fc19b57542917b3cedce72e': 'MLB',
    '5fc19b4e542917b3cedce72d': 'NBA',
    '60ee607964234495fe11737b': 'NCAAB',
    '6494682ed97d1a1e940b9d98': 'WNBA',
    '5fc19b43542917b3cedce72c': 'NHL',
    '60ee607964234495fe117391': 'MLS',
    '60ee607964234495fe11737f': 'Premier League',
    '62a264d540035a9fbc3830d5': 'ATP',
    '60ee607964234495fe1173f8': 'UFC',
}

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
            _classifier = False  # Mark as failed to avoid retrying
    return _classifier if _classifier is not False else None

def get_null_sport_bets():
    """Fetch all bets with null sport"""
    all_bets = []
    offset = 0
    batch_size = 1000

    while True:
        # Fetch additional fields needed for ML prediction
        url = f"{SUPABASE_URL}/rest/v1/bets?sport=is.null&select=id,raw_json,market,pick_name,bet_type,is_live,picks_count&limit={batch_size}&offset={offset}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()
        if not data:
            break

        all_bets.extend(data)

        if len(data) < batch_size:
            break

        offset += batch_size
        print(f"  Fetched {len(all_bets)} bets so far...")

    return all_bets

def update_bet(bet_id, sport, league, sport_id, league_id):
    """Update a single bet with sport and league"""
    url = f"{SUPABASE_URL}/rest/v1/bets?id=eq.{bet_id}"

    update_data = {
        'sport': sport,
        'league': league,
        'sport_id': sport_id,
        'league_id': league_id,
    }

    response = requests.patch(url, headers=headers, json=update_data)
    return response.ok

def main():
    print("Fetching bets with null sport...")
    bets = get_null_sport_bets()
    print(f"Found {len(bets)} bets with null sport")

    # Analyze
    has_sport_id = 0
    no_sport_id = 0
    unknown_sport_ids = set()
    unknown_league_ids = set()

    updates = []
    ml_candidates = []  # Bets that need ML prediction

    for bet in bets:
        bet_id = bet['id']
        raw_json = bet.get('raw_json') or {}
        picks = raw_json.get('picks') or []
        bet_type = raw_json.get('type', 'straight')

        # Handle parlays
        if bet_type == 'parlay':
            updates.append({
                'id': bet_id,
                'sport': 'Parlay',
                'league': 'Parlay',
                'sport_id': None,
                'league_id': None,
            })
            has_sport_id += 1
            continue

        if not picks:
            # No raw_json picks - try ML prediction
            ml_candidates.append(bet)
            no_sport_id += 1
            continue

        first_pick = picks[0] if isinstance(picks[0], dict) else {}
        linking = first_pick.get('linking_context') or {}
        sport_id = linking.get('sport')
        league_id = linking.get('league')

        if not sport_id:
            # No sport_id - try ML prediction
            ml_candidates.append(bet)
            no_sport_id += 1
            continue

        has_sport_id += 1

        sport = SPORT_ID_MAP.get(sport_id)
        league = LEAGUE_ID_MAP.get(league_id)

        if not sport:
            unknown_sport_ids.add(sport_id)
            # Unknown sport_id - try ML prediction
            ml_candidates.append(bet)
        else:
            # Successfully mapped sport
            updates.append({
                'id': bet_id,
                'sport': sport,
                'league': league,
                'sport_id': sport_id,
                'league_id': league_id,
            })

    print(f"\nAnalysis:")
    print(f"  Has sport_id in raw_json: {has_sport_id}")
    print(f"  No sport_id in raw_json: {no_sport_id}")
    print(f"  Can be updated via ID mapping: {len(updates)}")
    print(f"  Need ML prediction: {len(ml_candidates)}")

    if unknown_sport_ids:
        print(f"\n  Unknown sport IDs: {unknown_sport_ids}")
    if unknown_league_ids:
        print(f"  Unknown league IDs: {unknown_league_ids}")

    # Use ML classifier for remaining bets
    if ml_candidates:
        print(f"\nUsing ML classifier for {len(ml_candidates)} bets...")
        classifier = get_classifier()

        if classifier:
            try:
                # Prepare data for batch prediction
                df = pd.DataFrame([{
                    'id': b['id'],
                    'market': b.get('market'),
                    'pick_name': b.get('pick_name'),
                    'sport': None,  # Missing sport
                    'league': None,  # Missing league
                    'bet_type': b.get('bet_type', 'straight'),
                    'picks_count': b.get('picks_count', 1),
                    'is_live': b.get('is_live', False)
                } for b in ml_candidates])

                predictions, confidences = classifier.predict(df)

                # Add high-confidence predictions (>70%) to updates
                ml_updates = 0
                for i in range(len(df)):
                    sport_conf = predictions['sport_confidence'][i]
                    league_conf = predictions['league_confidence'][i]

                    # Only use predictions with >70% confidence
                    if sport_conf > 0.7:
                        updates.append({
                            'id': df.iloc[i]['id'],
                            'sport': predictions['predicted_sport'][i],
                            'league': predictions['predicted_league'][i] if league_conf > 0.7 else None,
                            'sport_id': None,
                            'league_id': None,
                        })
                        ml_updates += 1

                print(f"  ✅ ML classifier added {ml_updates} high-confidence predictions")

            except Exception as e:
                print(f"  ⚠️  ML prediction failed: {e}")

    if not updates:
        print("\nNo updates needed.")
        return

    print(f"\nUpdating {len(updates)} bets...")
    success = 0
    failed = 0

    for i, update in enumerate(updates):
        if update_bet(update['id'], update['sport'], update['league'], update['sport_id'], update['league_id']):
            success += 1
        else:
            failed += 1
            print(f"  Failed to update: {update['id']}")

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(updates)} ({success} success, {failed} failed)")

    print(f"\nDone! Updated {success} bets, {failed} failed")

if __name__ == '__main__':
    main()
