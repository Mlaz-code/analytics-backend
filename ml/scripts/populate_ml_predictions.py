#!/usr/bin/env python3
"""
Populate ML predictions table in Supabase
Fetches unique sport/league/market combinations and generates ML predictions
"""

import os
import sys
import json
import requests
from datetime import datetime
from pathlib import Path

# Add parent to path for predictor
sys.path.insert(0, str(Path(__file__).parent))
from predict_bet_profitability import BetProfitabilityPredictor

# Supabase config
SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://mnnjjvbaxzumfcgibtme.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_KEY', '')

# Load from .env if not in environment
if not SUPABASE_KEY:
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith('SUPABASE_SERVICE_KEY='):
                    SUPABASE_KEY = line.split('=', 1)[1].strip()
                    break

if not SUPABASE_KEY:
    print("Error: SUPABASE_SERVICE_KEY not found")
    sys.exit(1)

HEADERS = {
    'apikey': SUPABASE_KEY,
    'Authorization': f'Bearer {SUPABASE_KEY}',
    'Content-Type': 'application/json',
    'Prefer': 'return=minimal'
}

def create_table():
    """Create the ml_predictions table if it doesn't exist"""
    # Using Supabase's SQL editor would be better, but we can use RPC
    # For now, we'll assume the table exists or create via Supabase dashboard
    print("Note: Create table via Supabase dashboard SQL editor:")
    print("""
    CREATE TABLE IF NOT EXISTS ml_predictions (
        id SERIAL PRIMARY KEY,
        sport TEXT NOT NULL,
        league TEXT,
        market TEXT NOT NULL,
        direction TEXT,
        institution_name TEXT DEFAULT 'DraftKings',
        win_probability REAL,
        expected_roi REAL,
        bet_grade TEXT,
        confidence REAL,
        kelly_fraction REAL,
        profitable BOOLEAN,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        UNIQUE(sport, league, market, direction, institution_name)
    );

    CREATE INDEX IF NOT EXISTS idx_ml_predictions_lookup
    ON ml_predictions(sport, league, market, direction);
    """)

def get_unique_combinations():
    """Get unique sport/league/market/direction combinations from bets table"""
    # Query for distinct combinations with bet counts
    url = f"{SUPABASE_URL}/rest/v1/rpc/get_market_combinations"

    # Try RPC first, fall back to manual query
    try:
        response = requests.post(url, headers=HEADERS, json={})
        if response.ok:
            return response.json()
    except:
        pass

    # Manual approach - get all and dedupe
    print("Fetching unique combinations from bets table...")
    url = f"{SUPABASE_URL}/rest/v1/bets?select=sport,league,market,direction&status=in.(SETTLED_WIN,SETTLED_LOSS)&limit=10000"
    response = requests.get(url, headers=HEADERS)

    if not response.ok:
        print(f"Error fetching bets: {response.status_code} - {response.text}")
        return []

    bets = response.json()

    # Deduplicate
    seen = set()
    combinations = []
    for bet in bets:
        key = (bet.get('sport'), bet.get('league'), bet.get('market'), bet.get('direction'))
        if key not in seen and bet.get('sport') and bet.get('market'):
            seen.add(key)
            combinations.append({
                'sport': bet.get('sport'),
                'league': bet.get('league'),
                'market': bet.get('market'),
                'direction': bet.get('direction')
            })

    print(f"Found {len(combinations)} unique combinations")
    return combinations

def generate_predictions(combinations, predictor):
    """Generate ML predictions for each combination"""
    predictions = []

    for i, combo in enumerate(combinations):
        if combo['sport'] == 'Parlay':
            continue  # Skip parlays

        try:
            bet_data = {
                'sport': combo['sport'],
                'league': combo['league'] or '',
                'market': combo['market'],
                'institution_name': 'DraftKings',  # Default sportsbook
                'bet_type': 'straight',
                'american_odds': -110,
                'is_live': False,
                'clv_percentage': 0
            }

            pred = predictor.predict(bet_data)

            predictions.append({
                'sport': combo['sport'],
                'league': combo['league'],
                'market': combo['market'],
                'direction': combo['direction'],
                'institution_name': 'DraftKings',
                'win_probability': pred.get('win_probability'),
                'expected_roi': pred.get('expected_roi'),
                'bet_grade': pred.get('bet_grade'),
                'confidence': pred.get('confidence'),
                'kelly_fraction': pred.get('kelly_fraction'),
                'profitable': pred.get('profitable'),
                'updated_at': datetime.utcnow().isoformat()
            })

            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1}/{len(combinations)} predictions...")

        except Exception as e:
            print(f"Error predicting {combo}: {e}")
            continue

    return predictions

def upsert_predictions(predictions):
    """Insert or update predictions in Supabase"""
    if not predictions:
        print("No predictions to upsert")
        return

    # Upsert in batches
    batch_size = 100
    for i in range(0, len(predictions), batch_size):
        batch = predictions[i:i+batch_size]

        url = f"{SUPABASE_URL}/rest/v1/ml_predictions"
        headers = {**HEADERS, 'Prefer': 'resolution=merge-duplicates'}

        response = requests.post(url, headers=headers, json=batch)

        if response.ok:
            print(f"Upserted batch {i//batch_size + 1} ({len(batch)} records)")
        else:
            print(f"Error upserting batch: {response.status_code} - {response.text}")

def main():
    print("=" * 60)
    print("ML Predictions Population Script")
    print("=" * 60)

    # Show table creation SQL
    create_table()

    # Initialize predictor
    print("\nLoading ML models...")
    predictor = BetProfitabilityPredictor()

    if predictor.win_model is None:
        print("Error: ML models not loaded")
        sys.exit(1)

    print("Models loaded successfully")

    # Get unique combinations
    combinations = get_unique_combinations()

    if not combinations:
        print("No combinations found")
        sys.exit(1)

    # Generate predictions
    print(f"\nGenerating predictions for {len(combinations)} combinations...")
    predictions = generate_predictions(combinations, predictor)

    print(f"\nGenerated {len(predictions)} predictions")

    # Upsert to Supabase
    print("\nUpserting to Supabase...")
    upsert_predictions(predictions)

    print("\nDone!")

if __name__ == '__main__':
    main()
