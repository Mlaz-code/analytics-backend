#!/usr/bin/env python3
"""
Pikkit to Supabase Sync Script
Fetches bets from Pikkit API and syncs to Supabase
"""

import os
import sys
import json
import requests
from datetime import datetime
from typing import Dict, List, Set
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

# Configuration
PIKKIT_API_BASE = os.environ.get('PIKKIT_API_BASE', 'https://prod-website.pikkit.app')
PIKKIT_API_TOKEN = os.environ.get('PIKKIT_API_TOKEN', '')
SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://mnnjjvbaxzumfcgibtme.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_KEY', '')

# Mappings
INSTITUTION_MAP = {
    '60b6e38b111afa064096285a': 'DraftKings',
    '60b6e38b111afa0640962859': 'FanDuel',
    '60b6e38b111afa064096285b': 'BetMGM',
    '60b6e38b111afa064096285c': 'Caesars',
    '60b6e38b111afa064096285d': 'PointsBet',
}

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


def extract_time_from_id(object_id: str) -> str:
    """Extract timestamp from MongoDB ObjectId"""
    if not object_id or len(object_id) < 8:
        return None
    try:
        timestamp = int(object_id[:8], 16)
        return datetime.fromtimestamp(timestamp).isoformat()
    except:
        return None


def extract_event_from_context(pick_context: list) -> str:
    """Extract event name from pick context"""
    if not pick_context or not isinstance(pick_context, list):
        return None

    teams = []
    for item in pick_context:
        if item.get('type') == 'string':
            val = (item.get('value') or '').strip()
            if val and len(val) <= 5 and val.isupper() and val.isalpha():
                teams.append(val)

    if len(teams) >= 2:
        return f"{teams[0]} vs {teams[1]}"
    if len(teams) == 1:
        return teams[0]

    # Fallback: get first string
    for item in pick_context:
        if item.get('type') == 'string':
            val = (item.get('value') or '').strip()
            if val and val != '·' and len(val) > 3:
                return val[:100]

    return None


def predict_with_ml(pick_name: str, sport: str, market: str, league: str,
                    bet_type: str, is_live: bool, picks_count: int) -> Dict:
    """Use ML classifier to predict missing sport/league/market"""
    classifier = get_classifier()
    if not classifier:
        return {'sport': sport, 'league': league, 'market': market}

    # Create a dataframe for prediction
    df = pd.DataFrame([{
        'market': market,
        'pick_name': pick_name,
        'sport': sport,
        'league': league,
        'bet_type': bet_type,
        'picks_count': picks_count,
        'is_live': is_live
    }])

    try:
        predictions, confidences = classifier.predict(df)

        # Use predictions with >70% confidence
        result = {
            'sport': sport,
            'league': league,
            'market': market
        }

        if not sport and predictions['sport_confidence'][0] > 0.7:
            result['sport'] = predictions['predicted_sport'][0]

        if not league and predictions['league_confidence'][0] > 0.7:
            result['league'] = predictions['predicted_league'][0]

        if not market or market == 'Other':
            if predictions['market_confidence'][0] > 0.7:
                result['market'] = predictions['predicted_market'][0]

        return result

    except Exception as e:
        print(f"  ⚠️  ML prediction failed: {e}")
        return {'sport': sport, 'league': league, 'market': market}


def transform_bet(bet: Dict) -> Dict:
    """Transform Pikkit bet to Supabase schema"""
    picks = bet.get('picks', [])
    first_pick = picks[0] if picks else {}
    bet_type = bet.get('type', 'straight')
    pick_name = first_pick.get('pick_name', 'Unknown Bet')

    # Handle parlays
    if bet_type == 'parlay':
        sport = 'Parlay'
        league = 'Parlay'
        market = 'Parlay'
        event_name = None
        sport_id = None
        league_id = None
        team_id = None
        event_id = None
        market_id = None
        player_id = None
        outcome_id = None
    else:
        linking = first_pick.get('linking_context', {})

        sport_id = linking.get('sport')
        sport = SPORT_ID_MAP.get(sport_id) if sport_id else None

        league_id = linking.get('league')
        league = LEAGUE_ID_MAP.get(league_id) if league_id else None

        pick_context = first_pick.get('pick_context', [])
        event_name = extract_event_from_context(pick_context)

        team_id = linking.get('team')
        event_id = linking.get('event')
        market_id = linking.get('market')
        player_id = linking.get('player')
        outcome_id = linking.get('outcome')

        # Determine market type (order matters - check specific patterns first)
        market = 'Other'

        # Quarter/Half specific markets (check these FIRST before generic Moneyline/Spread/Total)
        if '1st Quarter Moneyline' in pick_name or 'Moneyline 1st Quarter' in pick_name:
            market = '1st Quarter Moneyline'
        elif '2nd Quarter Moneyline' in pick_name or 'Moneyline 2nd Quarter' in pick_name:
            market = '2nd Quarter Moneyline'
        elif '3rd Quarter Moneyline' in pick_name or 'Moneyline 3rd Quarter' in pick_name:
            market = '3rd Quarter Moneyline'
        elif '4th Quarter Moneyline' in pick_name or 'Moneyline 4th Quarter' in pick_name:
            market = '4th Quarter Moneyline'
        elif '1st Half Moneyline' in pick_name or 'Moneyline 1st Half' in pick_name:
            market = '1st Half Moneyline'
        elif '2nd Half Moneyline' in pick_name or 'Moneyline 2nd Half' in pick_name:
            market = '2nd Half Moneyline'
        # Team totals (check before generic Total) - Ice Hockey only
        elif 'Team Total Goals' in pick_name and sport == 'Ice Hockey':
            market = 'Team Total Goals'
        # Period totals - Ice Hockey only
        elif 'Total 1st Period' in pick_name and sport == 'Ice Hockey':
            market = '1st Period Total Goals'
        elif 'Total 2nd Period' in pick_name and sport == 'Ice Hockey':
            market = '2nd Period Total Goals'
        elif 'Total 3rd Period' in pick_name and sport == 'Ice Hockey':
            market = '3rd Period Total Goals'
        # Set winner - Tennis only
        elif '1st Set' in pick_name and sport == 'Tennis':
            market = '1st Set Winner'
        elif '2nd Set' in pick_name and sport == 'Tennis':
            market = '2nd Set Winner'
        elif '3rd Set' in pick_name and sport == 'Tennis':
            market = '3rd Set Winner'
        # Half totals - Basketball (check before generic Total)
        elif '1st Half Total' in pick_name and sport == 'Basketball':
            market = '1st Half Total'
        elif '2nd Half Total' in pick_name and sport == 'Basketball':
            market = '2nd Half Total'
        # Team half totals - Basketball
        elif ('Team Total 1st Half' in pick_name or '1st Half Team Total' in pick_name) and sport == 'Basketball':
            market = '1st Half Team Total'
        elif ('Team Total 2nd Half' in pick_name or '2nd Half Team Total' in pick_name) and sport == 'Basketball':
            market = '2nd Half Team Total'
        # Half spreads - Basketball
        elif ('Spread 1st Half' in pick_name or '1st Half Spread' in pick_name) and sport == 'Basketball':
            market = '1st Half Spread'
        elif ('Spread 2nd Half' in pick_name or '2nd Half Spread' in pick_name) and sport == 'Basketball':
            market = '2nd Half Spread'
        # Quarter totals - Basketball
        elif '1st Quarter Total' in pick_name and sport == 'Basketball':
            market = '1st Quarter Total'
        elif '2nd Quarter Total' in pick_name and sport == 'Basketball':
            market = '2nd Quarter Total'
        elif '3rd Quarter Total' in pick_name and sport == 'Basketball':
            market = '3rd Quarter Total'
        elif '4th Quarter Total' in pick_name and sport == 'Basketball':
            market = '4th Quarter Total'
        # Generic markets (only if not matched above)
        elif 'Moneyline' in pick_name or ' ML' in pick_name:
            market = 'Moneyline'
        elif 'Spread' in pick_name:
            market = 'Spread'
        elif 'Total' in pick_name or 'Over' in pick_name or 'Under' in pick_name:
            market = 'Total'
        elif any(x in pick_name for x in ['Points', 'Rebounds', 'Assists', 'Yards', 'TDs', 'Prop']):
            market = 'Player Props'

        # Use ML classifier to improve predictions for non-parlay bets
        # Only if sport, league, or market are missing/uncertain
        if not sport or not league or market == 'Other':
            ml_predictions = predict_with_ml(
                pick_name=pick_name,
                sport=sport,
                market=market,
                league=league,
                bet_type=bet_type,
                is_live=bet.get('is_live', False),
                picks_count=len(picks)
            )
            sport = ml_predictions['sport'] or sport
            league = ml_predictions['league'] or league
            market = ml_predictions['market'] or market

    # Sportsbook
    institution_id = bet.get('institution_id', '')
    sportsbook = INSTITUTION_MAP.get(institution_id, 'Unknown')

    # Odds conversion
    decimal_odds = bet.get('odds', 2.0)
    american_odds = None
    if decimal_odds and decimal_odds >= 2.0:
        american_odds = int((decimal_odds - 1) * 100)
    elif decimal_odds and decimal_odds > 1:
        american_odds = int(-100 / (decimal_odds - 1))

    # Status and outcome
    status = bet.get('status', 'UNKNOWN')
    is_win = 'WIN' in status.upper()
    is_settled = 'SETTLED' in status.upper()

    # Profit calculation
    amount = bet.get('amount', 0)
    profit = bet.get('profit')
    if profit is None and is_settled:
        if is_win:
            profit = amount * (decimal_odds - 1)
        else:
            profit = -amount

    # ROI
    roi = None
    if profit is not None and amount > 0:
        roi = (profit / amount) * 100

    return {
        'id': bet.get('_id'),
        'bet_type': bet_type,
        'odds': decimal_odds,
        'american_odds': american_odds,
        'amount': amount,
        'profit': profit,
        'roi': roi,
        'status': status,
        # Note: is_win and is_settled are generated columns in Supabase
        # They are computed from the status field automatically
        'time_placed': extract_time_from_id(bet.get('_id')),
        'institution_name': sportsbook,
        'pick_name': pick_name,
        'sport': sport,
        'league': league,
        'event_name': event_name,
        'market': market,
        'market_type': first_pick.get('market_type'),
        'picks_count': len(picks),
        'picks_json': picks,
        'picks_linking_context': [p.get('linking_context', {}) for p in picks],
        'raw_json': bet,
        'is_live': bet.get('is_live', False),
        'is_dfs': bet.get('is_dfs', False),
        'future': bet.get('future', False),
        'promotional_cash': bet.get('promotional_cash', False),
        'team_id': team_id,
        'event_id': event_id,
        'sport_id': sport_id,
        'league_id': league_id,
        'market_id': market_id,
        'player_id': player_id,
        'outcome_id': outcome_id,
    }


def fetch_pikkit_bets(offset: int = 0, limit: int = 50) -> List[Dict]:
    """Fetch bets from Pikkit API"""
    url = f"{PIKKIT_API_BASE}/user/bets?offset={offset}&limit={limit}"

    headers = {
        'Authorization': PIKKIT_API_TOKEN,
        'Accept': 'application/json',
        'Origin': 'https://app.pikkit.com',
        'User-Agent': 'Mozilla/5.0'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    return response.json()


def get_existing_bet_ids() -> Set[str]:
    """Get all existing bet IDs from Supabase"""
    all_ids = set()
    offset = 0
    batch_size = 1000

    while True:
        url = f"{SUPABASE_URL}/rest/v1/bets?select=id&limit={batch_size}&offset={offset}"

        headers = {
            'Authorization': f'Bearer {SUPABASE_KEY}',
            'apikey': SUPABASE_KEY
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()

        if not data:
            break

        for bet in data:
            all_ids.add(bet['id'])

        if len(data) < batch_size:
            break

        offset += batch_size

    return all_ids


def insert_bets(bets: List[Dict]) -> int:
    """Insert bets into Supabase"""
    if not bets:
        return 0

    url = f"{SUPABASE_URL}/rest/v1/bets"

    headers = {
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'apikey': SUPABASE_KEY,
        'Content-Type': 'application/json',
        'Prefer': 'resolution=merge-duplicates,return=representation'
    }

    response = requests.post(url, headers=headers, json=bets)

    if not response.ok:
        print(f"ERROR: {response.status_code} - {response.text}")
        # Try to identify which bet is causing the issue
        if len(bets) > 1:
            print(f"Trying to insert bets one at a time to identify the problem...")
            success_count = 0
            for i, bet in enumerate(bets):
                try:
                    single_response = requests.post(url, headers=headers, json=[bet])
                    if single_response.ok:
                        success_count += 1
                    else:
                        print(f"  Bet {i+1}/{len(bets)} failed: {bet.get('pick_name', 'Unknown')[:50]}")
                        print(f"    Error: {single_response.text}")
                except Exception as e:
                    print(f"  Bet {i+1}/{len(bets)} exception: {e}")
            return success_count
        else:
            print(f"Single bet error details:")
            print(json.dumps(bets[0], indent=2))
            response.raise_for_status()

    result = response.json()
    return len(result)


def sync_bets():
    """Main sync function"""
    print("=" * 60)
    print("Pikkit to Supabase Sync")
    print(f"Started: {datetime.now()}")
    print("=" * 60)
    print()

    # Get existing bets
    print("Fetching existing bets from Supabase...")
    existing_ids = get_existing_bet_ids()
    print(f"  Found {len(existing_ids)} existing bets")
    print()

    # Fetch from Pikkit
    all_new_bets = []
    offset = 0
    batch_size = 50

    while True:
        print(f"Fetching Pikkit bets (offset={offset})...")
        bets = fetch_pikkit_bets(offset, batch_size)

        if not bets:
            break

        print(f"  Received {len(bets)} bets")

        # Transform and filter
        for bet in bets:
            bet_id = bet.get('_id')
            if bet_id and bet_id not in existing_ids:
                transformed = transform_bet(bet)
                all_new_bets.append(transformed)

        if len(bets) < batch_size:
            break

        offset += batch_size

        # Safety limit
        if offset >= 500:
            print(f"  Reached pagination limit (500)")
            break

    print()
    print("=" * 60)
    print(f"Found {len(all_new_bets)} new bets to sync")

    if all_new_bets:
        # Show preview
        print()
        print("New bets preview:")
        for bet in all_new_bets[:5]:
            amount = bet.get('amount', 0)
            status = bet.get('status', 'UNKNOWN')
            pick = bet.get('pick_name', 'Unknown')[:50]
            print(f"  [{status}] ${amount} - {pick}...")
        if len(all_new_bets) > 5:
            print(f"  ... and {len(all_new_bets) - 5} more")

        print()
        print("Inserting bets to Supabase...")
        inserted = insert_bets(all_new_bets)
        print(f"  Inserted {inserted} bets")

    print()
    print("=" * 60)
    print(f"Sync completed: {datetime.now()}")
    print("=" * 60)
    print()

    return len(all_new_bets)


def trigger_model_retraining():
    """Trigger automated model retraining after successful sync"""
    import subprocess

    print()
    print("=" * 60)
    print("STARTING AUTOMATED MODEL RETRAINING")
    print("=" * 60)
    print()

    try:
        # Step 1: Fetch all bets
        print("Step 1/3: Fetching all bets from Supabase...")
        result = subprocess.run(
            ['python3', '/root/pikkit/ml/scripts/fetch_all_bets.py'],
            cwd='/root/pikkit/ml/scripts',
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"  ERROR fetching bets: {result.stderr}")
            return False

        print("  ✅ Bets fetched successfully")

        # Step 2: Feature engineering
        print("\nStep 2/3: Running feature engineering...")
        result = subprocess.run(
            ['python3', '/root/pikkit/ml/scripts/quick_market_features.py'],
            cwd='/root/pikkit/ml/scripts',
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            print(f"  ERROR in feature engineering: {result.stderr}")
            return False

        print("  ✅ Features engineered successfully")

        # Step 3: Train model
        print("\nStep 3/3: Training market prediction model...")
        result = subprocess.run(
            ['python3', '/root/pikkit/ml/scripts/train_market_model_from_features.py'],
            cwd='/root/pikkit/ml/scripts',
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            print(f"  ERROR in model training: {result.stderr}")
            return False

        print("  ✅ Model trained successfully")

        print()
        print("=" * 60)
        print("MODEL RETRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print()

        return True

    except subprocess.TimeoutExpired:
        print("  ERROR: Training timed out")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    try:
        new_bets = sync_bets()

        # Only retrain if new bets were added
        if new_bets > 0:
            print(f"\n{new_bets} new bets synced - triggering model retraining...")
            trigger_model_retraining()
        else:
            print("\nNo new bets - skipping model retraining")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
