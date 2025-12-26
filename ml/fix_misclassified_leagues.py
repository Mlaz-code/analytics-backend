#!/usr/bin/env python3
"""
Fix misclassified sport/league combinations in Supabase
Corrects obvious errors like "German BBL" under Soccer → Basketball
"""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv('/root/pikkit/.env')

SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://mnnjjvbaxzumfcgibtme.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_KEY') or os.environ.get('SUPABASE_ANON_KEY')

headers = {
    'Authorization': f'Bearer {SUPABASE_KEY}',
    'apikey': SUPABASE_KEY,
    'Content-Type': 'application/json',
}

# Define corrections: (current_sport, current_league) -> (correct_sport, correct_league)
CORRECTIONS = {
    # Basketball leagues misclassified as Soccer
    ('Soccer', 'German BBL'): ('Basketball', 'BBL'),

    # Soccer leagues misclassified as Basketball
    ('Basketball', 'La Liga'): ('Soccer', 'La Liga'),
    ('Basketball', 'Ligue 1'): ('Soccer', 'Ligue 1'),

    # American Football misclassified as Soccer
    ('Soccer', 'NCAA Football'): ('American Football', 'NCAAFB'),

    # Basketball misclassified as other sports
    ('Baseball', 'NBA'): ('Basketball', 'NBA'),
    ('Tennis', 'EuroLeague'): ('Basketball', 'EuroLeague'),

    # Standardize BBL naming
    ('Basketball', 'Germany - BBL'): ('Basketball', 'BBL'),
}

def fetch_misclassified_bets():
    """Fetch all bets that need corrections"""
    all_bets = []

    for (sport, league), (correct_sport, correct_league) in CORRECTIONS.items():
        # Build query with proper escaping
        url = f"{SUPABASE_URL}/rest/v1/bets?sport=eq.{sport}&league=eq.{league}&select=id,sport,league,pick_name"
        response = requests.get(url, headers=headers)

        if response.ok:
            bets = response.json()
            for bet in bets:
                bet['correction'] = {
                    'from_sport': sport,
                    'from_league': league,
                    'to_sport': correct_sport,
                    'to_league': correct_league
                }
            all_bets.extend(bets)
            print(f"  Found {len(bets)} bets: {sport}/{league} → {correct_sport}/{correct_league}")
        else:
            print(f"  ⚠️  Failed to fetch {sport}/{league}: {response.text}")

    return all_bets

def update_bet(bet_id, sport, league):
    """Update a single bet with corrected sport/league"""
    url = f"{SUPABASE_URL}/rest/v1/bets?id=eq.{bet_id}"

    update_data = {
        'sport': sport,
        'league': league,
    }

    response = requests.patch(url, headers=headers, json=update_data)
    return response.ok

def main():
    print("="*80)
    print("FIX MISCLASSIFIED SPORT/LEAGUE COMBINATIONS")
    print("="*80)
    print()

    print("Fetching misclassified bets...")
    bets = fetch_misclassified_bets()
    print(f"\nTotal bets to fix: {len(bets)}")
    print()

    if not bets:
        print("✅ No misclassified bets found!")
        return

    # Show summary
    print("="*80)
    print("SUMMARY OF CORRECTIONS")
    print("="*80)
    print()

    correction_summary = {}
    for bet in bets:
        corr = bet['correction']
        key = (corr['from_sport'], corr['from_league'], corr['to_sport'], corr['to_league'])
        if key not in correction_summary:
            correction_summary[key] = 0
        correction_summary[key] += 1

    for (from_sport, from_league, to_sport, to_league), count in sorted(correction_summary.items(), key=lambda x: -x[1]):
        print(f"{count:3} bets: {from_sport:25} / {from_league:25} → {to_sport:25} / {to_league}")

    print()
    print("="*80)
    print("SAMPLE BETS (first 10)")
    print("="*80)
    print()

    for i, bet in enumerate(bets[:10]):
        corr = bet['correction']
        pick = (bet.get('pick_name') or '')[:60]
        print(f"{i+1:2}. {bet['id']}")
        print(f"    Current: {corr['from_sport']:20} / {corr['from_league']}")
        print(f"    Correct: {corr['to_sport']:20} / {corr['to_league']}")
        print(f"    Pick:    {pick}")
        print()

    # Ask for confirmation (unless --yes flag is provided)
    print("="*80)

    auto_yes = '--yes' in sys.argv or '-y' in sys.argv

    if auto_yes:
        print(f"Auto-confirm enabled. Applying corrections to {len(bets)} bets...")
        response = 'yes'
    else:
        response = input(f"Apply these corrections to {len(bets)} bets? (yes/no): ")

    if response.lower() not in ['yes', 'y']:
        print("❌ Aborted. No changes made.")
        return

    print()
    print("Applying corrections...")
    success = 0
    failed = 0

    for i, bet in enumerate(bets):
        corr = bet['correction']
        if update_bet(bet['id'], corr['to_sport'], corr['to_league']):
            success += 1
        else:
            failed += 1
            print(f"  ❌ Failed to update: {bet['id']}")

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(bets)} ({success} success, {failed} failed)")

    print()
    print("="*80)
    print(f"✅ Done! Updated {success} bets, {failed} failed")
    print("="*80)

if __name__ == '__main__':
    main()
