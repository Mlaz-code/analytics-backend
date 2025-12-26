#!/usr/bin/env python3
"""
Analyze sport and league data quality in Supabase
Identifies inconsistencies, anomalies, and potential misclassifications
"""

import os
import pandas as pd
from collections import defaultdict
from supabase import create_client
from dotenv import load_dotenv

load_dotenv('/root/pikkit/.env')

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
supabase = create_client(url, key)

print("="*80)
print("SPORT AND LEAGUE DATA QUALITY ANALYSIS")
print("="*80)
print()

# Fetch all bets
print("Fetching all bets from Supabase...")
all_data = []
batch_size = 1000
offset = 0

while True:
    response = supabase.table('bets').select(
        'id,sport,league,market,pick_name,bet_type'
    ).order('id').range(offset, offset + batch_size - 1).execute()

    if not response.data:
        break

    all_data.extend(response.data)

    if len(response.data) < batch_size:
        break

    offset += batch_size

df = pd.DataFrame(all_data)
print(f"Total bets: {len(df)}")
print()

# Known valid sport/league combinations
VALID_COMBINATIONS = {
    'American Football': ['NFL', 'NCAAFB', 'AFL', 'CFL', 'UFL'],
    'Australian Football': ['AFL'],
    'Baseball': ['MLB', 'NCAAB', 'NPB', 'KBO', 'WBC', 'LMB', 'CPBL', 'NCAA Baseball'],
    'Basketball': ['NBA', 'NCAAB', 'WNBA', 'EuroLeague', 'Euroleague', 'NBL', 'BBL', 'French Pro A',
                   'International Basketball', 'International', 'ACB', 'VTB', 'German BBL', 'Turkish BSL',
                   'B.League', 'CBA', 'China - CBA', 'EuroCup', 'Italy - Lega Basket Serie A',
                   'Italian Lega Basket', 'Spanish ACB', 'Israeli Basketball', 'EuroLeague Women',
                   'Japan - B1 League', 'KBL', 'NCAAW', 'Review'],
    'Cricket': ['County Championship', 'IPL'],
    'Darts': ['PDC'],
    'Soccer': ['MLS', 'Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1',
               'Champions League', 'Europa League', 'World Cup', 'International', 'International Soccer',
               'European Leagues', 'English Leagues', 'Copa Libertadores', 'Chilean Primera',
               'Irish League', 'Liga MX', 'Greek Super League', 'J-League', 'UEFA Euro',
               'Argentine Primera', 'NWSL', 'Jamaica Premier League', 'Serbian SuperLiga',
               'UEFA Champions League', 'UEFA Europa League'],
    'Ice Hockey': ['NHL', 'KHL', 'SHL', 'Liiga', 'AHL'],
    'Tennis': ['ATP', 'WTA', 'ITF', 'Grand Slam', 'ATP/WTA', 'Review'],
    'MMA': ['UFC', 'Bellator', 'PFL'],
    'Golf': ['PGA', 'European Tour', 'LIV', 'PGA Tour'],
    'Rugby League': ['NRL', 'Super League'],
    'Table Tennis': ['ITTF', 'WTT', 'Czech Liga Pro'],
    'Parlay': ['Parlay'],
    'Multi-Sport': ['Multi-Sport'],
    'Unknown': ['Review'],
}

# Analysis 1: Sport/League Distribution
print("="*80)
print("1. SPORT AND LEAGUE DISTRIBUTION")
print("="*80)
print()

sport_counts = df['sport'].value_counts()
print("Sport Distribution:")
for sport, count in sport_counts.items():
    pct = count / len(df) * 100
    print(f"  {sport:30} {count:7,} ({pct:5.1f}%)")
print()

# Analysis 2: Sport/League Combinations
print("="*80)
print("2. SPORT/LEAGUE COMBINATIONS")
print("="*80)
print()

combinations = df.groupby(['sport', 'league']).size().reset_index(name='count')
combinations = combinations.sort_values(['sport', 'count'], ascending=[True, False])

for sport in sorted(df['sport'].dropna().unique()):
    sport_combos = combinations[combinations['sport'] == sport]
    print(f"\n{sport}:")
    for _, row in sport_combos.iterrows():
        league = row['league'] if pd.notna(row['league']) else '(NULL)'
        count = row['count']
        pct = count / sport_counts[sport] * 100

        # Flag suspicious combinations
        is_valid = True
        if sport in VALID_COMBINATIONS:
            if pd.notna(row['league']) and row['league'] not in VALID_COMBINATIONS[sport]:
                is_valid = False

        flag = "⚠️ " if not is_valid else "   "
        print(f"  {flag}{league:30} {count:6,} ({pct:5.1f}%)")

# Analysis 3: Potential Anomalies
print()
print("="*80)
print("3. POTENTIAL ANOMALIES")
print("="*80)
print()

anomalies = []

# Check for league mismatches
for sport, valid_leagues in VALID_COMBINATIONS.items():
    sport_data = df[df['sport'] == sport]
    if len(sport_data) == 0:
        continue

    invalid_leagues = sport_data[
        sport_data['league'].notna() &
        ~sport_data['league'].isin(valid_leagues + ['Parlay'])
    ]

    if len(invalid_leagues) > 0:
        for league in invalid_leagues['league'].unique():
            count = len(invalid_leagues[invalid_leagues['league'] == league])
            anomalies.append({
                'type': 'Invalid League',
                'sport': sport,
                'league': league,
                'count': count,
                'severity': 'HIGH'
            })

# Check for NULL leagues in non-parlay bets
null_leagues = df[
    (df['league'].isna()) &
    (df['bet_type'] != 'parlay') &
    (df['sport'] != 'Parlay')
]

if len(null_leagues) > 0:
    for sport in null_leagues['sport'].unique():
        if pd.isna(sport):
            continue
        count = len(null_leagues[null_leagues['sport'] == sport])
        anomalies.append({
            'type': 'NULL League',
            'sport': sport,
            'league': None,
            'count': count,
            'severity': 'MEDIUM'
        })

# Check for NULL sports
null_sports = df[df['sport'].isna()]
if len(null_sports) > 0:
    anomalies.append({
        'type': 'NULL Sport',
        'sport': None,
        'league': None,
        'count': len(null_sports),
        'severity': 'HIGH'
    })

# Print anomalies
if anomalies:
    print("Found the following anomalies:\n")
    for anom in sorted(anomalies, key=lambda x: (x['severity'], -x['count'])):
        print(f"[{anom['severity']:6}] {anom['type']:20} | "
              f"Sport: {str(anom['sport'])[:20]:20} | "
              f"League: {str(anom['league'])[:20]:20} | "
              f"Count: {anom['count']:,}")
else:
    print("✅ No anomalies detected!")

# Analysis 4: Sample Suspicious Bets
print()
print("="*80)
print("4. SAMPLE SUSPICIOUS BETS FOR REVIEW")
print("="*80)
print()

# Show examples of anomalous bets
if anomalies:
    for anom in anomalies[:5]:  # Top 5 anomalies
        if anom['type'] == 'NULL Sport':
            sample = df[df['sport'].isna()].head(5)
        elif anom['type'] == 'NULL League':
            sample = df[
                (df['sport'] == anom['sport']) &
                (df['league'].isna())
            ].head(5)
        else:  # Invalid League
            sample = df[
                (df['sport'] == anom['sport']) &
                (df['league'] == anom['league'])
            ].head(5)

        print(f"\n{anom['type']}: {anom['sport']} / {anom['league']} ({anom['count']} bets)")
        print("Sample bets:")
        for _, bet in sample.iterrows():
            pick = (bet['pick_name'] or '')[:60]
            print(f"  {bet['id']:25} | {pick}")

# Analysis 5: Recommendations
print()
print("="*80)
print("5. RECOMMENDATIONS")
print("="*80)
print()

recommendations = []

# Calculate stats
total_anomalies = sum(a['count'] for a in anomalies)
pct_anomalous = total_anomalies / len(df) * 100 if len(df) > 0 else 0

print(f"Total bets: {len(df):,}")
print(f"Anomalous bets: {total_anomalies:,} ({pct_anomalous:.2f}%)")
print()

if total_anomalies > 0:
    print("Recommended actions:")
    print()

    if any(a['type'] == 'NULL Sport' for a in anomalies):
        null_sport_count = next(a['count'] for a in anomalies if a['type'] == 'NULL Sport')
        print(f"1. Run backfill_sports.py to fill {null_sport_count:,} NULL sport values")
        recommendations.append('backfill')

    if any(a['type'] == 'NULL League' for a in anomalies):
        null_league_count = sum(a['count'] for a in anomalies if a['type'] == 'NULL League')
        print(f"2. Use ML classifier to predict {null_league_count:,} missing leagues")
        recommendations.append('ml_predict')

    if any(a['type'] == 'Invalid League' for a in anomalies):
        invalid_count = sum(a['count'] for a in anomalies if a['type'] == 'Invalid League')
        print(f"3. Review {invalid_count:,} invalid sport/league combinations")
        recommendations.append('manual_review')

    print()
    print("To fix these issues:")
    if 'backfill' in recommendations:
        print("  python3 /root/pikkit/backfill_sports.py")
    if 'ml_predict' in recommendations:
        print("  python3 /root/pikkit/ml/predict_non_parlay_leagues.py")
else:
    print("✅ Data quality is excellent! No action needed.")

print()
print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
