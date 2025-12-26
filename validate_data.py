#!/usr/bin/env python3
"""
Pikkit Data Validation Script
Compares Supabase data against Pikkit API (source of truth)
Generates report and optionally fixes mismatches
"""

import os
import sys
import json
import requests
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
import argparse

# Load Supabase config
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

# Tolerance for floating point comparisons
FLOAT_TOLERANCE = 0.01


class DataValidator:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.pikkit_bets: Dict[str, dict] = {}
        self.supabase_bets: Dict[str, dict] = {}
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'missing_in_supabase': [],
            'missing_in_pikkit': [],
            'status_mismatches': [],
            'amount_mismatches': [],
            'odds_mismatches': [],
            'profit_mismatches': [],
            'other_mismatches': [],
        }

    def fetch_pikkit_bets(self, limit: Optional[int] = None) -> int:
        """Fetch all bets from Pikkit API (source of truth)"""
        self.pikkit_bets = {}
        offset = 0
        batch_size = 50

        headers = {
            'Authorization': PIKKIT_API_TOKEN,
            'Accept': 'application/json',
            'Origin': 'https://app.pikkit.com',
            'User-Agent': 'Mozilla/5.0'
        }

        print("Fetching bets from Pikkit API (source of truth)...")

        while True:
            url = f"{PIKKIT_API_BASE}/user/bets?offset={offset}&limit={batch_size}"

            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"  Error fetching from Pikkit at offset {offset}: {e}")
                break

            bets = response.json()
            if not bets:
                break

            for bet in bets:
                bet_id = bet.get('_id')
                if bet_id:
                    self.pikkit_bets[bet_id] = bet

            if self.verbose or len(self.pikkit_bets) % 500 == 0:
                print(f"  Fetched {len(self.pikkit_bets)} bets...")

            if len(bets) < batch_size:
                break

            offset += batch_size

            # Optional limit for testing
            if limit and len(self.pikkit_bets) >= limit:
                break

        print(f"  Total Pikkit bets: {len(self.pikkit_bets)}")
        return len(self.pikkit_bets)

    def fetch_supabase_bets(self) -> int:
        """Fetch all bets from Supabase"""
        self.supabase_bets = {}
        offset = 0
        batch_size = 1000

        headers = {
            'Authorization': f'Bearer {SUPABASE_KEY}',
            'apikey': SUPABASE_KEY,
        }

        # Select fields needed for comparison (skip raw_json to save bandwidth)
        select_fields = ','.join([
            'id', 'status', 'amount', 'odds', 'profit', 'roi',
            'is_live', 'bet_type', 'pick_name', 'sport', 'league', 'market',
            'time_placed', 'institution_name'
        ])

        print("Fetching bets from Supabase...")

        while True:
            url = f"{SUPABASE_URL}/rest/v1/bets?select={select_fields}&limit={batch_size}&offset={offset}"

            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"  Error fetching from Supabase at offset {offset}: {e}")
                break

            bets = response.json()
            if not bets:
                break

            for bet in bets:
                bet_id = bet.get('id')
                if bet_id:
                    self.supabase_bets[bet_id] = bet

            if self.verbose or len(self.supabase_bets) % 1000 == 0:
                print(f"  Fetched {len(self.supabase_bets)} bets...")

            if len(bets) < batch_size:
                break

            offset += batch_size

        print(f"  Total Supabase bets: {len(self.supabase_bets)}")
        return len(self.supabase_bets)

    def _floats_equal(self, a: Optional[float], b: Optional[float]) -> bool:
        """Compare floats with tolerance"""
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return abs(float(a) - float(b)) <= FLOAT_TOLERANCE

    def _normalize_status(self, status: str) -> str:
        """Normalize status for comparison"""
        if not status:
            return ''
        return status.upper().strip()

    def compare_bets(self):
        """Compare all bets between Pikkit and Supabase"""
        print("\nComparing data...")

        pikkit_ids = set(self.pikkit_bets.keys())
        supabase_ids = set(self.supabase_bets.keys())

        # Find missing bets
        missing_in_supabase = pikkit_ids - supabase_ids
        missing_in_pikkit = supabase_ids - pikkit_ids
        common_ids = pikkit_ids & supabase_ids

        self.report['missing_in_supabase'] = list(missing_in_supabase)
        self.report['missing_in_pikkit'] = list(missing_in_pikkit)

        # Compare common bets
        matched = 0
        mismatched = 0

        for bet_id in common_ids:
            pikkit = self.pikkit_bets[bet_id]
            supabase = self.supabase_bets[bet_id]

            has_mismatch = False

            # Compare STATUS (most critical)
            p_status = self._normalize_status(pikkit.get('status', ''))
            s_status = self._normalize_status(supabase.get('status', ''))

            if p_status != s_status:
                has_mismatch = True
                self.report['status_mismatches'].append({
                    'id': bet_id,
                    'pikkit_status': p_status,
                    'supabase_status': s_status,
                    'pikkit_profit': pikkit.get('profit'),
                    'pick_name': supabase.get('pick_name', '')[:60],
                    'sport': supabase.get('sport'),
                    'needs_update': p_status.startswith('SETTLED') and not s_status.startswith('SETTLED'),
                })

            # Compare AMOUNT
            p_amount = pikkit.get('amount') or 0
            s_amount = supabase.get('amount') or 0

            if not self._floats_equal(p_amount, s_amount):
                has_mismatch = True
                self.report['amount_mismatches'].append({
                    'id': bet_id,
                    'pikkit_amount': p_amount,
                    'supabase_amount': s_amount,
                    'diff': abs(p_amount - s_amount),
                })

            # Compare ODDS
            p_odds = pikkit.get('odds') or 0
            s_odds = supabase.get('odds') or 0

            if not self._floats_equal(p_odds, s_odds):
                has_mismatch = True
                self.report['odds_mismatches'].append({
                    'id': bet_id,
                    'pikkit_odds': p_odds,
                    'supabase_odds': s_odds,
                })

            # Compare PROFIT (only if bet is settled in Pikkit)
            if p_status.startswith('SETTLED'):
                p_profit = pikkit.get('profit')
                s_profit = supabase.get('profit')

                if p_profit is not None and not self._floats_equal(p_profit, s_profit):
                    has_mismatch = True
                    self.report['profit_mismatches'].append({
                        'id': bet_id,
                        'pikkit_profit': p_profit,
                        'supabase_profit': s_profit,
                        'pikkit_status': p_status,
                    })

            if has_mismatch:
                mismatched += 1
            else:
                matched += 1

        # Build summary
        self.report['summary'] = {
            'pikkit_total': len(self.pikkit_bets),
            'supabase_total': len(self.supabase_bets),
            'common': len(common_ids),
            'matched': matched,
            'mismatched': mismatched,
            'missing_in_supabase': len(missing_in_supabase),
            'missing_in_pikkit': len(missing_in_pikkit),
            'status_mismatches': len(self.report['status_mismatches']),
            'amount_mismatches': len(self.report['amount_mismatches']),
            'odds_mismatches': len(self.report['odds_mismatches']),
            'profit_mismatches': len(self.report['profit_mismatches']),
            'pending_status_updates': len([m for m in self.report['status_mismatches'] if m.get('needs_update')]),
        }

        print(f"  Comparison complete: {matched} matched, {mismatched} mismatched")

    def print_report(self):
        """Print formatted report to console"""
        summary = self.report['summary']

        print("\n" + "=" * 70)
        print("PIKKIT vs SUPABASE DATA VALIDATION REPORT")
        print(f"Generated: {self.report['timestamp']}")
        print("=" * 70)

        print(f"\n{'SUMMARY':^70}")
        print("-" * 70)
        print(f"  Pikkit (source of truth):  {summary['pikkit_total']:,} bets")
        print(f"  Supabase:                  {summary['supabase_total']:,} bets")
        print(f"  Common bets:               {summary['common']:,}")
        print(f"  Matched (identical):       {summary['matched']:,}")
        print(f"  Mismatched (differences):  {summary['mismatched']:,}")

        print(f"\n{'SYNC STATUS':^70}")
        print("-" * 70)
        print(f"  Missing in Supabase:       {summary['missing_in_supabase']:,} (not synced)")
        print(f"  Missing in Pikkit:         {summary['missing_in_pikkit']:,} (orphaned)")

        print(f"\n{'MISMATCHES BY TYPE':^70}")
        print("-" * 70)
        print(f"  Status mismatches:         {summary['status_mismatches']:,}")
        print(f"    - Pending settlements:   {summary['pending_status_updates']:,}")
        print(f"  Amount mismatches:         {summary['amount_mismatches']:,}")
        print(f"  Odds mismatches:           {summary['odds_mismatches']:,}")
        print(f"  Profit mismatches:         {summary['profit_mismatches']:,}")

        # Show pending status updates (critical)
        pending = [m for m in self.report['status_mismatches'] if m.get('needs_update')]
        if pending:
            print(f"\n{'PENDING STATUS UPDATES (Settled in Pikkit, not in Supabase)':^70}")
            print("-" * 70)
            for i, m in enumerate(pending[:15]):
                profit_str = f"${m['pikkit_profit']:.2f}" if m['pikkit_profit'] else "N/A"
                print(f"  {m['id'][:24]}... {m['supabase_status']:10} -> {m['pikkit_status']:15} profit: {profit_str}")
            if len(pending) > 15:
                print(f"  ... and {len(pending) - 15} more")

        # Show sample status mismatches (other than pending updates)
        other_status = [m for m in self.report['status_mismatches'] if not m.get('needs_update')]
        if other_status:
            print(f"\n{'OTHER STATUS MISMATCHES (sample)':^70}")
            print("-" * 70)
            for i, m in enumerate(other_status[:10]):
                print(f"  {m['id'][:24]}... Pikkit: {m['pikkit_status']:15} Supabase: {m['supabase_status']}")
            if len(other_status) > 10:
                print(f"  ... and {len(other_status) - 10} more")

        # Show profit mismatches
        if self.report['profit_mismatches']:
            print(f"\n{'PROFIT MISMATCHES (sample)':^70}")
            print("-" * 70)
            for i, m in enumerate(self.report['profit_mismatches'][:10]):
                p_profit = m['pikkit_profit'] or 0
                s_profit = m['supabase_profit'] or 0
                diff = p_profit - s_profit
                print(f"  {m['id'][:24]}... Pikkit: ${p_profit:>8.2f}  Supabase: ${s_profit:>8.2f}  Diff: ${diff:>+.2f}")
            if len(self.report['profit_mismatches']) > 10:
                print(f"  ... and {len(self.report['profit_mismatches']) - 10} more")

        print("\n" + "=" * 70)

    def save_report(self, filepath: str):
        """Save full report to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.report, f, indent=2, default=str)
        print(f"\nFull report saved to: {filepath}")

    def fix_status_mismatches(self, dry_run: bool = True) -> int:
        """Fix status mismatches by updating Supabase with Pikkit data"""
        pending = [m for m in self.report['status_mismatches'] if m.get('needs_update')]

        if not pending:
            print("\nNo pending status updates to fix.")
            return 0

        if dry_run:
            print(f"\n[DRY RUN] Would fix {len(pending)} status mismatches")
            return len(pending)

        print(f"\nFixing {len(pending)} status mismatches...")

        headers = {
            'Authorization': f'Bearer {SUPABASE_KEY}',
            'apikey': SUPABASE_KEY,
            'Content-Type': 'application/json',
        }

        fixed = 0
        failed = 0

        for m in pending:
            bet_id = m['id']
            pikkit = self.pikkit_bets.get(bet_id, {})

            # Get updated values from Pikkit
            update_data = {
                'status': pikkit.get('status'),
                'profit': pikkit.get('profit'),
            }

            # Recalculate ROI
            amount = pikkit.get('amount') or 0
            profit = pikkit.get('profit') or 0
            if amount > 0:
                update_data['roi'] = (profit / amount) * 100

            url = f"{SUPABASE_URL}/rest/v1/bets?id=eq.{bet_id}"

            try:
                response = requests.patch(url, headers=headers, json=update_data, timeout=10)
                if response.ok:
                    fixed += 1
                else:
                    failed += 1
                    if self.verbose:
                        print(f"  Failed to update {bet_id}: {response.text}")
            except requests.RequestException as e:
                failed += 1
                if self.verbose:
                    print(f"  Error updating {bet_id}: {e}")

            if (fixed + failed) % 50 == 0:
                print(f"  Progress: {fixed + failed}/{len(pending)} ({fixed} fixed, {failed} failed)")

        print(f"\nFixed {fixed} bets, {failed} failed")
        return fixed

    def fix_profit_mismatches(self, dry_run: bool = True) -> int:
        """Fix profit mismatches by updating Supabase with Pikkit data"""
        mismatches = self.report['profit_mismatches']

        if not mismatches:
            print("\nNo profit mismatches to fix.")
            return 0

        if dry_run:
            print(f"\n[DRY RUN] Would fix {len(mismatches)} profit mismatches")
            return len(mismatches)

        print(f"\nFixing {len(mismatches)} profit mismatches...")

        headers = {
            'Authorization': f'Bearer {SUPABASE_KEY}',
            'apikey': SUPABASE_KEY,
            'Content-Type': 'application/json',
        }

        fixed = 0
        failed = 0

        for m in mismatches:
            bet_id = m['id']
            pikkit = self.pikkit_bets.get(bet_id, {})

            profit = pikkit.get('profit')
            amount = pikkit.get('amount') or 0

            update_data = {'profit': profit}
            if amount > 0 and profit is not None:
                update_data['roi'] = (profit / amount) * 100

            url = f"{SUPABASE_URL}/rest/v1/bets?id=eq.{bet_id}"

            try:
                response = requests.patch(url, headers=headers, json=update_data, timeout=10)
                if response.ok:
                    fixed += 1
                else:
                    failed += 1
            except requests.RequestException as e:
                failed += 1

            if (fixed + failed) % 50 == 0:
                print(f"  Progress: {fixed + failed}/{len(mismatches)} ({fixed} fixed, {failed} failed)")

        print(f"\nFixed {fixed} bets, {failed} failed")
        return fixed


def main():
    parser = argparse.ArgumentParser(description='Validate Pikkit data against Supabase')
    parser.add_argument('--fix', action='store_true', help='Fix mismatches (default: dry run)')
    parser.add_argument('--fix-status', action='store_true', help='Fix only status mismatches')
    parser.add_argument('--fix-profit', action='store_true', help='Fix only profit mismatches')
    parser.add_argument('--limit', type=int, help='Limit number of Pikkit bets to fetch (for testing)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--report', type=str, default='/root/pikkit/validation_report.json',
                        help='Path to save report')
    parser.add_argument('--full', action='store_true',
                        help='Fetch ALL bets from Pikkit (no limit, slower)')

    args = parser.parse_args()

    validator = DataValidator(verbose=args.verbose)

    # Fetch data
    validator.fetch_pikkit_bets(limit=args.limit)
    validator.fetch_supabase_bets()

    # Compare
    validator.compare_bets()

    # Print and save report
    validator.print_report()
    validator.save_report(args.report)

    # Fix if requested
    if args.fix or args.fix_status:
        validator.fix_status_mismatches(dry_run=not args.fix and not args.fix_status)

    if args.fix or args.fix_profit:
        validator.fix_profit_mismatches(dry_run=not args.fix and not args.fix_profit)

    # Return summary for scripting
    return validator.report['summary']


if __name__ == '__main__':
    main()
