#!/usr/bin/env python3
"""
Pikkit Validation Review Tool
Interactive tool to review validation reports and apply human-approved corrections
"""

import os
import sys
import json
import requests
from datetime import datetime
from typing import Dict, List, Optional

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
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_KEY', '')
REPORTS_DIR = '/root/pikkit/reports'
CORRECTIONS_DIR = '/root/pikkit/corrections'

# Ensure corrections directory exists
os.makedirs(CORRECTIONS_DIR, exist_ok=True)


def get_latest_report() -> Optional[str]:
    """Get the most recent validation report"""
    if not os.path.exists(REPORTS_DIR):
        return None
    reports = [f for f in os.listdir(REPORTS_DIR) if f.startswith('validation_') and f.endswith('.json')]
    if not reports:
        return None
    reports.sort(reverse=True)
    return os.path.join(REPORTS_DIR, reports[0])


def load_report(filepath: str) -> dict:
    """Load a validation report"""
    with open(filepath) as f:
        return json.load(f)


def display_summary(report: dict):
    """Display report summary"""
    summary = report.get('summary', {})
    print("\n" + "=" * 70)
    print("VALIDATION REPORT SUMMARY")
    print("=" * 70)
    print(f"  Generated: {report.get('timestamp', 'Unknown')}")
    print(f"  Pikkit bets: {summary.get('pikkit_total', 0):,}")
    print(f"  Supabase bets: {summary.get('supabase_total', 0):,}")
    print(f"  Matched: {summary.get('matched', 0):,}")
    print(f"  Mismatched: {summary.get('mismatched', 0):,}")
    print()
    print("  Issues found:")
    print(f"    - Status mismatches: {summary.get('status_mismatches', 0)}")
    print(f"    - Amount mismatches: {summary.get('amount_mismatches', 0)}")
    print(f"    - Odds mismatches: {summary.get('odds_mismatches', 0)}")
    print(f"    - Profit mismatches: {summary.get('profit_mismatches', 0)}")
    print(f"    - Missing in Supabase: {summary.get('missing_in_supabase', 0)}")
    print(f"    - Rogue (not in Pikkit): {summary.get('missing_in_pikkit', 0)}")
    print("=" * 70)


def review_status_mismatches(report: dict) -> List[dict]:
    """Review status mismatches and collect corrections"""
    mismatches = report.get('status_mismatches', [])
    if not mismatches:
        print("\nNo status mismatches to review.")
        return []

    corrections = []
    print(f"\n{'STATUS MISMATCHES':^70}")
    print("-" * 70)
    print(f"Found {len(mismatches)} status mismatches\n")

    # Group by type
    pending = [m for m in mismatches if m.get('needs_update')]
    other = [m for m in mismatches if not m.get('needs_update')]

    if pending:
        print(f"PENDING SETTLEMENTS ({len(pending)} bets settled in Pikkit, not in Supabase):")
        print("-" * 70)
        for i, m in enumerate(pending[:20], 1):
            profit = m.get('pikkit_profit')
            profit_str = f"${profit:.2f}" if profit is not None else "N/A"
            print(f"  {i:3}. {m['id'][:24]}...")
            print(f"       Supabase: {m['supabase_status']} -> Pikkit: {m['pikkit_status']}")
            print(f"       Profit: {profit_str} | {m.get('pick_name', '')[:50]}")
            print()

        if len(pending) > 20:
            print(f"  ... and {len(pending) - 20} more\n")

        response = input(f"\nApply all {len(pending)} pending status updates? [y/n/select]: ").strip().lower()

        if response == 'y':
            for m in pending:
                corrections.append({
                    'id': m['id'],
                    'action': 'update_status',
                    'field': 'status',
                    'old_value': m['supabase_status'],
                    'new_value': m['pikkit_status'],
                    'profit': m.get('pikkit_profit'),
                })
            print(f"  Added {len(pending)} corrections to queue")
        elif response == 'select':
            indices = input("Enter numbers to include (comma-separated, e.g., 1,3,5): ").strip()
            try:
                selected = [int(x.strip()) - 1 for x in indices.split(',') if x.strip()]
                for idx in selected:
                    if 0 <= idx < len(pending):
                        m = pending[idx]
                        corrections.append({
                            'id': m['id'],
                            'action': 'update_status',
                            'field': 'status',
                            'old_value': m['supabase_status'],
                            'new_value': m['pikkit_status'],
                            'profit': m.get('pikkit_profit'),
                        })
                print(f"  Added {len(selected)} corrections to queue")
            except ValueError:
                print("  Invalid input, skipping")

    return corrections


def review_profit_mismatches(report: dict) -> List[dict]:
    """Review profit mismatches and collect corrections"""
    mismatches = report.get('profit_mismatches', [])
    if not mismatches:
        print("\nNo profit mismatches to review.")
        return []

    corrections = []
    print(f"\n{'PROFIT MISMATCHES':^70}")
    print("-" * 70)
    print(f"Found {len(mismatches)} profit mismatches\n")

    for i, m in enumerate(mismatches[:20], 1):
        p_profit = m.get('pikkit_profit') or 0
        s_profit = m.get('supabase_profit') or 0
        diff = p_profit - s_profit
        print(f"  {i:3}. {m['id'][:24]}...")
        print(f"       Pikkit: ${p_profit:>8.2f}  Supabase: ${s_profit:>8.2f}  Diff: ${diff:>+.2f}")
        print()

    if len(mismatches) > 20:
        print(f"  ... and {len(mismatches) - 20} more\n")

    response = input(f"\nApply all {len(mismatches)} profit corrections from Pikkit? [y/n/select]: ").strip().lower()

    if response == 'y':
        for m in mismatches:
            corrections.append({
                'id': m['id'],
                'action': 'update_profit',
                'field': 'profit',
                'old_value': m['supabase_profit'],
                'new_value': m['pikkit_profit'],
            })
        print(f"  Added {len(mismatches)} corrections to queue")
    elif response == 'select':
        indices = input("Enter numbers to include (comma-separated): ").strip()
        try:
            selected = [int(x.strip()) - 1 for x in indices.split(',') if x.strip()]
            for idx in selected:
                if 0 <= idx < len(mismatches):
                    m = mismatches[idx]
                    corrections.append({
                        'id': m['id'],
                        'action': 'update_profit',
                        'field': 'profit',
                        'old_value': m['supabase_profit'],
                        'new_value': m['pikkit_profit'],
                    })
            print(f"  Added {len(selected)} corrections to queue")
        except ValueError:
            print("  Invalid input, skipping")

    return corrections


def review_rogue_entries(report: dict) -> List[dict]:
    """Review entries in Supabase but not in Pikkit"""
    rogue_ids = report.get('missing_in_pikkit', [])
    if not rogue_ids:
        print("\nNo rogue entries found.")
        return []

    corrections = []
    print(f"\n{'ROGUE ENTRIES (in Supabase, not in Pikkit)':^70}")
    print("-" * 70)
    print(f"Found {len(rogue_ids)} rogue entries\n")

    # Fetch details for first 20
    headers = {
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'apikey': SUPABASE_KEY,
    }

    sample_ids = rogue_ids[:20]
    id_filter = ','.join(sample_ids)
    url = f"{SUPABASE_URL}/rest/v1/bets?id=in.({id_filter})&select=id,pick_name,amount,status,time_placed"

    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.ok:
            bets = response.json()
            for i, bet in enumerate(bets, 1):
                print(f"  {i:3}. {bet['id'][:24]}...")
                print(f"       {bet.get('pick_name', 'Unknown')[:50]}")
                print(f"       Amount: ${bet.get('amount', 0):.2f} | Status: {bet.get('status', 'Unknown')}")
                print()
    except Exception as e:
        print(f"  Error fetching details: {e}")
        for i, rid in enumerate(sample_ids, 1):
            print(f"  {i:3}. {rid}")

    if len(rogue_ids) > 20:
        print(f"  ... and {len(rogue_ids) - 20} more\n")

    print("\nOptions:")
    print("  [d] Delete all rogue entries")
    print("  [s] Select specific entries to delete")
    print("  [i] Ignore (keep in Supabase)")
    print("  [f] Flag for manual review")

    response = input("\nChoice: ").strip().lower()

    if response == 'd':
        confirm = input(f"Confirm DELETE {len(rogue_ids)} entries? Type 'DELETE' to confirm: ").strip()
        if confirm == 'DELETE':
            for rid in rogue_ids:
                corrections.append({
                    'id': rid,
                    'action': 'delete',
                    'reason': 'rogue_entry',
                })
            print(f"  Added {len(rogue_ids)} deletions to queue")
    elif response == 's':
        indices = input("Enter numbers to delete (comma-separated): ").strip()
        try:
            selected = [int(x.strip()) - 1 for x in indices.split(',') if x.strip()]
            for idx in selected:
                if 0 <= idx < len(sample_ids):
                    corrections.append({
                        'id': sample_ids[idx],
                        'action': 'delete',
                        'reason': 'rogue_entry',
                    })
            print(f"  Added {len(selected)} deletions to queue")
        except ValueError:
            print("  Invalid input, skipping")
    elif response == 'f':
        # Save for manual review
        flag_file = os.path.join(CORRECTIONS_DIR, f"flagged_rogue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(flag_file, 'w') as f:
            json.dump({'rogue_ids': rogue_ids, 'flagged_at': datetime.now().isoformat()}, f, indent=2)
        print(f"  Flagged {len(rogue_ids)} entries for review: {flag_file}")

    return corrections


def apply_corrections(corrections: List[dict], dry_run: bool = True) -> dict:
    """Apply corrections to Supabase"""
    if not corrections:
        print("\nNo corrections to apply.")
        return {'applied': 0, 'failed': 0}

    if dry_run:
        print(f"\n[DRY RUN] Would apply {len(corrections)} corrections")
        return {'applied': len(corrections), 'failed': 0, 'dry_run': True}

    headers = {
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'apikey': SUPABASE_KEY,
        'Content-Type': 'application/json',
    }

    applied = 0
    failed = 0
    results = []

    print(f"\nApplying {len(corrections)} corrections...")

    for c in corrections:
        bet_id = c['id']
        action = c['action']

        try:
            if action == 'delete':
                url = f"{SUPABASE_URL}/rest/v1/bets?id=eq.{bet_id}"
                response = requests.delete(url, headers=headers, timeout=10)
            elif action in ('update_status', 'update_profit'):
                url = f"{SUPABASE_URL}/rest/v1/bets?id=eq.{bet_id}"
                update_data = {c['field']: c['new_value']}

                # Also update profit and ROI for status updates
                if action == 'update_status' and c.get('profit') is not None:
                    update_data['profit'] = c['profit']
                    # ROI will be calculated by Supabase trigger or we calculate here

                response = requests.patch(url, headers=headers, json=update_data, timeout=10)
            else:
                print(f"  Unknown action: {action}")
                failed += 1
                continue

            if response.ok:
                applied += 1
                results.append({'id': bet_id, 'action': action, 'status': 'success'})
            else:
                failed += 1
                results.append({'id': bet_id, 'action': action, 'status': 'failed', 'error': response.text})

        except Exception as e:
            failed += 1
            results.append({'id': bet_id, 'action': action, 'status': 'error', 'error': str(e)})

        if (applied + failed) % 25 == 0:
            print(f"  Progress: {applied + failed}/{len(corrections)}")

    print(f"\nCompleted: {applied} applied, {failed} failed")

    # Save correction log
    log_file = os.path.join(CORRECTIONS_DIR, f"applied_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(log_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'corrections': corrections,
            'results': results,
            'summary': {'applied': applied, 'failed': failed}
        }, f, indent=2)
    print(f"Correction log saved: {log_file}")

    return {'applied': applied, 'failed': failed}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Review validation report and apply corrections')
    parser.add_argument('--report', type=str, help='Path to validation report (default: latest)')
    parser.add_argument('--apply', action='store_true', help='Actually apply corrections (default: dry run)')
    parser.add_argument('--auto-status', action='store_true', help='Auto-approve all status updates')
    parser.add_argument('--auto-profit', action='store_true', help='Auto-approve all profit updates')
    args = parser.parse_args()

    # Load report
    report_path = args.report or get_latest_report()
    if not report_path or not os.path.exists(report_path):
        print("No validation report found. Run validate_data.py first.")
        sys.exit(1)

    print(f"Loading report: {report_path}")
    report = load_report(report_path)

    # Display summary
    display_summary(report)

    # Collect corrections
    all_corrections = []

    # Review each category
    if args.auto_status:
        pending = [m for m in report.get('status_mismatches', []) if m.get('needs_update')]
        for m in pending:
            all_corrections.append({
                'id': m['id'],
                'action': 'update_status',
                'field': 'status',
                'old_value': m['supabase_status'],
                'new_value': m['pikkit_status'],
                'profit': m.get('pikkit_profit'),
            })
        print(f"\nAuto-approved {len(pending)} status updates")
    else:
        all_corrections.extend(review_status_mismatches(report))

    if args.auto_profit:
        mismatches = report.get('profit_mismatches', [])
        for m in mismatches:
            all_corrections.append({
                'id': m['id'],
                'action': 'update_profit',
                'field': 'profit',
                'old_value': m['supabase_profit'],
                'new_value': m['pikkit_profit'],
            })
        print(f"\nAuto-approved {len(mismatches)} profit updates")
    else:
        all_corrections.extend(review_profit_mismatches(report))

    all_corrections.extend(review_rogue_entries(report))

    # Summary of corrections
    if all_corrections:
        print(f"\n{'CORRECTION SUMMARY':^70}")
        print("-" * 70)
        by_action = {}
        for c in all_corrections:
            action = c['action']
            by_action[action] = by_action.get(action, 0) + 1
        for action, count in by_action.items():
            print(f"  {action}: {count}")
        print(f"  TOTAL: {len(all_corrections)}")

        # Apply
        if args.apply:
            confirm = input("\nApply these corrections? [y/n]: ").strip().lower()
            if confirm == 'y':
                apply_corrections(all_corrections, dry_run=False)
            else:
                print("Corrections cancelled.")
        else:
            print("\n[DRY RUN MODE] Use --apply to actually apply corrections")
            # Save pending corrections
            pending_file = os.path.join(CORRECTIONS_DIR, f"pending_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(pending_file, 'w') as f:
                json.dump(all_corrections, f, indent=2)
            print(f"Pending corrections saved: {pending_file}")
    else:
        print("\nNo corrections needed!")


if __name__ == '__main__':
    main()
