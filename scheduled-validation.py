#!/usr/bin/env python3
"""
Automated validation report generator
Run nightly at 2 AM via cron: 0 2 * * * python3 /root/pikkit/scheduled-validation.py
"""

import os
import json
import sys
from datetime import datetime
from pathlib import Path

# Add pikkit to path
sys.path.insert(0, '/root/pikkit')

from validate_data import DataValidator

def generate_full_validation_report():
    """Generate complete validation report for all 28000 bets"""
    
    reports_dir = Path('/root/pikkit/reports')
    reports_dir.mkdir(exist_ok=True)
    
    log_file = Path('/var/log/pikkit-validation-report.log')
    log_file.parent.mkdir(exist_ok=True)
    
    def log_msg(msg):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{timestamp}] {msg}"
        print(line)
        with open(log_file, 'a') as f:
            f.write(line + '\n')
    
    try:
        log_msg("="*60)
        log_msg("STARTING AUTOMATED VALIDATION REPORT GENERATION")
        log_msg("="*60)
        
        # Initialize validator
        validator = DataValidator()
        
        # Fetch all Pikkit bets (no limit)
        log_msg("Fetching all Pikkit bets...")
        validator.fetch_pikkit_bets(limit=None)
        pikkit_count = len(validator.pikkit_bets)
        log_msg(f"✓ Fetched {pikkit_count} Pikkit bets")
        
        # Fetch all Supabase bets
        log_msg("Fetching all Supabase bets...")
        validator.fetch_supabase_bets()
        supabase_count = len(validator.supabase_bets)
        log_msg(f"✓ Fetched {supabase_count} Supabase bets")
        
        # Compare and generate report
        log_msg("Comparing bets and generating report...")
        validator.compare_bets()
        
        # Save report
        report_file = reports_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(validator.report, f, indent=2)
        
        log_msg(f"✓ Report saved to {report_file}")
        
        # Log summary
        summary = validator.report.get('summary', {})
        log_msg("")
        log_msg("VALIDATION SUMMARY")
        log_msg("-" * 60)
        log_msg(f"  Pikkit Total:        {summary.get('pikkit_total', 0):>10}")
        log_msg(f"  Supabase Total:      {summary.get('supabase_total', 0):>10}")
        log_msg(f"  Matched:             {summary.get('matched', 0):>10}")
        log_msg(f"  Status Mismatches:   {summary.get('status_mismatches', 0):>10}")
        log_msg(f"  Profit Mismatches:   {summary.get('profit_mismatches', 0):>10}")
        log_msg(f"  Missing in Pikkit:   {summary.get('missing_in_pikkit', 0):>10}")
        log_msg("-" * 60)
        
        total_issues = (
            len(summary.get('status_mismatches', [])) +
            len(summary.get('profit_mismatches', [])) +
            len(summary.get('missing_in_pikkit', []))
        )
        
        if total_issues == 0:
            log_msg("✓ NO ISSUES FOUND - All data is consistent!")
        else:
            log_msg(f"⚠ {total_issues} ISSUES FOUND - Review required")
        
        log_msg("="*60)
        log_msg("VALIDATION REPORT GENERATION COMPLETED SUCCESSFULLY")
        log_msg("="*60)
        
        return True
        
    except Exception as e:
        log_msg(f"✗ ERROR: {str(e)}")
        import traceback
        log_msg(traceback.format_exc())
        log_msg("="*60)
        log_msg("VALIDATION REPORT GENERATION FAILED")
        log_msg("="*60)
        return False

if __name__ == '__main__':
    success = generate_full_validation_report()
    sys.exit(0 if success else 1)
