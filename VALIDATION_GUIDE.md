# Pikkit Validation Review System

Automated validation of Pikkit bets against Supabase with interactive review and correction interface.

## System Architecture

```
┌─────────────────────────────────────────┐
│  Weekly Cron (Sundays 7am EST)         │
│  /root/pikkit/run_validation.sh        │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Validation Script                      │
│  /root/pikkit/validate_data.py          │
│  - Fetches all Pikkit bets              │
│  - Compares with Supabase               │
│  - Generates JSON report                │
└────────────────┬────────────────────────┘
                 │
                 ▼
      /root/pikkit/reports/
    validation_YYYY-MM-DD.json
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Web UI + API                           │
│  http://localhost:5001                  │
│  - Load latest/specific reports         │
│  - Select corrections interactively     │
│  - Apply with one click                 │
└─────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `/root/pikkit/validate_data.py` | Core validation script - compares Pikkit API vs Supabase |
| `/root/pikkit/review_validation.py` | CLI tool for interactive review (alternative to UI) |
| `/root/pikkit/validation-api.py` | Flask API backend serving UI and handling corrections |
| `/root/pikkit/validation-ui/` | Web UI (HTML/JS) |
| `/root/pikkit/run_validation.sh` | Cron wrapper script |
| `/root/pikkit/reports/` | JSON validation reports |
| `/root/pikkit/corrections/` | Applied correction logs |

## Usage

### Option 1: Web UI (Recommended)

1. **Access the UI:**
   ```
   http://192.168.4.80:5001
   ```

2. **Load Report:**
   - Click "📥 Load Latest Report"
   - Automatically loads most recent validation

3. **Review Issues:**
   - **Status Mismatches** - newly settled bets
   - **Profit Mismatches** - discrepancies in profit calculations
   - **Rogue Entries** - in Supabase but not in Pikkit

4. **Select Corrections:**
   - Check boxes next to items you want to fix
   - Summary shows what will be applied

5. **Apply:**
   - Click "✓ Apply Selected"
   - Confirm the action
   - Results shown immediately

### Option 2: CLI Tool

```bash
# Interactive review (dry run - shows what would change)
python3 /root/pikkit/review_validation.py

# Auto-approve status updates and apply them
python3 /root/pikkit/review_validation.py --auto-status --apply

# Review specific report
python3 /root/pikkit/review_validation.py --report /path/to/report.json

# Apply all corrections (dangerous - requires explicit input)
python3 /root/pikkit/review_validation.py --apply
```

### Manual Validation Run

```bash
# Test with 100 entries (fast)
python3 /root/pikkit/validate_data.py --limit 100

# Full validation (slow, fetches all Pikkit bets)
python3 /root/pikkit/validate_data.py --full

# Save to custom location
python3 /root/pikkit/validate_data.py --report /path/to/custom_report.json

# Verbose output
python3 /root/pikkit/validate_data.py -v --full
```

## Scheduled Runs

**Weekly validation:** Sundays at 7am EST (12pm UTC)
- Cron: `0 12 * * 0 /root/pikkit/run_validation.sh`
- Log: `/var/log/pikkit/validation.log`
- Report: `/root/pikkit/reports/validation_YYYY-MM-DD.json`

## Windows Access

Via Samba shared folder:

| Item | Windows Path |
|------|--------------|
| Reports | `T:\pikkit-validation-reports\` |
| Logs | `T:\pikkit-logs\` |
| Corrections | `T:\pikkit-corrections\` |

## Report Structure

```json
{
  "timestamp": "2025-12-22T17:00:00",
  "summary": {
    "pikkit_total": 18750,
    "supabase_total": 18650,
    "matched": 18600,
    "mismatched": 50,
    "missing_in_supabase": 100,
    "missing_in_pikkit": 50,
    "status_mismatches": 10,
    "amount_mismatches": 0,
    "odds_mismatches": 0,
    "profit_mismatches": 40
  },
  "status_mismatches": [
    {
      "id": "...",
      "pikkit_status": "SETTLED_WIN",
      "supabase_status": "PENDING",
      "pikkit_profit": 50.00,
      "needs_update": true,
      "pick_name": "NBA Moneyline"
    }
  ],
  "profit_mismatches": [...],
  "missing_in_supabase": [...],
  "missing_in_pikkit": [...]
}
```

## Correction Types

### 1. Status Updates
Bets that are settled in Pikkit but not yet marked as settled in Supabase.
- **Safe to apply:** Yes - brings Supabase current with Pikkit
- **Action:** Updates status and profit fields

### 2. Profit Mismatches
Discrepancies in calculated profit (rounding, calculation errors).
- **Safe to apply:** After review - verify profit is correct in Pikkit
- **Action:** Overwrites Supabase profit with Pikkit value

### 3. Rogue Entries
Entries in Supabase that don't exist in Pikkit (orphaned).
- **Safe to apply:** After review - investigate why they exist first
- **Action:** Deletes from Supabase

## Monitoring

Check API health:
```bash
systemctl status pikkit-validation-api
journalctl -u pikkit-validation-api -n 50 -f
```

Check logs:
```bash
tail -f /var/log/pikkit/validation-api.log
tail -f /var/log/pikkit/validation.log
```

## Troubleshooting

### "No validation reports found"
- Run validation manually: `python3 /root/pikkit/validate_data.py --full`
- Check cron ran: `grep pikkit /var/log/cron*`

### UI shows blank panels
- Clear browser cache (Ctrl+Shift+Del)
- Refresh page
- Check API logs: `journalctl -u pikkit-validation-api -n 100`

### Corrections not applying
- Check Supabase credentials in `/root/pikkit/.env`
- Verify API logs for error details
- Try applying single item first to test

### Port already in use
Change port in `/root/pikkit/validation-api.py` and restart service

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serve UI |
| `/api/validation/reports` | GET | List available reports |
| `/api/validation/report/<filename>` | GET | Get specific report |
| `/api/validation/apply` | POST | Apply corrections |
| `/api/validation/stats` | GET | Get overall statistics |

## Backup & History

Reports are automatically kept for 30 days in `/root/pikkit/reports/`
- Older reports automatically deleted
- Correction logs saved in `/root/pikkit/corrections/`
- Full audit trail preserved

## Future Enhancements

- [ ] Email notifications on validation issues
- [ ] Webhook integration (n8n, Telegram)
- [ ] Automated corrections for specific categories
- [ ] Dashboard showing validation trends
- [ ] Export reports to CSV/Excel
- [ ] Batch operations across multiple reports
