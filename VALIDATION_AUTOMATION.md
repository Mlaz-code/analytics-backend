# Automated Validation Report Generation

## Overview
Automated validation reports are generated nightly at **2 AM UTC** when API traffic is low, analyzing all ~28,000 bets in both Pikkit and Supabase databases.

## Schedule
- **Time**: 2:00 AM daily (UTC)
- **Frequency**: Once per day
- **Coverage**: All bets (no limit)
- **Timezone**: System timezone (typically UTC on Tools VM)

## Files

### Cron Job
- **Location**: System crontab
- **Command**: `crontab -l` to view
- **Entry**: `0 2 * * * /usr/bin/python3 /root/pikkit/scheduled-validation.py`

### Scripts
1. **scheduled-validation.py**
   - Main Python script that generates full validation reports
   - Fetches all Pikkit bets (no limit parameter)
   - Fetches all Supabase bets
   - Compares and generates report
   - Saves report to `/root/pikkit/reports/`
   - Logs to `/var/log/pikkit-validation-report.log`

2. **generate-report.sh**
   - Bash wrapper script (optional fallback)
   - Calls validation API endpoint
   - Located at `/root/pikkit/generate-report.sh`

## Reports

### Location
Reports are saved to: `/root/pikkit/reports/`

### Naming Convention
`validation_YYYYMMDD_HHMMSS.json`

Example: `validation_20251222_020000.json`

### Report Contents
Each report includes:
- **Timestamp**: When the report was generated
- **Summary**: Statistics and counts
  - `pikkit_total`: Total bets from Pikkit
  - `supabase_total`: Total bets from Supabase
  - `matched`: Number of consistent bets
  - `status_mismatches`: Bets with wrong status
  - `profit_mismatches`: Bets with wrong profit amount
  - `missing_in_pikkit`: Bets in Supabase but not Pikkit
- **Detailed Issues**: 
  - Status mismatches with old/new values
  - Profit mismatches with differences
  - Rogue entries (extra in Supabase)

## Logs

### Log File
`/var/log/pikkit-validation-report.log`

### View Logs
```bash
# View recent validation runs
tail -100 /var/log/pikkit-validation-report.log

# Search for specific date
grep "2025-12-22" /var/log/pikkit-validation-report.log

# Watch logs in real-time
tail -f /var/log/pikkit-validation-report.log
```

## Manual Execution

### Run Report Generation Manually
```bash
# Python script (recommended)
python3 /root/pikkit/scheduled-validation.py

# Or via API
curl "http://localhost:5001/api/validation/generate"
```

### View Generated Reports
```bash
# List all reports
ls -lh /root/pikkit/reports/

# View latest report
cat /root/pikkit/reports/validation_*.json | tail -1

# View specific report
cat /root/pikkit/reports/validation_20251222_020000.json | jq .
```

## Cron Management

### Add New Schedule
```bash
# Edit crontab
crontab -e

# Add entry (e.g., for 3 AM instead)
# 0 3 * * * /usr/bin/python3 /root/pikkit/scheduled-validation.py
```

### View Current Jobs
```bash
crontab -l
```

### Remove Cron Job
```bash
crontab -r
```

## Troubleshooting

### Reports Not Generated
1. Check cron is running: `service cron status`
2. Check logs: `tail /var/log/pikkit-validation-report.log`
3. Test manually: `python3 /root/pikkit/scheduled-validation.py`
4. Verify API is running: `curl http://localhost:5001/`

### API Not Responding
1. Check if validation API is running: `ps aux | grep validation-api`
2. Check port 5001: `netstat -tulpn | grep 5001`
3. Check validation API logs: `tail /tmp/validation.log`
4. Restart API: `python3 /root/pikkit/validation-api.py`

### Permission Issues
```bash
# Fix log file permissions
sudo touch /var/log/pikkit-validation-report.log
sudo chmod 666 /var/log/pikkit-validation-report.log
```

## Integration with Validation UI

The generated reports are automatically available in the validation UI:
- Access: `http://192.168.4.80:5001/`
- Reports list: `/api/validation/reports`
- Specific report: `/api/validation/report/<filename>`

## Performance Notes

- **Processing Time**: ~2-5 minutes for 28,000 bets (depending on network)
- **Report Size**: ~2-5 MB
- **Best Time**: 2 AM UTC (low API traffic)
- **Cleanup**: Old reports are retained indefinitely; manually delete if needed

## Next Steps

1. Monitor first few automated runs
2. Review `/var/log/pikkit-validation-report.log` after 2 AM
3. Check generated reports in `/root/pikkit/reports/`
4. Adjust time if needed via `crontab -e`
5. Set up alerts for issues found (future enhancement)
