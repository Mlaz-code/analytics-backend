#!/bin/bash
# Automated Pikkit Data Validation
# Runs daily via cron to ensure Supabase data matches Pikkit source of truth

set -e

SCRIPT_DIR="/root/pikkit"
LOG_DIR="/var/log/pikkit"
REPORT_DIR="/root/pikkit/reports"

# Create directories if needed
mkdir -p "$LOG_DIR" "$REPORT_DIR"

# Generate dated report filename
DATE=$(date +%Y-%m-%d)
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
REPORT_FILE="$REPORT_DIR/validation_$DATE.json"
LOG_FILE="$LOG_DIR/validation.log"

echo "============================================" >> "$LOG_FILE"
echo "Validation started: $(date)" >> "$LOG_FILE"

# Run validation (report only - no auto-fix, requires human review)
cd "$SCRIPT_DIR"
python3 validate_data.py \
    --full \
    --report "$REPORT_FILE" \
    >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

echo "Validation completed: $(date)" >> "$LOG_FILE"
echo "Exit code: $EXIT_CODE" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Parse report for summary (if jq is available)
if command -v jq &> /dev/null && [ -f "$REPORT_FILE" ]; then
    MISMATCHED=$(jq -r '.summary.mismatched // 0' "$REPORT_FILE")
    MISSING=$(jq -r '.summary.missing_in_supabase // 0' "$REPORT_FILE")
    STATUS_FIXES=$(jq -r '.summary.pending_status_updates // 0' "$REPORT_FILE")

    # Log summary
    echo "Summary: $MISMATCHED mismatches, $MISSING missing, $STATUS_FIXES status fixes" >> "$LOG_FILE"

    # Alert if significant issues (optional: integrate with Telegram)
    if [ "$MISMATCHED" -gt 10 ]; then
        echo "WARNING: High mismatch count ($MISMATCHED)" >> "$LOG_FILE"
    fi
fi

# Cleanup old reports (keep last 30 days)
find "$REPORT_DIR" -name "validation_*.json" -mtime +30 -delete 2>/dev/null || true

exit $EXIT_CODE
