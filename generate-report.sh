#!/bin/bash
# Generate validation report for all 28000 bets overnight

set -e

# Log file
LOG_FILE="/var/log/pikkit-validation-report.log"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Start report generation
log_message "Starting automated validation report generation..."

cd /root/pikkit

# Generate report via API (no limit = all bets)
log_message "Generating report for all bets..."
RESPONSE=$(curl -s "http://localhost:5001/api/validation/generate")

# Check if successful
if echo "$RESPONSE" | grep -q '"success":true'; then
    log_message "✓ Report generated successfully"
    echo "$RESPONSE" | jq . >> "$LOG_FILE" 2>/dev/null || echo "$RESPONSE" >> "$LOG_FILE"
else
    log_message "✗ Failed to generate report"
    echo "$RESPONSE" >> "$LOG_FILE"
    exit 1
fi

log_message "Validation report generation completed"
