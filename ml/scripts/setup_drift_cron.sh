#!/bin/bash
# Setup automated drift checking and retraining cron jobs

set -e

echo "Setting up Pikkit ML drift detection cron jobs..."

# Ensure log directory exists
mkdir -p /var/log/pikkit-ml

# Load environment variables
if [ -f /root/pikkit/.env ]; then
    export $(grep -v '^#' /root/pikkit/.env | xargs)
fi

# Create cron job for drift checking (daily at 2 AM)
DRIFT_CHECK_CRON="0 2 * * * cd /root/pikkit/ml && /usr/bin/python3 /root/pikkit/ml/scripts/check_drift_and_retrain.py >> /var/log/pikkit-ml/drift-cron.log 2>&1"

# Create cron job for weekly baseline update (Sunday at 3 AM)
BASELINE_UPDATE_CRON="0 3 * * 0 cd /root/pikkit/ml && /usr/bin/python3 /root/pikkit/ml/scripts/create_baseline_reference.py >> /var/log/pikkit-ml/baseline-update.log 2>&1"

# Check if cron jobs already exist
crontab -l 2>/dev/null | grep -q "check_drift_and_retrain.py" && {
    echo "Drift checking cron job already exists, updating..."
    crontab -l 2>/dev/null | grep -v "check_drift_and_retrain.py" | crontab -
} || echo "Adding new drift checking cron job..."

crontab -l 2>/dev/null | grep -q "create_baseline_reference.py" && {
    echo "Baseline update cron job already exists, updating..."
    crontab -l 2>/dev/null | grep -v "create_baseline_reference.py" | crontab -
} || echo "Adding new baseline update cron job..."

# Add cron jobs
(crontab -l 2>/dev/null; echo "$DRIFT_CHECK_CRON") | crontab -
(crontab -l 2>/dev/null; echo "$BASELINE_UPDATE_CRON") | crontab -

echo ""
echo "Cron jobs installed successfully!"
echo ""
echo "Drift checking: Daily at 2:00 AM"
echo "Baseline update: Weekly on Sunday at 3:00 AM"
echo ""
echo "View current cron jobs:"
echo "  crontab -l"
echo ""
echo "View logs:"
echo "  tail -f /var/log/pikkit-ml/drift-cron.log"
echo "  tail -f /var/log/pikkit-ml/baseline-update.log"
echo ""
echo "Manual drift check:"
echo "  python3 /root/pikkit/ml/scripts/check_drift_and_retrain.py"
echo ""
echo "Create initial baseline (run this first!):"
echo "  python3 /root/pikkit/ml/scripts/create_baseline_reference.py"
