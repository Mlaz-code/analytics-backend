#!/bin/bash
# Weekly XGBoost Model Retraining Script
# Runs every Sunday at 2am UTC

SCRIPT_DIR="/root/pikkit/ml/scripts"
LOG_FILE="/var/log/pikkit-ml-retrain.log"

echo "========================================" >> "$LOG_FILE"
echo "$(date): Starting weekly ML retrain" >> "$LOG_FILE"

# Source environment variables if they exist
if [ -f /root/pikkit/.env ]; then
    set -a  # Export all variables
    source /root/pikkit/.env
    set +a
fi

# Export for Python to use
export SUPABASE_URL
export SUPABASE_KEY

# Run training script
cd "$SCRIPT_DIR"
python3 train_market_profitability_model.py >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "$(date): Training completed successfully" >> "$LOG_FILE"
else
    echo "$(date): Training failed with exit code $EXIT_CODE" >> "$LOG_FILE"
fi

echo "========================================" >> "$LOG_FILE"

exit $EXIT_CODE
