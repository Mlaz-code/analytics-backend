#!/usr/bin/env python3
"""
Automated Drift Checking and Retraining Script
Run via cron to monitor data drift and trigger retraining when needed
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, '/root/pikkit/ml')

from monitoring.drift_detector import DriftDetector, ModelDriftDetector
import requests
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/pikkit-ml/drift-check.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://mnnjjvbaxzumfcgibtme.supabase.co')
SUPABASE_SERVICE_KEY = os.environ.get('SUPABASE_SERVICE_KEY', '')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
N8N_WEBHOOK_URL = os.environ.get('N8N_WEBHOOK_URL', '')

DRIFT_REPORT_PATH = '/root/pikkit/ml/reports/drift_reports'
BASELINE_DATA_PATH = '/root/pikkit/ml/data/baseline_reference.parquet'

# Ensure directories exist
Path(DRIFT_REPORT_PATH).mkdir(parents=True, exist_ok=True)
Path('/var/log/pikkit-ml').mkdir(parents=True, exist_ok=True)


def send_telegram_alert(message: str):
    """Send alert via Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not configured, skipping alert")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        response = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }, timeout=10)
        response.raise_for_status()
        logger.info("Telegram alert sent successfully")
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")


def fetch_recent_data(days: int = 7) -> pd.DataFrame:
    """
    Fetch recent betting data from Supabase.

    Args:
        days: Number of days to look back

    Returns:
        DataFrame with recent bet data
    """
    logger.info(f"Fetching {days} days of recent data from Supabase")

    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Construct Supabase query
        headers = {
            'apikey': SUPABASE_SERVICE_KEY,
            'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
            'Content-Type': 'application/json'
        }

        # Query for settled bets in date range
        params = {
            'is_settled': 'eq.true',
            'created_at': f'gte.{start_date.isoformat()}',
            'select': 'sport,league,market,institution_name,bet_type,american_odds,clv_percentage,clv_ev,is_live',
            'limit': '50000',
            'order': 'created_at.desc'
        }

        url = f"{SUPABASE_URL}/rest/v1/bets"
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        df = pd.DataFrame(data)

        # Calculate implied probability from american_odds
        if 'american_odds' in df.columns:
            def calc_implied_prob(odds):
                if pd.isna(odds) or odds == 0:
                    return 0.5
                if odds < 0:
                    return abs(odds) / (abs(odds) + 100)
                else:
                    return 100 / (odds + 100)

            df['implied_prob'] = df['american_odds'].apply(calc_implied_prob)

        logger.info(f"Fetched {len(df)} recent bets")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch recent data: {e}")
        return pd.DataFrame()


def load_baseline_data() -> pd.DataFrame:
    """
    Load baseline reference data.

    This should be historical "good" data that represents normal operation.
    """
    baseline_path = Path(BASELINE_DATA_PATH)

    if not baseline_path.exists():
        logger.warning(f"Baseline data not found at {baseline_path}")
        logger.info("You need to create baseline reference data by running:")
        logger.info("  python3 /root/pikkit/ml/scripts/create_baseline_reference.py")
        return pd.DataFrame()

    try:
        df = pd.read_parquet(baseline_path)
        logger.info(f"Loaded baseline data: {len(df)} samples")
        return df
    except Exception as e:
        logger.error(f"Failed to load baseline data: {e}")
        return pd.DataFrame()


def check_drift():
    """Main drift checking workflow"""
    logger.info("=" * 60)
    logger.info("DRIFT DETECTION CHECK STARTED")
    logger.info("=" * 60)

    # Load baseline data
    baseline_df = load_baseline_data()
    if baseline_df.empty:
        logger.error("Cannot perform drift detection without baseline data")
        return False

    # Fetch recent data
    recent_df = fetch_recent_data(days=7)
    if recent_df.empty:
        logger.error("No recent data available for drift detection")
        return False

    if len(recent_df) < 100:
        logger.warning(f"Insufficient recent data ({len(recent_df)} samples), need at least 100")
        return False

    # Initialize drift detector
    categorical_features = ['sport', 'league', 'market', 'institution_name', 'bet_type']
    numerical_features = ['implied_prob', 'clv_percentage', 'clv_ev']

    detector = DriftDetector(
        reference_data=baseline_df,
        categorical_features=categorical_features,
        numerical_features=numerical_features
    )

    # Detect drift
    logger.info("Running drift detection...")
    results = detector.detect_drift(recent_df)
    report = detector.generate_drift_report(results)

    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = Path(DRIFT_REPORT_PATH) / f'drift_report_{timestamp}.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Drift report saved to: {report_file}")

    # Log results
    logger.info(f"Features checked: {report['total_features_checked']}")
    logger.info(f"Features with drift: {report['features_with_drift']}")
    logger.info(f"Drifted features: {report['drifted_features']}")
    logger.info(f"Requires retraining: {report['requires_retraining']}")

    # Send alert if drift detected
    if report['requires_retraining']:
        alert_message = (
            f"<b>🚨 Data Drift Detected</b>\n\n"
            f"Timestamp: {report['timestamp']}\n"
            f"Features with drift: {report['features_with_drift']}\n"
            f"Drifted features: {', '.join(report['drifted_features'])}\n\n"
            f"<b>Action Required:</b> Model retraining recommended\n\n"
            f"Report: {report_file}"
        )
        send_telegram_alert(alert_message)

        # Trigger n8n webhook for retraining workflow
        if N8N_WEBHOOK_URL:
            try:
                requests.post(N8N_WEBHOOK_URL, json={
                    'type': 'drift_detected',
                    'drift_report': report,
                    'requires_retraining': True
                }, timeout=10)
                logger.info("n8n retraining webhook triggered")
            except Exception as e:
                logger.error(f"Failed to trigger n8n webhook: {e}")

    logger.info("=" * 60)
    logger.info("DRIFT DETECTION CHECK COMPLETED")
    logger.info("=" * 60)

    return report['requires_retraining']


if __name__ == '__main__':
    try:
        requires_retraining = check_drift()
        sys.exit(0 if not requires_retraining else 1)  # Exit 1 if retraining needed
    except Exception as e:
        logger.error(f"Drift check failed: {e}", exc_info=True)
        sys.exit(2)  # Exit 2 on error
