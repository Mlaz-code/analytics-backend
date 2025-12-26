# Pikkit ML Pipeline Enhancement Implementation Guide

## Overview

Three major enhancements have been added to the Pikkit ML pipeline:

1. **Drift Detection** - Automated monitoring of data and model drift with alerting
2. **Automated Retraining** - Scheduled and event-driven model retraining workflows
3. **Feature Store** - Centralized feature management with Feast for consistent training/serving

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ENHANCED ML PIPELINE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────┐         │
│  │ Data Sources │───▶│ Feature Store│───▶│ Drift Detector│         │
│  │ (Supabase)   │    │ (Feast)      │    │ (PSI, KS test)│         │
│  └──────────────┘    └──────────────┘    └───────────────┘         │
│                            │                      │                 │
│                            ▼                      ▼                 │
│                      ┌──────────────┐      ┌──────────────┐        │
│                      │  Training    │◀─────│ Auto Retrain │        │
│                      │  Pipeline    │      │ Orchestrator │        │
│                      └──────────────┘      └──────────────┘        │
│                            │                      │                 │
│                            ▼                      ▼                 │
│                      ┌──────────────┐      ┌──────────────┐        │
│                      │ Model        │      │  Alerting    │        │
│                      │ Registry     │      │  (Telegram)  │        │
│                      └──────────────┘      └──────────────┘        │
│                            │                                        │
│                            ▼                                        │
│                      ┌──────────────┐                               │
│                      │   Serving    │                               │
│                      │   (FastAPI)  │                               │
│                      └──────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
```

## Phase 1: Drift Detection

### What's Included

- `ml/monitoring/drift_detector.py` - Statistical drift detection (PSI, KS test)
- `ml/scripts/check_drift_and_retrain.py` - Automated drift checking script
- `ml/scripts/create_baseline_reference.py` - Create baseline reference data
- `mlops/app/main.py` - New `/api/v1/drift-report` endpoint

### Setup Steps

#### 1. Create Baseline Reference Data

This establishes the "normal" data distribution for comparison:

```bash
cd /root/pikkit/ml
python3 scripts/create_baseline_reference.py
```

This will:
- Fetch 30 days of historical betting data
- Save to `/root/pikkit/ml/data/baseline_reference.parquet`
- Log summary statistics

#### 2. Test Drift Detection

Run a manual drift check:

```bash
python3 scripts/check_drift_and_retrain.py
```

This will:
- Compare recent 7 days vs baseline
- Generate drift report in `/root/pikkit/ml/reports/drift_reports/`
- Send Telegram alert if drift detected
- Exit code 0 = no drift, 1 = drift detected, 2 = error

#### 3. Setup Automated Checking (Cron)

Install cron jobs for automated drift monitoring:

```bash
cd /root/pikkit/ml
./scripts/setup_drift_cron.sh
```

This adds:
- Daily drift check at 2:00 AM
- Weekly baseline update on Sundays at 3:00 AM

View cron jobs:
```bash
crontab -l
```

View logs:
```bash
tail -f /var/log/pikkit-ml/drift-cron.log
tail -f /var/log/pikkit-ml/baseline-update.log
```

#### 4. Test API Endpoint

The FastAPI ML API now has a drift monitoring endpoint:

```bash
# If API is running
curl http://localhost:8001/api/v1/drift-report

# Or with parameters
curl "http://localhost:8001/api/v1/drift-report?days_lookback=7&min_samples=100"
```

### How It Works

**PSI (Population Stability Index)** for categorical features:
- Compares distribution of categories between baseline and current data
- PSI < 0.1: No drift
- 0.1 ≤ PSI < 0.2: Moderate drift
- PSI ≥ 0.25: Significant drift (retrain recommended)

**KS Test (Kolmogorov-Smirnov)** for numerical features:
- Tests if distributions are significantly different
- p-value < 0.05 indicates drift

**Retraining Trigger**:
- 3+ features with drift, OR
- Any feature with "high" severity drift

---

## Phase 2: Automated Retraining

### What's Included

- Drift-triggered retraining via cron
- Baseline refresh mechanism
- Telegram alerting for drift events
- Integration with n8n webhooks

### How It Works

1. **Daily Drift Check** (2:00 AM via cron)
   - Compares last 7 days vs baseline
   - If drift detected → Telegram alert + n8n webhook

2. **Manual Retraining**
   ```bash
   cd /root/pikkit/ml
   python3 scripts/train_market_profitability_model.py
   ```

3. **Automated Deployment** (via n8n or CI/CD)
   ```bash
   cd /root/pikkit/mlops
   ./scripts/deploy.sh canary  # Deploy as canary for testing
   ./scripts/deploy.sh deploy  # Full blue-green deployment
   ```

### Configuration

Edit drift checking frequency in crontab:
```bash
crontab -e

# Change from daily at 2 AM to every 12 hours:
0 */12 * * * cd /root/pikkit/ml && python3 scripts/check_drift_and_retrain.py
```

---

## Phase 3: Feature Store (Feast)

### What's Included

- `ml/feature_store/feature_repo/` - Feast feature repository
- `ml/feature_store/features.py` - Feature view definitions
- `ml/feature_store/materialize_features.py` - Feature computation script
- `ml/feature_store/README.md` - Detailed usage guide

### Setup Steps

#### 1. Install Feast

```bash
pip install feast
```

#### 2. Initialize Feature Store

```bash
cd /root/pikkit/ml/feature_store/feature_repo
feast apply
```

This registers:
- 3 entities: `sport_market_key`, `institution_name`, `sport_league_key`
- 3 feature views: `historical_performance`, `institution_features`, `league_features`

#### 3. Materialize Features

Compute and store features from Supabase data:

```bash
python3 /root/pikkit/ml/feature_store/materialize_features.py
```

This will:
- Fetch 90 days of betting data
- Compute aggregated features
- Save to offline store (parquet files in `ml/feature_store/data/`)
- Materialize to online store (SQLite for fast serving)

#### 4. Verify Features

```bash
cd /root/pikkit/ml/feature_store/feature_repo
feast feature-views list
feast entities list
```

#### 5. Setup Daily Feature Updates

Add to crontab:
```bash
0 1 * * * cd /root/pikkit/ml && python3 feature_store/materialize_features.py >> /var/log/pikkit-ml/features.log 2>&1
```

### Using Features in Training

```python
from feast import FeatureStore

fs = FeatureStore(repo_path='/root/pikkit/ml/feature_store/feature_repo')

# Get historical features for training
training_df = fs.get_historical_features(
    entity_df=bets_df,  # Your bets with timestamps
    features=[
        "historical_performance:sport_win_rate",
        "historical_performance:sport_market_roi",
        "institution_features:institution_win_rate",
        "league_features:league_avg_edge"
    ]
).to_df()
```

### Using Features in Serving (FastAPI)

Add to `mlops/app/main.py`:

```python
from feast import FeatureStore

# Initialize in lifespan
fs = FeatureStore(repo_path='/root/pikkit/ml/feature_store/feature_repo')

# Get online features for a bet
async def enrich_bet_with_features(bet: BetPredictionRequest):
    entity = {
        "sport_market_key": f"{bet.sport}_{bet.market}",
        "institution_name": bet.institution_name,
        "sport_league_key": f"{bet.sport}_{bet.league}"
    }

    features = fs.get_online_features(
        entity_rows=[entity],
        features=[
            "historical_performance:sport_market_win_rate",
            "institution_features:institution_roi",
            "league_features:league_roi"
        ]
    ).to_dict()

    return features
```

---

## Testing the Complete Pipeline

### 1. End-to-End Test

```bash
# Step 1: Create baseline
python3 /root/pikkit/ml/scripts/create_baseline_reference.py

# Step 2: Materialize features
python3 /root/pikkit/ml/feature_store/materialize_features.py

# Step 3: Check drift
python3 /root/pikkit/ml/scripts/check_drift_and_retrain.py

# Step 4: Verify API endpoints
curl http://localhost:8001/api/v1/drift-report
curl http://localhost:8001/health
```

### 2. Simulate Drift

To test drift detection alerts:

```python
import pandas as pd
import numpy as np

# Load baseline
baseline = pd.read_parquet('/root/pikkit/ml/data/baseline_reference.parquet')

# Create drifted data (shift distributions)
drifted = baseline.copy()
drifted['implied_prob'] = drifted['implied_prob'] * 1.2  # 20% shift
drifted['sport'] = drifted['sport'].replace({'Basketball': 'Hockey'})  # Category shift

# Save as current data
drifted.to_parquet('/root/pikkit/ml/data/current_test.parquet')

# Now run drift detector on this test data
```

---

## Monitoring and Maintenance

### Daily Checks

```bash
# View drift logs
tail -50 /var/log/pikkit-ml/drift-cron.log

# View feature materialization logs
tail -50 /var/log/pikkit-ml/features.log

# Check latest drift report
ls -ltr /root/pikkit/ml/reports/drift_reports/ | tail -1
cat /root/pikkit/ml/reports/drift_reports/drift_report_*.json | jq .
```

### Weekly Tasks

- Review drift reports for trends
- Verify baseline is still representative
- Check feature store freshness
- Monitor model performance metrics

### Monthly Tasks

- Recreate baseline with recent data
- Review and tune drift thresholds
- Audit feature quality
- Update feature definitions if needed

---

## Troubleshooting

### Drift Detection Not Working

**Issue**: No drift reports generated

```bash
# Check if baseline exists
ls -l /root/pikkit/ml/data/baseline_reference.parquet

# If missing, create it
python3 /root/pikkit/ml/scripts/create_baseline_reference.py

# Check cron is running
crontab -l | grep drift
systemctl status cron
```

### Feature Store Errors

**Issue**: "Feature view not found"

```bash
cd /root/pikkit/ml/feature_store/feature_repo
feast apply
```

**Issue**: "No data in offline store"

```bash
python3 /root/pikkit/ml/feature_store/materialize_features.py
```

### Telegram Alerts Not Sending

```bash
# Check environment variables
echo $TELEGRAM_BOT_TOKEN
echo $TELEGRAM_CHAT_ID

# Verify in .env file
grep TELEGRAM /root/pikkit/.env

# Test manual alert
curl -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
  -d chat_id="$TELEGRAM_CHAT_ID" \
  -d text="Test alert"
```

---

## Next Steps

1. **Integrate with Training Pipeline**
   - Modify `scripts/train_market_profitability_model.py` to use Feast features
   - Add drift checking before training
   - Implement automatic retraining on drift

2. **Enhance Monitoring**
   - Add Grafana dashboards for drift metrics
   - Set up Prometheus metrics for feature freshness
   - Track retraining frequency and success rates

3. **A/B Testing**
   - Implement outcome tracking for canary deployments
   - Compare model performance between versions
   - Automated promotion based on statistical significance

4. **Advanced Features**
   - Real-time feature computation
   - Feature importance tracking
   - Automated feature engineering
   - Cross-validation with time-series awareness

---

## File Structure

```
/root/pikkit/ml/
├── monitoring/
│   ├── __init__.py
│   └── drift_detector.py          # Statistical drift detection
├── feature_store/
│   ├── feature_repo/
│   │   ├── feature_store.yaml     # Feast configuration
│   │   └── features.py            # Feature definitions
│   ├── data/                      # Feature parquet files (generated)
│   ├── materialize_features.py    # Feature computation
│   └── README.md                  # Feature store guide
├── scripts/
│   ├── check_drift_and_retrain.py # Automated drift checking
│   ├── create_baseline_reference.py # Baseline creation
│   └── setup_drift_cron.sh        # Cron setup script
├── data/
│   ├── baseline_reference.parquet # Drift baseline (generated)
│   └── current_test.parquet       # Test data (optional)
├── reports/
│   └── drift_reports/             # JSON drift reports (generated)
└── ML_PIPELINE_ENHANCEMENT_GUIDE.md # This file
```

---

## Support

For issues or questions:
- Check logs in `/var/log/pikkit-ml/`
- Review drift reports in `ml/reports/drift_reports/`
- Read feature store README: `ml/feature_store/README.md`
- Test components individually before end-to-end testing
