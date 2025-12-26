# Pikkit Feature Store (Feast)

Centralized feature management for training and serving.

## Setup

### 1. Install Feast

```bash
pip install feast
```

### 2. Initialize Feature Store

```bash
cd /root/pikkit/ml/feature_store/feature_repo
feast apply
```

This registers the feature definitions with the feature store.

### 3. Materialize Features

Fetch data from Supabase and compute features:

```bash
python3 /root/pikkit/ml/feature_store/materialize_features.py
```

This script:
- Fetches 90 days of betting data from Supabase
- Computes aggregated features (performance, institution, league metrics)
- Saves to offline store (parquet files)
- Materializes to online store (SQLite) for low-latency serving

### 4. Verify Features

```bash
cd /root/pikkit/ml/feature_store/feature_repo
feast feature-views list
feast entities list
```

## Feature Views

### historical_performance
Sport + Market level performance metrics:
- `sport_win_rate`: Win rate for the sport
- `sport_roi`: ROI for the sport
- `sport_market_win_rate`: Win rate for sport-market combination
- `sport_market_roi`: ROI for sport-market combination
- `sport_market_count`: Number of bets for this combination
- `avg_clv`: Average CLV across bets
- `recent_win_rate_10`: Win rate of last 10 bets
- `recent_roi_10`: ROI of last 10 bets

**Entity**: `sport_market_key` (e.g., "Basketball_Spread")

### institution_features
Sportsbook-specific metrics:
- `institution_win_rate`: Win rate at this sportsbook
- `institution_roi`: ROI at this sportsbook
- `institution_count`: Total bets at this sportsbook
- `institution_avg_odds`: Average odds offered
- `institution_avg_clv`: Average CLV
- `institution_sharp_pct`: Percentage of bets with positive CLV

**Entity**: `institution_name` (e.g., "DraftKings")

### league_features
League-specific patterns:
- `league_win_rate`: Win rate for this league
- `league_roi`: ROI for this league
- `league_avg_edge`: Average betting edge
- `league_total_bets`: Total bets in this league
- `league_recent_performance`: Recent ROI trend

**Entity**: `sport_league_key` (e.g., "Basketball_NBA")

## Usage in Training

```python
from feast import FeatureStore

fs = FeatureStore(repo_path='/root/pikkit/ml/feature_store/feature_repo')

# Get training features
training_df = fs.get_historical_features(
    entity_df=entity_df,  # DataFrame with entity keys and timestamps
    features=[
        "historical_performance:sport_win_rate",
        "historical_performance:sport_market_win_rate",
        "institution_features:institution_roi",
        "league_features:league_avg_edge"
    ]
).to_df()
```

## Usage in Serving (Real-time Inference)

```python
from feast import FeatureStore

fs = FeatureStore(repo_path='/root/pikkit/ml/feature_store/feature_repo')

# Get online features for a single bet
features = fs.get_online_features(
    entity_rows=[{
        "sport_market_key": "Basketball_Spread",
        "institution_name": "DraftKings",
        "sport_league_key": "Basketball_NBA"
    }],
    features=[
        "historical_performance:sport_market_win_rate",
        "historical_performance:sport_market_roi",
        "institution_features:institution_win_rate",
        "league_features:league_roi"
    ]
).to_dict()
```

## Maintenance

### Update Features (Run Daily)

```bash
python3 /root/pikkit/ml/feature_store/materialize_features.py
```

### Cron Job (Automated Daily Updates)

Add to crontab:
```
0 1 * * * cd /root/pikkit/ml && python3 feature_store/materialize_features.py >> /var/log/pikkit-ml/feature-materialization.log 2>&1
```

### Check Feature Store Status

```bash
cd /root/pikkit/ml/feature_store/feature_repo
feast feature-views list
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

## Troubleshooting

**Error: "No module named 'feast'"**
```bash
pip install feast
```

**Error: "Feature view not found"**
```bash
cd /root/pikkit/ml/feature_store/feature_repo
feast apply
```

**Error: "No data in offline store"**
```bash
python3 /root/pikkit/ml/feature_store/materialize_features.py
```

## Next Steps

1. Integrate with training pipeline (`train_market_profitability_model.py`)
2. Use online features in FastAPI serving (`mlops/app/main.py`)
3. Set up automated materialization cron job
4. Monitor feature freshness and quality
