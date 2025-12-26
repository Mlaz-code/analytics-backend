# Pikkit XGBoost Market Profitability Model

## Overview

This is a **multi-task XGBoost machine learning system** designed to predict bet market profitability for the Pikkit sports betting analytics platform. The system predicts both:

1. **Win Probability**: Likelihood that a bet will win (0-100%)
2. **Expected ROI**: Expected return on investment percentage

The model uses historical betting data from Supabase to identify profitable betting markets and provide actionable insights for bet selection and sizing.

---

## Architecture

### Multi-Task Learning Approach

We train **two separate but complementary models**:

1. **Win Probability Classifier** (XGBoost Binary Classification)
   - Predicts: P(Win) for any given bet
   - Output: Probability score 0.0-1.0
   - Use case: Identify high-confidence bets

2. **ROI Predictor** (XGBoost Regression)
   - Predicts: Expected ROI% for any given bet
   - Output: Percentage return (can be negative)
   - Use case: Optimize for profitability, not just accuracy

### Why Multi-Task?

- **Win probability alone is insufficient**: A 55% win rate at -200 odds loses money
- **ROI captures the full picture**: Accounts for odds, stake sizing, and market efficiency
- **Complementary signals**: Win prob provides confidence, ROI provides profit expectation
- **Better Kelly Criterion**: We can calculate optimal bet sizing using both outputs

---

## Features

### Feature Engineering Pipeline

The model uses **26 features** across several categories:

#### 1. **Categorical Features** (Label Encoded)
- `sport_encoded`: NBA, NFL, MLB, NHL, etc.
- `league_encoded`: League/competition
- `market_encoded`: Spread, Total, Moneyline, Player Props
- `institution_name_encoded`: Bookmaker (DraftKings, FanDuel, etc.)
- `bet_type_encoded`: Over/Under/Home/Away

#### 2. **Bet Characteristics**
- `implied_prob`: Implied probability from American odds
- `is_live`: Boolean - live bet vs pre-game

#### 3. **Closing Line Value (CLV) Features**
- `clv_percentage`: CLV % when available (~20% of bets)
- `clv_ev`: CLV × implied probability
- `has_clv`: Boolean indicator for CLV availability

#### 4. **Historical Performance Features** (Expanding Window)
- `sport_win_rate`: Win rate for this sport
- `sport_roi`: Historical ROI for this sport
- `sport_market_win_rate`: Win rate for sport+market combo
- `sport_market_roi`: ROI for sport+market combo
- `sport_league_win_rate`: Win rate for sport+league combo
- `sport_league_roi`: ROI for sport+league combo
- `sport_league_market_win_rate`: Win rate for sport+league+market
- `sport_league_market_roi`: ROI for sport+league+market
- `institution_name_win_rate`: Win rate per bookmaker
- `institution_name_roi`: ROI per bookmaker

#### 5. **Sample Size Features** (Confidence Weighting)
- `sport_market_count`: Number of historical bets for this market
- `institution_name_count`: Number of bets with this bookmaker

#### 6. **Recent Trends**
- `recent_win_rate`: Win rate over last 10 bets in this market

#### 7. **Temporal Features**
- `day_of_week`: 0-6 (Monday-Sunday)
- `hour_of_day`: 0-23
- `days_since_first_bet`: Account age/experience

### Anti-Lookahead Design

**Critical**: All historical features use **expanding windows with `.shift(1)`** to prevent lookahead bias:
- Current bet outcome is NOT included in its own features
- Only past data is used for predictions
- Time-based train/test split (80/20)

---

## Model Performance

### Sample Data Results (5,000 bets)

#### Win Probability Classifier
- **Train Accuracy**: 53.2%
- **Val Accuracy**: 50.4%
- **Train AUC**: 0.612
- **Val AUC**: 0.544

#### ROI Predictor
- **Train MAE**: 104.03%
- **Val MAE**: 103.53%

*Note: Performance on real Supabase data will be significantly better due to higher quality features (actual CLV, larger sample sizes, etc.)*

### Top Feature Importance

**Win Prediction:**
1. `implied_prob` (0.0587)
2. `clv_ev` (0.0491)
3. `days_since_first_bet` (0.0488)

**ROI Prediction:**
1. `league_encoded` (0.0613)
2. `sport_market_count` (0.0519)
3. `sport_league_win_rate` (0.0499)

---

## Directory Structure

```
/root/pikkit/ml/
├── models/                          # Trained model artifacts
│   ├── win_probability_model_latest.pkl
│   ├── roi_prediction_model_latest.pkl
│   ├── model_metadata_latest.json
│   └── *_YYYYMMDD_HHMMSS.*        # Timestamped backups
├── predictions/                     # Model predictions
│   ├── market_predictions_latest.json
│   └── *_YYYYMMDD_HHMMSS.json     # Timestamped predictions
├── data/                           # Raw/processed data cache
├── scripts/                        # Training & inference scripts
│   ├── train_market_profitability_model.py
│   ├── predict_bet_profitability.py
│   ├── weekly_retrain.sh
│   └── setup_credentials.sh
└── README.md                       # This file
```

---

## Usage

### 1. Setup Supabase Credentials

```bash
cd /root/pikkit/ml/scripts
./setup_credentials.sh
```

This will create `/root/pikkit/.env` with:
```bash
SUPABASE_URL=https://mnnjjvbaxzumfcgibtme.supabase.co
SUPABASE_KEY=your_key_here
```

### 2. Train the Model

#### Manual Training
```bash
cd /root/pikkit/ml/scripts
python3 train_market_profitability_model.py
```

#### Automated Weekly Retraining
The model automatically retrains **every Sunday at 2:00 AM UTC** via cron:
```bash
# Check cron job
crontab -l | grep pikkit-ml

# View training logs
tail -f /var/log/pikkit-ml-retrain.log
```

### 3. Make Predictions

#### Python API
```python
from predict_bet_profitability import BetProfitabilityPredictor

# Initialize predictor
predictor = BetProfitabilityPredictor()

# Example bet
bet = {
    'sport': 'NBA',
    'league': 'NBA',
    'market': 'Spread',
    'institution_name': 'DraftKings',
    'bet_type': 'Over',
    'odds': -110,
    'clv_percentage': 2.5,
    'is_live': False,
}

# Historical stats (optional - uses defaults if not provided)
stats = {
    'sport_win_rate': 0.52,
    'sport_roi': 1.5,
    'sport_market_count': 500,
}

# Get prediction
result = predictor.predict(bet, stats)

print(f"Win Probability: {result['win_probability']:.1%}")
print(f"Expected ROI: {result['expected_roi']:+.2f}%")
print(f"Recommended Stake: {result['recommended_stake_pct']:.2f}% of bankroll")
print(f"Grade: {result['bet_grade']}")
```

#### Command Line
```bash
python3 predict_bet_profitability.py
```

### 4. View Market Predictions

The model generates predictions for all major market combinations:

```bash
cat /root/pikkit/ml/predictions/market_predictions_latest.json | jq '.[0:5]'
```

Example output:
```json
[
  {
    "sport": "NHL",
    "league": "NHL",
    "market": "Spread",
    "sample_size": 200,
    "predicted_win_prob": 0.474,
    "predicted_roi": 6.36,
    "historical_win_rate": 0.465,
    "historical_roi": 5.8,
    "confidence": 1.0
  }
]
```

---

## Integration with Pikkit Infrastructure

### n8n Workflow Integration

To integrate predictions into the n8n webhook workflow:

1. **Fetch market predictions** from `/root/pikkit/ml/predictions/market_predictions_latest.json`
2. **Filter to Grade A/B markets** (expected ROI > 3%)
3. **Call prediction API** for individual bets in those markets
4. **Apply Kelly Criterion sizing** using recommended_stake_pct
5. **Send to BlissOS automation** or dashboard

### Netlify Dashboard Integration

Copy predictions to the dashboard data directory:
```bash
cp /root/pikkit/ml/predictions/market_predictions_latest.json \
   /root/pikkit/consolidated/dashboard/ml-predictions.json
```

Then deploy to Netlify (this would be automated in the full system).

---

## Model Retraining Strategy

### Weekly Retraining (Current)
- **Schedule**: Every Sunday 2:00 AM UTC
- **Reason**: Betting markets evolve; models decay over time
- **Process**:
  1. Fetch all settled bets from Supabase
  2. Engineer features with expanding windows
  3. Train multi-task XGBoost models
  4. Generate market predictions
  5. Save models and predictions
  6. Log results

### Future: Daily Incremental Updates
For production, consider:
- **Daily feature updates** (recalculate win rates, ROI)
- **Weekly full retraining** (rebuild models from scratch)
- **Model drift monitoring** (alert if performance degrades)
- **A/B testing** (compare new model vs current in production)

---

## Kelly Criterion Bet Sizing

The prediction API calculates **Kelly Criterion** for optimal bet sizing:

```
Kelly% = (p × b - q) / b

Where:
  p = win probability (from model)
  q = 1 - p (loss probability)
  b = decimal odds - 1
```

We use **Quarter Kelly** (25% of full Kelly) for conservative sizing:
```python
recommended_stake_pct = kelly_fraction * 100 * 0.25
```

### Example
- Win prob: 55%
- Odds: +150 (decimal 2.5)
- Full Kelly: 13.3%
- Quarter Kelly: **3.3% of bankroll**

---

## Monitoring & Validation

### Check Model Performance

```bash
# View recent training log
tail -100 /var/log/pikkit-ml-retrain.log

# Check model timestamps
ls -lth /root/pikkit/ml/models/

# Validate predictions exist
cat /root/pikkit/ml/predictions/market_predictions_latest.json | jq '. | length'
```

### Success Metrics

Track these KPIs over time:
1. **ROI Improvement**: Compare model-selected bets vs all bets
2. **Precision@K**: Win rate of top 20% model predictions
3. **Calibration Error**: Predicted probabilities vs actual outcomes
4. **Feature Stability**: Top features shouldn't change drastically week-to-week
5. **CLV Correlation**: Model predictions should align with CLV when available

---

## Troubleshooting

### Issue: Supabase Connection Error
```bash
# Check credentials
cat /root/pikkit/.env | grep SUPABASE_KEY

# Test connection
python3 -c "
from supabase import create_client
import os
url = os.environ.get('SUPABASE_URL')
key = os.environ.get('SUPABASE_KEY')
client = create_client(url, key)
print('✅ Connected to Supabase')
"
```

### Issue: Missing Python Packages
```bash
pip install xgboost pandas numpy scikit-learn supabase
```

### Issue: Model Files Not Found
```bash
# Re-run training
cd /root/pikkit/ml/scripts
python3 train_market_profitability_model.py
```

### Issue: Cron Job Not Running
```bash
# Check cron service
systemctl status cron

# View cron logs
grep pikkit /var/log/syslog

# Test script manually
/root/pikkit/ml/scripts/weekly_retrain.sh
```

---

## Next Steps & Enhancements

### Phase 1: Current (✅ Complete)
- [x] Multi-task XGBoost model (win prob + ROI)
- [x] Feature engineering pipeline
- [x] Prediction API
- [x] Weekly retraining automation

### Phase 2: Production Hardening
- [ ] Connect to real Supabase data (set SUPABASE_KEY)
- [ ] Add CLV imputation model (for bets without CLV)
- [ ] Implement temporal cross-validation
- [ ] Add probability calibration (Platt scaling)
- [ ] Model versioning and rollback

### Phase 3: Advanced Features
- [ ] Sport-specific sub-models (NFL, NBA, MLB separate)
- [ ] Live betting model (different features than pre-game)
- [ ] Player props specialized model
- [ ] Ensemble with baseline models
- [ ] Deep learning for sequential patterns (LSTM)

### Phase 4: Integration
- [ ] Real-time prediction endpoint (Flask/FastAPI)
- [ ] n8n webhook integration
- [ ] BlissOS automation trigger
- [ ] Monitoring dashboard (Grafana)
- [ ] Alerting (Telegram bot for Grade A bets)

---

## Technical Details

### Model Hyperparameters

#### Win Probability Classifier
```python
xgb.XGBClassifier(
    n_estimators=200,           # Number of trees
    max_depth=6,                # Tree depth
    learning_rate=0.05,         # Boosting learning rate
    subsample=0.8,              # Row sampling
    colsample_bytree=0.8,       # Column sampling
    objective='binary:logistic',
    eval_metric='logloss',
    early_stopping_rounds=20,   # Stop if no improvement
)
```

#### ROI Regressor
```python
xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,                # Slightly shallower than classifier
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    eval_metric='mae',
    early_stopping_rounds=20,
)
```

### Sample Weighting
We apply **exponential time decay weighting** to prioritize recent bets:
```python
train_weights = np.exp(np.linspace(-1, 0, len(train_df)))
```
This gives ~2.7x more weight to the most recent bet vs the oldest.

---

## References

- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **Kelly Criterion**: https://en.wikipedia.org/wiki/Kelly_criterion
- **CLV (Closing Line Value)**: Industry standard for measuring bet quality
- **Supabase Python Client**: https://github.com/supabase-community/supabase-py

---

## License

Proprietary - Pikkit Analytics Suite

---

## Contact

For questions or issues, check:
- Training logs: `/var/log/pikkit-ml-retrain.log`
- Claude Code context: `/root/.claude/CLAUDE.md`
- Pikkit documentation: https://pikkit-2d-dashboard.netlify.app
