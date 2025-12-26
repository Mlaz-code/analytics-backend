# Pikkit XGBoost Model - Variables Reference

## Input Features (26 Total)

### 1. Categorical Features (5)
| Variable | Description | Example Values | Source |
|----------|-------------|----------------|--------|
| `sport_encoded` | Sport type (encoded as integer) | Basketball=0, American Football=1, Baseball=2, Ice Hockey=3, Tennis=4 | Supabase: `sport` |
| `league_encoded` | League within sport | NBA=0, NCAAB=1, NFL=2, NCAAF=3, MLB=4, NHL=5, etc. | Supabase: `league` |
| `market_encoded` | Betting market type | Spread=0, Total=1, Moneyline=2, Player Props=3, etc. | Supabase: `market` |
| `institution_name_encoded` | Bookmaker | DraftKings=0, FanDuel=1, BetMGM=2, etc. | Supabase: `institution_name` |
| `bet_type_encoded` | Type of wager | straight=0, parlay=1, etc. | Supabase: `bet_type` |

**Note**: Encoding is done via LabelEncoder during training. Mappings are saved in `model_metadata_latest.json`.

---

### 2. Bet Characteristics (2)
| Variable | Description | Formula | Range |
|----------|-------------|---------|-------|
| `implied_prob` | Probability implied by American odds | If odds < 0: `abs(odds)/(abs(odds)+100)`<br>If odds > 0: `100/(odds+100)` | 0.0 - 1.0 |
| `is_live` | Live bet vs pre-game | 1 if live, 0 if pre-game | 0 or 1 |

**Example**:
- Odds of -110 → implied_prob = 0.524 (52.4%)
- Odds of +150 → implied_prob = 0.400 (40.0%)

---

### 3. Closing Line Value (CLV) Features (3)
| Variable | Description | Example | Notes |
|----------|-------------|---------|-------|
| `clv_percentage` | % difference from closing line | +2.5% means you got 2.5% better odds than close | ~20% of bets have CLV data |
| `clv_ev` | Expected value from CLV | `clv_percentage × implied_prob` | Combines CLV with bet quality |
| `has_clv` | CLV data available? | 1 if yes, 0 if no | Used to weight CLV features |

**What is CLV?**: Professional bettors track CLV (closing line value) - the difference between the odds you got and the final "closing" odds before the game. Consistently beating the closing line is a strong indicator of long-term profitability.

---

### 4. Historical Performance Features (12)

#### Sport Level (2)
| Variable | Description | Example |
|----------|-------------|---------|
| `sport_win_rate` | Historical win rate for this sport | 0.485 = 48.5% win rate in Basketball |
| `sport_roi` | Historical ROI for this sport | -2.3 = -2.3% average ROI in Basketball |

#### Sport + Market Level (2)
| Variable | Description | Example |
|----------|-------------|---------|
| `sport_market_win_rate` | Win rate for sport+market combo | Basketball Spreads: 51.2% |
| `sport_market_roi` | ROI for sport+market combo | Basketball Spreads: +0.8% |

#### Sport + League Level (2)
| Variable | Description | Example |
|----------|-------------|---------|
| `sport_league_win_rate` | Win rate for sport+league | Basketball NCAAB: 49.1% |
| `sport_league_roi` | ROI for sport+league | Basketball NCAAB: -1.2% |

#### Sport + League + Market Level (2)
| Variable | Description | Example |
|----------|-------------|---------|
| `sport_league_market_win_rate` | Win rate for full combo | NCAAB Spreads: 52.3% |
| `sport_league_market_roi` | ROI for full combo | NCAAB Spreads: +1.5% |

#### Bookmaker Level (2)
| Variable | Description | Example |
|----------|-------------|---------|
| `institution_name_win_rate` | Win rate with this bookmaker | DraftKings: 50.2% |
| `institution_name_roi` | ROI with this bookmaker | DraftKings: -0.5% |

**Important**: All historical features use **expanding windows** with `.shift(1)` to avoid lookahead bias. The current bet's outcome is NOT included in its own features.

---

### 5. Sample Size Features (2)
| Variable | Description | Purpose |
|----------|-------------|---------|
| `sport_market_count` | Number of historical bets for this market | Higher = more reliable statistics |
| `institution_name_count` | Number of historical bets with this bookmaker | Used for confidence weighting |

**Example**: If `sport_market_count = 500`, we have 500 historical bets for this market, making the win rate more trustworthy than if we only had 10 bets.

---

### 6. Recent Trends (1)
| Variable | Description | Window |
|----------|-------------|--------|
| `recent_win_rate` | Win rate over last 10 bets in this market | Rolling 10-bet window |

**Why this matters**: Captures short-term momentum and recent market shifts that long-term averages miss.

---

### 7. Temporal Features (3)
| Variable | Description | Range |
|----------|-------------|-------|
| `day_of_week` | Day bet was placed | 0=Monday, 1=Tuesday, ..., 6=Sunday |
| `hour_of_day` | Hour bet was placed (UTC) | 0-23 |
| `days_since_first_bet` | Account age in days | 0-365+ |

**Pattern discovered**: ROI varies significantly by day of week (10.67% feature importance)!

---

## Target Variables (What the Model Predicts)

### Model 1: Win Probability Classifier
**Predicts**: `won` (0 = loss, 1 = win)
**Output**: Probability from 0.0 to 1.0
**Use case**: Confidence scoring, filtering low-probability bets

### Model 2: ROI Regressor
**Predicts**: `roi` (percentage return on investment)
**Output**: Expected ROI% (negative values indicate expected loss)
**Use case**: Bet selection, identifying +EV opportunities

---

## Derived Outputs (Calculated from Model Predictions)

| Variable | Description | Formula |
|----------|-------------|---------|
| `win_probability` | Predicted win chance | From Win Classifier (0.0-1.0) |
| `expected_roi` | Predicted ROI | From ROI Regressor (-100 to +100+) |
| `kelly_fraction` | Optimal bet size (Kelly Criterion) | `(p × b - q) / b` where p=win_prob, b=decimal_odds-1, q=1-p |
| `recommended_stake_pct` | Conservative bet size | `kelly_fraction × 100 × 0.25` (Quarter Kelly) |
| `confidence` | Prediction confidence | `min(1.0, sample_size / 100)` |
| `bet_grade` | Quality grade | A/B/C/D/F based on ROI + confidence |

---

## Feature Importance (From Real Data - 27,115 Bets)

### Win Prediction
1. **bet_type_encoded** (50.68%) - Straight bets perform very differently than parlays
2. **implied_prob** (11.06%) - Odds quality matters
3. **recent_win_rate** (5.81%) - Recent momentum is predictive

### ROI Prediction
1. **recent_win_rate** (12.94%) - Short-term trends drive profitability
2. **day_of_week** (10.67%) - Some days are more profitable than others
3. **implied_prob** (8.76%) - Better odds → better ROI

---

## Data Flow Example

### Input Bet
```json
{
  "sport": "Basketball",
  "league": "NCAAB",
  "market": "Spread",
  "institution_name": "DraftKings",
  "bet_type": "straight",
  "american_odds": -110,
  "is_live": false,
  "clv_percentage": 2.5
}
```

### Feature Engineering
```
sport_encoded = 0 (Basketball)
league_encoded = 1 (NCAAB)
market_encoded = 0 (Spread)
institution_name_encoded = 0 (DraftKings)
bet_type_encoded = 0 (straight)
implied_prob = 0.524 (from -110 odds)
is_live = 0
clv_percentage = 2.5
clv_ev = 2.5 × 0.524 = 1.31
has_clv = 1

# Historical features (fetched from database)
sport_win_rate = 0.485
sport_roi = -2.3
sport_market_win_rate = 0.512
sport_market_roi = 0.8
... (etc)

# Sample size
sport_market_count = 450
institution_name_count = 1200

# Recent trends
recent_win_rate = 0.548

# Temporal
day_of_week = 3 (Thursday)
hour_of_day = 14
days_since_first_bet = 180
```

### Model Output
```json
{
  "win_probability": 0.523,
  "expected_roi": 1.2,
  "kelly_fraction": 0.015,
  "recommended_stake_pct": 0.38,
  "confidence": 0.82,
  "bet_grade": "C",
  "profitable": true
}
```

**Interpretation**:
- 52.3% chance of winning
- Expected +1.2% ROI
- Bet 0.38% of bankroll (Quarter Kelly)
- Grade C (marginal +EV bet)

---

## Anti-Lookahead Protection

All historical features use **expanding windows** with `.shift(1)`:
```python
df['sport_win_rate'] = (
    df.groupby('sport')['won']
    .transform(lambda x: x.shift(1).expanding().mean())
    .fillna(0.5)  # Prior: 50% win rate
)
```

This ensures the **current bet's outcome is never used to predict itself**.

---

## Missing Data Handling

| Scenario | Strategy |
|----------|----------|
| No CLV data | `clv_percentage = 0`, `has_clv = 0` |
| New sport/market | Historical features default to 0.5 (50% win rate), 0% ROI |
| Missing odds | `implied_prob = 0.5` (default 50/50) |

---

## Model Updates

- **Training Data**: 27,115 settled bets (as of 2025-12-21)
- **Update Frequency**: Weekly (Sunday 2am UTC)
- **Feature Drift Monitoring**: Top 10 features should remain stable
- **Retrain Trigger**: If validation accuracy drops >5%

---

## Usage in Prediction API

```python
from predict_bet_profitability import BetProfitabilityPredictor

predictor = BetProfitabilityPredictor()

bet = {
    'sport': 'Basketball',
    'league': 'NBA',
    'market': 'Total',
    'institution_name': 'FanDuel',
    'odds': -105
}

result = predictor.predict(bet)
# Returns: win_probability, expected_roi, kelly_fraction, bet_grade, etc.
```

---

## References

- **Feature Engineering**: `/root/pikkit/ml/scripts/train_market_profitability_model.py` (lines 176-284)
- **Model Metadata**: `/root/pikkit/ml/models/model_metadata_latest.json`
- **Predictions**: `/root/pikkit/ml/predictions/market_predictions_latest.json`
- **Training Logs**: `/var/log/pikkit-ml-retrain.log`
