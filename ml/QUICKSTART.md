# Pikkit XGBoost Model - Quick Start Guide

## 🚀 5-Minute Setup

### Step 1: Set Supabase Credentials
```bash
cd /root/pikkit/ml/scripts
./setup_credentials.sh
# Enter your Supabase service key when prompted
```

### Step 2: Train Model on Real Data
```bash
python3 train_market_profitability_model.py
```

### Step 3: Make Predictions
```bash
python3 predict_bet_profitability.py
```

Done! The model is now ready to use.

---

## 📊 Using the Model

### Python API Example

```python
from predict_bet_profitability import BetProfitabilityPredictor

# Load models
predictor = BetProfitabilityPredictor()

# Predict for a bet
bet = {
    'sport': 'NBA',
    'league': 'NBA',
    'market': 'Spread',
    'institution_name': 'DraftKings',
    'bet_type': 'Over',
    'odds': -110,
    'clv_percentage': 2.5,  # Optional
    'is_live': False
}

result = predictor.predict(bet)

# Use predictions
print(f"Win Probability: {result['win_probability']:.1%}")
print(f"Expected ROI: {result['expected_roi']:+.2f}%")
print(f"Stake: {result['recommended_stake_pct']:.2f}% of bankroll")
print(f"Grade: {result['bet_grade']}")
```

### Market Predictions

View profitable markets:
```bash
cat /root/pikkit/ml/predictions/market_predictions_latest.json | jq '.[0:10]'
```

Filter Grade A markets:
```bash
cat /root/pikkit/ml/predictions/market_predictions_latest.json | \
  jq '[.[] | select(.predicted_roi > 5)]'
```

---

## 🔄 Maintenance

### View Training Logs
```bash
tail -f /var/log/pikkit-ml-retrain.log
```

### Manual Retrain
```bash
/root/pikkit/ml/scripts/weekly_retrain.sh
```

### Check Cron Status
```bash
crontab -l | grep pikkit-ml
```

---

## 🎯 Decision Rules

### When to Bet
- **Grade A** (ROI > 5%, confidence > 70%): ✅ Strong bet
- **Grade B** (ROI > 3%, confidence > 50%): ✅ Good bet
- **Grade C** (ROI > 0%, confidence > 30%): ⚠️ Marginal bet
- **Grade D** (ROI > -2%): ❌ Avoid
- **Grade F** (ROI < -2%): ❌ Definitely avoid

### Bet Sizing
Use the `recommended_stake_pct` which applies **Quarter Kelly**:
- Grade A with 5% Kelly → Bet 1.25% of bankroll
- Grade B with 3% Kelly → Bet 0.75% of bankroll
- Never exceed 5% of bankroll on a single bet

---

## 🔧 Troubleshooting

### Can't connect to Supabase
```bash
source /root/pikkit/.env
echo $SUPABASE_KEY  # Should show your key
```

### Models not found
```bash
ls -lh /root/pikkit/ml/models/
# Re-train if empty:
python3 train_market_profitability_model.py
```

### Prediction errors
Check that you're providing all required fields:
- `sport`, `league`, `market`, `institution_name`, `bet_type`, `odds`

---

## 📚 Full Documentation

See `/root/pikkit/ml/README.md` for:
- Complete architecture details
- Feature engineering explanation
- Advanced usage
- Integration guides
- Performance metrics

---

## 🆘 Support

Check logs:
- Training: `/var/log/pikkit-ml-retrain.log`
- Cron: `grep pikkit-ml /var/log/syslog`

Manual intervention:
```bash
# Stop automatic retraining
crontab -e  # Comment out the pikkit-ml line

# Restart automatic retraining
crontab -e  # Uncomment the pikkit-ml line
```
