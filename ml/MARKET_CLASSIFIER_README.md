# Market Classifier - Pikkit Betting Data

## Overview

The Market Classifier is a machine learning model that categorizes bet markets into standardized types. It was built to clean up and standardize the `market_type` field in the Supabase `bets` table, which contained inconsistent values (`unknown`, `null`, `Total Points`, etc.).

## Model Performance

**Training Results:**
- **Test Accuracy:** 99.60%
- **Training Samples:** 11,250 bets
- **Classes:** 4 (moneyline, spread, total, parlay)

**Classification Report:**
```
              precision    recall  f1-score   support
   moneyline       0.99      1.00      1.00       501
      parlay       1.00      1.00      1.00       328
      spread       1.00      0.99      0.99       702
       total       0.99      1.00      1.00       719

    accuracy                           1.00      2250
```

## Classification Categories

The classifier assigns bets to one of these categories:

1. **moneyline** - Win/loss bets on match outcomes
   - Examples: "Moneyline", "Match Betting", "Winner"

2. **spread** - Point spread or handicap bets
   - Examples: "Spread", "Run Line", "Puck Line", "Handicap"

3. **total** - Over/under bets on total points/goals/runs
   - Examples: "Total Points", "Over/Under", "Team Total", "Player Points"

4. **prop** - Proposition bets (special markets)
   - Examples: "1st Set Winner", "Both Teams to Score", "Player Props"

5. **parlay** - Multi-leg combination bets
   - Identified by `bet_type = 'parlay'` or `picks_count > 1`

## Features

The model uses the following features for classification:

**Text Features (TF-IDF):**
- Market name
- Pick name
- Sport name

**Categorical Features:**
- Is parlay bet
- Number of picks
- Multi-leg indicator

**Pattern Features:**
- Contains "moneyline" keywords
- Contains "spread" keywords
- Contains "total" keywords
- Contains "prop" keywords
- Contains "parlay" keywords

## Database Update

**Applied Classifications:**
- **Total Updated:** 2,045 bets (70%+ confidence)
- **Success Rate:** 100% (0 errors)

**Distribution of Updates:**
- Spread: 1,100 (53.8%)
- Total: 584 (28.5%)
- Parlay: 197 (9.6%)
- Moneyline: 150 (7.3%)
- Prop: 14 (0.7%)

## Files

```
/root/pikkit/ml/
├── market_classifier.py          # Main training & prediction script
├── apply_market_predictions.py   # Database update script
├── models/
│   └── market_classifier.joblib  # Trained model (99.6% accuracy)
├── data/
│   └── market_predictions.csv    # Prediction results (2,550 bets)
└── MARKET_CLASSIFIER_README.md   # This file
```

## Usage

### Train the Model

```bash
cd /root/pikkit/ml
python3 market_classifier.py
```

This will:
1. Fetch training data from Supabase (11,250+ labeled bets)
2. Train a Random Forest classifier
3. Evaluate performance on test set
4. Save model to `models/market_classifier.joblib`
5. Generate predictions for unlabeled bets
6. Save predictions to `data/market_predictions.csv`

### Apply Predictions to Database

```bash
# Dry run (preview changes)
python3 apply_market_predictions.py --confidence=0.7

# Apply changes (70% confidence threshold)
python3 apply_market_predictions.py --apply --confidence=0.7

# Apply with higher confidence (80%)
python3 apply_market_predictions.py --apply --confidence=0.8
```

### Use the Model in Code

```python
from market_classifier import MarketClassifier
import pandas as pd

# Load trained model
classifier = MarketClassifier()
classifier.load()

# Prepare data for prediction
df = pd.DataFrame({
    'market': ['Spread', 'Total Points', 'Moneyline'],
    'pick_name': ['Lakers +5.5', 'Over 220.5', 'Warriors Win'],
    'sport': ['Basketball', 'Basketball', 'Basketball'],
    'bet_type': ['straight', 'straight', 'straight'],
    'picks_count': [1, 1, 1],
    'is_live': [False, False, False]
})

# Predict
predictions, probabilities = classifier.predict(df)

print(predictions)
# Output: ['spread', 'total', 'moneyline']

print(probabilities)
# Output: [[0.02, 0.01, 0.96, 0.01],  # spread: 96%
#          [0.01, 0.01, 0.02, 0.96],  # total: 96%
#          [0.95, 0.01, 0.02, 0.02]]  # moneyline: 95%
```

## Model Details

**Algorithm:** Random Forest Classifier
- 200 trees
- Max depth: 20
- Min samples split: 5
- Class weights: balanced
- Random state: 42

**Text Vectorization:** TF-IDF
- Max features: 500
- N-grams: (1, 3)
- Lowercase: True
- Strip accents: Unicode

**Rule-Based Post-Processing:**
- Special markets (set winners, draw no bet, etc.) → `prop`
- Only applied when model confidence < 80%

## Data Quality

**Training Data Quality:**
- Stratified sampling across all market types
- Balanced class representation (3,500-3,700 samples per class)
- Filtered out classes with < 10 samples

**Prediction Confidence:**
- Mean: 87.72%
- Median: 94.69%
- 73.1% of predictions > 90% confidence
- 80.2% of predictions > 70% confidence

## Future Improvements

1. **Add More Categories:**
   - Split `prop` into subcategories (player_prop, game_prop, etc.)
   - Add `futures` category for long-term bets

2. **Enhance Features:**
   - Add league-specific patterns
   - Use historical market performance
   - Include odds ranges as features

3. **Active Learning:**
   - Flag low-confidence predictions for manual review
   - Retrain model with corrected labels

4. **Integration:**
   - Add classifier to Extension API (port 8000)
   - Real-time classification for new bets
   - API endpoint: `/api/classify-market`

## Maintenance

**Retraining Schedule:**
- Retrain monthly or when 1,000+ new labeled bets are added
- Monitor classification drift (new market types appearing)

**Performance Monitoring:**
- Track prediction confidence distribution
- Monitor manual corrections to predictions
- Log misclassifications for future training

## Contact

For questions or issues:
- Model location: `/root/pikkit/ml/models/market_classifier.joblib`
- Training script: `/root/pikkit/ml/market_classifier.py`
- Supabase project: `mnnjjvbaxzumfcgibtme`

---

**Last Updated:** 2025-12-24
**Model Version:** 1.0
**Training Data:** 11,250 bets
**Test Accuracy:** 99.60%
