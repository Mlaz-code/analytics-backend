# Granular Market Classifier - Pikkit Betting Data

## Overview

The Granular Market Classifier is a sport-aware machine learning model that predicts specific market types across 79+ categories. Unlike the basic classifier (4 categories), this model can distinguish between fine-grained markets like:

- **American Football**: "Player Passing Yards", "Player Receiving Yards", "1st Quarter Spread", etc.
- **Baseball**: "Pitcher Strikeouts", "1st 5 Innings Total", "Player Home Runs", etc.
- **Basketball**: "Player Points", "Player PRA", "1st Half Spread", "Player 3-Pointers", etc.
- **Tennis**: "1st Set Winner", "2nd Set Winner", "Total Games", "Set Handicap", etc.
- **Ice Hockey**: "Total Goals", "Puck Line", "Player Shots", "1st Period Total", etc.
- **Soccer**: "Total Goals", "Both Teams to Score", "Draw No Bet", etc.

## Model Performance

**Training Results:**
- **Test Accuracy:** 99.23%
- **Training Samples:** 12,390 bets
- **Market Types:** 79 categories
- **Algorithm:** Gradient Boosting (200 estimators)

**Top 20 Market Classification:**
```
                        precision    recall  f1-score   support
             Moneyline       0.99      0.99      0.99       598
                Spread       1.00      1.00      1.00       354
          Total Points       1.00      0.98      0.99       151
        1st Set Winner       1.00      1.00      1.00       110
       1st Half Spread       1.00      1.00      1.00       107
        2nd Set Winner       1.00      1.00      1.00       100
        1st Half Total       1.00      1.00      1.00        72
    Pitcher Strikeouts       1.00      1.00      1.00        63
                Parlay       1.00      0.98      0.99        58
     Team Total Points       1.00      1.00      1.00        52
```

## Comparison: Basic vs Granular Classifier

| Feature | Basic Classifier | Granular Classifier |
|---------|------------------|---------------------|
| **Categories** | 4 (moneyline, spread, total, parlay) | 79+ sport-specific markets |
| **Use Case** | High-level categorization | Specific market identification |
| **Accuracy** | 99.60% | 99.23% |
| **Training Data** | 11,250 bets | 12,390 bets |
| **Algorithm** | Random Forest | Gradient Boosting |
| **Sport-Aware** | No | Yes |
| **File** | `market_classifier.py` | `granular_market_classifier.py` |

## Supported Markets by Sport

### American Football (19 markets)
- Spread, Moneyline, Total Points
- Player Passing Yards, Player Receiving Yards, Player Rushing Yards
- Player Completions, Player Receptions, Player Passing TDs
- 1st Quarter: Spread, Moneyline, Total
- 1st Half: Spread, Moneyline, Total, Team Total
- Team Total Points
- Player Rushing Attempts, Player Longest Reception

### Baseball (19 markets)
- Moneyline, Run Line, Total Runs
- Pitcher Strikeouts, Pitcher Outs, Pitcher Earned Runs, Pitcher Hits Allowed
- Player Home Runs, Player Hits, Player Total Bases
- 1st Inning Total, 1st 3 Innings Total
- 1st 5 Innings: Moneyline, Run Line, Total
- Team Total Runs
- Other Player Props, Other

### Basketball (30 markets)
- Spread, Moneyline, Total Points
- Player Points, Player Rebounds, Player Assists
- Player PRA, Player P+R, Player R+A, Player P+A
- Player 3-Pointers, Player Turnovers, Player Blocks, Player Steals+Blocks
- Player Double-Double, First Basket
- 1st Quarter: Spread, Moneyline, Total
- 2nd Quarter: Spread, Moneyline, Total
- 3rd Quarter: Spread, Moneyline, Total
- 4th Quarter: Moneyline, Total
- 1st Half: Spread, Moneyline, Total, Team Total
- 2nd Half: Spread, Total
- Team Total Points

### Tennis (10 markets)
- Moneyline, Spread
- 1st Set Winner, 2nd Set Winner, Total Sets
- Total Games, Set Handicap, Game Handicap
- Game Props, Other Player Props

### Ice Hockey (9 markets)
- Moneyline, Puck Line
- Total Goals, Team Total Goals, 1st Period Total
- Player Points, Player Shots, Player Saves, Player Goals

### Soccer (10 markets)
- Moneyline, Spread, Total, Total Goals
- Team Total Goals, Team Total Corners
- Draw No Bet, Both Teams to Score
- 1st Half: Total, Total Goals, Team Total

## Features

The model uses these features for classification:

**Text Features (TF-IDF):**
- Market name
- Pick name (player names, selections)
- Sport name
- League name
- Combined text with pipe separators

**Sport Encoding:**
- Label-encoded sport for sport-specific patterns

**Pattern Features:**
- Has "over"/"under" keywords
- Contains player names (initials pattern)
- Contains numbers (point totals, spreads)
- Contains +/- symbols (spreads, odds)

**Categorical Features:**
- Is parlay bet
- Number of picks
- Is live bet

**Period/Quarter Indicators:**
- 1st half, 1st quarter
- 2nd half, 2nd quarter
- Period/inning markers

## Files

```
/root/pikkit/ml/
├── granular_market_classifier.py      # Training & core logic
├── predict_granular_markets.py        # Generate predictions
├── apply_granular_predictions.py      # Apply to database (when ready)
├── GRANULAR_CLASSIFIER_README.md      # This file
├── models/
│   └── granular_market_classifier.joblib  # Trained model (99.23% accuracy)
└── data/
    └── granular_market_predictions.csv    # Prediction results
```

## Usage

### 1. Train the Model (Already Done)

```bash
cd /root/pikkit/ml
python3 granular_market_classifier.py
```

### 2. Generate Predictions

```bash
# Generate predictions for bets with NULL/generic markets
python3 predict_granular_markets.py
```

This creates `/root/pikkit/ml/data/granular_market_predictions.csv` with predictions for:
- Bets with NULL market
- Bets labeled as "Other", "Parlay", "Game Props", "Player Props"

### 3. Review Predictions

```bash
# View predictions
cat data/granular_market_predictions.csv | head -50

# Check prediction confidence distribution
python3 -c "
import pandas as pd
df = pd.read_csv('data/granular_market_predictions.csv')
print(f'Mean confidence: {df[\"prediction_confidence\"].mean():.2%}')
print(df['predicted_market'].value_counts())
"
```

### 4. Apply Predictions (When Ready)

```bash
# Dry run (preview changes)
python3 apply_granular_predictions.py --confidence=0.8

# Apply to database (80%+ confidence)
python3 apply_granular_predictions.py --apply --confidence=0.8

# Apply with higher confidence threshold (90%)
python3 apply_granular_predictions.py --apply --confidence=0.9
```

## Use in Code

```python
from granular_market_classifier import GranularMarketClassifier
import pandas as pd

# Load model
classifier = GranularMarketClassifier()
classifier.load()

# Prepare data
df = pd.DataFrame({
    'market': ['Other', None, 'Parlay'],
    'pick_name': [
        'Patrick Mahomes · Over 275.5 Passing Yards',
        'Lakers -5.5',
        'Over 220.5 · Total'
    ],
    'sport': ['American Football', 'Basketball', 'Basketball'],
    'league': ['NFL', 'NBA', 'NBA'],
    'bet_type': ['straight', 'straight', 'straight'],
    'picks_count': [1, 1, 1],
    'is_live': [False, False, False]
})

# Predict
predictions, probabilities = classifier.predict(df)

print(predictions)
# Output: ['Player Passing Yards', 'Spread', 'Total Points']

print(probabilities)
# Output: [0.98, 0.99, 0.97]
```

## When to Use Which Classifier

### Use Basic Classifier (`market_classifier.py`) When:
- You need high-level market categorization
- Working with new sports not in training data
- Building dashboards with simplified categories
- Performance analysis by broad market type

### Use Granular Classifier (`granular_market_classifier.py`) When:
- You need specific market identification
- Building detailed market-level reports
- Analyzing player prop performance by stat type
- Correcting misclassified or missing market names
- Creating market-specific bet recommendations

## Prediction Confidence

**Current Results (595 candidates):**
- Mean confidence: **99.59%**
- Median confidence: **100.00%**
- >90% confidence: 591 (99.3%)
- >80% confidence: 591 (99.3%)

**Recommendation:**
- Use **80%+ confidence** for automatic application
- Review **60-80% confidence** predictions manually
- Flag **<60% confidence** for human verification

## Known Limitations

1. **"Other" Markets:**
   - Some markets are legitimately ambiguous (e.g., "1st Run", "Fight Props")
   - Model correctly identifies these as "Other"
   - Consider creating sport-specific categories for these

2. **Parlay Detection:**
   - Parlays are correctly identified with 100% confidence
   - These don't need reclassification

3. **New Market Types:**
   - Model only knows 79 markets from training
   - New market types will be classified to closest match
   - Retrain periodically with new markets

4. **Sport-Specific Props:**
   - Some sports have unique prop types not in training
   - May be classified as generic "Player Props" or "Other"

## Future Enhancements

1. **Add More Markets:**
   - Expand to 150+ market types
   - Include more sports (Golf, Darts, Esports)
   - Add futures and exotic markets

2. **Hierarchical Classification:**
   - Level 1: Sport
   - Level 2: Market Type (spread/total/prop)
   - Level 3: Specific Market (Player Passing Yards)

3. **Active Learning:**
   - Flag low-confidence predictions
   - User feedback loop for corrections
   - Incremental retraining

4. **API Integration:**
   - Real-time market classification endpoint
   - Batch prediction API
   - Confidence scoring for new bets

## Maintenance

**Retraining Schedule:**
- Retrain quarterly or when 5,000+ new bets added
- Monitor for new market types appearing in data
- Track prediction accuracy on new data

**Performance Monitoring:**
- Track confidence distribution over time
- Log incorrect predictions when found
- Monitor market type distribution shifts

## Contact

- Model: `/root/pikkit/ml/models/granular_market_classifier.joblib`
- Training: `/root/pikkit/ml/granular_market_classifier.py`
- Supabase: `mnnjjvbaxzumfcgibtme`

---

**Last Updated:** 2025-12-24
**Model Version:** 1.0
**Training Data:** 12,390 bets across 6 sports
**Test Accuracy:** 99.23%
**Market Categories:** 79
