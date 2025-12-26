# Market Prediction API Documentation

## Overview

The Market Prediction API provides real-time recommendations for betting markets based on historical performance data. It uses XGBoost machine learning models to predict future winrate and ROI, returning a 0-100 recommendation score with take/skip decisions.

## Endpoints

### POST /api/v1/predict-market

Predict future performance for a specific market combination.

**Request Body:**
```json
{
  "sport": "Basketball",
  "league": "NBA",
  "market": "Player Points"
}
```

**Response:**
```json
{
  "sport": "Basketball",
  "league": "NBA",
  "market": "Player Points",
  "market_key": "Basketball|NBA|Player Points",

  "predicted_winrate": 0.456,
  "predicted_roi": -5.70,
  "confidence": 0.976,

  "recommendation_score": 43.2,
  "grade": "D",
  "should_take": false,
  "explanation": "SKIP: Low winrate (45.6%) | Low ROI (-5.7%)",

  "historical_bets": 100,
  "historical_winrate": 0.48,
  "historical_roi": -72.73
}
```

### GET /api/v1/predict-market/{sport}/{league}/{market}

Get market prediction via URL parameters (browser-friendly).

**Example:**
```
GET http://192.168.4.80:8001/api/v1/predict-market/Basketball/NBA/Player%20Points
```

## Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `sport` | string | Sport name |
| `league` | string | League name |
| `market` | string | Market type |
| `market_key` | string | Combined identifier |
| `predicted_winrate` | float | Predicted win rate (0-1) |
| `predicted_roi` | float | Predicted ROI (%) |
| `confidence` | float | Prediction confidence (0-1) |
| `recommendation_score` | float | 0-100 recommendation score |
| `grade` | string | Letter grade (A/B/C/D/F) |
| `should_take` | boolean | Take/skip recommendation |
| `explanation` | string | Human-readable explanation |
| `historical_bets` | integer | Number of historical bets |
| `historical_winrate` | float | Historical win rate (0-1) |
| `historical_roi` | float | Historical ROI (%) |

## Scoring System

### Recommendation Score (0-100)

The recommendation score is calculated using three components:

```
winrate_score = (predicted_winrate - 0.5) × 100
roi_score = predicted_roi / 5
confidence_score = confidence × 50

recommendation_score = clip(winrate_score + roi_score + confidence_score, 0, 100)
```

### Letter Grades

| Score Range | Grade |
|-------------|-------|
| 85-100 | A |
| 75-84 | B |
| 60-74 | C |
| 40-59 | D |
| 0-39 | F |

### Take/Skip Decision

A market is recommended (should_take = true) if ALL conditions are met:
- Predicted winrate ≥ 53%
- Predicted ROI ≥ 3%
- Confidence ≥ 60%

## Example Usage

### Python

```python
import requests

response = requests.post(
    "http://192.168.4.80:8001/api/v1/predict-market",
    json={
        "sport": "Basketball",
        "league": "NCAAB",
        "market": "Total Points"
    }
)

result = response.json()
print(f"Grade: {result['grade']}")
print(f"Score: {result['recommendation_score']:.0f}/100")
print(f"Decision: {result['explanation']}")
```

### JavaScript (Chrome Extension)

```javascript
async function predictMarket(sport, league, market) {
  const response = await fetch('http://192.168.4.80:8001/api/v1/predict-market', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sport, league, market })
  });

  const result = await response.json();

  return {
    score: result.recommendation_score,
    grade: result.grade,
    shouldTake: result.should_take,
    explanation: result.explanation
  };
}

// Usage
const prediction = await predictMarket('Basketball', 'NBA', 'Player Points');
console.log(`${prediction.grade} - ${prediction.explanation}`);
```

### cURL

```bash
# POST request
curl -X POST "http://192.168.4.80:8001/api/v1/predict-market" \
  -H "Content-Type: application/json" \
  -d '{"sport": "Basketball", "league": "NBA", "market": "Player Points"}' \
  | jq '.'

# GET request
curl "http://192.168.4.80:8001/api/v1/predict-market/Basketball/NBA/Player%20Points" | jq '.'
```

## Sample Predictions

### High-Quality Market (Grade A)
```json
{
  "market_key": "Baseball|MLB|Pitcher Strikeouts",
  "recommendation_score": 87,
  "grade": "A",
  "should_take": true,
  "explanation": "TAKE: Expected 58.2% winrate with +8.5% ROI (confidence: 95%)",
  "historical_bets": 250
}
```

### Poor-Quality Market (Grade F)
```json
{
  "market_key": "Basketball|NCAAB|Total Points",
  "recommendation_score": 39,
  "grade": "F",
  "should_take": false,
  "explanation": "SKIP: Low winrate (46.1%) | Low ROI (-34.4%)",
  "historical_bets": 100
}
```

### Cold-Start Market (Limited Data)
```json
{
  "market_key": "Soccer|MLS|Moneyline",
  "recommendation_score": 50,
  "grade": "D",
  "should_take": false,
  "explanation": "SKIP: Low confidence (40%)",
  "historical_bets": 5
}
```

## Model Details

### Architecture
- Three XGBoost regression models:
  - **Winrate Model**: Predicts future win rate
  - **ROI Model**: Predicts future return on investment
  - **Confidence Model**: Predicts prediction reliability

### Features
- Historical performance (all-time winrate, ROI)
- Recent performance (last 10 and 20 bets)
- Momentum indicators (trend direction)
- Variance metrics (consistency)

### Training Data
- 16,192 training samples from 448 unique markets
- 27,209 settled bets from Supabase
- Time-based train/test split

### Performance Metrics
- Winrate MAE: 19.47%
- ROI MAE: 52.11
- Confidence MAE: 18.55%

## Integration Notes

### For Chrome Extension
1. Call API when bet opportunity appears
2. Display recommendation score and grade in UI
3. Show take/skip recommendation prominently
4. Include historical context (# of bets, historical performance)

### For Dashboard
1. Fetch predictions for all active markets
2. Sort by recommendation score
3. Filter by grade (show A/B grades only)
4. Update predictions hourly as new bets settle

## Error Handling

### 500 Internal Server Error
- Models not loaded yet (wait 5-10 seconds after server start)
- Invalid market combination
- Database connection issues

### Solutions
1. Check `/health` endpoint for model status
2. Verify sport/league/market names match database values
3. Ensure Supabase credentials are configured

## Production Deployment

The API runs on port 8001 (Blue) and 8002 (Green) for zero-downtime updates:

```bash
cd /root/pikkit/mlops
./scripts/deploy.sh deploy
./scripts/deploy.sh status
```

Model files location: `/root/pikkit/ml/models/market_*_latest.pkl`
