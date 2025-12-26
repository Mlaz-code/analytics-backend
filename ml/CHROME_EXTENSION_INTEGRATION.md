# Chrome Extension ML Integration Guide

## Overview

The Pikkit XGBoost ML models are now integrated into the Chrome extension to provide real-time AI-powered bet predictions alongside historical statistics.

## Architecture

```
OddsJam Page → Chrome Extension → Background Worker → ML API (Tools VM) → XGBoost Models
                      ↓
              Display ML Predictions + Historical Stats
```

## Components

### 1. Flask ML API Server (Tools VM: 192.168.4.80:5000)

**Location**: `/root/pikkit/ml/api/ml_prediction_api.py`

**Endpoints**:
- `GET /health` - Health check
- `GET /ml-predict` - Single bet prediction
- `POST /batch-predict` - Multiple bet predictions
- `GET /model-info` - Model metadata

**Service**: `pikkit-ml-api.service` (systemd)

**Start/Stop**:
```bash
systemctl start pikkit-ml-api
systemctl stop pikkit-ml-api
systemctl status pikkit-ml-api
systemctl restart pikkit-ml-api
```

**Logs**: `/var/log/pikkit-ml-api.log`

### 2. Chrome Extension Background Worker

**File**: `/root/pikkit-oddsjam-extension/background.js`

**New Function**:
```javascript
async function fetchMLPrediction(betData) {
    const params = new URLSearchParams({
        sport: betData.sport || '',
        league: betData.league || '',
        market: betData.market || '',
        institution_name: betData.book || '',
        bet_type: betData.betType || 'straight',
        odds: betData.odds || -110,
        is_live: betData.isLive || false,
        clv_percentage: betData.clv || 0
    });

    try {
        const response = await fetch(`http://192.168.4.80:5000/ml-predict?${params}`);
        if (!response.ok) throw new Error(`ML API error: ${response.status}`);
        return response.json();
    } catch (error) {
        debugLog('ML prediction unavailable:', error.message);
        return null; // Graceful degradation
    }
}
```

**Message Handler**:
```javascript
if (request.action === 'fetchMLPrediction') {
    fetchMLPrediction(request.data)
        .then(sendResponse)
        .catch(error => sendResponse({ error: error.message }));
    return true;
}
```

### 3. Chrome Extension Content Script

**File**: `/root/pikkit-oddsjam-extension/content.js`

**Fetches Both APIs in Parallel**:
```javascript
// Fetch score and ML prediction in parallel
const [scoreData, mlPrediction] = await Promise.all([
    fetchBetScore(betData),
    fetchMLPrediction(betData)
]);

// Merge ML prediction into scoreData
if (mlPrediction && scoreData) {
    scoreData.ml = {
        score: mlPrediction.win_probability * 100,
        confidence: mlPrediction.confidence,
        grade: mlPrediction.bet_grade,
        expectedRoi: mlPrediction.expected_roi,
        kellyFraction: mlPrediction.kelly_fraction,
        recommendedStake: mlPrediction.recommended_stake_pct,
        profitable: mlPrediction.profitable,
        recommendation: mlPrediction.profitable ? 'BET' : 'SKIP'
    };
}
```

## ML Prediction Response

```json
{
    "win_probability": 0.523,
    "expected_roi": 1.2,
    "kelly_fraction": 0.015,
    "recommended_stake_pct": 0.38,
    "confidence": 0.82,
    "bet_grade": "C",
    "profitable": true,
    "timestamp": "2025-12-21T01:50:03.155050"
}
```

## UI Display

The ML predictions are displayed in the bet card badge:

- **Grade Badge**: A/B/C/D/F colored indicator
- **Win Probability**: Displayed as ML score (0-100)
- **Expected ROI**: Shown alongside historical ROI
- **Recommendation**: BET (green) or SKIP (red)

**Grade Colors**:
- **A**: Green (#00c853) - Strong bet (ROI > 5%, confidence > 70%)
- **B**: Light green (#4caf50) - Good bet (ROI > 3%, confidence > 50%)
- **C**: Yellow (#ffc107) - Marginal bet (ROI > 0%, confidence > 30%)
- **D**: Orange (#ff9800) - Weak bet (ROI > -2%)
- **F**: Red (#f44336) - Avoid (ROI < -2%)

## Testing

### 1. Test API Endpoint Directly

```bash
# Health check
curl http://192.168.4.80:5000/health

# Single prediction
curl "http://192.168.4.80:5000/ml-predict?sport=Basketball&league=NBA&market=Spread&institution_name=DraftKings&odds=-110"
```

### 2. Test in Chrome Extension

1. Load unpacked extension from `/root/pikkit-oddsjam-extension/`
2. Navigate to https://oddsjam.com/
3. Check bet cards for ML grade badges
4. Open Chrome DevTools Console
5. Look for ML prediction logs

### 3. Debugging

**Check API logs**:
```bash
tail -f /var/log/pikkit-ml-api.log
```

**Check Chrome Console**:
- Look for "ML prediction unavailable" errors
- Verify fetch requests to 192.168.4.80:5000

**Common Issues**:
- API not running: `systemctl status pikkit-ml-api`
- Network blocked: Check manifest.json host_permissions
- CORS errors: Flask-CORS should handle this

## Graceful Degradation

If the ML API is unavailable:
- Extension continues to work with historical stats only
- No ML grade badge shown
- Errors logged to debug console
- No user-facing error messages

## Performance

- ML predictions fetch in parallel with historical stats (no added latency)
- Typical ML prediction response time: <100ms
- Model loads once on API startup (100ms), then cached in memory

## Future Enhancements

1. **Caching**: Cache ML predictions for identical bets
2. **Batch Predictions**: Predict all visible bets in single API call
3. **Real-time Updates**: WebSocket for live model updates
4. **A/B Testing**: Compare ML vs historical accuracy
5. **User Feedback**: Track user actions on ML recommendations

## Maintenance

**Weekly Model Retraining**:
- Cron job: `0 2 * * 0` (Sunday 2 AM)
- Script: `/root/pikkit/ml/scripts/weekly_retrain.sh`
- Logs: `/var/log/pikkit-ml-retrain.log`

**API Service Updates**:
```bash
# Update code
cd /root/pikkit/ml/api
# Edit ml_prediction_api.py

# Restart service
systemctl restart pikkit-ml-api

# Check status
systemctl status pikkit-ml-api
```

## Deployment Checklist

- [x] Flask API server running on Tools VM
- [x] Systemd service configured and enabled
- [x] Background worker updated with ML fetch function
- [x] Content script fetches and displays ML predictions
- [x] Manifest.json allows 192.168.4.80 access
- [ ] Extension reloaded in Chrome
- [ ] End-to-end testing on OddsJam

## Contact

For issues or questions:
- Check logs: `/var/log/pikkit-ml-api.log`
- Review model training: `/var/log/pikkit-ml-training-real-data-final.log`
- See model metadata: `/root/pikkit/ml/models/model_metadata_latest.json`
