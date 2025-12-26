# Drift Monitoring Dashboard - Setup Complete! 🎯

## What Was Created

### 1. Interactive Web Dashboard
**Location:** `/root/pikkit/mlops/app/templates/drift_dashboard.html`

A beautiful, modern dashboard featuring:
- Real-time drift metrics with auto-refresh (every 5 minutes)
- Interactive Chart.js visualizations
- Feature-level drift analysis
- Severity classification (high/moderate/low)
- Retraining recommendations
- Responsive design with gradient background

### 2. FastAPI Endpoints

**Dashboard UI:** `GET /drift-dashboard`
- Serves the interactive HTML dashboard
- No authentication required (add if needed)
- Accessible from browser

**JSON API:** `GET /api/v1/drift-report`
- Returns latest drift report in JSON format
- Includes report age and metadata
- Perfect for programmatic access or integration

### 3. Updated ML API
**Location:** `/root/pikkit/mlops/app/main.py`

Added:
- Jinja2 templates support
- Drift report endpoint (updated to read actual reports)
- Dashboard rendering endpoint
- NaN/Inf handling for JSON serialization

## Access the Dashboard

### Local Development (if ML API is running)

```bash
# Dashboard UI
http://localhost:8001/drift-dashboard

# JSON API
http://localhost:8001/api/v1/drift-report

# Or with custom parameters
http://localhost:8001/api/v1/drift-report?days_lookback=14&min_samples=200
```

### Production (after deployment)

```bash
# Blue instance
http://192.168.4.80:8001/drift-dashboard

# Green instance
http://192.168.4.80:8002/drift-dashboard

# Canary instance (if running)
http://192.168.4.80:8003/drift-dashboard
```

## Deploy to Production

### Option 1: Rebuild and Deploy ML API

```bash
cd /root/pikkit/mlops

# Rebuild Docker image with new dashboard
docker-compose build pikkit-ml-api-blue

# Start blue instance
docker-compose up -d pikkit-ml-api-blue

# Verify dashboard
curl -I http://localhost:8001/drift-dashboard
```

### Option 2: Full Blue-Green Deployment

```bash
cd /root/pikkit/mlops
./scripts/deploy.sh deploy
```

This will:
1. Build new Docker images for both blue and green
2. Deploy to both instances
3. Health check both
4. Make them available

### Option 3: Quick Test Without Docker

You can test the endpoints immediately without Docker:

```bash
cd /root/pikkit/mlops
uvicorn app.main:app --reload --port 8001
```

Then visit: `http://localhost:8001/drift-dashboard`

## Integration with Pikkit Dashboard

Your Pikkit dashboard is hosted on Netlify at:
`https://pikkit-2d-dashboard.netlify.app`

### Integration Option 1: iframe Embed

Add to your Netlify dashboard HTML/React:

```html
<div class="drift-monitor">
  <h2>ML Drift Monitor</h2>
  <iframe
    src="http://192.168.4.80:8001/drift-dashboard"
    width="100%"
    height="900px"
    frameborder="0"
    style="border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  </iframe>
</div>
```

### Integration Option 2: Fetch API Data

Create a custom dashboard component that fetches drift data:

```javascript
// In your React/Vue component
async function fetchDriftData() {
  try {
    const response = await fetch('http://192.168.4.80:8001/api/v1/drift-report');
    const data = await response.json();

    // Use the data in your dashboard
    return {
      totalFeatures: data.total_features_checked,
      driftedFeatures: data.features_with_drift,
      requiresRetraining: data.requires_retraining,
      reportAge: data.report_age_hours,
      details: data.feature_details
    };
  } catch (error) {
    console.error('Failed to fetch drift data:', error);
    return null;
  }
}

// Update every 5 minutes
setInterval(fetchDriftData, 5 * 60 * 1000);
```

### Integration Option 3: Navigation Link

Add a menu item or card that links to the dashboard:

```html
<a
  href="http://192.168.4.80:8001/drift-dashboard"
  target="_blank"
  class="dashboard-card">
  <h3>🎯 ML Drift Monitor</h3>
  <p>Monitor data drift and model performance</p>
  <span class="badge">Live</span>
</a>
```

## Dashboard Features

### Summary Statistics
- **Total Features Checked:** Number of features analyzed for drift
- **Features Drifted:** Count of features showing significant drift
- **Drift Severity:** Average drift score across all features
- **Retraining Status:** Whether model retraining is recommended

### Visualizations

1. **Drift Scores Chart** (Bar Chart)
   - Shows drift score for each feature
   - Color-coded by severity (green/yellow/orange/red)
   - Highlights which features need attention

2. **Category Distribution** (Donut Chart)
   - Breaks down features by severity level
   - Shows proportion of: No Drift, Low, Moderate, High

3. **Detailed Table**
   - Feature name and detection method
   - Drift score with 3 decimal precision
   - Severity badge (visual indicator)
   - Status (Drifted/Stable)
   - Additional details (PSI/KS statistics)

### Auto-Refresh
- Automatically refreshes every 5 minutes
- Manual refresh via 🔄 button
- Shows last update timestamp

## Testing

### Test the Dashboard

```bash
# 1. Check drift report endpoint
curl http://localhost:8001/api/v1/drift-report | jq .

# 2. Check dashboard loads
curl -I http://localhost:8001/drift-dashboard

# 3. Open in browser
firefox http://localhost:8001/drift-dashboard
# or
google-chrome http://localhost:8001/drift-dashboard
```

### Verify Data Flow

```bash
# 1. Check latest drift report exists
ls -lt /root/pikkit/ml/reports/drift_reports/ | head -5

# 2. View report content
cat /root/pikkit/ml/reports/drift_reports/drift_report_*.json | jq . | head -50

# 3. Test dashboard renders data
curl -s http://localhost:8001/api/v1/drift-report | jq '{
  features: .total_features_checked,
  drifted: .features_with_drift,
  retrain: .requires_retraining,
  age_hours: .report_age_hours
}'
```

## Monitoring

### Dashboard Logs

```bash
# ML API logs (Docker)
docker-compose logs -f pikkit-ml-api-blue | grep drift

# ML API logs (direct)
tail -f /var/log/pikkit-ml/drift-cron.log
```

### Health Check

```bash
# Check ML API is healthy
curl http://localhost:8001/health

# Check drift report is fresh (< 24 hours old)
curl -s http://localhost:8001/api/v1/drift-report | jq .report_age_hours
```

## Customization

### Change Refresh Interval

Edit `/root/pikkit/mlops/app/templates/drift_dashboard.html`:

```javascript
// Line ~730: Change from 5 to 10 minutes
setInterval(loadDashboard, 10 * 60 * 1000);
```

### Modify Color Scheme

Update the gradient background:

```css
/* Line ~15: Change gradient colors */
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
```

### Add Custom Metrics

Edit the stats grid HTML (around line 140):

```html
<div class="stat-card">
    <div class="stat-label">Your Custom Metric</div>
    <div class="stat-value" id="customMetric">-</div>
</div>
```

Then fetch and update in JavaScript:

```javascript
document.getElementById('customMetric').textContent = customValue;
```

## Troubleshooting

### Dashboard shows "No drift reports found"

Run drift detection:
```bash
python3 /root/pikkit/ml/scripts/check_drift_and_retrain.py
```

### Dashboard won't load (500 error)

Check templates directory exists:
```bash
ls -l /root/pikkit/mlops/app/templates/drift_dashboard.html
```

### API returns old data

Drift reports are generated daily at 2 AM via cron. To get fresh data:
```bash
# Run drift check manually
python3 /root/pikkit/ml/scripts/check_drift_and_retrain.py

# Verify cron job is installed
crontab -l | grep drift
```

### Can't access from remote browser

Check firewall rules allow port 8001:
```bash
# On Tools VM
sudo ufw status
sudo ufw allow 8001/tcp  # if needed
```

## Next Steps

1. **Deploy to production** - Rebuild Docker images and deploy
2. **Integrate with Netlify dashboard** - Choose iframe, API, or link approach
3. **Set up monitoring** - Add Grafana dashboard for drift metrics
4. **Configure alerts** - Ensure Telegram notifications work
5. **Customize colors/branding** - Match Pikkit color scheme

## Files Created/Modified

```
/root/pikkit/
├── mlops/
│   ├── app/
│   │   ├── main.py                          # MODIFIED: Added dashboard endpoints
│   │   └── templates/
│   │       └── drift_dashboard.html         # NEW: Dashboard UI
│   ├── requirements.txt                     # MODIFIED: Added jinja2
│   ├── DRIFT_DASHBOARD.md                   # NEW: Dashboard documentation
│   └── Dockerfile                           # Already supports templates/
├── ml/
│   ├── monitoring/
│   │   └── drift_detector.py                # MODIFIED: Added NaN/Inf handling
│   └── DRIFT_DASHBOARD_SETUP.md             # NEW: This file
```

## Support

For issues or questions:
- Check logs: `docker-compose logs -f pikkit-ml-api-blue`
- View drift reports: `ls -l /root/pikkit/ml/reports/drift_reports/`
- Test endpoints: `curl http://localhost:8001/api/v1/drift-report`
- Review documentation: `/root/pikkit/mlops/DRIFT_DASHBOARD.md`

---

**Dashboard is ready to use!** 🚀

Access it at: `http://localhost:8001/drift-dashboard` (after deployment)
