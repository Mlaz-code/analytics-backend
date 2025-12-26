# Integrating Drift Monitor into Pikkit Bet Analytics Dashboard

## Overview

Add the ML Drift Monitor to your existing Pikkit dashboard at https://app.chocopancake.com/

**Drift Monitor URL:** https://pikkit-drift-monitor.netlify.app

## Integration Options

### Option 1: Add Navigation Link (Quickest)

Add to your navigation menu in `index.html`:

```html
<!-- In your navigation section -->
<a href="https://pikkit-drift-monitor.netlify.app" target="_blank" class="nav-link">
  🎯 ML Drift
</a>
```

Or add to your existing navigation structure:

```html
<nav>
  <a href="/">🏠 Dashboard</a>
  <a href="/markets">📊 Markets</a>
  <a href="/live">🔴 Live</a>
  <a href="/insights">💡 Insights</a>
  <a href="/labs">🧪 Labs</a>
  <a href="/ev">💰 +EV</a>
  <a href="/opportunities">🎯 Tracker</a>
  <a href="http://192.168.4.80:5001">✓ Validation</a>
  <!-- ADD THIS -->
  <a href="https://pikkit-drift-monitor.netlify.app" target="_blank">🎯 ML Drift</a>
</nav>
```

### Option 2: Embed as Dedicated Page (Recommended)

Create a new page `/drift` that embeds the monitor:

**File:** `drift.html` (or add to your routing)

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Drift Monitor - Pikkit Analytics</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a202c;
        }

        .drift-container {
            width: 100%;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .drift-header {
            background: #2d3748;
            padding: 15px 30px;
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .drift-header h1 {
            margin: 0;
            font-size: 20px;
        }

        .back-link {
            color: #667eea;
            text-decoration: none;
            font-size: 14px;
        }

        .back-link:hover {
            color: #7c8ff0;
        }

        iframe {
            flex: 1;
            border: none;
            width: 100%;
        }
    </style>
</head>
<body>
    <div class="drift-container">
        <div class="drift-header">
            <h1>🎯 ML Drift Monitor</h1>
            <a href="/" class="back-link">← Back to Dashboard</a>
        </div>
        <iframe src="https://pikkit-drift-monitor.netlify.app"
                title="ML Drift Monitor"
                allow="fullscreen">
        </iframe>
    </div>
</body>
</html>
```

Then add to navigation:
```html
<a href="/drift">🎯 ML Drift</a>
```

### Option 3: Embed in Labs Section

Add to your Labs page as an iframe:

```html
<!-- In your Labs section -->
<div class="lab-section">
    <h2>ML Drift Monitoring</h2>
    <p>Real-time monitoring of data drift and model performance</p>

    <div class="drift-embed">
        <iframe
            src="https://pikkit-drift-monitor.netlify.app"
            width="100%"
            height="900px"
            frameborder="0"
            style="border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        </iframe>
    </div>
</div>
```

### Option 4: Dashboard Widget (Compact View)

Add as a compact widget on the main dashboard:

```html
<!-- Add to your main dashboard -->
<div class="dashboard-widget drift-widget">
    <div class="widget-header">
        <h3>🎯 ML Drift Status</h3>
        <a href="https://pikkit-drift-monitor.netlify.app" target="_blank">
            View Full →
        </a>
    </div>
    <div class="widget-content">
        <iframe
            src="https://pikkit-drift-monitor.netlify.app"
            width="100%"
            height="400px"
            frameborder="0"
            style="border-radius: 4px;">
        </iframe>
    </div>
</div>

<style>
.drift-widget {
    background: white;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.widget-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 2px solid #e2e8f0;
}

.widget-header h3 {
    margin: 0;
    font-size: 18px;
    color: #2d3748;
}

.widget-header a {
    color: #667eea;
    text-decoration: none;
    font-size: 14px;
}

.widget-header a:hover {
    color: #5568d3;
}
</style>
```

## Implementation Steps

### Step 1: Choose Integration Method

Select one of the options above based on your preference.

### Step 2: Update bet-analytics Project

```bash
# Clone or edit your bet-analytics project
# Add the chosen integration code
# Commit and push changes

# OR deploy directly to Netlify
cd /path/to/bet-analytics
# Add your changes
netlify deploy --prod
```

### Step 3: Configure API Endpoint

The drift monitor needs to connect to your ML API:

**API Endpoint:** `http://192.168.4.80:8001/api/v1/drift-report`

Users will configure this in the browser when they first visit the drift monitor.

### Step 4: Update CORS (if needed)

If the drift monitor can't fetch data, update ML API CORS:

**File:** `/root/pikkit/mlops/app/main.py`

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Or specific domains:
        "https://app.chocopancake.com",
        "https://pikkit-drift-monitor.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Quick Test Links

**Main Dashboard:** https://app.chocopancake.com/
**Drift Monitor:** https://pikkit-drift-monitor.netlify.app
**ML API:** http://192.168.4.80:8001/api/v1/drift-report

## Example: Full Page Integration

Here's a complete example of adding drift monitoring as a new page:

**File:** Create `public/drift.html` in your bet-analytics repo

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Drift Monitor | Pikkit Analytics</title>
    <link rel="icon" type="image/svg+xml" href="/favicon.svg">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: white;
        }

        .topnav {
            background: #1e293b;
            padding: 1rem 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }

        .topnav-links {
            display: flex;
            gap: 2rem;
            align-items: center;
        }

        .topnav a {
            color: #94a3b8;
            text-decoration: none;
            font-size: 14px;
            transition: color 0.2s;
        }

        .topnav a:hover,
        .topnav a.active {
            color: #667eea;
        }

        .drift-frame {
            width: 100%;
            height: calc(100vh - 60px);
            border: none;
        }
    </style>
</head>
<body>
    <nav class="topnav">
        <div class="topnav-links">
            <a href="/">🏠 Dashboard</a>
            <a href="/markets">📊 Markets</a>
            <a href="/live">🔴 Live</a>
            <a href="/insights">💡 Insights</a>
            <a href="/labs">🧪 Labs</a>
            <a href="/ev">💰 +EV</a>
            <a href="/opportunities">🎯 Tracker</a>
            <a href="/drift" class="active">🎯 ML Drift</a>
            <a href="http://192.168.4.80:5001">✓ Validation</a>
        </div>
    </nav>

    <iframe
        src="https://pikkit-drift-monitor.netlify.app"
        class="drift-frame"
        title="ML Drift Monitoring Dashboard">
    </iframe>
</body>
</html>
```

Then update your main navigation to include the link.

## Recommended: Option 2 (Dedicated Page)

For the best user experience, I recommend **Option 2** - creating a dedicated `/drift` page that maintains your navigation but embeds the full drift monitor.

This provides:
- ✓ Consistent navigation experience
- ✓ Full-screen drift monitoring
- ✓ Easy to maintain
- ✓ Professional appearance

## Need Help?

If you need help implementing any of these options, I can:
1. Create the HTML files for you
2. Update your bet-analytics project
3. Deploy the changes to Netlify
4. Test the integration

Just let me know which option you prefer!
