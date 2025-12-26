# Pikkit Analytics Backend

**Sports betting analytics backend infrastructure** with ML training pipelines, API services, and data quality monitoring.

## 🏗️ Architecture Overview

This repository contains the **backend infrastructure** for the Pikkit sports analytics platform:

- **Flask Extension API** (port 8000) - REST API for Chrome extension
- **FastAPI ML API** (ports 8001/8002) - Production ML inference with blue-green deployment
- **ML Training Pipeline** - XGBoost models with drift detection and feature stores
- **Data Pipeline** - Quality gates, validation, and lineage tracking
- **MLOps Infrastructure** - Docker deployments, monitoring, CI/CD

## 🔗 Related Repositories

- **[analytics-dashboard](https://github.com/Mlaz-code/analytics-dashboard)** - Frontend dashboard and Netlify serverless functions
- **Bet Advisor Extension** - Chrome extension (consumer of extension API)

## 🚀 Quick Start

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt
cd mlops && pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your Supabase credentials

# 3. Run Extension API (Flask)
python3 extension-api.py
# Runs on http://localhost:8000

# 4. Run ML API (FastAPI)
cd mlops && docker-compose up pikkit-ml-api-blue
# Runs on http://localhost:8001
```

### Production Deployment

```bash
# Deploy ML API with blue-green strategy
cd mlops && ./scripts/deploy.sh deploy

# Check deployment status
./scripts/deploy.sh status

# Rollback if needed
./scripts/deploy.sh rollback
```

## 📁 Directory Structure

```
pikkit/
├── extension-api.py         # Flask API server (port 8000)
├── ml/                      # ML training infrastructure
│   ├── pipeline/            # Hyperparameter tuning (Optuna)
│   ├── data_pipeline/       # Data quality & validation
│   ├── monitoring/          # Drift detection (PSI, KS test)
│   ├── feature_store/       # Feast feature repository
│   ├── scripts/             # Training & automation scripts
│   ├── models/              # Trained models (.pkl, .joblib)
│   └── data/                # Baseline reference data
├── mlops/                   # Production ML infrastructure
│   ├── app/                 # FastAPI application
│   ├── tests/               # Pytest test suite
│   ├── Dockerfile           # Multi-stage production build
│   ├── docker-compose.yml   # Blue-green deployment config
│   └── scripts/             # Deployment automation
├── validation/              # Data validation scripts
└── reports/                 # Generated analysis reports
```

## 🔌 API Endpoints

### Extension API (Flask - port 8000)

```bash
# Score a bet based on historical performance
GET /api/score-bet?sport=Basketball&market=Spread&league=NBA

# Get dashboard analytics
GET /api/dashboard

# Sync bets to Supabase
POST /api/sync
```

### ML API (FastAPI - ports 8001/8002)

```bash
# Predict bet profitability
POST /api/predict
{
  "sport": "Basketball",
  "league": "NBA",
  "market": "Spread",
  "institution": "DraftKings"
}

# Get drift detection report
GET /api/drift

# Health check
GET /health
GET /ready
```

## 🧪 ML Pipeline

### Feature Engineering

```bash
# Extract features from Supabase
python3 ml/scripts/fetch_all_bets.py

# Engineer advanced features
python3 ml/scripts/advanced_feature_engineering.py

# Materialize to feature store
cd ml/feature_store/feature_repo && feast apply
python3 ml/feature_store/materialize_features.py
```

### Model Training

```bash
# Hyperparameter optimization (Optuna)
python3 ml/scripts/hyperparameter_optimization.py

# Train ensemble models
python3 ml/scripts/ensemble_models.py

# Calibrate probabilities
python3 ml/scripts/model_calibration.py

# Train market prediction model
python3 ml/scripts/train_market_prediction_model.py
```

### Drift Detection

```bash
# Create baseline reference (run once)
python3 ml/scripts/create_baseline_reference.py

# Manual drift check
python3 ml/scripts/check_drift_and_retrain.py

# Install automated monitoring (cron)
./ml/scripts/setup_drift_cron.sh
```

**Automated Schedule:**
- Daily drift checks: 2 AM
- Weekly baseline refresh: Sunday 3 AM
- Telegram alerts on drift detection

## 📊 Data Pipeline

### Quality Gates

```python
from ml.data_pipeline.quality_gates import DataQualityGate

gate = DataQualityGate()
result = gate.validate_data(df)
if not result.passed:
    print(f"Quality check failed: {result.failures}")
```

### Drift Detection

```python
from ml.monitoring.drift_detector import DriftDetector

detector = DriftDetector()
result = detector.detect_drift(current_data, reference_data)
print(f"Drift detected: {result.has_drift}")
print(f"PSI scores: {result.psi_scores}")
```

## 🧪 Testing

```bash
# Run ML API tests
cd mlops && pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Test drift detection
pytest ml/monitoring/ -v
```

## 🔐 Environment Variables

Required in `.env`:

```bash
# Supabase
SUPABASE_URL=https://mnnjjvbaxzumfcgibtme.supabase.co
SUPABASE_KEY=your_anon_key
SUPABASE_SERVICE_KEY=your_service_key

# Telegram (for alerts)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# ML API
MODEL_DIR=./ml/models
ENVIRONMENT=production
```

## 📦 Deployment Architecture

```
┌─────────────────────────────────────────────┐
│          Netlify (analytics-dashboard)      │
│  ┌─────────────────────────────────────┐   │
│  │  Frontend HTML/JS/CSS               │   │
│  │  Netlify Functions (proxies)        │   │
│  └──────────────┬──────────────────────┘   │
│                 │ HTTPS                     │
└─────────────────┼─────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│       Tools VM (192.168.4.80)               │
│  ┌──────────────────────────────────────┐  │
│  │  Flask Extension API (port 8000)     │  │
│  └──────────────────────────────────────┘  │
│  ┌──────────────────────────────────────┐  │
│  │  FastAPI ML API Blue (port 8001)     │  │
│  │  FastAPI ML API Green (port 8002)    │  │
│  └──────────────────────────────────────┘  │
│  ┌──────────────────────────────────────┐  │
│  │  Nginx Load Balancer                 │  │
│  └──────────────────────────────────────┘  │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│          Supabase (Database)                │
│  - Bets table                               │
│  - Opportunities table                      │
│  - ML predictions table                     │
└─────────────────────────────────────────────┘
```

## 🔄 CI/CD Pipeline

GitHub Actions workflow:
1. Run ML tests (`pytest`)
2. Validate OpenAPI contract
3. Build Docker images
4. Deploy to blue/green environments
5. Run smoke tests
6. Update monitoring dashboards

## 📈 Monitoring

- **Prometheus** - Metrics collection (prediction latency, error rates)
- **Grafana** - Visualization dashboards
- **Loki** - Log aggregation
- **n8n Webhooks** - Drift alerts to Telegram

## 🛠️ Development

### Adding a New Endpoint

1. Add route to `extension-api.py` or `mlops/app/main.py`
2. Update OpenAPI spec in `openapi.yaml`
3. Add tests in `mlops/tests/test_api.py`
4. Deploy via `./mlops/scripts/deploy.sh deploy`

### Training a New Model

1. Prepare data in `ml/data/`
2. Create training script in `ml/scripts/`
3. Save model to `ml/models/` with version suffix
4. Update symlink: `model_latest.pkl → model_20250126.pkl`
5. Test with `pytest mlops/tests/`

## 📝 License

MIT

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

For issues or questions:
- Open a GitHub issue
- Check [analytics-dashboard](https://github.com/Mlaz-code/analytics-dashboard) for frontend issues
- Review [CLAUDE.md](./CLAUDE.md) for codebase documentation
