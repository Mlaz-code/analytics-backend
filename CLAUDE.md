# Module: pikkit

<!-- AUTO-MANAGED: module-description -->
## Purpose

**Pikkit Sports Analytics Suite** - Backend API and ML system for sports betting analytics. Provides bet scoring, historical performance analysis, and opportunity capture for the Pikkit Chrome extension.

Key responsibilities:
- Flask API server (port 8000) for Chrome extension
- FastAPI ML inference server (port 8001/8002) for production predictions
- Supabase integration for bet tracking and performance data
- ML models with hyperparameter optimization, ensembles, and calibration
- Data pipeline with validation, quality gates, and lineage tracking
- MLOps infrastructure with blue-green deployments and CI/CD

<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: architecture -->
## Module Architecture

```
pikkit/
├── .env                        # Credentials (single source of truth)
├── extension-api.py            # Flask API server (port 8000)
├── validate_data.py            # Data validation logic
├── sync_to_supabase.py         # Supabase sync utilities
├── review_validation.py        # Validation review system
├── scheduled-validation.py     # Automated validation
├── ml/                         # Machine learning pipeline
│   ├── config/                 # Training configuration YAML
│   ├── pipeline/               # Training pipeline components
│   │   ├── config.py           # Pipeline configuration
│   │   ├── optimizer.py        # Optuna hyperparameter tuning
│   │   └── experiment_tracker.py # Experiment logging
│   ├── data_pipeline/          # Data quality and processing
│   │   ├── schemas.py          # Pydantic validation schemas
│   │   ├── quality_gates.py    # Data quality checks
│   │   ├── anomaly_detection.py # Anomaly detection
│   │   ├── lineage.py          # Data lineage tracking
│   │   └── incremental.py      # CDC and incremental loads
│   ├── monitoring/             # Drift detection and performance tracking
│   │   ├── __init__.py         # DriftDetector, ModelDriftDetector exports
│   │   └── drift_detector.py   # PSI, KS test, model degradation detection
│   ├── feature_store/          # Feast feature store (centralized features)
│   │   ├── feature_repo/       # Feast repository
│   │   │   ├── feature_store.yaml # SQLite online/file offline stores
│   │   │   └── features.py     # Feature views, entities
│   │   ├── materialize_features.py # Compute & materialize features
│   │   └── data/               # Parquet feature data
│   ├── scripts/                # ML training and automation scripts
│   │   ├── advanced_feature_engineering.py
│   │   ├── ensemble_models.py  # Stacking, voting ensembles
│   │   ├── model_calibration.py # Probability calibration
│   │   ├── hyperparameter_optimization.py
│   │   ├── create_baseline_reference.py # Drift baseline creation
│   │   ├── setup_drift_cron.sh # Install drift monitoring cron
│   │   └── check_drift_and_retrain.py # Automated drift checks
│   ├── data/                   # Baseline reference data for drift
│   ├── lstm/                   # LSTM prediction models
│   └── rl/                     # Reinforcement learning
├── mlops/                      # Production ML infrastructure
│   ├── app/                    # FastAPI application
│   │   └── main.py             # Async ML serving API
│   ├── tests/                  # Pytest test suite
│   │   ├── conftest.py         # Test fixtures (models, sample data)
│   │   └── __init__.py         # Test package init
│   ├── Dockerfile              # Multi-stage production build
│   ├── docker-compose.yml      # Blue-green deployment config
│   ├── scripts/deploy.sh       # Deployment orchestration
│   ├── .github/workflows/      # CI/CD pipelines
│   ├── monitoring/             # Prometheus metrics config
│   └── nginx/                  # Load balancer config
├── reports/                    # Generated reports
└── corrections/                # Data correction scripts
```

**External Services:**
- Supabase: `https://mnnjjvbaxzumfcgibtme.supabase.co`
- n8n Webhook: `https://n8n.chocopancake.com/webhook/pikkit-analysis`
- Dashboard: `https://pikkit-2d-dashboard.netlify.app`

**Deployment Ports:**
- Flask API: 8000
- ML API Blue: 8001
- ML API Green: 8002
- ML API Canary: 8003

<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: build-commands -->
## Build & Run Commands

```bash
# Extension API
python3 /root/pikkit/extension-api.py  # Port 8000

# ML API - Production Deployment
cd /root/pikkit/mlops && ./scripts/deploy.sh deploy
cd /root/pikkit/mlops && ./scripts/deploy.sh rollback
cd /root/pikkit/mlops && ./scripts/deploy.sh status

# ML API - Development
cd /root/pikkit/mlops && docker-compose up pikkit-ml-api-blue
cd /root/pikkit/mlops && docker-compose logs -f pikkit-ml-api-blue

# ML API - Blue-Green Deployment
cd /root/pikkit/mlops && docker-compose --profile blue-green up -d

# ML Training Pipeline
cd /root/pikkit/ml && python3 scripts/hyperparameter_optimization.py
cd /root/pikkit/ml && python3 scripts/ensemble_models.py
cd /root/pikkit/ml && python3 scripts/model_calibration.py

# Data Pipeline
cd /root/pikkit/ml && python3 -m data_pipeline.quality_gates
cd /root/pikkit/ml && python3 -m data_pipeline.anomaly_detection

# Drift Detection & Baseline
python3 /root/pikkit/ml/scripts/create_baseline_reference.py
python3 /root/pikkit/ml/scripts/check_drift_and_retrain.py
/root/pikkit/ml/scripts/setup_drift_cron.sh  # Install cron jobs

# Feature Store
cd /root/pikkit/ml/feature_store/feature_repo && feast apply
python3 /root/pikkit/ml/feature_store/materialize_features.py

# Validation
python3 /root/pikkit/validate_data.py
python3 /root/pikkit/scheduled-validation.py

# Testing
cd /root/pikkit/mlops && pytest tests/
cd /root/pikkit/mlops && pytest tests/ -v
```

<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: conventions -->
## Module-Specific Conventions

- **Credentials**: All secrets in `.env`, use `python-dotenv`
- **APIs**: Flask (extension) with CORS, FastAPI (ML) with async handlers
- **Supabase**: Use service key for backend, anon key for proxy
- **Data Validation**: Pydantic schemas in `data_pipeline/schemas.py`
- **Configuration**: YAML for training config (`ml/config/training_config.yaml`)
- **Deployment**: Blue-green via `mlops/scripts/deploy.sh`
- **Monitoring**: Prometheus metrics exposed on `/metrics` endpoint
- **Drift Detection**: Dataclass results with DriftResult, industry-standard PSI thresholds (0.1/0.2/0.25)
- **Feature Store**: Feast with 3 feature views (historical_performance, institution_features, league_features)
- **Cron Jobs**: Daily drift checks (2 AM), weekly baseline updates (Sunday 3 AM) via `setup_drift_cron.sh`
- **Scripts**: Executable permissions, shebang `#!/usr/bin/env python3`
- **ML Models**: Pickle format with metadata JSON, versioned with `_latest` suffix
- **Testing**: Pytest with session-scoped fixtures, TemporaryDirectory for test models

<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: dependencies -->
## Key Dependencies

**Core APIs:**
- **Flask**: Web framework for extension API (port 8000)
- **FastAPI**: Async ML inference API (ports 8001/8002)
- **flask-cors**: CORS handling for Chrome extension
- **uvicorn**: ASGI server for FastAPI

**ML Stack:**
- **xgboost**: Gradient boosting models
- **scikit-learn**: Feature engineering, ensembles, calibration
- **optuna**: Hyperparameter optimization
- **pandas/numpy**: Data processing
- **scipy**: Statistical tests (KS test for drift detection)
- **feast**: Feature store for centralized feature management

**Data Quality:**
- **pydantic**: Schema validation and data quality
- **python-dotenv**: Environment variable management

**Storage & APIs:**
- **requests**: HTTP client for Supabase API
- **Supabase Python SDK**: Direct database access

**Monitoring:**
- **prometheus_client**: Metrics collection and export

**Testing:**
- **pytest**: Test framework with fixtures and parametrization
- **scikit-learn**: GradientBoosting models for test fixtures

<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: patterns -->
## Detected Patterns

**MLOps Infrastructure:**
- **Blue-Green Deployment**: Separate containers (blue/green) for zero-downtime updates
- **Canary Deployment**: 10% traffic routing to new model versions for testing
- **Health Checks**: `/health` and `/ready` endpoints with Docker healthcheck integration
- **Model Caching**: Async model loading with hot reload detection via file mtime
- **Prometheus Metrics**: Counter/Histogram/Gauge for predictions, latency, errors

**Data Pipeline:**
- **Schema Validation**: Pydantic models with validators for data consistency
- **Quality Gates**: Automated data quality checks with configurable thresholds
- **Anomaly Detection**: Statistical methods (IQR, Z-score) for outlier detection
- **Data Lineage**: Track data transformations and dependencies
- **Incremental Loading**: CDC pattern for efficient data refresh
- **Drift Detection**: PSI (Population Stability Index) for categorical features, KS test for numerical features
- **Model Performance Tracking**: Degradation detection with configurable thresholds per metric
- **Feature Store**: Feast-based centralized features with online (SQLite) and offline (parquet) stores
- **Automated Retraining**: Cron-based drift checks (daily 2 AM) with Telegram alerting
- **Baseline Management**: Weekly baseline refresh (Sunday 3 AM) for drift comparison

**ML Training:**
- **Hyperparameter Optimization**: Optuna with pruning and early stopping
- **Ensemble Models**: Stacking and voting classifiers for improved predictions
- **Model Calibration**: Platt scaling and isotonic regression for probability calibration
- **Feature Engineering**: Lookback windows, rolling aggregations, temporal features
- **Experiment Tracking**: Metadata persistence with model versioning

**Configuration:**
- **YAML-based Config**: Central training configuration in `ml/config/training_config.yaml`
- **Environment Variables**: Runtime config via `.env` with docker-compose integration
- **Multi-stage Builds**: Dockerfile with builder and production stages for smaller images

**Testing:**
- **Session-scoped Fixtures**: TemporaryDirectory pattern for isolated test models
- **Test Model Generation**: GradientBoosting classifiers/regressors trained on synthetic data
- **Metadata Consistency**: Test fixtures mirror production schema (26 features, encoders)
- **Environment Isolation**: Test-specific env vars (ENVIRONMENT=test, MODEL_DIR=/tmp/test_models)

<!-- END AUTO-MANAGED -->

<!-- MANUAL -->
## Notes

**Run API**: `python3 /root/pikkit/extension-api.py` (port 8000)

**Sync Netlify Env**: `./sync-netlify-env.sh`

**BlissOS Scripts**:
- `/root/pikkit-autotap-tools.sh`
- `/root/pikkit-autotap-telegram.sh`

<!-- END MANUAL -->
