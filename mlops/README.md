# Pikkit ML API - MLOps Infrastructure

Production-grade ML serving infrastructure for the Pikkit Sports Betting prediction system.

## Architecture Overview

```
                                   +------------------+
                                   |   GitHub Actions |
                                   |   CI/CD Pipeline |
                                   +--------+---------+
                                            |
                                            v
+------------+      +---------+      +------+-------+      +---------------+
|   Chrome   |----->|  Nginx  |----->|   FastAPI    |----->|  Prometheus   |
|  Extension |      |   LB    |      |   ML API     |      |   + Grafana   |
+------------+      +----+----+      +------+-------+      +---------------+
                         |                  |
                         v                  v
                  +------+------+    +------+-------+
                  | Blue | Green|    |    Models    |
                  | 8001 | 8002 |    |  XGBoost PKL |
                  +------+------+    +--------------+
                         |
                         v
                  +------+------+
                  |   Canary    |
                  |    8003     |
                  +-------------+
```

## Quick Start

### Local Development

```bash
# Copy environment file
cp .env.example .env

# Edit with your credentials
vim .env

# Start development server with hot reload
docker-compose --profile dev up

# Access API at http://localhost:8000
```

### Production Deployment

```bash
# Deploy using blue-green strategy
./scripts/deploy.sh deploy --version 1.0.0

# Check deployment status
./scripts/deploy.sh status

# View logs
./scripts/deploy.sh logs
```

## Directory Structure

```
mlops/
├── app/                          # FastAPI application
│   └── main.py                   # Main API with async endpoints
├── tests/                        # Unit tests
│   └── test_api.py
├── scripts/                      # Deployment scripts
│   ├── deploy.sh                 # Blue-green deployment
│   └── setup-n8n-webhook.sh      # n8n integration
├── monitoring/                   # Observability configs
│   ├── prometheus-pikkit.yml     # Prometheus scrape config
│   ├── alerts-pikkit.yml         # Alert rules
│   └── grafana-dashboard-pikkit-ml.json
├── nginx/                        # Load balancer
│   └── nginx.conf                # Weighted routing config
├── .github/workflows/            # CI/CD
│   └── ci-cd.yml                 # Full pipeline
├── Dockerfile                    # Multi-stage build
├── docker-compose.yml            # Service definitions
└── requirements.txt              # Python dependencies
```

## API Endpoints

### Prediction Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict` | POST | Single bet prediction |
| `/api/v1/predict` | GET | Single prediction (query params) |
| `/api/v1/batch-predict` | POST | Batch predictions (max 100) |
| `/api/v1/model-info` | GET | Model metadata |
| `/api/v1/reload-models` | POST | Force model reload |

### Health & Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (for LB) |
| `/ready` | GET | Readiness probe (K8s) |
| `/live` | GET | Liveness probe (K8s) |
| `/metrics` | GET | Prometheus metrics |

### Legacy Endpoints (backward compatibility)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ml-predict` | GET | Legacy prediction |
| `/batch-predict` | POST | Legacy batch |

## Deployment Strategies

### Blue-Green Deployment

Zero-downtime deployments with instant rollback capability.

```bash
# Deploy new version
./scripts/deploy.sh deploy --version 1.2.0

# If issues detected, rollback
./scripts/deploy.sh rollback
```

### Canary Deployment

Test new models with 10% of traffic.

```bash
# Deploy canary with new model
./scripts/deploy.sh canary --version 1.2.0-rc1

# Monitor metrics at /metrics
# If successful, promote to production
./scripts/deploy.sh promote

# Or rollback
docker-compose --profile canary down
```

## CI/CD Pipeline

The GitHub Actions pipeline includes:

1. **Lint** - Code quality checks (Black, isort, Ruff, MyPy)
2. **Test** - Unit tests with coverage
3. **Model Validation** - Performance benchmarks
4. **Build** - Multi-stage Docker build
5. **Security Scan** - Trivy vulnerability scanning
6. **Deploy** - Blue-green deployment to Tools VM
7. **Verify** - Smoke tests and load testing

### Required Secrets

Configure in GitHub repository settings:

| Secret | Description |
|--------|-------------|
| `TOOLS_VM_SSH_KEY` | SSH private key for deployment |
| `TELEGRAM_BOT_TOKEN` | Telegram notifications |
| `TELEGRAM_CHAT_ID` | Telegram chat ID |

## Monitoring

### Prometheus Metrics

Add to `/root/monitoring-stack/prometheus/prometheus.yml`:

```yaml
# Include Pikkit ML API
- job_name: 'pikkit-ml-api'
  # See monitoring/prometheus-pikkit.yml
```

### Alert Rules

Add to `/root/monitoring-stack/prometheus/alerts/`:

```bash
cp monitoring/alerts-pikkit.yml /root/monitoring-stack/prometheus/alerts/
```

### Grafana Dashboard

Import `monitoring/grafana-dashboard-pikkit-ml.json` into Grafana.

## n8n Integration

High-grade bets (A/B) trigger n8n webhook for:
- Telegram notifications
- Logging to Supabase

Setup:
```bash
./scripts/setup-n8n-webhook.sh
```

## Model Management

Models are loaded from `MODEL_PATH` (default: `/root/pikkit/ml/models`).

Required files:
- `win_probability_model_latest.pkl` - Win probability XGBoost
- `roi_prediction_model_latest.pkl` - ROI prediction XGBoost
- `model_metadata_latest.json` - Feature encoders

Hot reload: API checks for model updates every 60 seconds.

Force reload:
```bash
curl -X POST http://localhost:8000/api/v1/reload-models
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_VERSION` | 1.0.0 | Application version |
| `ENVIRONMENT` | development | Environment name |
| `MODEL_DIR` | /app/models | Model directory |
| `SUPABASE_URL` | - | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | - | Supabase service key |
| `TELEGRAM_BOT_TOKEN` | - | Telegram bot token |
| `TELEGRAM_CHAT_ID` | - | Telegram chat ID |
| `N8N_WEBHOOK_URL` | - | n8n webhook URL |

## Troubleshooting

### Models not loading

```bash
# Check model files exist
ls -la /root/pikkit/ml/models/

# Check container logs
docker logs pikkit-ml-api-blue

# Force reload
curl -X POST http://localhost:8001/api/v1/reload-models
```

### High latency

```bash
# Check P95 latency
curl http://localhost:8000/metrics | grep pikkit_prediction_latency

# Check active requests
curl http://localhost:8000/metrics | grep pikkit_active_requests

# Scale horizontally if needed
docker-compose up -d --scale pikkit-ml-api-blue=2
```

### Deployment failure

```bash
# Check deployment status
./scripts/deploy.sh status

# View container health
docker inspect pikkit-ml-api-blue | jq '.[0].State.Health'

# Rollback
./scripts/deploy.sh rollback
```

## Performance Benchmarks

| Metric | Target | Typical |
|--------|--------|---------|
| Single prediction latency (P95) | <100ms | 25-50ms |
| Batch prediction (100 items) | <500ms | 150-200ms |
| Model load time | <5s | 2-3s |
| Memory usage | <2GB | 500MB-1GB |

## License

Proprietary - Cornerstone Management
