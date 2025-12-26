#!/usr/bin/env python3
"""
Pikkit ML API - Production FastAPI Application
High-performance async ML serving with model caching and batch predictions
"""

import os
import sys
import json
import pickle
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from functools import lru_cache
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, validator
import httpx
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pikkit-ml-api")

# Environment configuration
SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://mnnjjvbaxzumfcgibtme.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_KEY', '')
SUPABASE_ANON_KEY = os.environ.get('SUPABASE_ANON_KEY', '')
MODEL_DIR = os.environ.get('MODEL_DIR', '/app/models')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
N8N_WEBHOOK_URL = os.environ.get('N8N_WEBHOOK_URL', '')
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'development')
VERSION = os.environ.get('APP_VERSION', '1.0.0')

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'pikkit_predictions_total',
    'Total number of predictions made',
    ['sport', 'market', 'grade']
)
PREDICTION_LATENCY = Histogram(
    'pikkit_prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
MODEL_LOAD_TIME = Gauge(
    'pikkit_model_load_timestamp',
    'Timestamp when models were last loaded'
)
ACTIVE_REQUESTS = Gauge(
    'pikkit_active_requests',
    'Number of active requests'
)
HIGH_GRADE_BETS = Counter(
    'pikkit_high_grade_bets_total',
    'Total number of A/B grade bets detected',
    ['grade']
)


class ModelCache:
    """Thread-safe model cache with lazy loading and hot reload support"""

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.win_model = None
        self.roi_model = None
        self.metadata = None
        self.feature_names = []
        self.encoders = {}
        self.loaded_at = None
        self._lock = asyncio.Lock()

    async def load_models(self) -> bool:
        """Load models from disk with async lock"""
        async with self._lock:
            try:
                logger.info(f"Loading models from {self.model_dir}")

                # Load win probability model
                win_model_path = self.model_dir / "win_probability_model_latest.pkl"
                with open(win_model_path, 'rb') as f:
                    self.win_model = pickle.load(f)
                logger.info(f"Loaded win model: {win_model_path}")

                # Load ROI prediction model
                roi_model_path = self.model_dir / "roi_prediction_model_latest.pkl"
                with open(roi_model_path, 'rb') as f:
                    self.roi_model = pickle.load(f)
                logger.info(f"Loaded ROI model: {roi_model_path}")

                # Load metadata
                metadata_path = self.model_dir / "model_metadata_latest.json"
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)

                self.feature_names = self.metadata.get('features', [])
                self.encoders = {
                    k: {i: v for i, v in enumerate(vals)}
                    for k, vals in self.metadata.get('encoders', {}).items()
                }

                self.loaded_at = datetime.utcnow()
                MODEL_LOAD_TIME.set(self.loaded_at.timestamp())

                logger.info(f"Models loaded successfully at {self.loaded_at}")
                logger.info(f"Model timestamp: {self.metadata.get('timestamp', 'unknown')}")
                logger.info(f"Features: {len(self.feature_names)}")

                return True

            except Exception as e:
                logger.error(f"Failed to load models: {e}")
                return False

    async def reload_if_updated(self) -> bool:
        """Reload models if files have been updated"""
        try:
            metadata_path = self.model_dir / "model_metadata_latest.json"
            mtime = datetime.fromtimestamp(metadata_path.stat().st_mtime)

            if self.loaded_at is None or mtime > self.loaded_at:
                logger.info("Detected model update, reloading...")
                return await self.load_models()
            return False
        except Exception as e:
            logger.error(f"Error checking for model updates: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self.win_model is not None and self.roi_model is not None

    def encode_categorical(self, value: str, encoder_name: str) -> int:
        """Encode categorical variable with unknown handling"""
        if encoder_name not in self.encoders:
            return 0

        encoder = self.encoders[encoder_name]
        for idx, val in encoder.items():
            if val == value:
                return idx
        return 0  # Default for unknown categories


# Global model cache
model_cache = ModelCache()


# Pydantic models for request/response validation
class BetPredictionRequest(BaseModel):
    """Request model for single bet prediction"""
    sport: str = Field(..., description="Sport type (Basketball, American Football, etc.)")
    league: str = Field(..., description="League (NBA, NFL, NCAAB, etc.)")
    market: str = Field(..., description="Market type (Spread, Total, Moneyline, etc.)")
    institution_name: str = Field(..., description="Bookmaker (DraftKings, FanDuel, etc.)")
    bet_type: str = Field(default="straight", description="Bet type (straight, parlay, etc.)")
    odds: int = Field(default=-110, description="American odds (e.g., -110, +150)")
    is_live: bool = Field(default=False, description="Is live bet")
    clv_percentage: float = Field(default=0.0, description="Closing line value percentage")

    @validator('odds')
    def validate_odds(cls, v):
        if v == 0 or v < -10000 or v > 10000:
            raise ValueError("Invalid odds value")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "sport": "Basketball",
                "league": "NBA",
                "market": "Spread",
                "institution_name": "DraftKings",
                "bet_type": "straight",
                "odds": -110,
                "is_live": False,
                "clv_percentage": 2.5
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    bets: List[BetPredictionRequest] = Field(..., description="List of bets to predict")

    @validator('bets')
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError("Maximum batch size is 100")
        if len(v) == 0:
            raise ValueError("At least one bet is required")
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    win_probability: float
    expected_roi: float
    kelly_fraction: float
    recommended_stake_pct: float
    confidence: float
    bet_grade: str
    profitable: bool
    timestamp: str
    model_version: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: bool
    model_timestamp: Optional[str]
    environment: str
    version: str
    uptime_seconds: float


class MarketAnalysisRequest(BaseModel):
    """Request for market analysis"""
    sport: Optional[str] = None
    league: Optional[str] = None
    market: Optional[str] = None
    min_sample_size: int = Field(default=30, ge=10, le=1000)
    top_n: int = Field(default=20, ge=1, le=100)


# Utility functions
async def send_telegram_alert(message: str):
    """Send alert to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    try:
        async with httpx.AsyncClient() as client:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            await client.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            }, timeout=10)
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")


async def trigger_n8n_webhook(data: Dict):
    """Trigger n8n webhook for high-grade bets"""
    if not N8N_WEBHOOK_URL:
        return

    try:
        async with httpx.AsyncClient() as client:
            await client.post(N8N_WEBHOOK_URL, json=data, timeout=10)
    except Exception as e:
        logger.error(f"Failed to trigger n8n webhook: {e}")


def prepare_features(bet: BetPredictionRequest, historical_stats: Dict = None) -> pd.DataFrame:
    """Prepare features for prediction"""
    cache = model_cache

    # Calculate implied probability
    if bet.odds < 0:
        implied_prob = abs(bet.odds) / (abs(bet.odds) + 100)
    else:
        implied_prob = 100 / (bet.odds + 100)

    # Build feature dict
    features_dict = {}

    # Categorical encodings
    features_dict['sport_encoded'] = cache.encode_categorical(bet.sport, 'sport')
    features_dict['league_encoded'] = cache.encode_categorical(bet.league, 'league')
    features_dict['market_encoded'] = cache.encode_categorical(bet.market, 'market')
    features_dict['institution_name_encoded'] = cache.encode_categorical(bet.institution_name, 'institution_name')
    features_dict['bet_type_encoded'] = cache.encode_categorical(bet.bet_type, 'bet_type')

    # Bet characteristics
    features_dict['implied_prob'] = implied_prob
    features_dict['is_live'] = int(bet.is_live)

    # CLV features
    features_dict['clv_percentage'] = bet.clv_percentage
    features_dict['clv_ev'] = bet.clv_percentage * implied_prob
    features_dict['has_clv'] = int(bet.clv_percentage != 0)

    # Historical performance (use defaults if not provided)
    if historical_stats is None:
        historical_stats = {}

    defaults = {
        'sport_win_rate': 0.5,
        'sport_roi': 0.0,
        'sport_market_win_rate': 0.5,
        'sport_market_roi': 0.0,
        'sport_league_win_rate': 0.5,
        'sport_league_roi': 0.0,
        'sport_league_market_win_rate': 0.5,
        'sport_league_market_roi': 0.0,
        'institution_name_win_rate': 0.5,
        'institution_name_roi': 0.0,
        'sport_market_count': 100,
        'institution_name_count': 100,
        'recent_win_rate': 0.5,
        'day_of_week': datetime.now().weekday(),
        'hour_of_day': datetime.now().hour,
        'days_since_first_bet': 180,
    }

    for key, default_val in defaults.items():
        features_dict[key] = historical_stats.get(key, default_val)

    return pd.DataFrame([{f: features_dict.get(f, 0) for f in cache.feature_names}])


def grade_bet(win_prob: float, expected_roi: float, confidence: float) -> str:
    """Assign grade to bet (A/B/C/D/F)"""
    if expected_roi > 5 and confidence > 0.7:
        return 'A'
    elif expected_roi > 3 and confidence > 0.5:
        return 'B'
    elif expected_roi > 0 and confidence > 0.3:
        return 'C'
    elif expected_roi > -2:
        return 'D'
    else:
        return 'F'


async def make_prediction(bet: BetPredictionRequest, background_tasks: BackgroundTasks = None) -> PredictionResponse:
    """Make prediction for a single bet"""
    cache = model_cache

    if not cache.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Prepare features
    X = prepare_features(bet)

    # Get predictions
    win_prob = float(cache.win_model.predict_proba(X)[0, 1])
    expected_roi = float(cache.roi_model.predict(X)[0])

    # Calculate Kelly Criterion
    if bet.odds < 0:
        decimal_odds = 1 + (100 / abs(bet.odds))
    else:
        decimal_odds = 1 + (bet.odds / 100)

    b = decimal_odds - 1
    p = win_prob
    q = 1 - p
    kelly_fraction = max(0, (p * b - q) / b)

    # Confidence and grade
    confidence = min(1.0, 100 / 100)  # Would use actual sample size in production
    grade = grade_bet(win_prob, expected_roi, confidence)

    # Update metrics
    PREDICTION_COUNTER.labels(
        sport=bet.sport,
        market=bet.market,
        grade=grade
    ).inc()

    # Send alerts for high-grade bets
    if grade in ['A', 'B']:
        HIGH_GRADE_BETS.labels(grade=grade).inc()

        if background_tasks:
            alert_message = (
                f"<b>High Grade Bet Alert</b>\n\n"
                f"Grade: <b>{grade}</b>\n"
                f"Sport: {bet.sport}\n"
                f"League: {bet.league}\n"
                f"Market: {bet.market}\n"
                f"Book: {bet.institution_name}\n"
                f"Win Prob: {win_prob:.1%}\n"
                f"Expected ROI: {expected_roi:+.2f}%\n"
                f"Kelly: {kelly_fraction:.3f}"
            )
            background_tasks.add_task(send_telegram_alert, alert_message)
            background_tasks.add_task(trigger_n8n_webhook, {
                "type": "high_grade_bet",
                "grade": grade,
                "bet": bet.dict(),
                "prediction": {
                    "win_probability": win_prob,
                    "expected_roi": expected_roi,
                    "kelly_fraction": kelly_fraction
                }
            })

    return PredictionResponse(
        win_probability=win_prob,
        expected_roi=expected_roi,
        kelly_fraction=kelly_fraction,
        recommended_stake_pct=kelly_fraction * 100 * 0.25,  # Quarter Kelly
        confidence=confidence,
        bet_grade=grade,
        profitable=expected_roi > 0,
        timestamp=datetime.utcnow().isoformat(),
        model_version=cache.metadata.get('timestamp') if cache.metadata else None
    )


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    logger.info("Starting Pikkit ML API...")
    app.state.start_time = datetime.utcnow()

    # Load models
    success = await model_cache.load_models()
    if not success:
        logger.error("Failed to load models on startup")

    # Start background model reload task
    async def periodic_model_check():
        while True:
            await asyncio.sleep(60)  # Check every minute
            await model_cache.reload_if_updated()

    task = asyncio.create_task(periodic_model_check())

    yield

    # Shutdown
    task.cancel()
    logger.info("Shutting down Pikkit ML API...")


# Create FastAPI application
app = FastAPI(
    title="Pikkit ML API",
    description="Production ML serving API for sports betting predictions",
    version=VERSION,
    lifespan=lifespan,
    docs_url="/docs" if ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if ENVIRONMENT != "production" else None,
)

# Configure templates for dashboard
templates = Jinja2Templates(directory="/app/templates" if os.path.exists("/app/templates") else "/root/pikkit/mlops/app/templates")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request tracking middleware
@app.middleware("http")
async def track_requests(request: Request, call_next):
    ACTIVE_REQUESTS.inc()
    start_time = datetime.utcnow()

    try:
        response = await call_next(request)
        return response
    finally:
        ACTIVE_REQUESTS.dec()
        duration = (datetime.utcnow() - start_time).total_seconds()

        if request.url.path.startswith("/api/"):
            PREDICTION_LATENCY.observe(duration)


# API Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for load balancers and monitoring"""
    uptime = (datetime.utcnow() - app.state.start_time).total_seconds()

    return HealthResponse(
        status="healthy" if model_cache.is_loaded else "degraded",
        models_loaded=model_cache.is_loaded,
        model_timestamp=model_cache.metadata.get('timestamp') if model_cache.metadata else None,
        environment=ENVIRONMENT,
        version=VERSION,
        uptime_seconds=uptime
    )


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness probe for Kubernetes"""
    if not model_cache.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ready"}


@app.get("/live", tags=["Health"])
async def liveness_check():
    """Liveness probe for Kubernetes"""
    return {"status": "alive"}


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_bet(bet: BetPredictionRequest, background_tasks: BackgroundTasks):
    """
    Predict win probability and expected ROI for a single bet.

    Returns prediction with confidence score and bet grade (A/B/C/D/F).
    High-grade bets (A/B) trigger Telegram alerts and n8n webhooks.
    """
    with PREDICTION_LATENCY.time():
        return await make_prediction(bet, background_tasks)


@app.get("/api/v1/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_bet_get(
    sport: str,
    league: str,
    market: str,
    institution_name: str,
    bet_type: str = "straight",
    odds: int = -110,
    is_live: bool = False,
    clv_percentage: float = 0.0,
    background_tasks: BackgroundTasks = None
):
    """GET endpoint for prediction (useful for browser testing)"""
    bet = BetPredictionRequest(
        sport=sport,
        league=league,
        market=market,
        institution_name=institution_name,
        bet_type=bet_type,
        odds=odds,
        is_live=is_live,
        clv_percentage=clv_percentage
    )
    return await make_prediction(bet, background_tasks)


@app.post("/api/v1/batch-predict", tags=["Predictions"])
async def batch_predict(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """
    Batch prediction endpoint for multiple bets.

    Maximum batch size: 100 bets.
    Returns predictions for all bets with summary statistics.
    """
    predictions = []
    errors = []

    for i, bet in enumerate(request.bets):
        try:
            pred = await make_prediction(bet, background_tasks)
            predictions.append({
                "index": i,
                "bet": bet.dict(),
                "prediction": pred.dict()
            })
        except Exception as e:
            errors.append({
                "index": i,
                "bet": bet.dict(),
                "error": str(e)
            })

    # Calculate summary
    if predictions:
        grades = [p["prediction"]["bet_grade"] for p in predictions]
        avg_roi = np.mean([p["prediction"]["expected_roi"] for p in predictions])
        avg_win_prob = np.mean([p["prediction"]["win_probability"] for p in predictions])
    else:
        grades = []
        avg_roi = 0
        avg_win_prob = 0

    return {
        "predictions": predictions,
        "errors": errors,
        "summary": {
            "total": len(request.bets),
            "successful": len(predictions),
            "failed": len(errors),
            "grade_distribution": {g: grades.count(g) for g in set(grades)},
            "average_expected_roi": float(avg_roi),
            "average_win_probability": float(avg_win_prob)
        }
    }


@app.get("/api/v1/model-info", tags=["Model"])
async def model_info():
    """Get model metadata and feature information"""
    if not model_cache.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    return {
        "metadata": model_cache.metadata,
        "feature_names": model_cache.feature_names,
        "feature_count": len(model_cache.feature_names),
        "loaded_at": model_cache.loaded_at.isoformat() if model_cache.loaded_at else None,
        "encoders": {k: list(v.values()) for k, v in model_cache.encoders.items()}
    }


@app.post("/api/v1/reload-models", tags=["Model"])
async def reload_models():
    """Force reload models from disk"""
    success = await model_cache.load_models()
    if success:
        return {"status": "success", "message": "Models reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload models")


@app.get("/api/v1/drift-report", tags=["Monitoring"])
async def get_drift_report(
    days_lookback: int = 7,
    min_samples: int = 100
):
    """
    Get most recent data drift detection report.

    Returns the latest drift report from the automated monitoring system.
    If no recent report exists, returns status information.

    Args:
        days_lookback: Number of days of recent data to analyze (default: 7)
        min_samples: Minimum samples required for drift detection (default: 100)
    """
    try:
        import glob

        # Find most recent drift report
        drift_reports_dir = Path("/root/pikkit/ml/reports/drift_reports")

        if not drift_reports_dir.exists():
            return {
                "status": "no_reports",
                "message": "No drift reports found. Run drift detection first.",
                "configuration": {
                    "days_lookback": days_lookback,
                    "min_samples": min_samples,
                    "categorical_features": ["sport", "league", "market", "institution_name", "bet_type"],
                    "numerical_features": ["implied_prob", "clv_percentage", "clv_ev"]
                },
                "next_steps": [
                    "Run: python3 /root/pikkit/ml/scripts/check_drift_and_retrain.py",
                    "Or wait for automated daily check at 2:00 AM"
                ]
            }

        # Get most recent report
        report_files = sorted(drift_reports_dir.glob("drift_report_*.json"), reverse=True)

        if not report_files:
            return {
                "status": "no_reports",
                "message": "No drift reports found in directory.",
                "next_steps": ["Run drift detection: python3 /root/pikkit/ml/scripts/check_drift_and_retrain.py"]
            }

        latest_report = report_files[0]

        with open(latest_report, 'r') as f:
            report_data = json.load(f)

        # Add metadata
        report_data['report_file'] = str(latest_report.name)
        report_data['report_age_hours'] = round(
            (datetime.utcnow() - datetime.fromisoformat(report_data['timestamp'])).total_seconds() / 3600,
            1
        )

        return report_data

    except Exception as e:
        logger.error(f"Drift detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")


@app.get("/drift-dashboard", response_class=HTMLResponse, tags=["Monitoring"])
async def drift_dashboard(request: Request):
    """
    Serve the interactive drift monitoring dashboard.

    Provides real-time visualization of:
    - Data drift metrics and trends
    - Feature-level drift scores
    - Retraining recommendations
    - Historical drift patterns
    """
    try:
        return templates.TemplateResponse("drift_dashboard.html", {"request": request})
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return HTMLResponse(
            content=f"""
            <html>
                <body style="font-family: sans-serif; padding: 40px; text-align: center;">
                    <h1>⚠️ Dashboard Error</h1>
                    <p>Failed to load drift dashboard: {str(e)}</p>
                    <p><a href="/api/v1/drift-report">View JSON report instead</a></p>
                </body>
            </html>
            """,
            status_code=500
        )


# Legacy endpoint compatibility (for existing Chrome extension)
@app.get("/ml-predict", tags=["Legacy"])
async def ml_predict_legacy(
    sport: str = "",
    league: str = "",
    market: str = "",
    institution_name: str = "",
    bet_type: str = "straight",
    odds: int = -110,
    is_live: str = "false",
    clv_percentage: float = 0.0,
    background_tasks: BackgroundTasks = None
):
    """Legacy ML predict endpoint for backward compatibility"""
    if not all([sport, league, market, institution_name]):
        raise HTTPException(status_code=400, detail="Missing required fields")

    bet = BetPredictionRequest(
        sport=sport,
        league=league,
        market=market,
        institution_name=institution_name,
        bet_type=bet_type,
        odds=odds,
        is_live=is_live.lower() == "true",
        clv_percentage=clv_percentage
    )

    pred = await make_prediction(bet, background_tasks)
    return pred.dict()


@app.post("/batch-predict", tags=["Legacy"])
async def batch_predict_legacy(request: Dict, background_tasks: BackgroundTasks):
    """Legacy batch predict endpoint"""
    bets = request.get("bets", [])
    if not bets:
        raise HTTPException(status_code=400, detail="No bets provided")

    predictions = []
    for bet_data in bets:
        try:
            bet = BetPredictionRequest(**bet_data)
            pred = await make_prediction(bet, background_tasks)
            predictions.append(pred.dict())
        except Exception as e:
            predictions.append({"error": str(e)})

    return {
        "predictions": predictions,
        "count": len(predictions)
    }


# =============================================================================
# MARKET PREDICTION ENDPOINTS (NEW)
# =============================================================================

from app.market_prediction import (
    MarketPredictor,
    MarketPredictionRequest,
    MarketPredictionResponse,
    get_predictor
)

@app.post("/api/v1/predict-market", response_model=MarketPredictionResponse, tags=["Market Prediction"])
async def predict_market_performance(request: MarketPredictionRequest):
    """
    Predict future performance for a specific market combination.

    Returns:
    - Predicted winrate and ROI for future bets in this market
    - 0-100 recommendation score (like existing XGBoost model)
    - Letter grade (A/B/C/D/F)
    - Take/Skip recommendation with explanation

    Example:
        POST /api/v1/predict-market
        {
            "sport": "Basketball",
            "league": "NBA",
            "market": "Player Points"
        }

        Response:
        {
            "sport": "Basketball",
            "league": "NBA",
            "market": "Player Points",
            "predicted_winrate": 0.52,
            "predicted_roi": 5.3,
            "confidence": 0.75,
            "recommendation_score": 78,
            "grade": "B",
            "should_take": true,
            "explanation": "TAKE: Expected 52% winrate with +5.3% ROI (confidence: 75%)",
            "historical_bets": 456,
            "historical_winrate": 0.495,
            "historical_roi": -1.2
        }
    """
    try:
        predictor = get_predictor()
        result = await predictor.predict_market(
            sport=request.sport,
            league=request.league,
            market=request.market
        )
        return result

    except Exception as e:
        logger.error(f"Market prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/predict-market/{sport}/{league}/{market}", response_model=MarketPredictionResponse, tags=["Market Prediction"])
async def predict_market_get(sport: str, league: str, market: str):
    """
    Predict market performance via GET request (for easy browser testing).

    Example: GET /api/v1/predict-market/Basketball/NBA/Player%20Points
    """
    try:
        predictor = get_predictor()
        result = await predictor.predict_market(sport=sport, league=league, market=market)
        return result

    except Exception as e:
        logger.error(f"Market prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info"
    )
