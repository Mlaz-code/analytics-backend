"""
Market Prediction Extension for Pikkit ML API
Predicts future performance of market combinations (sport/league/market)
Returns 0-100 recommendation score like existing XGBoost model
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import httpx
import logging

logger = logging.getLogger("market-prediction")

# Check both production and development paths
if os.path.exists('/app/models'):
    MODEL_DIR = Path('/app/models')
else:
    MODEL_DIR = Path(os.environ.get('MODEL_DIR', '/root/pikkit/ml/models'))

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_KEY')


class MarketPredictionRequest(BaseModel):
    """Request for market performance prediction"""
    sport: str = Field(..., description="Sport name (e.g., Basketball, Baseball)")
    league: str = Field(..., description="League name (e.g., NBA, MLB)")
    market: str = Field(..., description="Market type (e.g., Moneyline, Spread)")


class MarketPredictionResponse(BaseModel):
    """Response with market performance prediction"""
    sport: str
    league: str
    market: str
    market_key: str

    # Predictions
    predicted_winrate: float = Field(..., description="Predicted win rate (0-1)")
    predicted_roi: float = Field(..., description="Predicted ROI (%)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")

    # Scoring (0-100 like existing model)
    recommendation_score: float = Field(..., description="0-100 recommendation score")
    grade: str = Field(..., description="Letter grade (A/B/C/D/F)")
    should_take: bool = Field(..., description="Take/skip recommendation")
    explanation: str = Field(..., description="Human-readable explanation")

    # Historical context
    historical_bets: int = Field(..., description="Number of historical bets")
    historical_winrate: float = Field(..., description="Historical win rate")
    historical_roi: float = Field(..., description="Historical ROI")


class MarketPredictor:
    """Market performance predictor with model caching"""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.winrate_model = None
        self.roi_model = None
        self.confidence_model = None
        self.scaler = None
        self.feature_cols = []
        self.loaded = False

    def load_models(self):
        """Load market prediction models from disk"""
        if self.loaded:
            return

        try:
            logger.info(f"Loading market prediction models from {self.model_dir}")

            # Load models
            with open(self.model_dir / "market_winrate_latest.pkl", 'rb') as f:
                self.winrate_model = pickle.load(f)

            with open(self.model_dir / "market_roi_latest.pkl", 'rb') as f:
                self.roi_model = pickle.load(f)

            with open(self.model_dir / "market_confidence_latest.pkl", 'rb') as f:
                self.confidence_model = pickle.load(f)

            with open(self.model_dir / "market_scaler_latest.pkl", 'rb') as f:
                self.scaler = pickle.load(f)

            # Feature names
            self.feature_cols = [
                'hist_bets', 'hist_winrate', 'hist_roi',
                'last10_winrate', 'last10_roi',
                'last20_winrate', 'last20_roi',
                'winrate_momentum', 'roi_momentum', 'roi_std'
            ]

            self.loaded = True
            logger.info("✅ Market prediction models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading market models: {e}")
            raise

    async def get_market_history(self, sport: str, league: str, market: str, limit: int = 100) -> pd.DataFrame:
        """Fetch recent betting history for this market from Supabase"""

        # Get env variables here to ensure they're loaded
        supabase_url = os.environ.get('SUPABASE_URL')
        supabase_key = os.environ.get('SUPABASE_SERVICE_KEY')

        if not supabase_url or not supabase_key:
            logger.error("Supabase credentials not configured")
            return pd.DataFrame()

        headers = {
            'apikey': supabase_key,
            'Authorization': f'Bearer {supabase_key}'
        }

        params = {
            'is_settled': 'eq.true',
            'sport': f'eq.{sport}',
            'league': f'eq.{league}',
            'market': f'eq.{market}',
            'select': 'is_win,roi,created_at',
            'order': 'created_at.desc',
            'limit': str(limit)
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f'{supabase_url}/rest/v1/bets',
                headers=headers,
                params=params,
                timeout=10.0
            )

            if response.status_code != 200:
                logger.error(f"Supabase error: {response.status_code} - {response.text[:200]}")
                return pd.DataFrame()

            data = response.json()
            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            return df.sort_values('created_at')

    def calculate_features(self, history_df: pd.DataFrame) -> Dict:
        """Calculate features from betting history"""

        # Fill NaN values in ROI column (from null in database)
        if len(history_df) > 0 and 'roi' in history_df.columns:
            history_df['roi'] = history_df['roi'].fillna(0)

        if len(history_df) < 20:
            # Not enough data - return cold-start features
            return {
                'hist_bets': len(history_df),
                'hist_winrate': float(history_df['is_win'].mean()) if len(history_df) > 0 else 0.5,
                'hist_roi': float(history_df['roi'].mean()) if len(history_df) > 0 else 0,
                'last10_winrate': float(history_df.tail(10)['is_win'].mean()) if len(history_df) >= 10 else 0.5,
                'last10_roi': float(history_df.tail(10)['roi'].mean()) if len(history_df) >= 10 else 0,
                'last20_winrate': float(history_df.tail(20)['is_win'].mean()) if len(history_df) >= 20 else 0.5,
                'last20_roi': float(history_df.tail(20)['roi'].mean()) if len(history_df) >= 20 else 0,
                'winrate_momentum': 0,
                'roi_momentum': 0,
                'roi_std': 0,
                'cold_start': True
            }

        features = {
            'hist_bets': len(history_df),
            'hist_winrate': float(history_df['is_win'].mean()),
            'hist_roi': float(history_df['roi'].mean()),
            'last10_winrate': float(history_df.tail(10)['is_win'].mean()),
            'last10_roi': float(history_df.tail(10)['roi'].mean()),
            'last20_winrate': float(history_df.tail(20)['is_win'].mean()),
            'last20_roi': float(history_df.tail(20)['roi'].mean()),
            'winrate_momentum': float(history_df.tail(10)['is_win'].mean() - history_df['is_win'].mean()),
            'roi_momentum': float(history_df.tail(10)['roi'].mean() - history_df['roi'].mean()),
            'roi_std': float(history_df['roi'].std()) if len(history_df) > 1 else 0,
            'cold_start': False
        }

        # Replace any NaN values with safe defaults
        for key in features:
            if isinstance(features[key], float) and np.isnan(features[key]):
                if 'winrate' in key:
                    features[key] = 0.5
                else:
                    features[key] = 0.0

        return features

    def predict(self, features: Dict) -> Dict:
        """Make prediction from features with 0-100 scoring"""

        # Convert to DataFrame
        X = pd.DataFrame([features])[self.feature_cols]

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        pred_winrate = float(np.clip(self.winrate_model.predict(X_scaled)[0], 0, 1))
        pred_roi = float(self.roi_model.predict(X_scaled)[0])
        pred_confidence = float(np.clip(self.confidence_model.predict(X_scaled)[0], 0, 1))

        # Handle NaN values (replace with safe defaults)
        if np.isnan(pred_winrate):
            pred_winrate = 0.5
        if np.isnan(pred_roi):
            pred_roi = 0.0
        if np.isnan(pred_confidence):
            pred_confidence = 0.5

        # Calculate 0-100 recommendation score (same formula as existing model)
        winrate_score = (pred_winrate - 0.5) * 100  # Normalize around 50%
        roi_score = pred_roi / 5  # Normalize ROI
        confidence_score = pred_confidence * 50

        recommendation_score = float(np.clip(
            winrate_score + roi_score + confidence_score,
            0, 100
        ))

        # Handle NaN in final score
        if np.isnan(recommendation_score):
            recommendation_score = 0.0

        # Letter grade
        if recommendation_score >= 85:
            grade = 'A'
        elif recommendation_score >= 75:
            grade = 'B'
        elif recommendation_score >= 60:
            grade = 'C'
        elif recommendation_score >= 40:
            grade = 'D'
        else:
            grade = 'F'

        # Should take decision (same thresholds as training)
        should_take = (
            pred_winrate >= 0.53 and
            pred_roi >= 3.0 and
            pred_confidence >= 0.6
        )

        # Explanation
        if should_take:
            explanation = f"TAKE: Expected {pred_winrate*100:.1f}% winrate with {pred_roi:+.1f}% ROI (confidence: {pred_confidence*100:.0f}%)"
        else:
            reasons = []
            if pred_winrate < 0.53:
                reasons.append(f"Low winrate ({pred_winrate*100:.1f}%)")
            if pred_roi < 3.0:
                reasons.append(f"Low ROI ({pred_roi:+.1f}%)")
            if pred_confidence < 0.6:
                reasons.append(f"Low confidence ({pred_confidence*100:.0f}%)")
            explanation = f"SKIP: {' | '.join(reasons)}"

        return {
            'predicted_winrate': pred_winrate,
            'predicted_roi': pred_roi,
            'confidence': pred_confidence,
            'recommendation_score': recommendation_score,
            'grade': grade,
            'should_take': should_take,
            'explanation': explanation
        }

    async def predict_market(self, sport: str, league: str, market: str) -> MarketPredictionResponse:
        """Full prediction pipeline for a market"""

        # Ensure models are loaded
        if not self.loaded:
            self.load_models()

        # Get betting history
        history_df = await self.get_market_history(sport, league, market)

        # Calculate features
        features = self.calculate_features(history_df)
        cold_start = features.pop('cold_start', False)

        # Make prediction
        prediction = self.predict(features)

        # Build response
        return MarketPredictionResponse(
            sport=sport,
            league=league,
            market=market,
            market_key=f"{sport}|{league}|{market}",
            **prediction,
            historical_bets=features['hist_bets'],
            historical_winrate=features['hist_winrate'],
            historical_roi=features['hist_roi']
        )


# Global predictor instance
_predictor = None

def get_predictor() -> MarketPredictor:
    """Get or create global predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = MarketPredictor()
    return _predictor
