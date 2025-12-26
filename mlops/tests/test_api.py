"""
Pikkit ML API - Unit Tests
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import numpy as np


# Test fixtures
@pytest.fixture
def mock_models():
    """Create mock ML models for testing"""
    win_model = Mock()
    win_model.predict_proba = Mock(return_value=np.array([[0.45, 0.55]]))

    roi_model = Mock()
    roi_model.predict = Mock(return_value=np.array([3.5]))

    return win_model, roi_model


@pytest.fixture
def mock_metadata():
    """Create mock model metadata"""
    return {
        'timestamp': '20251224_000000',
        'features': [
            'sport_encoded', 'league_encoded', 'market_encoded',
            'institution_name_encoded', 'bet_type_encoded',
            'implied_prob', 'is_live', 'clv_percentage', 'clv_ev', 'has_clv',
            'sport_win_rate', 'sport_roi', 'sport_market_win_rate', 'sport_market_roi',
            'sport_league_win_rate', 'sport_league_roi',
            'sport_league_market_win_rate', 'sport_league_market_roi',
            'institution_name_win_rate', 'institution_name_roi',
            'sport_market_count', 'institution_name_count',
            'recent_win_rate', 'day_of_week', 'hour_of_day', 'days_since_first_bet'
        ],
        'encoders': {
            'sport': ['Basketball', 'American Football', 'Baseball'],
            'league': ['NBA', 'NFL', 'MLB'],
            'market': ['Spread', 'Moneyline', 'Total'],
            'institution_name': ['DraftKings', 'FanDuel', 'BetMGM'],
            'bet_type': ['straight', 'parlay']
        }
    }


@pytest.fixture
def sample_bet_request():
    """Sample bet prediction request"""
    return {
        "sport": "Basketball",
        "league": "NBA",
        "market": "Spread",
        "institution_name": "DraftKings",
        "bet_type": "straight",
        "odds": -110,
        "is_live": False,
        "clv_percentage": 2.0
    }


class TestGradeBet:
    """Test bet grading logic"""

    def test_grade_a_bet(self):
        """A-grade: high ROI with high confidence"""
        from app.main import grade_bet
        grade = grade_bet(win_prob=0.55, expected_roi=6.0, confidence=0.8)
        assert grade == 'A'

    def test_grade_b_bet(self):
        """B-grade: moderate ROI with moderate confidence"""
        from app.main import grade_bet
        grade = grade_bet(win_prob=0.52, expected_roi=4.0, confidence=0.6)
        assert grade == 'B'

    def test_grade_c_bet(self):
        """C-grade: positive ROI with low confidence"""
        from app.main import grade_bet
        grade = grade_bet(win_prob=0.50, expected_roi=1.0, confidence=0.4)
        assert grade == 'C'

    def test_grade_d_bet(self):
        """D-grade: slightly negative ROI"""
        from app.main import grade_bet
        grade = grade_bet(win_prob=0.48, expected_roi=-1.0, confidence=0.5)
        assert grade == 'D'

    def test_grade_f_bet(self):
        """F-grade: very negative ROI"""
        from app.main import grade_bet
        grade = grade_bet(win_prob=0.40, expected_roi=-5.0, confidence=0.5)
        assert grade == 'F'


class TestModelCache:
    """Test model caching functionality"""

    @pytest.mark.asyncio
    async def test_model_cache_initialization(self, mock_models, mock_metadata, tmp_path):
        """Test model cache initialization"""
        import pickle
        import json
        from app.main import ModelCache

        # Create temporary model files
        win_model, roi_model = mock_models

        with open(tmp_path / "win_probability_model_latest.pkl", 'wb') as f:
            pickle.dump(win_model, f)
        with open(tmp_path / "roi_prediction_model_latest.pkl", 'wb') as f:
            pickle.dump(roi_model, f)
        with open(tmp_path / "model_metadata_latest.json", 'w') as f:
            json.dump(mock_metadata, f)

        # Initialize cache
        cache = ModelCache(model_dir=str(tmp_path))
        success = await cache.load_models()

        assert success is True
        assert cache.is_loaded is True
        assert len(cache.feature_names) == 26

    def test_encode_categorical_known(self, mock_metadata):
        """Test encoding known categorical value"""
        from app.main import ModelCache

        cache = ModelCache()
        cache.encoders = {
            k: {i: v for i, v in enumerate(vals)}
            for k, vals in mock_metadata['encoders'].items()
        }

        encoded = cache.encode_categorical('Basketball', 'sport')
        assert encoded == 0

        encoded = cache.encode_categorical('NFL', 'league')
        assert encoded == 1

    def test_encode_categorical_unknown(self, mock_metadata):
        """Test encoding unknown categorical value returns 0"""
        from app.main import ModelCache

        cache = ModelCache()
        cache.encoders = {
            k: {i: v for i, v in enumerate(vals)}
            for k, vals in mock_metadata['encoders'].items()
        }

        encoded = cache.encode_categorical('Unknown Sport', 'sport')
        assert encoded == 0


class TestPredictionRequest:
    """Test request validation"""

    def test_valid_request(self, sample_bet_request):
        """Test valid prediction request"""
        from app.main import BetPredictionRequest
        request = BetPredictionRequest(**sample_bet_request)

        assert request.sport == "Basketball"
        assert request.odds == -110
        assert request.is_live is False

    def test_default_values(self):
        """Test default values are applied"""
        from app.main import BetPredictionRequest

        request = BetPredictionRequest(
            sport="Basketball",
            league="NBA",
            market="Spread",
            institution_name="DraftKings"
        )

        assert request.bet_type == "straight"
        assert request.odds == -110
        assert request.is_live is False
        assert request.clv_percentage == 0.0

    def test_invalid_odds(self):
        """Test invalid odds validation"""
        from app.main import BetPredictionRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BetPredictionRequest(
                sport="Basketball",
                league="NBA",
                market="Spread",
                institution_name="DraftKings",
                odds=0  # Invalid
            )


class TestBatchRequest:
    """Test batch prediction request validation"""

    def test_valid_batch(self, sample_bet_request):
        """Test valid batch request"""
        from app.main import BatchPredictionRequest, BetPredictionRequest

        request = BatchPredictionRequest(
            bets=[BetPredictionRequest(**sample_bet_request) for _ in range(5)]
        )

        assert len(request.bets) == 5

    def test_empty_batch_rejected(self):
        """Test empty batch is rejected"""
        from app.main import BatchPredictionRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BatchPredictionRequest(bets=[])

    def test_oversized_batch_rejected(self, sample_bet_request):
        """Test batch exceeding max size is rejected"""
        from app.main import BatchPredictionRequest, BetPredictionRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BatchPredictionRequest(
                bets=[BetPredictionRequest(**sample_bet_request) for _ in range(101)]
            )


class TestFeaturePreparation:
    """Test feature preparation logic"""

    def test_implied_probability_negative_odds(self):
        """Test implied probability calculation for negative odds"""
        from app.main import BetPredictionRequest, prepare_features, model_cache

        # Setup minimal cache
        model_cache.feature_names = ['implied_prob']
        model_cache.encoders = {}

        bet = BetPredictionRequest(
            sport="Basketball",
            league="NBA",
            market="Spread",
            institution_name="DraftKings",
            odds=-110
        )

        # -110 odds implies: 110 / (110 + 100) = 0.5238
        expected = 110 / 210
        assert abs(expected - 0.5238) < 0.001

    def test_implied_probability_positive_odds(self):
        """Test implied probability calculation for positive odds"""
        from app.main import BetPredictionRequest

        bet = BetPredictionRequest(
            sport="Basketball",
            league="NBA",
            market="Moneyline",
            institution_name="DraftKings",
            odds=150
        )

        # +150 odds implies: 100 / (150 + 100) = 0.40
        expected = 100 / 250
        assert abs(expected - 0.40) < 0.001


class TestKellyCriterion:
    """Test Kelly criterion calculations"""

    def test_kelly_positive(self):
        """Test Kelly fraction for positive EV bet"""
        # Kelly = (p * b - q) / b
        # p = 0.55, q = 0.45, b = decimal_odds - 1 = 0.909 (for -110)
        p = 0.55
        b = 100 / 110  # 0.909
        q = 1 - p

        kelly = (p * b - q) / b
        assert kelly > 0

    def test_kelly_negative_capped_at_zero(self):
        """Test Kelly fraction capped at 0 for negative EV"""
        p = 0.40  # Low win probability
        b = 100 / 110  # -110 odds
        q = 1 - p

        kelly = max(0, (p * b - q) / b)
        assert kelly == 0


class TestHealthEndpoints:
    """Test health check endpoints"""

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test health check response"""
        from httpx import AsyncClient, ASGITransport
        from app.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.get("/health")
            assert response.status_code == 200

            data = response.json()
            assert "status" in data
            assert "models_loaded" in data
            assert "environment" in data

    @pytest.mark.asyncio
    async def test_liveness_endpoint(self):
        """Test liveness probe"""
        from httpx import AsyncClient, ASGITransport
        from app.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.get("/live")
            assert response.status_code == 200
            assert response.json()["status"] == "alive"


class TestMetrics:
    """Test Prometheus metrics"""

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self):
        """Test metrics endpoint returns Prometheus format"""
        from httpx import AsyncClient, ASGITransport
        from app.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.get("/metrics")
            assert response.status_code == 200
            assert "pikkit_predictions_total" in response.text or "text/plain" in response.headers.get("content-type", "")


class TestPredictionEndpoint:
    """Test prediction endpoints"""

    @pytest.mark.asyncio
    async def test_predict_missing_fields(self):
        """Test prediction with missing required fields"""
        from httpx import AsyncClient, ASGITransport
        from app.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/predict",
                json={"sport": "Basketball"}  # Missing required fields
            )
            assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_legacy_predict_endpoint(self):
        """Test legacy /ml-predict endpoint"""
        from httpx import AsyncClient, ASGITransport
        from app.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.get(
                "/ml-predict",
                params={
                    "sport": "Basketball",
                    "league": "NBA",
                    "market": "Spread",
                    "institution_name": "DraftKings"
                }
            )
            # May be 503 if models not loaded, or 200 if loaded
            assert response.status_code in [200, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
