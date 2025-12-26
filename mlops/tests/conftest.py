"""
Pytest configuration and fixtures for Pikkit ML API tests
"""

import os
import sys
import json
import pickle
import pytest
import tempfile
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'app'))

# Set test environment variables before importing app
os.environ.setdefault('ENVIRONMENT', 'test')
os.environ.setdefault('MODEL_DIR', '/tmp/test_models')


@pytest.fixture(scope='session')
def test_model_dir():
    """Create temporary model directory with test models"""
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)

        # Create dummy models
        np.random.seed(42)
        X = np.random.rand(100, 26)
        y_win = np.random.randint(0, 2, 100)
        y_roi = np.random.randn(100) * 10

        win_model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        win_model.fit(X, y_win)

        roi_model = GradientBoostingRegressor(n_estimators=10, random_state=42)
        roi_model.fit(X, y_roi)

        # Save models
        with open(model_dir / 'win_probability_model_latest.pkl', 'wb') as f:
            pickle.dump(win_model, f)

        with open(model_dir / 'roi_prediction_model_latest.pkl', 'wb') as f:
            pickle.dump(roi_model, f)

        # Save metadata
        metadata = {
            'timestamp': 'test_20251224_000000',
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
                'sport': ['Basketball', 'American Football', 'Baseball', 'Ice Hockey'],
                'league': ['NBA', 'NFL', 'MLB', 'NHL', 'NCAAB', 'NCAAF'],
                'market': ['Spread', 'Moneyline', 'Total', 'Player Props'],
                'institution_name': ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars', 'ESPN BET'],
                'bet_type': ['straight', 'parlay', 'round_robin_3']
            }
        }

        with open(model_dir / 'model_metadata_latest.json', 'w') as f:
            json.dump(metadata, f)

        # Set environment variable
        os.environ['MODEL_DIR'] = str(model_dir)

        yield model_dir


@pytest.fixture
def sample_bet():
    """Sample bet prediction request data"""
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


@pytest.fixture
def sample_bets():
    """Sample batch of bets"""
    return [
        {
            "sport": "Basketball",
            "league": "NBA",
            "market": "Spread",
            "institution_name": "DraftKings",
            "odds": -110
        },
        {
            "sport": "American Football",
            "league": "NFL",
            "market": "Moneyline",
            "institution_name": "FanDuel",
            "odds": 150
        },
        {
            "sport": "Ice Hockey",
            "league": "NHL",
            "market": "Total",
            "institution_name": "BetMGM",
            "odds": -115
        }
    ]
