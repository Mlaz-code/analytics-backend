#!/usr/bin/env python3
"""
Pikkit Bet Profitability Prediction API
Real-time inference for new betting opportunities
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List

# Model paths
MODEL_DIR = '/root/pikkit/ml/models'


class BetProfitabilityPredictor:
    """Predict win probability and expected ROI for betting opportunities"""

    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.win_model = None
        self.roi_model = None
        self.metadata = None
        self.features = None
        self.encoders = None
        self.load_models()

    def load_models(self):
        """Load trained models and metadata"""
        print("📦 Loading models...")

        # Load win probability model
        win_model_path = f"{self.model_dir}/win_probability_model_latest.pkl"
        with open(win_model_path, 'rb') as f:
            self.win_model = pickle.load(f)
        print(f"  ✅ Loaded win model from {win_model_path}")

        # Load ROI prediction model
        roi_model_path = f"{self.model_dir}/roi_prediction_model_latest.pkl"
        with open(roi_model_path, 'rb') as f:
            self.roi_model = pickle.load(f)
        print(f"  ✅ Loaded ROI model from {roi_model_path}")

        # Load metadata
        metadata_path = f"{self.model_dir}/model_metadata_latest.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.features = self.metadata['features']
        self.encoders = {k: {i: v for i, v in enumerate(vals)}
                        for k, vals in self.metadata['encoders'].items()}

        print(f"  ✅ Loaded metadata (trained: {self.metadata['timestamp']})")
        print(f"  ✅ Using {len(self.features)} features\n")

    def encode_categorical(self, value, encoder_name):
        """Encode categorical variable (with unknown handling)"""
        if encoder_name not in self.encoders:
            return 0  # Default encoding for unknown categories

        encoder = self.encoders[encoder_name]
        # Find the value in the encoder
        for idx, val in encoder.items():
            if val == value:
                return idx

        # Return most common value (0) if not found
        return 0

    def prepare_features_for_bet(self, bet_data: Dict, historical_stats: Dict = None) -> pd.DataFrame:
        """
        Prepare features for a single bet

        Args:
            bet_data: Dict with bet information
                Required: sport, league, market, institution_name, bet_type, odds
                Optional: clv_percentage, is_live
            historical_stats: Dict with historical performance stats
                Optional: sport_win_rate, sport_roi, etc.
        """
        # Calculate implied probability
        odds = bet_data.get('odds', -110)
        if odds < 0:
            implied_prob = abs(odds) / (abs(odds) + 100)
        else:
            implied_prob = 100 / (odds + 100)

        # Build feature dict
        features_dict = {}

        # Categorical encodings
        features_dict['sport_encoded'] = self.encode_categorical(
            bet_data.get('sport', ''), 'sport')
        features_dict['league_encoded'] = self.encode_categorical(
            bet_data.get('league', ''), 'league')
        features_dict['market_encoded'] = self.encode_categorical(
            bet_data.get('market', ''), 'market')
        features_dict['institution_name_encoded'] = self.encode_categorical(
            bet_data.get('institution_name', ''), 'institution_name')
        features_dict['bet_type_encoded'] = self.encode_categorical(
            bet_data.get('bet_type', ''), 'bet_type')

        # Bet characteristics
        features_dict['implied_prob'] = implied_prob
        features_dict['is_live'] = int(bet_data.get('is_live', False))

        # CLV features
        clv_pct = bet_data.get('clv_percentage', 0)
        features_dict['clv_percentage'] = clv_pct
        features_dict['clv_ev'] = clv_pct * implied_prob
        features_dict['has_clv'] = int(clv_pct != 0)

        # Historical performance (use defaults if not provided)
        if historical_stats is None:
            historical_stats = {}

        # Default values based on overall market averages
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
            'days_since_first_bet': 180,  # Assume 6 months of history
        }

        for key, default_val in defaults.items():
            features_dict[key] = historical_stats.get(key, default_val)

        # Create dataframe with features in correct order
        return pd.DataFrame([{f: features_dict.get(f, 0) for f in self.features}])

    def predict(self, bet_data: Dict, historical_stats: Dict = None) -> Dict:
        """
        Predict win probability and expected ROI for a bet

        Returns:
            Dict with prediction results
        """
        # Prepare features
        X = self.prepare_features_for_bet(bet_data, historical_stats)

        # Get predictions
        win_prob = self.win_model.predict_proba(X)[0, 1]
        expected_roi = self.roi_model.predict(X)[0]

        # Calculate Kelly Criterion for bet sizing
        # Kelly = (p * b - q) / b, where p = win prob, q = 1-p, b = decimal odds - 1
        odds = bet_data.get('odds', -110)
        if odds < 0:
            decimal_odds = 1 + (100 / abs(odds))
        else:
            decimal_odds = 1 + (odds / 100)

        b = decimal_odds - 1
        p = win_prob
        q = 1 - p
        kelly_fraction = max(0, (p * b - q) / b)  # Don't bet if kelly is negative

        # Confidence based on sample size
        sample_size = historical_stats.get('sport_market_count', 100) if historical_stats else 100
        confidence = min(1.0, sample_size / 100)

        return {
            'win_probability': float(win_prob),
            'expected_roi': float(expected_roi),
            'kelly_fraction': float(kelly_fraction),
            'recommended_stake_pct': float(kelly_fraction * 100 * 0.25),  # Quarter Kelly
            'confidence': float(confidence),
            'profitable': bool(expected_roi > 0),
            'bet_grade': self._grade_bet(win_prob, expected_roi, confidence),
            'timestamp': datetime.now().isoformat(),
        }

    def _grade_bet(self, win_prob, expected_roi, confidence):
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

    def batch_predict(self, bets: List[Dict], historical_stats_list: List[Dict] = None) -> List[Dict]:
        """Predict for multiple bets"""
        if historical_stats_list is None:
            historical_stats_list = [None] * len(bets)

        results = []
        for bet, stats in zip(bets, historical_stats_list):
            try:
                prediction = self.predict(bet, stats)
                prediction['bet_id'] = bet.get('id', None)
                results.append(prediction)
            except Exception as e:
                print(f"Error predicting bet {bet.get('id')}: {e}")
                continue

        return results


def main():
    """Example usage"""
    print("\n" + "=" * 60)
    print("🎯 PIKKIT BET PROFITABILITY PREDICTOR")
    print("=" * 60 + "\n")

    # Initialize predictor
    predictor = BetProfitabilityPredictor()

    # Example bets
    example_bets = [
        {
            'id': 'bet_001',
            'sport': 'NBA',
            'league': 'NBA',
            'market': 'Spread',
            'institution_name': 'DraftKings',
            'bet_type': 'Over',
            'odds': -110,
            'clv_percentage': 2.5,
            'is_live': False,
        },
        {
            'id': 'bet_002',
            'sport': 'NFL',
            'league': 'NFL',
            'market': 'Total',
            'institution_name': 'FanDuel',
            'bet_type': 'Under',
            'odds': 105,
            'clv_percentage': -1.0,
            'is_live': True,
        },
        {
            'id': 'bet_003',
            'sport': 'NHL',
            'league': 'NHL',
            'market': 'Moneyline',
            'institution_name': 'BetMGM',
            'bet_type': 'Home',
            'odds': 150,
            'is_live': False,
        }
    ]

    # Example historical stats (would come from database in production)
    example_stats = [
        {
            'sport_win_rate': 0.52,
            'sport_roi': 1.5,
            'sport_market_win_rate': 0.51,
            'sport_market_roi': 0.8,
            'sport_market_count': 500,
            'institution_name_win_rate': 0.50,
            'institution_name_roi': 0.0,
            'institution_name_count': 200,
        },
        None,  # Test with no historical stats
        {
            'sport_win_rate': 0.48,
            'sport_roi': -0.5,
            'sport_market_count': 150,
        }
    ]

    # Batch predict
    print("🔮 Making predictions...")
    predictions = predictor.batch_predict(example_bets, example_stats)

    # Display results
    print("\n" + "=" * 60)
    print("📊 PREDICTIONS")
    print("=" * 60 + "\n")

    for pred in predictions:
        print(f"Bet ID: {pred.get('bet_id', 'N/A')}")
        print(f"  Grade: {pred['bet_grade']}")
        print(f"  Win Probability: {pred['win_probability']:.1%}")
        print(f"  Expected ROI: {pred['expected_roi']:+.2f}%")
        print(f"  Kelly Fraction: {pred['kelly_fraction']:.3f}")
        print(f"  Recommended Stake: {pred['recommended_stake_pct']:.2f}% of bankroll")
        print(f"  Confidence: {pred['confidence']:.1%}")
        print(f"  Profitable: {'✅ YES' if pred['profitable'] else '❌ NO'}")
        print()

    print("=" * 60)
    print("✅ Prediction complete!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
