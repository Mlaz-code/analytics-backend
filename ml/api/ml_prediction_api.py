#!/usr/bin/env python3
"""
Flask API for real-time ML bet predictions
Serves the trained XGBoost models via HTTP endpoint
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
from pathlib import Path

# Add parent directory to path to import predictor
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from predict_bet_profitability import BetProfitabilityPredictor

app = Flask(__name__)
CORS(app)  # Enable CORS for Chrome extension

# Initialize predictor (loads models on startup)
predictor = BetProfitabilityPredictor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': predictor.win_model is not None and predictor.roi_model is not None,
        'model_timestamp': predictor.metadata.get('timestamp', 'unknown') if predictor.metadata else 'unknown'
    })

@app.route('/ml-predict', methods=['GET', 'POST'])
def ml_predict():
    """
    ML prediction endpoint

    Query Parameters / JSON Body:
        sport: str - Sport type (Basketball, American Football, etc.)
        league: str - League (NBA, NFL, NCAAB, etc.)
        market: str - Market type (Spread, Total, Moneyline, etc.)
        institution_name: str - Bookmaker (DraftKings, FanDuel, etc.)
        bet_type: str - Bet type (straight, parlay, etc.) [optional, default: straight]
        odds: int - American odds (e.g., -110, +150) [optional, default: -110]
        is_live: bool - Is live bet [optional, default: false]
        clv_percentage: float - Closing line value % [optional]

    Returns:
        JSON with predictions:
        {
            "win_probability": 0.523,
            "expected_roi": 1.2,
            "kelly_fraction": 0.015,
            "recommended_stake_pct": 0.38,
            "confidence": 0.82,
            "bet_grade": "C",
            "profitable": true,
            "feature_summary": {...}
        }
    """
    try:
        # Get parameters from GET or POST
        if request.method == 'POST':
            data = request.get_json() or {}
        else:
            data = request.args.to_dict()

        # Extract bet parameters
        bet_data = {
            'sport': data.get('sport', ''),
            'league': data.get('league', ''),
            'market': data.get('market', ''),
            'institution_name': data.get('institution_name', ''),
            'bet_type': data.get('bet_type', 'straight'),
            'american_odds': int(data.get('odds', -110)),
            'is_live': data.get('is_live', 'false').lower() == 'true',
            'clv_percentage': float(data.get('clv_percentage', 0))
        }

        # Validate required fields
        required_fields = ['sport', 'league', 'market', 'institution_name']
        missing_fields = [f for f in required_fields if not bet_data.get(f)]

        if missing_fields:
            return jsonify({
                'error': f"Missing required fields: {', '.join(missing_fields)}",
                'required': required_fields
            }), 400

        # Get prediction
        prediction = predictor.predict(bet_data)

        return jsonify(prediction)

    except ValueError as e:
        return jsonify({'error': f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for multiple bets

    Request Body:
        {
            "bets": [
                {...bet_data...},
                {...bet_data...}
            ]
        }

    Returns:
        {
            "predictions": [...],
            "count": N
        }
    """
    try:
        data = request.get_json()
        bets = data.get('bets', [])

        if not bets:
            return jsonify({'error': 'No bets provided'}), 400

        predictions = []
        for bet in bets:
            try:
                pred = predictor.predict(bet)
                predictions.append(pred)
            except Exception as e:
                predictions.append({'error': str(e)})

        return jsonify({
            'predictions': predictions,
            'count': len(predictions)
        })

    except Exception as e:
        return jsonify({'error': f"Batch prediction failed: {str(e)}"}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """
    Get model metadata and feature importance
    """
    try:
        return jsonify({
            'metadata': predictor.metadata,
            'feature_names': predictor.feature_names,
            'feature_count': len(predictor.feature_names) if predictor.feature_names else 0
        })
    except Exception as e:
        return jsonify({'error': f"Failed to get model info: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found', 'available_endpoints': [
        'GET /health',
        'GET|POST /ml-predict',
        'POST /batch-predict',
        'GET /model-info'
    ]}), 404

if __name__ == '__main__':
    # Run on port 5000, accessible from network
    app.run(host='0.0.0.0', port=5000, debug=False)
