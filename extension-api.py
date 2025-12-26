#!/usr/bin/env python3
"""
Pikkit Extension Backend API
Serves bet scoring and opportunity capture endpoints for the Chrome extension
Connects to Supabase to get historical betting performance data
"""

import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

# Configuration from .env
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://mnnjjvbaxzumfcgibtme.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_KEY', '')  # Use service key for backend
SUPABASE_ANON_KEY = os.environ.get('SUPABASE_ANON_KEY', '')  # For proxy endpoints

app = Flask(__name__)
CORS(app)

def get_bet_performance(sport, market, book, league='', player='', bet_type='straight'):
    """
    Get historical performance for a specific bet type
    Queries Supabase to find matching bets and calculate statistics
    """
    try:
        if not SUPABASE_KEY:
            return {
                'success': False,
                'error': 'Supabase credentials not configured'
            }
        
        headers = {
            'Authorization': f'Bearer {SUPABASE_KEY}',
            'apikey': SUPABASE_KEY,
            'Content-Type': 'application/json',
        }
        
        # Build query to find matching bets from Supabase
        # Query format: SELECT * FROM bets WHERE sport=? AND market=? AND book=? AND league=?
        query = f"sport=eq.{sport}&market=eq.{market}&book=eq.{book}"
        
        if league:
            query += f"&league=eq.{league}"
        
        if player:
            query += f"&player=ilike.{player}"
        
        # Query the bets table
        url = f"{SUPABASE_URL}/rest/v1/bets?{query}&select=*"
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if not response.ok:
            return {
                'success': False,
                'error': f'Supabase query failed: {response.status_code}'
            }
        
        bets = response.json() if isinstance(response.json(), list) else []
        
        if not bets:
            return {
                'success': True,
                'performance': {
                    'totalBets': 0,
                    'totalWagered': 0,
                    'totalProfit': 0,
                    'roi': 0,
                    'winRate': 0,
                    'wins': 0,
                    'losses': 0
                },
                'evaluation': {
                    'score': 'unknown',
                    'scoreValue': 0,
                    'recommendation': 'No historical data for this market',
                    'warnings': []
                }
            }
        
        # Calculate statistics
        total_wagered = sum(float(b.get('wagered', 0)) for b in bets)
        total_profit = sum(float(b.get('profit', 0)) for b in bets)
        total_bets = len(bets)
        wins = sum(1 for b in bets if float(b.get('profit', 0)) > 0)
        losses = total_bets - wins
        roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
        win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
        
        # Determine score based on ROI and win rate
        if roi >= 5 and win_rate >= 52:
            score = 'good'
        elif roi >= 0 and win_rate >= 48:
            score = 'neutral'
        elif roi >= -5 and win_rate >= 44:
            score = 'warning'
        else:
            score = 'danger'
        
        return {
            'success': True,
            'performance': {
                'totalBets': total_bets,
                'totalWagered': round(total_wagered, 2),
                'totalProfit': round(total_profit, 2),
                'roi': round(roi, 1),
                'winRate': round(win_rate, 1),
                'wins': wins,
                'losses': losses
            },
            'evaluation': {
                'score': score,
                'scoreValue': round(roi, 1),
                'recommendation': f'{score.capitalize()} performing market based on {total_bets} bets',
                'warnings': []
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@app.route('/api/score-bet', methods=['GET'])
def score_bet():
    """
    Score a specific bet based on historical performance
    Query parameters: sport, market, book, league, player, betType
    """
    try:
        sport = request.args.get('sport', '')
        market = request.args.get('market', '')
        book = request.args.get('book', '')
        league = request.args.get('league', '')
        player = request.args.get('player', '')
        bet_type = request.args.get('betType', 'straight')
        
        if not all([sport, market, book]):
            return jsonify({
                'success': False,
                'error': 'Missing required parameters: sport, market, book'
            }), 400
        
        result = get_bet_performance(sport, market, book, league, player, bet_type)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/capture-opportunities', methods=['POST'])
def capture_opportunities():
    """
    Capture identified opportunities to Supabase
    """
    try:
        data = request.json
        opportunities = data.get('opportunities', [])
        
        if not opportunities:
            return jsonify({
                'success': False,
                'error': 'No opportunities provided'
            }), 400
        
        # Log opportunities capture (implement as needed)
        print(f"Captured {len(opportunities)} opportunities at {datetime.now()}")
        
        return jsonify({
            'success': True,
            'captured': len(opportunities)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/supabase/query', methods=['GET'])
def supabase_query():
    """
    Generic Supabase query proxy endpoint
    Keeps credentials server-side while allowing frontend to query Supabase

    Query parameters:
        table: Table name (required)
        select: Columns to select (default: *)
        filter: Query filters (e.g., "sport=eq.Basketball&limit=10")
        order: Order by clause (e.g., "created_at.desc")
        limit: Limit results (default: 100)
    """
    try:
        table = request.args.get('table')
        if not table:
            return jsonify({'error': 'table parameter required'}), 400

        select = request.args.get('select', '*')
        filter_params = request.args.get('filter', '')
        order = request.args.get('order', '')
        limit = request.args.get('limit', '100')

        # Build query URL
        query_parts = [f"select={select}"]
        if filter_params:
            query_parts.append(filter_params)
        if order:
            query_parts.append(f"order={order}")
        if limit:
            query_parts.append(f"limit={limit}")

        query_string = '&'.join(query_parts)
        url = f"{SUPABASE_URL}/rest/v1/{table}?{query_string}"

        headers = {
            'Authorization': f'Bearer {SUPABASE_ANON_KEY}',
            'apikey': SUPABASE_ANON_KEY,
            'Content-Type': 'application/json'
        }

        response = requests.get(url, headers=headers, timeout=10)

        if not response.ok:
            return jsonify({
                'error': f'Supabase query failed: {response.status_code}',
                'details': response.text
            }), response.status_code

        return jsonify(response.json())

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/supabase/last-sync', methods=['GET'])
def last_sync():
    """
    Get last Pikkit to Supabase sync time
    Returns the most recent bet creation timestamp
    """
    try:
        url = f"{SUPABASE_URL}/rest/v1/bets?select=created_at&order=created_at.desc&limit=1"

        headers = {
            'Authorization': f'Bearer {SUPABASE_ANON_KEY}',
            'apikey': SUPABASE_ANON_KEY,
            'Content-Type': 'application/json'
        }

        response = requests.get(url, headers=headers, timeout=10)

        if not response.ok:
            return jsonify({
                'success': False,
                'error': f'Supabase query failed: {response.status_code}'
            }), response.status_code

        data = response.json()

        if data and len(data) > 0:
            return jsonify({
                'success': True,
                'last_sync': data[0]['created_at']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No data found'
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'supabase_configured': bool(SUPABASE_ANON_KEY and SUPABASE_KEY)
    })


if __name__ == '__main__':
    print("Pikkit Extension Backend API")
    print(f"Supabase URL: {SUPABASE_URL}")
    print("Starting server on http://localhost:8000")
    print("Press Ctrl+C to stop\n")
    
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=False,
        use_reloader=False
    )
