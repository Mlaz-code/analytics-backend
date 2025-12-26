#!/usr/bin/env python3
"""
Pikkit Validation Review API
Serves validation reports and applies corrections
Run with: python3 validation-api.py
"""

import os
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Configuration
REPORTS_DIR = '/root/pikkit/reports'
CORRECTIONS_DIR = '/root/pikkit/corrections'
VALIDATION_UI_DIR = '/root/pikkit/validation-ui'

# Load Supabase config
env_file = '/root/pikkit/.env'
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://mnnjjvbaxzumfcgibtme.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_KEY', '')
PIKKIT_API_BASE = os.environ.get('PIKKIT_API_BASE', 'https://prod-website.pikkit.app')
PIKKIT_API_TOKEN = os.environ.get('PIKKIT_API_TOKEN', '')

# Create Flask app
app = Flask(__name__, static_folder=VALIDATION_UI_DIR, static_url_path='')
CORS(app)

# Ensure directories exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CORRECTIONS_DIR, exist_ok=True)


@app.route('/')
def index():
    """Serve the UI"""
    return send_from_directory(VALIDATION_UI_DIR, 'index.html')


@app.route('/api/validation/generate', methods=['GET'])
def generate_report():
    """Generate a validation report (all bets or limited sample)"""
    try:
        from validate_data import DataValidator

        # Get optional limit parameter from query string
        limit = request.args.get('limit', None)
        if limit:
            try:
                limit = int(limit)
            except ValueError:
                limit = None

        validator = DataValidator()
        print(f'Fetching Pikkit bets{f" (limit: {limit})" if limit else " (all)"}...')
        validator.fetch_pikkit_bets(limit=limit)
        print(f'Fetched {len(validator.pikkit_bets)} Pikkit bets')

        print('Fetching Supabase bets...')
        validator.fetch_supabase_bets()
        print(f'Fetched {len(validator.supabase_bets)} Supabase bets')

        print('Comparing bets...')
        validator.compare_bets()

        # Use the report data from the validator object
        report = validator.report

        return jsonify({'success': True, 'report': report})
    except Exception as e:
        print(f'Error generating report: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/validation/reports')
def get_reports():
    """Get list of available validation reports"""
    try:
        reports = []
        if os.path.exists(REPORTS_DIR):
            for f in os.listdir(REPORTS_DIR):
                if f.startswith('validation_') and f.endswith('.json'):
                    path = os.path.join(REPORTS_DIR, f)
                    stat = os.stat(path)
                    reports.append({
                        'filename': f,
                        'timestamp': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'size': stat.st_size
                    })
        reports.sort(key=lambda x: x['timestamp'], reverse=True)
        return jsonify(reports)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/validation/report/<filename>')
def get_report(filename):
    """Get a specific validation report"""
    try:
        # Prevent directory traversal
        if '..' in filename or '/' in filename:
            return jsonify({'error': 'Invalid filename'}), 400

        filepath = os.path.join(REPORTS_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Report not found'}), 404

        with open(filepath) as f:
            report = json.load(f)
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/validation/apply', methods=['POST'])
def apply_corrections():
    """Apply corrections to Supabase"""
    try:
        data = request.json
        corrections = data.get('corrections', [])

        if not corrections:
            return jsonify({'error': 'No corrections provided'}), 400

        headers = {
            'Authorization': f'Bearer {SUPABASE_KEY}',
            'apikey': SUPABASE_KEY,
            'Content-Type': 'application/json',
        }

        applied = 0
        failed = 0
        results = []

        for c in corrections:
            bet_id = c['id']
            action = c['action']

            try:
                if action == 'delete':
                    url = f"{SUPABASE_URL}/rest/v1/bets?id=eq.{bet_id}"
                    response = requests.delete(url, headers=headers, timeout=10)
                elif action in ('update_status', 'update_profit'):
                    url = f"{SUPABASE_URL}/rest/v1/bets?id=eq.{bet_id}"
                    update_data = {c['field']: c['new_value']}

                    # Also update profit and ROI for status updates
                    if action == 'update_status' and c.get('profit') is not None:
                        update_data['profit'] = c['profit']

                    response = requests.patch(url, headers=headers, json=update_data, timeout=10)
                else:
                    results.append({
                        'id': bet_id,
                        'action': action,
                        'status': 'error',
                        'error': 'Unknown action'
                    })
                    failed += 1
                    continue

                if response.ok:
                    applied += 1
                    results.append({
                        'id': bet_id,
                        'action': action,
                        'status': 'success'
                    })
                else:
                    failed += 1
                    results.append({
                        'id': bet_id,
                        'action': action,
                        'status': 'failed',
                        'error': response.text[:200]
                    })

            except Exception as e:
                failed += 1
                results.append({
                    'id': bet_id,
                    'action': action,
                    'status': 'error',
                    'error': str(e)
                })

        # Save correction log
        log_file = os.path.join(
            CORRECTIONS_DIR,
            f"applied_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(log_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'corrections': corrections,
                'results': results,
                'summary': {'applied': applied, 'failed': failed}
            }, f, indent=2)

        return jsonify({
            'applied': applied,
            'failed': failed,
            'results': results,
            'log': log_file
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/validation/stats')
def get_stats():
    """Get overall validation statistics"""
    try:
        stats = {
            'reports_count': 0,
            'corrections_applied': 0,
            'last_validation': None,
            'issues_found': 0
        }

        # Count reports
        if os.path.exists(REPORTS_DIR):
            reports = [f for f in os.listdir(REPORTS_DIR) if f.endswith('.json')]
            stats['reports_count'] = len(reports)

            # Get latest report stats
            if reports:
                latest = sorted(reports, reverse=True)[0]
                latest_path = os.path.join(REPORTS_DIR, latest)
                with open(latest_path) as f:
                    report = json.load(f)
                    summary = report.get('summary', {})
                    stats['last_validation'] = report.get('timestamp')
                    stats['issues_found'] = (
                        summary.get('mismatched', 0) +
                        summary.get('missing_in_supabase', 0) +
                        summary.get('missing_in_pikkit', 0)
                    )

        # Count applied corrections
        if os.path.exists(CORRECTIONS_DIR):
            corrections = [f for f in os.listdir(CORRECTIONS_DIR) if f.startswith('applied_')]
            stats['corrections_applied'] = len(corrections)

        return jsonify(stats)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Pikkit Validation API")
    print(f"Reports directory: {REPORTS_DIR}")
    print(f"Corrections directory: {CORRECTIONS_DIR}")
    print(f"UI directory: {VALIDATION_UI_DIR}")
    print()
    print("Starting server on http://localhost:5001")
    print("Press Ctrl+C to stop")
    print()

    app.run(
        host='0.0.0.0',
        port=5001,
        debug=False,
        use_reloader=False
    )
