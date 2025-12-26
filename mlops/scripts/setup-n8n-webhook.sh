#!/bin/bash
# =============================================================================
# Setup n8n Webhook for High-Grade Bet Alerts
# Creates/updates n8n workflow for handling ML API notifications
# =============================================================================

set -euo pipefail

# Configuration
N8N_BASE_URL="${N8N_BASE_URL:-https://n8n.chocopancake.com}"
WORKFLOW_NAME="Pikkit ML High-Grade Bet Handler"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Create workflow JSON
create_workflow_json() {
    cat << 'EOF'
{
  "name": "Pikkit ML High-Grade Bet Handler",
  "nodes": [
    {
      "parameters": {
        "httpMethod": "POST",
        "path": "pikkit-high-grade-bet",
        "responseMode": "onReceived",
        "responseData": "allEntries"
      },
      "id": "webhook",
      "name": "Webhook",
      "type": "n8n-nodes-base.webhook",
      "typeVersion": 1,
      "position": [250, 300]
    },
    {
      "parameters": {
        "conditions": {
          "string": [
            {
              "value1": "={{ $json.grade }}",
              "operation": "equals",
              "value2": "A"
            }
          ]
        }
      },
      "id": "if-grade-a",
      "name": "Is Grade A?",
      "type": "n8n-nodes-base.if",
      "typeVersion": 1,
      "position": [450, 300]
    },
    {
      "parameters": {
        "chatId": "={{ $env.TELEGRAM_CHAT_ID }}",
        "text": "=🎯 <b>A-GRADE BET ALERT</b>\n\n<b>Sport:</b> {{ $json.bet.sport }}\n<b>League:</b> {{ $json.bet.league }}\n<b>Market:</b> {{ $json.bet.market }}\n<b>Book:</b> {{ $json.bet.institution_name }}\n<b>Odds:</b> {{ $json.bet.odds }}\n\n<b>Prediction:</b>\n• Win Prob: {{ ($json.prediction.win_probability * 100).toFixed(1) }}%\n• Expected ROI: {{ $json.prediction.expected_roi.toFixed(2) }}%\n• Kelly: {{ ($json.prediction.kelly_fraction * 100).toFixed(2) }}%\n\n⏰ {{ new Date().toLocaleString() }}",
        "additionalFields": {
          "parse_mode": "HTML"
        }
      },
      "id": "telegram-a",
      "name": "Telegram A-Grade",
      "type": "n8n-nodes-base.telegram",
      "typeVersion": 1,
      "position": [650, 200],
      "credentials": {
        "telegramApi": {
          "id": "1",
          "name": "Telegram API"
        }
      }
    },
    {
      "parameters": {
        "chatId": "={{ $env.TELEGRAM_CHAT_ID }}",
        "text": "=📊 <b>B-GRADE BET ALERT</b>\n\n<b>Sport:</b> {{ $json.bet.sport }}\n<b>League:</b> {{ $json.bet.league }}\n<b>Market:</b> {{ $json.bet.market }}\n<b>Book:</b> {{ $json.bet.institution_name }}\n<b>Odds:</b> {{ $json.bet.odds }}\n\n<b>Prediction:</b>\n• Win Prob: {{ ($json.prediction.win_probability * 100).toFixed(1) }}%\n• Expected ROI: {{ $json.prediction.expected_roi.toFixed(2) }}%\n\n⏰ {{ new Date().toLocaleString() }}",
        "additionalFields": {
          "parse_mode": "HTML"
        }
      },
      "id": "telegram-b",
      "name": "Telegram B-Grade",
      "type": "n8n-nodes-base.telegram",
      "typeVersion": 1,
      "position": [650, 400],
      "credentials": {
        "telegramApi": {
          "id": "1",
          "name": "Telegram API"
        }
      }
    },
    {
      "parameters": {
        "operation": "insert",
        "table": "ml_alerts",
        "columns": "grade,sport,league,market,institution_name,odds,win_probability,expected_roi,kelly_fraction,created_at",
        "values": "={{ $json.grade }},{{ $json.bet.sport }},{{ $json.bet.league }},{{ $json.bet.market }},{{ $json.bet.institution_name }},{{ $json.bet.odds }},{{ $json.prediction.win_probability }},{{ $json.prediction.expected_roi }},{{ $json.prediction.kelly_fraction }},{{ new Date().toISOString() }}"
      },
      "id": "supabase",
      "name": "Log to Supabase",
      "type": "n8n-nodes-base.supabase",
      "typeVersion": 1,
      "position": [850, 300],
      "credentials": {
        "supabaseApi": {
          "id": "1",
          "name": "Supabase API"
        }
      }
    }
  ],
  "connections": {
    "Webhook": {
      "main": [
        [
          {
            "node": "Is Grade A?",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Is Grade A?": {
      "main": [
        [
          {
            "node": "Telegram A-Grade",
            "type": "main",
            "index": 0
          }
        ],
        [
          {
            "node": "Telegram B-Grade",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Telegram A-Grade": {
      "main": [
        [
          {
            "node": "Log to Supabase",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Telegram B-Grade": {
      "main": [
        [
          {
            "node": "Log to Supabase",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  },
  "active": true,
  "settings": {},
  "tags": ["pikkit", "ml", "alerts"]
}
EOF
}

# Main
main() {
    log_info "Setting up n8n webhook workflow for Pikkit ML API"
    log_info "n8n URL: $N8N_BASE_URL"

    # Generate workflow JSON
    WORKFLOW_FILE="/tmp/pikkit-n8n-workflow.json"
    create_workflow_json > "$WORKFLOW_FILE"

    log_info "Workflow configuration saved to: $WORKFLOW_FILE"
    log_info ""
    log_info "To import this workflow into n8n:"
    log_info "1. Open n8n at $N8N_BASE_URL"
    log_info "2. Go to Workflows > Import from File"
    log_info "3. Select $WORKFLOW_FILE"
    log_info "4. Configure credentials for Telegram and Supabase"
    log_info "5. Activate the workflow"
    log_info ""
    log_info "Webhook URL will be: $N8N_BASE_URL/webhook/pikkit-high-grade-bet"
    log_info ""
    log_info "Update your .env with:"
    log_info "  N8N_WEBHOOK_URL=$N8N_BASE_URL/webhook/pikkit-high-grade-bet"
}

main "$@"
