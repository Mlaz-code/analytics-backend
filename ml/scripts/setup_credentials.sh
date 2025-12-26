#!/bin/bash
# Setup Supabase credentials for Pikkit ML
# NOTE: This script is DEPRECATED. Use the main .env file at /root/pikkit/.env instead.

echo "🔐 Checking Supabase credentials..."

# Check if credentials already exist
if [ -f /root/pikkit/.env ]; then
    echo "✅ Credentials file already exists at /root/pikkit/.env"

    # Export for current session
    set -a  # Export all variables
    source /root/pikkit/.env
    set +a

    echo ""
    echo "✅ Environment configured!"
    echo "   SUPABASE_URL: $SUPABASE_URL"
    echo "   SUPABASE_SERVICE_KEY: ${SUPABASE_SERVICE_KEY:0:20}..."
    echo ""
    echo "To use in scripts, run: source /root/pikkit/.env"
else
    echo "❌ Credentials file not found at /root/pikkit/.env"
    echo ""
    echo "Please create the .env file with the following structure:"
    echo ""
    echo "SUPABASE_URL=https://mnnjjvbaxzumfcgibtme.supabase.co"
    echo "SUPABASE_ANON_KEY=your_anon_key"
    echo "SUPABASE_SERVICE_KEY=your_service_key"
    echo ""
    echo "See /root/pikkit/.env.example for a full template."
    exit 1
fi
