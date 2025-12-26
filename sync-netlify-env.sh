#!/bin/bash
##############################################################################
# Netlify Environment Variable Sync Script
# Syncs credentials from /root/pikkit/.env to Netlify site
# Keeps Netlify deployment in sync with local configuration
##############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load .env file
ENV_FILE="/root/pikkit/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}❌ Error: .env file not found at $ENV_FILE${NC}"
    exit 1
fi

# Source the .env file
export $(grep -v '^#' "$ENV_FILE" | grep -v '^$' | xargs)

# Netlify site ID (from .env or hardcoded)
SITE_ID="${NETLIFY_SITE_ID:-689bd4cd16c931b7c90336bf}"

echo -e "${BLUE}🔄 Syncing environment variables to Netlify...${NC}"
echo -e "${BLUE}Site ID: $SITE_ID${NC}\n"

# Check if netlify CLI is installed
if ! command -v netlify &> /dev/null; then
    echo -e "${YELLOW}⚠️  Netlify CLI not installed. Installing...${NC}"
    npm install -g netlify-cli
fi

# Authenticate if needed
echo -e "${YELLOW}🔐 Authenticating with Netlify...${NC}"
export NETLIFY_AUTH_TOKEN="$NETLIFY_TOKEN"

# Function to set environment variable
set_env_var() {
    local key=$1
    local value=$2
    local context=${3:-"production,deploy-preview,branch-deploy"}

    if [ -z "$value" ]; then
        echo -e "${YELLOW}⚠️  Skipping $key (empty value)${NC}"
        return
    fi

    echo -e "${GREEN}✅ Setting $key${NC}"
    netlify env:set "$key" "$value" --site "$SITE_ID" --context "$context" 2>&1 | grep -v "Warning" || true
}

# Sync Supabase credentials
echo -e "\n${BLUE}📦 Supabase Credentials${NC}"
set_env_var "SUPABASE_URL" "$SUPABASE_URL"
set_env_var "SUPABASE_ANON_KEY" "$SUPABASE_ANON_KEY"

# Sync API URLs
echo -e "\n${BLUE}🔌 API Endpoints${NC}"
set_env_var "BACKEND_API_URL" "$EXTENSION_API_URL"
set_env_var "ML_API_URL" "$ML_API_URL"
set_env_var "N8N_WEBHOOK_URL" "$N8N_WEBHOOK_URL"

# Sync dashboard config
echo -e "\n${BLUE}🎨 Dashboard Config${NC}"
set_env_var "DASHBOARD_URL" "$DASHBOARD_URL"

# Optional: Sync Telegram credentials (if needed for notifications)
echo -e "\n${BLUE}📱 Telegram (Optional)${NC}"
read -p "Sync Telegram credentials? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    set_env_var "TELEGRAM_BOT_TOKEN" "$TELEGRAM_BOT_TOKEN"
    set_env_var "TELEGRAM_CHAT_ID" "$TELEGRAM_CHAT_ID"
fi

echo -e "\n${GREEN}✅ Environment variables synced successfully!${NC}"
echo -e "${YELLOW}💡 Tip: Redeploy your site for changes to take effect${NC}"
echo -e "${BLUE}   Run: netlify deploy --prod --site $SITE_ID${NC}\n"

# List all environment variables
echo -e "${BLUE}📋 Current Netlify Environment Variables:${NC}"
netlify env:list --site "$SITE_ID" 2>&1 | grep -v "Warning" || echo "Could not list variables"
