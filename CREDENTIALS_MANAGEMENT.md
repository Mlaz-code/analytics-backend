# Credentials Management Guide

## Overview
All Pikkit credentials are now centralized in `/root/pikkit/.env` as a single source of truth.

**Hybrid Network Detection**: Frontend applications use smart detection to automatically choose the best method:
- **Local Network**: Uses backend proxy (fast, secure)
- **External Network**: Falls back to direct Supabase access (works anywhere)

See [HYBRID_NETWORK_DETECTION.md](HYBRID_NETWORK_DETECTION.md) for technical details.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   /root/pikkit/.env                         │
│              (Single Source of Truth)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ├──────────────────┐
                       ▼                  ▼
           ┌───────────────────┐  ┌──────────────────┐
           │  Backend API      │  │  Sync Scripts    │
           │  (192.168.4.80)   │  │  - sync-to-supabase│
           │  Port: 8000       │  │  - sync-netlify  │
           └─────────┬─────────┘  └──────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
┌──────────────┐ ┌────────────┐ ┌─────────────┐
│  Extension   │ │ Dashboard  │ │  Netlify    │
│  (Local)     │ │  (Local)   │ │  (Cloud)    │
└──────────────┘ └────────────┘ └─────────────┘
```

## File Structure

```
/root/pikkit/
├── .env                        # ✅ Single source of truth
├── .env.example                # Template for new setups
├── CREDENTIALS_AUDIT.md        # Audit results
├── CREDENTIALS_MANAGEMENT.md   # This file
├── extension-api.py            # ✅ Backend proxy API
├── sync-netlify-env.sh         # ✅ Netlify sync script
└── sync_to_supabase.py         # Uses .env

/root/analytics-dashboard/
├── config.js                   # ✅ Runtime configuration
├── inject-env.js               # ✅ Build-time env injection
├── netlify.toml                # ✅ Netlify configuration
└── supabase-client.js          # ✅ Updated to use proxy

/root/pikkit-oddsjam-extension/
└── popup.js                    # ✅ Updated to use proxy
```

## .env File Structure

```bash
# Supabase (Database)
SUPABASE_URL=https://mnnjjvbaxzumfcgibtme.supabase.co
SUPABASE_ANON_KEY=eyJhbGc...  # Safe for frontend
SUPABASE_SERVICE_KEY=sb_secret_...  # Backend only!

# Netlify (Deployment)
NETLIFY_TOKEN=nfc_...
NETLIFY_SITE_ID=689bd4cd16c931b7c90336bf

# Telegram (Notifications)
TELEGRAM_BOT_TOKEN=8459356740:AAG...
TELEGRAM_CHAT_ID=8254786612

# n8n (Workflows)
N8N_WEBHOOK_URL=https://n8n.chocopancake.com/webhook/pikkit-analysis
N8N_WORKFLOW_ID=fUof5TUXMcNQAB7x

# API Endpoints
ML_API_URL=http://192.168.4.80:5000
EXTENSION_API_URL=http://192.168.4.80:8000
DASHBOARD_URL=https://app.chocopancake.com
```

## Backend API Endpoints

### 1. Supabase Query Proxy
```bash
GET /api/supabase/query
```

**Query Parameters:**
- `table` (required) - Table name
- `select` (optional) - Columns to select (default: *)
- `filter` (optional) - Query filters
- `order` (optional) - Order by clause
- `limit` (optional) - Limit results (default: 100)

**Example:**
```javascript
fetch('http://192.168.4.80:8000/api/supabase/query?table=bets&select=*&order=created_at.desc&limit=10')
```

### 2. Last Sync Time
```bash
GET /api/supabase/last-sync
```

**Response:**
```json
{
  "success": true,
  "last_sync": "2025-12-22T13:19:32.11572+00:00"
}
```

### 3. Health Check
```bash
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-23T...",
  "supabase_configured": true
}
```

## Usage Guide

### Local Development

**1. Extension:**
```javascript
// Extension automatically uses local backend proxy
fetch('http://192.168.4.80:8000/api/supabase/last-sync')
```

**2. Dashboard:**
```javascript
// Dashboard detects config and uses proxy
const client = new SupabaseClient(); // Automatically uses proxy
const data = await client.request('bets?select=*&limit=10');
```

### Netlify Deployment

**1. Sync environment variables:**
```bash
cd /root/pikkit
./sync-netlify-env.sh
```

**2. Deploy to Netlify:**
```bash
cd /root/analytics-dashboard
netlify deploy --prod
```

The build process (`inject-env.js`) will:
- Read env vars from Netlify
- Generate `config.js` with production settings
- Dashboard will use proxy or fallback to direct access

## Syncing Netlify Environment Variables

### Automatic Sync (Recommended)

```bash
cd /root/pikkit
./sync-netlify-env.sh
```

This script:
1. Reads `/root/pikkit/.env`
2. Authenticates with Netlify
3. Sets environment variables on Netlify site
4. Lists current variables for verification

### Manual Sync (via Netlify UI)

1. Go to: https://app.netlify.com/sites/689bd4cd16c931b7c90336bf/settings/deploys#environment
2. Add/update environment variables:
   - `BACKEND_API_URL` = Your backend URL
   - `ML_API_URL` = Your ML API URL
   - `SUPABASE_URL` = Supabase URL
   - `SUPABASE_ANON_KEY` = Supabase anon key

### Testing Sync

```bash
# Check Netlify env vars
netlify env:list --site 689bd4cd16c931b7c90336bf

# Test build locally
cd /root/analytics-dashboard
node inject-env.js
cat config.js  # Should show injected values
```

## Security Best Practices

### ✅ DO:
- Keep `.env` file secure (chmod 600)
- Use backend proxy for Supabase queries
- Use SUPABASE_SERVICE_KEY only in backend
- Rotate keys periodically
- Use HTTPS for API endpoints in production

### ❌ DON'T:
- Commit `.env` to git
- Expose SERVICE_KEY to frontend
- Hardcode credentials in code
- Share credentials in chat/email
- Use same key for dev/prod

## Credential Rotation

When rotating credentials:

1. **Update .env file:**
   ```bash
   nano /root/pikkit/.env
   ```

2. **Sync to Netlify:**
   ```bash
   ./sync-netlify-env.sh
   ```

3. **Restart backend API:**
   ```bash
   pkill -f extension-api.py
   cd /root/pikkit && python3 extension-api.py &
   ```

4. **Redeploy dashboard:**
   ```bash
   cd /root/analytics-dashboard
   netlify deploy --prod
   ```

5. **Reload extension:**
   - Go to `chrome://extensions`
   - Click refresh icon on Bet Advisor

## Troubleshooting

### Extension shows "No data"
- Check backend API is running: `curl http://192.168.4.80:8000/api/health`
- Check .env has correct keys: `cat /root/pikkit/.env | grep SUPABASE_ANON_KEY`
- Check logs: View browser console

### Dashboard not loading data
- Check config: Open browser console, see `window.PIKKIT_CONFIG`
- Check proxy: `curl http://192.168.4.80:8000/api/supabase/query?table=bets&limit=1`
- Check Supabase client: Look for "Proxy request failed" in console

### Netlify deployment issues
- Verify env vars: `netlify env:list`
- Check build logs in Netlify UI
- Verify inject-env.js ran: Look for "Injecting environment variables"

### Backend API not responding
- Check if running: `ps aux | grep extension-api`
- Check logs: `journalctl -u extension-api -n 50`
- Check port: `netstat -tulpn | grep 8000`
- Restart: `cd /root/pikkit && python3 extension-api.py`

## Monitoring

### Check backend health
```bash
curl http://192.168.4.80:8000/api/health
```

### Check Supabase connection
```bash
curl "http://192.168.4.80:8000/api/supabase/query?table=bets&limit=1"
```

### Check last sync time
```bash
curl http://192.168.4.80:8000/api/supabase/last-sync
```

## Future Improvements

- [ ] Add credential expiry monitoring
- [ ] Automate Netlify sync on .env changes
- [ ] Add rate limiting to proxy endpoints
- [ ] Implement caching layer (Redis)
- [ ] Add API authentication for extension
- [ ] Create credential rotation automation
- [ ] Add secret management (Vault/AWS Secrets)

---

**Last Updated:** 2025-12-23
**Maintainer:** Pikkit Team
