# Credentials Audit Report
Generated: 2025-12-23

## Executive Summary
Found **multiple hardcoded credentials** across the codebase that need centralization.

## 1. Supabase Credentials

### Current State
**✅ Backend (Python)** - Already using .env
- `/root/pikkit/extension-api.py` - Reads from env vars
- `/root/pikkit/sync_to_supabase.py` - Reads from env vars
- `/root/pikkit/.env` - Contains correct keys

**❌ Extension (JavaScript)** - Hardcoded
- `/root/pikkit-oddsjam-extension/popup.js:28-31` - Hardcoded URL and key

**❌ Dashboard (JavaScript)** - Hardcoded
- `/root/analytics-dashboard/supabase-client.js:4-5` - Hardcoded URL and key

### Keys Found
```
URL: https://mnnjjvbaxzumfcgibtme.supabase.co

ANON_KEY (in .env - EXPIRED):
sb_publishable_sWDtc6LjmIBE6k0ntZLlaw_MJF8t2a1

ANON_KEY (in JS files - CURRENT):
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1ubmpqdmJheHp1bWZjZ2lidG1lIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTk3NjA3ODksImV4cCI6MjA3NTMzNjc4OX0.N3up-YsvfACOaH2bIIE39ErKV3Y_GFlkHioypJhTvmc

SERVICE_KEY (in .env):
sb_secret_dLTyMJeDF7UxOriGMNIbhQ_eBBM8H7i
```

## 2. Netlify Credentials

### Current State
**📝 Documented** - Only in CLAUDE.md
- `/root/.claude/CLAUDE.md` - Token documented but not used in code

### Keys Found
```
NETLIFY_TOKEN: nfc_ZGtuPt6rv1dKVc5vkmJXhXxJycS3CtFS1920
```

**Usage:** Not currently used in automated scripts (manual deployment)

## 3. Telegram Bot Credentials

### Current State
**📝 Documented** - Only in CLAUDE.md
- `/root/.claude/CLAUDE.md` - Bot token documented

### Keys Found
```
TELEGRAM_BOT_TOKEN: 8459356740:AAGyqKe85uWOX81VFmYn0FgqpcyMwnlcWHU
TELEGRAM_CHAT_ID: 8254786612
```

**Usage:** Likely used in `/root/pikkit-autotap-telegram.sh` (script not found on Tools VM)

## 4. Other API Endpoints

### n8n Webhook
```
URL: https://n8n.chocopancake.com/webhook/pikkit-analysis
Workflow ID: fUof5TUXMcNQAB7x
```

**Status:** Public webhook, no auth required

---

## Priority Actions

### High Priority (Security Risk)
1. ✅ **Update .env with current Supabase ANON key** - Mismatch between .env and JS files
2. ❌ **Remove hardcoded Supabase keys from:**
   - Extension popup.js
   - Dashboard supabase-client.js

### Medium Priority (Best Practice)
3. ❌ **Add to .env:**
   - NETLIFY_TOKEN
   - TELEGRAM_BOT_TOKEN
   - TELEGRAM_CHAT_ID
   - N8N_WEBHOOK_URL

### Low Priority (Optional)
4. ❌ **Document which keys are safe to expose** (anon keys vs service keys)

---

## Recommendation

**Implement Backend Proxy Pattern:**
- Extension/Dashboard → Backend API → Supabase
- Keeps secrets server-side
- Single source of truth in .env

**Next Steps:**
1. Update .env with all credentials
2. Create backend proxy endpoints
3. Update extension to use proxy
4. Update dashboard to use proxy
5. Remove hardcoded credentials
