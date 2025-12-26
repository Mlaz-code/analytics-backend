# Hybrid Network Detection Guide

## Overview

Pikkit uses **smart network detection** to provide the best experience on both local and external networks:

- **Local Network (192.168.4.x)**: Uses fast local backend proxy (secure, no credentials exposed)
- **External Network**: Falls back to direct Supabase access (works anywhere)

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Extension / Dashboard                       │
│                                                          │
│  1. Try local proxy (2 second timeout)                  │
│     ├─ Success → Use proxy (fast & secure)              │
│     └─ Fail → Fall back to direct Supabase              │
└─────────────────────────────────────────────────────────┘
```

### Extension Smart Detection (popup.js)

```javascript
// 1. Try local proxy first (192.168.4.80:8000)
fetch('http://192.168.4.80:8000/api/supabase/last-sync', {
    signal: abortController.signal, // 2 second timeout
})

// 2. If proxy fails, use direct Supabase
fetch('https://mnnjjvbaxzumfcgibtme.supabase.co/rest/v1/bets...', {
    headers: { 'apikey': SUPABASE_ANON_KEY }
})
```

**Timeouts:**
- Proxy attempt: 2 seconds
- Direct Supabase: Standard fetch timeout (~30s)

### Dashboard Smart Detection (config.js + supabase-client.js)

```javascript
// 1. Quick network detection (1 second)
fetch('http://192.168.4.80:8000/api/health', {
    signal: abortController.signal
})

// 2. Sets window.PIKKIT_CONFIG.LOCAL_NETWORK flag

// 3. SupabaseClient automatically uses:
//    - requestViaProxy() if proxy reachable
//    - Direct Supabase if proxy unreachable
```

## Network Detection Logic

### Extension
- **Timeout**: 2 seconds for proxy
- **Retry**: None (immediate fallback)
- **Caching**: Per-request (no session caching)
- **Logging**: Console logs show which method is used

### Dashboard
- **Timeout**: 1 second for initial detection
- **Retry**: None (immediate fallback)
- **Caching**: Session-level (sets LOCAL_NETWORK flag)
- **Logging**: Console logs network type

## Behavior by Environment

### Local Network (at home)
```
Extension:
  ✅ Tries proxy → Success → Uses 192.168.4.80:8000
  ⏱️  Response time: ~50ms (local network)
  🔒 Credentials: Stay on server

Dashboard:
  ✅ Detects local network → Uses proxy
  ⏱️  Response time: ~50ms
  🔒 Credentials: Stay on server
```

### External Network (coffee shop, mobile, cloud)
```
Extension:
  ⚠️  Tries proxy → Timeout (2s) → Falls back to direct
  ✅ Uses direct Supabase access
  ⏱️  Response time: ~200-500ms (internet)
  🔓 Uses ANON key (safe for frontend - RLS protected)

Dashboard:
  ⚠️  Detects external network → Falls back to direct
  ✅ Uses direct Supabase access
  ⏱️  Response time: ~200-500ms
  🔓 Uses ANON key (safe for frontend)
```

### Netlify Deployment
```
Dashboard:
  ⚠️  Proxy unreachable (private IP)
  ✅ Automatically uses direct Supabase
  ⏱️  Response time: ~200-500ms
  🔓 Uses ANON key (safe)
```

## Security Considerations

### Is the ANON Key Safe to Expose?
**YES** - The Supabase ANON key is designed for frontend use:

✅ **Row Level Security (RLS)**: Supabase enforces access control
✅ **Read-only by default**: Can't modify data without RLS rules
✅ **Publicly documented**: Supabase shows this in their docs
✅ **Expires**: Our key expires in 2035

### What About SERVICE_KEY?
**NEVER EXPOSE** - The SERVICE_KEY bypasses RLS:

❌ Never send to frontend
❌ Never commit to git
✅ Only used in backend (extension-api.py)
✅ Only stored in .env

## Performance Comparison

| Environment | Method | Latency | Security |
|------------|---------|---------|----------|
| Local Network | Proxy | ~50ms | ⭐⭐⭐ Best |
| External | Direct Supabase | ~300ms | ⭐⭐ Good (RLS) |
| Netlify | Direct Supabase | ~200ms | ⭐⭐ Good (RLS) |

## Console Logging

### Extension Console (Local)
```
✅ Using local proxy API
```

### Extension Console (External)
```
⚠️ Proxy unavailable, using direct Supabase access
✅ Using direct Supabase access
```

### Dashboard Console (Local)
```
✅ Local network detected - using proxy for best performance
```

### Dashboard Console (External)
```
⚠️ External network detected - using direct Supabase access
```

## Testing

### Test Local Network Detection

**Extension:**
```javascript
// Open extension popup
// Check console - should show:
"✅ Using local proxy API"
```

**Dashboard:**
```javascript
// Open dashboard
// Check console - should show:
"✅ Local network detected"
// Check PIKKIT_CONFIG:
window.PIKKIT_CONFIG.LOCAL_NETWORK // should be true
```

### Test External Network Detection

**Method 1: Disable proxy**
```bash
# Stop backend API
pkill -f extension-api.py

# Open extension/dashboard
# Should fall back to direct Supabase
```

**Method 2: Test from external network**
```bash
# Open dashboard on Netlify
https://pikkit-2d-dashboard.netlify.app

# Check console - should show:
"⚠️ External network detected"
```

## Troubleshooting

### Extension stuck on "Loading..."

**Problem**: Both proxy and direct access failing

**Check:**
```javascript
// Open browser console
// Look for errors
```

**Fix:**
1. Check internet connection
2. Check Supabase status: https://status.supabase.io
3. Verify ANON key hasn't expired

### Dashboard shows "External network" when on local network

**Problem**: Proxy not responding fast enough

**Check:**
```bash
# Test proxy manually
curl http://192.168.4.80:8000/api/health

# Should return quickly
{"status":"healthy",...}
```

**Fix:**
1. Restart backend API
2. Check firewall isn't blocking port 8000
3. Increase timeout in config.js

### Dashboard loads slowly

**Problem**: Waiting for proxy timeout before falling back

**Solution**: Normal behavior when proxy is unreachable. The timeout is intentionally short (1-2 seconds) to minimize delay.

## Best Practices

### ✅ DO:
- Let automatic detection work (no user config needed)
- Monitor console logs to verify detection
- Use proxy when available (faster + more secure)
- Accept fallback when necessary (works everywhere)

### ❌ DON'T:
- Hardcode one method or the other
- Increase timeouts too much (slows down external users)
- Expose SERVICE_KEY to frontend
- Disable fallback (breaks external access)

## Future Enhancements

Potential improvements:

1. **Network Detection Caching**
   - Cache detection result in localStorage
   - Re-check every 5 minutes
   - Faster subsequent loads

2. **Adaptive Timeout**
   - Measure actual response times
   - Adjust timeout dynamically
   - Optimize for user's network

3. **User Override**
   - Settings option to force proxy or direct
   - Debug mode for testing
   - Performance metrics display

4. **Cloud Proxy Option**
   - Deploy proxy to Optima VPS
   - Use `https://api.chocopancake.com`
   - Works everywhere, keeps credentials secure

---

**Last Updated:** 2025-12-23
**Related Docs:** CREDENTIALS_MANAGEMENT.md, CREDENTIALS_AUDIT.md
