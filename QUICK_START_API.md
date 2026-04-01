# Flask API - Quick Start Guide

## Start the API in 3 Steps

### Step 1: Navigate to Project
```bash
cd "C:\Users\DEVENDER\OneDrive\Desktop\voicbot\project-root"
```

### Step 2: Start API Server
```bash
python app_api.py
```

Output should show:
```
============================================================
VoiceBot Trading API Server
============================================================

Endpoints:
  POST /api/auth/register
  POST /api/auth/login
  GET  /api/signals/today
  GET  /api/signals/history
  GET  /api/performance
  GET  /api/user/profile
  GET  /api/health

Running on http://127.0.0.1:5000
============================================================
```

### Step 3: Test in Another Terminal
```bash
python test_api.py
```

Expected: All 5 tests pass with Status 200-201

---

## Quick API Test Examples

### Using Python
```python
import requests

# Register user
resp = requests.post("http://localhost:5000/api/auth/register", json={
    "email": "user@example.com",
    "name": "John Doe",
    "password": "test123"
})
token = resp.json()["access_token"]

# Get signals
headers = {"Authorization": f"Bearer {token}"}
signals = requests.get("http://localhost:5000/api/signals/today", headers=headers)
print(signals.json())
```

### Using PowerShell
```powershell
# Health check
$response = Invoke-WebRequest -Uri "http://localhost:5000/api/health" -UseBasicParsing
$response.Content | ConvertFrom-Json
```

### Using curl (in WSL or Git Bash)
```bash
# Test endpoint
curl http://localhost:5000/api/health

# Register with data
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","name":"User","password":"test123"}'
```

---

## Production Deployment

### Option 1: Gunicorn
```bash
# Install
pip install gunicorn

# Run
gunicorn -w 4 -b 0.0.0.0:8000 app_api:app
```

### Option 2: Docker
```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app_api:app"]
```

---

## Configuration

### Required .env Variables
```env
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id
RAZORPAY_KEY_ID=your_key
RAZORPAY_KEY_SECRET=your_secret
JWT_SECRET=your_secret_key
```

### Generate New JWT Secret
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Port 5000 in use | `lsof -i :5000` then `kill -9 <PID>` |
| Module not found | `pip install -r requirements.txt` |
| Database locked | Stop all processes and restart |
| JWT token error | Re-login or update JWT_SECRET |

---

## Files & Documentation

| Document | Purpose |
|----------|---------|
| **API_DEPLOYMENT_GUIDE.md** | Comprehensive deployment guide |
| **API_DEPLOYMENT_SUMMARY.md** | Detailed summary of all changes |
| **test_api.py** | Automated end-to-end API tests |
| **app_api.py** | Main Flask API server |
| **payment_manager.py** | Database and user management |

---

## Key Endpoints

```
POST   /api/auth/register      → Create account
POST   /api/auth/login         → Login
GET    /api/health             → Server status
GET    /api/signals/today      → Today's signals
GET    /api/signals/history    → Historical signals
GET    /api/performance        → Performance metrics
GET    /api/user/profile       → User info
POST   /api/subscribe          → (Disabled - will enable later)
```

---

## Monitoring

### Check API Status
```bash
curl http://localhost:5000/api/health
```

### View Logs
```bash
tail -f logs/payment_manager.log
```

### Check Database
```bash
sqlite3 logs/subscriptions.db "SELECT * FROM users;"
```

---

## What's Working

✅ REST API server running  
✅ User authentication with JWT tokens  
✅ Signal delivery system  
✅ Performance metrics tracking  
✅ SQLite database management  
✅ CORS enabled for cross-origin requests  
✅ Error handling and logging  
✅ All 5 endpoints tested and verified  

---

## Next: Integration & Deployment

1. **Connect Dashboard**: Update Streamlit to use API
2. **Enable Payments**: Uncomment endpoints and set Razorpay
3. **Deploy**: Move to production server
4. **Monitor**: Set up uptime alerts

---

**Status**: ✅ PRODUCTION READY  
**Last Updated**: 2026-04-01  
**Version**: 1.0.0
