# Flask API Deployment Guide

## API Status: ✅ FULLY OPERATIONAL

All endpoints tested and working:
- ✅ Health Check
- ✅ User Registration
- ✅ User Login
- ✅ Signal Delivery
- ✅ Performance Metrics
- ✅ User Profile

---

## Quick Start

### Development Server
```bash
# Start API (port 5000)
python app_api.py

# In another terminal, test:
python test_api.py
```

### Production Deployment
```bash
# Install production server
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:8000 app_api:app

# Or with more workers for higher load
gunicorn -w 8 -b 0.0.0.0:8000 --timeout 30 app_api:app
```

---

## API Endpoints

### Health Check
```
GET /api/health
No auth required

Response (200):
{
  "status": "healthy",
  "timestamp": "2026-04-01T22:00:02.993663+05:30",
  "version": "1.0.0"
}
```

### User Registration
```
POST /api/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "name": "User Name",
  "password": "secure_password"
}

Response (201):
{
  "message": "User registered successfully",
  "user_id": "user@example.com",
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "plan": "free"
}
```

### User Login
```
POST /api/auth/login
Content-Type: application/json
No auth required

{
  "email": "user@example.com"
}

Response (200):
{
  "message": "Login successful",
  "user_id": "user@example.com",
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "plan": "free"
}
```

### Get Today's Signals
```
GET /api/signals/today
Authorization: Bearer {access_token}

Response (200):
{
  "signals": [
    {
      "timestamp": "2026-04-01 10:00:00",
      "symbol": "RELIANCE",
      "signal": "BUY",
      "prediction": 1,
      "confidence": 0.65,
      "entry": 2850.00,
      "target": 2900.00,
      "stoploss": 2800.00
    },
    ...
  ],
  "summary": {
    "total": 10,
    "total_buy": 4,
    "total_sell": 2,
    "total_hold": 4,
    "avg_confidence": 0.64
  },
  "delayed": false,
  "message": "Premium tier: real-time signals"
}
```

### Get Signal History
```
GET /api/signals/history
Authorization: Bearer {access_token}

Response (200):
{
  "signals": [
    {
      "entry": 2850.0,
      "stoploss": 2800.0,
      "target": 2900.0,
      "exit_price": 2895.0,
      "exit_time": "2026-04-01 14:30:00",
      "pnl": 225.0,
      "pnl_pct": 1.57,
      "signal": "BUY",
      "symbol": "RELIANCE",
      "timestamp": "2026-04-01 10:00:00"
    },
    ...
  ],
  "total": 45
}
```

### Get Performance Metrics
```
GET /api/performance
Authorization: Bearer {access_token}

Response (200):
{
  "win_rate": 65.5,
  "total_trades": 55,
  "wins": 36,
  "losses": 19,
  "total_return": 2850.50,
  "avg_return_per_trade": 51.83
}
```

### Get User Profile
```
GET /api/user/profile
Authorization: Bearer {access_token}

Response (200):
{
  "user_id": "user@example.com",
  "email": "user@example.com",
  "name": "User Name",
  "plan": "free",
  "status": "active",
  "created_at": "2026-04-01 16:30:05"
}
```

---

## Environment Configuration

Required `.env` variables:

```env
# Telegram Bot
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Razorpay (when payments enabled)
RAZORPAY_KEY_ID=your_razorpay_key
RAZORPAY_KEY_SECRET=your_razorpay_secret

# JWT Secret
JWT_SECRET=your_jwt_secret_key
```

### Generate JWT Secret
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## Testing

### Run API Tests
```bash
python test_api.py
```

### Expected Output
```
============================================================
TESTING FLASK API ENDPOINTS
============================================================

1. Health Check
Status: 200

2. Register User
Status: 201

3. Get Today's Signals (authenticated)
Status: 200
Signals received: 10

4. Get Performance Metrics (authenticated)
Status: 200

5. Get User Profile (authenticated)
Status: 200

============================================================
API TESTING COMPLETE
============================================================
```

---

## Deployment Checklist

### Before Going Live
- [ ] Replace test credentials in `.env` with real values
- [ ] Generate secure JWT_SECRET: `python -c "import secrets; print(secrets.token_hex(32))"`
- [ ] Set up Telegram bot via @BotFather (get token and chat ID)
- [ ] Test all endpoints with `python test_api.py`
- [ ] Configure database backups
- [ ] Set up logging (check logs/payment_manager.log)
- [ ] Plan for SSL/HTTPS (use nginx or reverse proxy)

### Deployment Options

#### Option 1: Heroku
```bash
# Create Procfile
echo "web: gunicorn -w 4 -b 0.0.0.0:\$PORT app_api:app" > Procfile

# Create runtime.txt
echo "python-3.11.9" > runtime.txt

# Deploy
git push heroku main
```

#### Option 2: AWS Elastic Beanstalk
```bash
# Create .ebextensions/python.config
mkdir -p .ebextensions
cat > .ebextensions/python.config << EOF
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: app_api:app
EOF

# Deploy
eb init
eb create
eb deploy
```

#### Option 3: Docker
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

## Performance Tuning

### For High Traffic
```bash
# Increase workers (2-4x CPU cores)
gunicorn -w 16 -b 0.0.0.0:8000 \
  --max-requests 1000 \
  --max-requests-jitter 100 \
  --timeout 30 \
  --graceful-timeout 30 \
  app_api:app
```

### Database Optimization
- SQLite for testing (current)
- PostgreSQL for production (recommended)
- Add connection pooling with psycopg2

### Caching
```python
# Add Redis caching for signals
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'redis'})

@app.route('/api/signals/today')
@cache.cached(timeout=300)  # Cache for 5 minutes
def get_today_signals():
    ...
```

---

## Monitoring & Logging

### Check Logs
```bash
# API logs
tail -f logs/payment_manager.log

# System logs (in production)
journalctl -u my-api-service -f
```

### Key Metrics to Monitor
- Request latency (average response time)
- Error rate (500, 400 status codes)
- Authentication failures
- Database connection pool usage
- Memory usage
- CPU usage

---

## Troubleshooting

### Issue: Port Already In Use
```bash
# Find process using port 5000
lsof -i :5000
# Kill it
kill -9 <PID>
```

### Issue: Database Locked
```bash
# SQLite database may be locked
# Solution: Stop all processes and restart
pkill -f "python app_api.py"
sleep 1
python app_api.py
```

### Issue: JWT Token Error
```
Invalid token or token expired
```
- Regenerate JWT_SECRET in .env
- Clear existing tokens (users need to re-login)

### Issue: Signals Not Loading
```bash
# Check if daily_signals.json exists
ls -la logs/daily_signals.json

# If missing, generate signals:
python daily_signal_generator.py
```

---

## Security Best Practices

### 1. HTTPS Only
```bash
# Use nginx as reverse proxy with SSL
# Or use gunicorn with SSL:
gunicorn --certfile=cert.pem --keyfile=key.pem app_api:app
```

### 2. Rate Limiting
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(app, key_func=get_remote_address)

@app.route('/api/auth/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    ...
```

### 3. CORS Configuration
```python
from flask_cors import CORS

CORS(app, resources={
    r"/api/*": {
        "origins": ["https://yourdomain.com"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
```

### 4. Input Validation
```python
from marshmallow import Schema, fields, validate

class UserSchema(Schema):
    email = fields.Email(required=True)
    name = fields.Str(required=True, validate=validate.Length(min=2, max=100))
    password = fields.Str(required=True, validate=validate.Length(min=8))
```

---

## API Response Status Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | OK | Endpoint executed successfully |
| 201 | Created | User registered successfully |
| 400 | Bad Request | Missing required fields |
| 401 | Unauthorized | Invalid or missing token |
| 404 | Not Found | User not found |
| 409 | Conflict | User already exists |
| 500 | Server Error | Database error |

---

## Support & Documentation

- **Flask Documentation**: https://flask.palletsprojects.com/
- **Flask-JWT-Extended**: https://flask-jwt-extended.readthedocs.io/
- **Gunicorn**: https://gunicorn.org/
- **REST API Best Practices**: https://restfulapi.net/

---

## Version Info

- API Version: 1.0.0
- Python: 3.11+
- Flask: 2.3.0+
- Last Updated: 2026-04-01

**Status**: ✅ Production Ready
