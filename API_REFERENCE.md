# Flask REST API - Complete Reference

**Status:** ⚠️ Payment endpoints temporarily disabled (coming back later)

**Base URL:** `http://localhost:5000` (development) or your deployed domain

**Authentication:** JWT Bearer token (except `/register`, `/health`)

**Active Endpoints:**
- User authentication (register, login)
- Signal delivery (/api/signals/today, /api/signals/history)
- Performance metrics (/api/performance)
- User profile (/api/user/profile)

**Disabled (Coming Soon):**
- Payment subscription (/api/subscribe)
- Razorpay webhook (/api/webhook/razorpay)
- Subscription status (/api/user/subscription)
- Payment history (/api/user/payments)

---

## 🔑 Authentication

### Get JWT Token

**Endpoint:** `POST /api/auth/login`

```bash
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "secure_password"
  }'
```

**Response (Success):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user_id": "user123",
  "plan": "premium"
}
```

**Usage in subsequent requests:**
```bash
curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  http://localhost:5000/api/signals/today
```

---

## 👤 User Management

### Register New User

**Endpoint:** `POST /api/auth/register`

```bash
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "newuser@example.com",
    "password": "secure_password123",
    "name": "John Doe"
  }'
```

**Response:**
```json
{
  "user_id": "user_12345",
  "email": "newuser@example.com",
  "plan": "free",
  "access_token": "eyJ..."
}
```

---

### Get User Profile

**Endpoint:** `GET /api/user/profile`  
**Auth:** ✅ Required

```bash
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:5000/api/user/profile
```

**Response:**
```json
{
  "user_id": "user123",
  "email": "user@example.com",
  "name": "John Doe",
  "plan": "premium",
  "joined_date": "2024-04-01",
  "last_login": "2024-04-10 14:30:00"
}
```

---

## 📊 Signals & Trading Data

### Get Today's Signals

**Endpoint:** `GET /api/signals/today`  
**Auth:** ✅ Required

```bash
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:5000/api/signals/today
```

**Response (Free Tier - Yesterday's signals):**
```json
{
  "date": "2024-04-09",
  "tier": "free",
  "delayed": true,
  "signals": [
    {
      "ticker": "RELIANCE.NS",
      "action": "BUY",
      "confidence": 0.72,
      "expected_return": "2.5%",
      "entry_price": 2850.50,
      "timestamp": "2024-04-09 08:30:00"
    },
    {
      "ticker": "INFY.NS",
      "action": "SELL",
      "confidence": 0.68,
      "expected_return": "1.8%",
      "entry_price": 1520.00,
      "timestamp": "2024-04-09 08:30:00"
    }
  ],
  "summary": {
    "total_signals": 8,
    "buy_signals": 5,
    "sell_signals": 3,
    "avg_confidence": 0.70
  }
}
```

**Response (Premium Tier - Real-time signals):**
```json
{
  "date": "2024-04-10",
  "tier": "premium",
  "delayed": false,
  "signals": [
    {
      "ticker": "TCS.NS",
      "action": "BUY",
      "confidence": 0.75,
      "expected_return": "3.2%",
      "entry_price": 3850.00,
      "timestamp": "2024-04-10 08:30:00"
    }
  ],
  "summary": {
    "total_signals": 10,
    "buy_signals": 6,
    "sell_signals": 4,
    "avg_confidence": 0.72
  }
}
```

---

### Get Signal History

**Endpoint:** `GET /api/signals/history?days=30`  
**Auth:** ✅ Required  
**Parameters:**
- `days` (optional): Number of days to retrieve (default: 30)

```bash
curl -H "Authorization: Bearer TOKEN" \
  'http://localhost:5000/api/signals/history?days=7'
```

**Response:**
```json
{
  "total_signals": 56,
  "period": "7 days",
  "signals": [
    {
      "date": "2024-04-09",
      "ticker": "RELIANCE.NS",
      "action": "BUY",
      "confidence": 0.72,
      "entry_price": 2850.50,
      "exit_price": 2920.00,
      "return_pct": 2.43,
      "status": "completed"
    },
    {
      "date": "2024-04-08",
      "ticker": "INFY.NS",
      "action": "SELL",
      "confidence": 0.68,
      "entry_price": 1520.00,
      "exit_price": 1495.50,
      "return_pct": 1.61,
      "status": "completed"
    }
  ]
}
```

---

## 📈 Performance Metrics

### Get Performance Metrics

**Endpoint:** `GET /api/performance?period=30`  
**Auth:** ✅ Required  
**Parameters:**
- `period` (optional): Period in days (default: 30)

```bash
curl -H "Authorization: Bearer TOKEN" \
  'http://localhost:5000/api/performance?period=14'
```

**Response:**
```json
{
  "period_days": 14,
  "traded_signals": 28,
  "winning_trades": 19,
  "losing_trades": 9,
  "win_rate": 0.678,
  "total_return_pct": 24.5,
  "avg_return_per_trade": 1.75,
  "std_deviation": 0.45,
  "sharpe_ratio": 1.82,
  "max_drawdown": 3.2,
  "best_trade": {
    "ticker": "TCS.NS",
    "return_pct": 5.2,
    "date": "2024-04-05"
  },
  "worst_trade": {
    "ticker": "WIPRO.NS",
    "return_pct": -2.1,
    "date": "2024-04-03"
  }
}
```

---

## 💳 Payment & Subscription

### Create Subscription Order

**Endpoint:** `POST /api/subscribe`  
**Auth:** ✅ Required

```bash
curl -X POST http://localhost:5000/api/subscribe \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "plan": "premium_trial"
  }'
```

**Available Plans:**
- `free` - ₹0/month (limited features)
- `premium_trial` - ₹99 (1st month, then ₹299)
- `premium` - ₹299/month (recurring)

**Response (Razorpay Order Created):**
```json
{
  "order_id": "order_GXkLYb6bfgLgHI",
  "amount": 9900,
  "amount_due": 9900,
  "currency": "INR",
  "plan": "premium_trial",
  "razorpay_key_id": "YOUR_KEY_ID",
  "user_id": "user123",
  "created_at": "2024-04-10 10:30:00"
}
```

**Use this in Razorpay Payment Form:**
```javascript
// Frontend JavaScript
var options = {
  "key": response.razorpay_key_id,
  "amount": response.amount,
  "currency": "INR",
  "name": "Trading Signal Service",
  "description": response.plan,
  "order_id": response.order_id,
  "handler": function (response){
    // Send payment_id to backend for verification
    verifyPayment(response.razorpay_payment_id, response.razorpay_order_id);
  }
};
var rzp1 = new Razorpay(options);
```

---

### Verify & Process Payment (Webhook)

**Endpoint:** `POST /api/webhook/razorpay`  
**Auth:** ❌ Not required (signature-verified)

This endpoint is called automatically by Razorpay after payment.

**Razorpay sends:**
```json
{
  "razorpay_order_id": "order_GXkLYb6bfgLgHI",
  "razorpay_payment_id": "pay_GXkLYeB8QOsngI",
  "razorpay_signature": "signature_hash"
}
```

**Our API verifies signature and:**
1. Calls `PaymentManager.verify_payment()`
2. Upgrades user subscription in database
3. Sends confirmation email
4. Returns success response

**Response:**
```json
{
  "status": "success",
  "user_id": "user123",
  "new_plan": "premium_trial",
  "message": "Payment processed, subscription activated"
}
```

---

### Get Subscription Status

**Endpoint:** `GET /api/user/subscription`  
**Auth:** ✅ Required

```bash
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:5000/api/user/subscription
```

**Response:**
```json
{
  "user_id": "user123",
  "current_plan": "premium",
  "status": "active",
  "start_date": "2024-03-01",
  "next_renewal": "2024-05-01",
  "auto_renew": true,
  "days_remaining": 21
}
```

---

### Get Payment History

**Endpoint:** `GET /api/user/payments`  
**Auth:** ✅ Required

```bash
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:5000/api/user/payments
```

**Response:**
```json
{
  "total_payments": 3,
  "payments": [
    {
      "payment_id": "pay_GXkLYeB8QOsngI",
      "order_id": "order_GXkLYb6bfgLgHI",
      "amount": 9900,
      "currency": "INR",
      "plan": "premium_trial",
      "date": "2024-04-10",
      "status": "completed"
    },
    {
      "payment_id": "pay_FWjKXdA7PXsngH",
      "order_id": "order_FWjKXcA6EXsngH",
      "amount": 29900,
      "currency": "INR",
      "plan": "premium",
      "date": "2024-05-10",
      "status": "completed"
    }
  ]
}
```

---

## 🏥 Health & System

### Health Check

**Endpoint:** `GET /api/health`  
**Auth:** ❌ Not required

```bash
curl http://localhost:5000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-04-10 14:30:00",
  "version": "1.0.0",
  "database": "connected",
  "telegram_bot": "connected"
}
```

---

## 🔌 Integration Examples

### Example 1: User Registration & Login

```python
import requests

# 1. Register
register_response = requests.post(
  'http://localhost:5000/api/auth/register',
  json={
    'email': 'investor@example.com',
    'password': 'secure123',
    'name': 'Investor Name'
  }
)

token = register_response.json()['access_token']

# 2. Get user profile
profile = requests.get(
  'http://localhost:5000/api/user/profile',
  headers={'Authorization': f'Bearer {token}'}
)

print(profile.json())
```

---

### Example 2: Get Signals & Check Performance

```python
import requests

token = 'your_access_token'
headers = {'Authorization': f'Bearer {token}'}

# Get today's signals
signals = requests.get(
  'http://localhost:5000/api/signals/today',
  headers=headers
).json()

print(f"Today's signals: {signals['summary']['total_signals']}")

# Get performance metrics
performance = requests.get(
  'http://localhost:5000/api/performance?period=7',
  headers=headers
).json()

print(f"Win rate: {performance['win_rate']*100:.1f}%")
print(f"Total return: {performance['total_return_pct']:.1f}%")
```

---

### Example 3: Upgrade to Premium

```python
import requests

token = 'your_access_token'
headers = {'Authorization': f'Bearer {token}'}

# Create subscription order
order = requests.post(
  'http://localhost:5000/api/subscribe',
  headers=headers,
  json={'plan': 'premium_trial'}
).json()

print(f"Order created: {order['order_id']}")
print(f"Amount: ₹{order['amount']/100}")

# In your frontend, use Razorpay.checkout() with:
# - order['razorpay_key_id']
# - order['order_id']
# - order['amount']
```

---

### Example 4: Process Payment Webhook

```python
# This runs on your backend when Razorpay sends webhook
from payment_manager import PaymentManager

payload = request.json  # From Razorpay

pm = PaymentManager()
if pm.verify_payment(payload):
    pm.handle_payment_success(payload)
    return {'status': 'success'}, 200
else:
    return {'status': 'failed'}, 400
```

---

## ⚠️ Error Responses

### 400 - Bad Request
```json
{
  "error": "Invalid email format",
  "code": "INVALID_INPUT"
}
```

### 401 - Unauthorized
```json
{
  "error": "Token expired",
  "code": "TOKEN_EXPIRED"
}
```

### 404 - Not Found
```json
{
  "error": "User not found",
  "code": "USER_NOT_FOUND"
}
```

### 500 - Server Error
```json
{
  "error": "Database connection failed",
  "code": "INTERNAL_ERROR"
}
```

---

## 📚 Rate Limiting

All endpoints are rate-limited to **100 requests per minute**.

If limit exceeded:
```json
{
  "error": "Rate limit exceeded",
  "retry_after": 60
}
```

---

## 🔐 Security Best Practices

1. **Never expose tokens** in URLs or logs
2. **Always use HTTPS** in production
3. **Keep JWT_SECRET safe** - change regularly
4. **Validate all inputs** on frontend and backend
5. **Handle sensitive data** securely (never log passwords)
6. **Test webhooks** with Razorpay's test credentials first

---

## 💻 Running the API

```bash
# Development (auto-reload)
python app_api.py

# Production (with gunicorn)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app_api:app

# With logging
python app_api.py > logs/api.log 2>&1 &
```

---

## ✅ Quick Test Checklist

```bash
# 1. Health check
curl http://localhost:5000/api/health

# 2. Register user
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"test123","name":"Test"}'

# 3. Login (get token)
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"test123"}'

# 4. Get today's signals (replace TOKEN)
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:5000/api/signals/today

# 5. Get performance
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:5000/api/performance

# 6. Create payment order (replace TOKEN)
curl -X POST http://localhost:5000/api/subscribe \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"plan":"premium_trial"}'
```

---

## 📞 Support

For issues with:
- **API calls:** Check endpoint documentation above
- **Authentication:** Verify JWT token format
- **Payments:** See Razorpay dashboard logs
- **Signals:** Check signal generator logs in `logs/`

Generated: 2024-04-10
Last Updated: Week 6 - Full-Stack Deployment
