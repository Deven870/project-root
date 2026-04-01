# Monetization Stack - Quick Start (15 mins)

**⚠️ Status:** Payment system temporarily disabled - will be re-enabled later

**Week 6 Core Components Ready:**
- ✅ Signal generation & delivery
- ✅ Telegram bot
- ✅ Streamlit dashboard
- ✅ User authentication API

**Not Currently Available:**
- ⏳ Razorpay payments
- ⏳ Subscription management

---

## ⚡ 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create Configuration File
```bash
# Create .env file (don't commit to git!)
cat > .env << 'EOF'
# Telegram
TELEGRAM_BOT_TOKEN=your_token_from_botfather
TELEGRAM_CHAT_ID=-1001234567890

# Razorpay  
RAZORPAY_KEY_ID=your_key_id
RAZORPAY_KEY_SECRET=your_key_secret

# API
JWT_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")

# Database
DATABASE_URL=sqlite:///logs/subscriptions.db
EOF
```

### 3. Get Your Credentials

**Telegram Bot:**
1. Open Telegram, search for `@BotFather`
2. Send `/start`, then `/newbot`
3. Follow prompts, copy BOT_TOKEN to .env
4. Create Telegram group, add bot, copy CHAT_ID to .env

**Razorpay Account:**
1. Sign up at razorpay.com
2. Get API keys from Dashboard
3. Copy KEY_ID and KEY_SECRET to .env

### 4. Test Each Component

```bash
# Test signal generation
python daily_signal_generator.py
# Look for: logs/daily_signals.json ✅

# Test Telegram bot
python telegram_signal_bot.py test
# Should print: "✅ Telegram bot connected"

# Test API
python app_api.py &
curl http://localhost:5000/api/health
# Should return: {"status": "healthy"}

# Test Dashboard
streamlit run dashboard.py
# Opens: http://localhost:8501
```

---

## 🚀 Start Everything (Production Setup)

### Option A: All Services at Once

```bash
# Terminal 1: Signal generator (already runs on task scheduler)
# Or manually:
python run_daily_scheduler.py

# Terminal 2: Telegram bot daemon
python telegram_signal_bot.py daemon &

# Terminal 3: Flask API
python app_api.py &

# Terminal 4: Streamlit dashboard
streamlit run dashboard.py
```

### Option B: Docker (Recommended for deployment)

```bash
# Build image
docker build -t voicebot-monetization .

# Run all services
docker-compose up
```

---

## ✅ Verification Checklist

**Signal Generation** (8:30 AM IST)
- [ ] logs/daily_signals.json created
- [ ] Contains 8-10 signals with BUY/SELL and confidence scores
- [ ] Timestamp shows 08:30 IST

**Telegram Delivery** (9:00 AM IST)
- [ ] Message arrives in Telegram group
- [ ] Shows signals with emojis and formatting
- [ ] Entry price and confidence visible

**Dashboard** (localhost:8501)
- [ ] Displays today's signals
- [ ] Shows metrics (win rate, returns)
- [ ] Subscription tiers visible
- [ ] Sign up button works

**API** (localhost:5000)
- [ ] `/api/health` returns 200
- [ ] `/api/auth/register` creates user
- [ ] `/api/signals/today` returns today's signals (free tier = yesterday's)
- [ ] `/api/user/profile` shows user details

**Payments**
- [ ] `/api/subscribe` creates Razorpay order
- [ ] Razorpay button appears on dashboard
- [ ] Test payment completes (use test card)
- [ ] User upgraded to premium
- [ ] `logs/subscriptions.db` has new payment record

---

## 📊 File Outputs (Verify These Exist)

```
logs/
├── daily_signals.json          # Today's signals (refreshes 8:30 AM)
├── today_trades.csv            # Manual exit entry sheet
├── paper_trading.json          # All historical trades
├── validation_tracker.json     # Win rate & metrics summary
├── subscriptions.db            # User subscriptions & payments
└── telegram_history.json       # Sent Telegram messages log
```

---

## 🎯 Immediate Next Steps

### Today
```bash
# 1. Setup credentials
vi .env  # Add TELEGRAM_BOT_TOKEN, RAZORPAY_KEY_ID

# 2. Test locally
python app_api.py
streamlit run dashboard.py  # In another terminal

# 3. Manual test
# Open http://localhost:8501
# Sign up with test email
# Click "Subscribe" → Test payment on Razorpay
```

### Tomorrow (After signals run)
```bash
# 4. Check signal generation
cat logs/daily_signals.json | jq '.summary'

# 5. Verify telegram received signals
# Check telegram group - signals should appear at 9 AM

# 6. Check database recorded metrics
sqlite3 logs/subscriptions.db "SELECT * FROM subscriptions LIMIT 1"
```

### This Week
```bash
# 7. Deploy to production
# See MONETIZATION_STACK_GUIDE.md "Path 2: Cloud Deployment"

# 8. Configure payment webhooks
# Razorpay dashboard → Webhooks → Add your API endpoint

# 9. Beta test with 10 users
# Share dashboard link, process 5-10 test payments
```

---

## 💡 Common First-Time Issues & Fixes

| Problem | Quick Fix |
|---------|-----------|
| "No module named 'telegram'" | `pip install requests` |
| "TELEGRAM_BOT_TOKEN not found" | Check `.env` exists and has correct key |
| "sqlite3 database is locked" | Kill Python processes: `pkill -f python` |
| "Port 5000 already in use" | Change port: `python app_api.py --port 5001` |
| "Streamlit not found" | `pip install streamlit` |
| "No signals in dashboard" | Check time is after 08:30 AM IST OR manually run signal generator |
| "Can't connect to Razorpay" | Test in **test mode** first (free test card: 4111111111111111) |

---

## 📱 Testing Payment Flow

**Use Razorpay's test card (works in test mode):**
```
Card: 4111111111111111
Exp: 12/25
CVV: 123
OTP: 123456
```

**In production, real payments require real cards.**

---

## 🔄 Daily Operations (Automation Status)

**8:30 AM IST** - ✅ AUTOMATED
- Signal generation runs
- Outputs to logs/daily_signals.json

**9:00 AM IST** - ✅ AUTOMATED
- Telegram bot sends signals (if daemon running)

**3:30 PM IST** - 📝 MANUAL
- You enter exit prices in logs/today_trades.csv

**Periodic** - ⏰ SCHEDULED
- Validation tracker updates
- Subscription billing (Razorpay auto)
- Payment confirmations sent

---

## 💾 Required Directories (Auto-created)

```bash
# These are created automatically on first run:
mkdir -p logs/backup
mkdir -p logs/archive

# Backup daily if possible
cp logs/subscriptions.db logs/backup/subscriptions_$(date +%Y%m%d).db
```

---

## 🎓 Once Everything Works

### For Users
1. Share dashboard URL: `http://your-domain.com/dashboard`
2. They sign up (free tier)
3. They see yesterday's signals
4. They click "Upgrade to Premium" → Pay ₹99
5. They get today's signals in real-time + Telegram alerts

### For You
1. Monitor earnings: `sqlite3 logs/subscriptions.db "SELECT SUM(amount) FROM payments WHERE status='completed'"`
2. Track growth: `sqlite3 logs/subscriptions.db "SELECT COUNT(*) FROM subscriptions WHERE plan='premium'"`
3. Serve more signals: Scale to all NSE tickers (expand watchlist)
4. Add more services: SMS alerts, WhatsApp bot, etc.

---

## 📞 Emergency Debugging

**If signals don't generate:**
```bash
# Check signal generator logs
tail -50 logs/signal_generator.log

# Run manually to see errors
python daily_signal_generator.py --debug
```

**If Telegram doesn't send:**
```bash
# Test telegram connection
python -c "from telegram_signal_bot import TelegramBot; TelegramBot().test_connection()"

# Check token format
echo $TELEGRAM_BOT_TOKEN | head -c 20
```

**If API returns errors:**
```bash
# See server logs
tail -50 logs/flask_api.log

# Test endpoint directly
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"test123","name":"Test"}'
```

---

## 📈 Success Metrics

**Week 6 (Now):**
- [ ] All 4 components deploy locally without errors
- [ ] Signal generator + Telegram + Dashboard + API all working
- [ ] Test payment processes successfully

**Week 7:**
- [ ] Bot live in Telegram group (50 beta users)
- [ ] Dashboard accessible to public
- [ ] 5-10 paying users signed up
- [ ] First ₹500-₹1000 in revenue

**Week 8:**
- [ ] 100+ dashboard visitors
- [ ] 20-30 paying users (₹6-8k revenue)
- [ ] Signal accuracy validated (55%+ win rate)

**Month 2:**
- [ ] ₹35k MRR target
- [ ] 300+ paid users
- [ ] Expand to all NSE top 100 stocks

---

## 🎯 Your Action Now

### Right Now (5 minutes)
```bash
pip install -r requirements.txt
# Copy API keys to .env
```

### Next 10 minutes
```bash
python app_api.py &
streamlit run dashboard.py
# Open http://localhost:8501 and test sign up
```

### Next Hour
```bash
# Test entire flow:
# 1. Register user
# 2. View signals (free tier)
# 3. Click subscribe
# 4. Complete payment with test card
# 5. Verify premium signal in response
```

**Then:** Follow MONETIZATION_STACK_GUIDE.md for deployment

---

## ✨ What's Deployed

| Component | Status | Command |
|-----------|--------|---------|
| Signal Generator | ✅ Ready | Runs 8:30 AM auto |
| Telegram Bot | ✅ Ready | `python telegram_signal_bot.py daemon` |
| Dashboard | ✅ Ready | `streamlit run dashboard.py` |
| REST API | ✅ Ready | `python app_api.py` |
| Payments | ✅ Ready | Razorpay integration |
| Database | ✅ Ready | Auto-initializes |

**All systems go. Ready to launch! 🚀**

---

Next: Go to **MONETIZATION_STACK_GUIDE.md** for production deployment
