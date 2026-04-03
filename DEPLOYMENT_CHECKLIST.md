# 🚀 VoiceBot System - Getting Started Checklist

## ✅ Pre-Deployment Checklist

Before running your system, work through this checklist step by step.

---

## Phase 1: Environment Setup

### Step 1: Verify Python Installation
```bash
python --version  # Should be 3.10 or higher
```
- [ ] Python 3.10+ installed

### Step 2: Create Virtual Environment
```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```
- [ ] Virtual environment activated (should see (venv) in prompt)

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```
- [ ] All packages installed successfully
- [ ] No errors or warnings

---

## Phase 2: Configuration Setup

### Step 5: Create .env File
```bash
cp .env.template .env
```
- [ ] .env file created

### Step 6: Update Configuration

Edit `.env` and fill in the following (at minimum):

```
# Edit these values
FINNHUB_API_KEY=your_api_key
TELEGRAM_BOT_TOKEN=your_bot_token  (optional)
TELEGRAM_CHAT_ID=your_chat_id      (optional)
STOCK_SYMBOL=RELIANCE.NS           (change if needed)
```

- [ ] FINNHUB_API_KEY set
- [ ] STOCK_SYMBOL configured
- [ ] Database settings reviewed
- [ ] Other credentials filled in (optional features)

### Step 7: Add to .gitignore
```bash
echo ".env" >> .gitignore
echo "credentials.json" >> .gitignore
```
- [ ] .env added to .gitignore (never commit secrets!)

---

## Phase 3: Database Initialization

### Step 8: Verify Database Configuration
```bash
python -c "from system_config import get_config; print(get_config().get('DB_TYPE')); print(get_config().get('DB_PATH'))"
```

If using SQLite (default):
- Database will be created automatically at `data/voicebot.db`

If using PostgreSQL:
1. Install PostgreSQL locally or use Docker
2. Update DB_* settings in .env
3. Create database:
   ```bash
   psql -U postgres -c "CREATE DATABASE voicebot;"
   ```

- [ ] Database configuration verified
- [ ] PostgreSQL running (if applicable)

### Step 9: Initialize Database Tables
```bash
python -c "from database import init_db; init_db()"
```

You should see: `✓ Database initialized: sqlite:///data/voicebot.db`

- [ ] Database tables created successfully

---

## Phase 4: System Validation

### Step 10: Run Integration Tests
```bash
python test_system_integration.py
```

You should see:
```
✓ Configuration Loading
✓ Logging System
✓ Database Connection
✓ Database Models
✓ Health Monitoring
✓ System Orchestration

TEST RESULTS: 6 passed, 0 failed
✓ ALL TESTS PASSED - System is ready!
```

- [ ] All integration tests passed
- [ ] No failures reported

### Step 11: Check System Health
```bash
python system_launcher.py --health
```

Output should show:
```
✓ HEALTHY system
5/6 or 6/6 services healthy
```

- [ ] System health check passed
- [ ] Most services showing as healthy

---

## Phase 5: Component Testing

### Step 12: Start Individual Components

#### Test API Server
```bash
python system_launcher.py --api-only
```

In another terminal:
```bash
curl http://127.0.0.1:5000/api/health
```

Should return:
```json
{
    "status": "ok",
    "timestamp": "..."
}
```

- [ ] API server starts without errors
- [ ] Health endpoint responds

#### Test Dashboard  
```bash
python system_launcher.py --dashboard-only
```

Should see:
```
You can now view your Streamlit app in your browser.
URL: http://127.0.0.1:8501
```

Open http://127.0.0.1:8501 in browser
- [ ] Dashboard loads without errors
- [ ] Page displays correctly

#### Test Scheduler
```bash
python system_launcher.py --scheduler-only
```

Should see:
```
✓ Scheduler started and running
```

- [ ] Scheduler starts without errors

---

## Phase 6: Full System Test

### Step 13: Start All Services
```bash
python system_launcher.py
```

Wait for all services to start. You should see:
```
SYSTEM STARTUP COMPLETE
Started 3 component(s):
  - API (PID: xxxx)
  - Dashboard (PID: xxxx)
  - Scheduler (PID: xxxx)

System is running. Press Ctrl+C to shutdown.
```

- [ ] API server started (Port 5000)
- [ ] Dashboard started (Port 8501)
- [ ] Scheduler started (background)
- [ ] No error messages

### Step 14: Access Services

**API:**
- URL: http://127.0.0.1:5000
- Test: `curl http://127.0.0.1:5000/api/health`

**Dashboard:**
- URL: http://127.0.0.1:8501
- Access via browser

**Check Logs:**
```bash
tail -f logs/voicebot.log
```

- [ ] API accessible and responding
- [ ] Dashboard dashboard loads
- [ ] Logs being generated
- [ ] No errors in logs

---

## Phase 7: Authentication Setup

### Step 15: Create First User

Option A: Via API
```bash
curl -X POST http://127.0.0.1:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "email": "admin@example.com",
    "password": "SecurePassword123!"
  }'
```

Option B: Via Dashboard
- Navigate to http://127.0.0.1:8501
- Look for signup/register option

- [ ] First user account created

### Step 16: Test Login

```bash
curl -X POST http://127.0.0.1:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "SecurePassword123!"
  }'
```

You should receive an access token:
```json
{
    "access_token": "eyJhbGc...",
    "token_type": "Bearer"
}
```

- [ ] Login successful
- [ ] Access token received

---

## Phase 8: Production Preparation

### Step 17: Update Security Settings

If deploying to production:

```env
# In .env
ENVIRONMENT=production
JWT_SECRET=long-random-secret-key-change-this
DEBUG_MODE=false
ENABLE_CORS=false  # Restrict to your domain
CORS_ORIGINS=https://yourdomain.com
```

- [ ] ENVIRONMENT set to production
- [ ] JWT_SECRET changed
- [ ] DEBUG_MODE disabled
- [ ] CORS configured

### Step 18: Setup Database Backups

```bash
# Create backup directory
mkdir -p backups

# For SQLite
cp data/voicebot.db backups/voicebot_$(date +%Y%m%d_%H%M%S).db

# For PostgreSQL
pg_dump -U postgres voicebot > backups/voicebot_$(date +%Y%m%d_%H%M%S).sql
```

- [ ] Backup strategy planned
- [ ] First backup created

### Step 19: Review Logs Configuration

In `.env`:
```env
LOG_LEVEL=INFO          # Change to ERROR for production
LOGS_DIRECTORY=logs
SQL_ECHO=false          # Keep false in production
```

- [ ] Log level appropriate for environment
- [ ] Log directory exists
- [ ] SQL_ECHO disabled

---

## Phase 9: Feature Enablement

### Step 20: Configure External Services (Optional)

#### Telegram Notifications
1. Create Telegram Bot: Message @BotFather
2. Get token: `TELEGRAM_BOT_TOKEN=5xx...`
3. Get Chat ID: Forward message from bot and extract ID
4. In `.env`:
   ```env
   TELEGRAM_BOT_TOKEN=your_token
   TELEGRAM_CHAT_ID=your_chat_id
   ENABLE_TELEGRAM_NOTIFICATIONS=true
   ```

- [ ] Telegram setup (optional)

#### Google Sheets Sync
1. Create Google Cloud Project
2. Enable Sheets API
3. Create Service Account
4. Download credentials → `service_account.json`
5. In `.env`:
   ```env
   GOOGLE_SHEETS_ID=your_sheet_id
   SERVICE_ACCOUNT_FILE=service_account.json
   ```

- [ ] Google Sheets setup (optional)

#### Live Trading (If Applicable)
```env
ENABLE_LIVE_TRADING=true
BROKER_TYPE=zerodha  # or others
ZERODHA_API_KEY=your_key
```

- [ ] Live trading configured (optional)

---

## Phase 10: Monitoring & Maintenance

### Step 21: Setup Monitoring

```bash
# Monitor API responses
watch -n 5 'curl -s http://127.0.0.1:5000/api/health | python -m json.tool'

# Monitor logs
tail -f logs/voicebot.log

# Monitor system health daily
python system_launcher.py --health

# Check running processes
ps aux | grep voicebot
```

- [ ] Monitoring plan created
- [ ] Health check command documented

### Step 22: Planning Maintenance

- [ ] Regular log cleanup scheduled
- [ ] Database backup schedule created
- [ ] Update schedule for dependencies planned
- [ ] Monitoring alerts configured
- [ ] Documentation updated

---

## 🎉 Completion Checklist Summary

### Environment Setup
- [x] Python 3.10+
- [x] Virtual environment
- [x] Dependencies installed

### Configuration
- [x] .env created and configured
- [x] Secrets not in version control
- [x] Database configured

### Validation
- [x] Integration tests passed
- [x] Health checks passed
- [x] All components tested individually
- [x] Full system startup works

### Authentication
- [x] User account created
- [x] Login successful
- [x] Token generation working

### Production Ready
- [x] Security settings configured
- [x] Logging configured
- [x] Backups planned
- [x] Monitoring setup

### Optional Features
- [x] Telegram configured (if needed)
- [x] Google Sheets configured (if needed)
- [x] Live trading configured (if needed)

---

## 📋 Daily Operations Checklist

### Every Startup
```bash
# 1. Activate environment
source venv/bin/activate

# 2. Check system health
python system_launcher.py --health

# 3. Start services
python system_launcher.py
```

### Daily Review
```bash
# Monitor logs
tail -n 100 logs/voicebot.log

# Check recent trades
# (via Dashboard at http://127.0.0.1:8501)

# Review system health
python system_launcher.py --health
```

### Weekly Maintenance
```bash
# Backup database
cp data/voicebot.db backups/voicebot_weekly_$(date +%Y%m%d).db

# Review logs for errors
grep ERROR logs/voicebot.log

# Check for updates
pip list --outdated
```

### Monthly Review
```bash
# Full system test
python test_system_integration.py

# Database cleanup
# (archive old records if needed)

# Update documentation
# (if any changes needed)
```

---

## 🆘 Troubleshooting Quick Links

If you encounter issues:

1. **Port Already in Use**: See "Troubleshooting" in QUICK_REFERENCE.md
2. **Database Errors**: See "Database Connection" in SYSTEM_SETUP_GUIDE.md
3. **Configuration Issues**: Check .env values and run `python system_launcher.py --config-only`
4. **Logging Problems**: See "Logging System" in system_logger.py
5. **External Service Errors**: Check TELEGRAM_BOT_TOKEN, API keys, etc.

---

## 📞 Next Steps After Setup

1. ✅ **Immediately**: Verify all checklist items above
2. 📊 **Next**: Configure trading parameters in dashboard
3. 📈 **Then**: Run backtests on historical data
4. 🔔 **Setup**: Configure alerts and notifications
5. 📡 **Deploy**: Move to production when ready

---

## 📅 Setup Timeline

- **Phase 1-2** (Environment & Config): 5-15 minutes
- **Phase 3-4** (Database & Validation): 5-10 minutes
- **Phase 5-6** (Component Testing): 10-15 minutes
- **Phase 7-8** (Auth & Production): 10-20 minutes
- **Phase 9-10** (Features & Monitoring): 15-30 minutes

**Total Time**: 45 minutes to 90 minutes for complete setup

---

## ✅ Sign-Off

- [ ] Entire checklist reviewed
- [ ] All tests passed
- [ ] System ready for use
- [ ] Backups taken
- [ ] Monitoring configured
- [ ] Team notified (if applicable)

---

**Date Completed**: _______________  
**Completed By**: _______________  
**Ready for Production**: ✅ / ❌  

---

**For Updated Help**: See SYSTEM_SETUP_GUIDE.md and QUICK_REFERENCE.md

**Last Updated**: April 3, 2026
