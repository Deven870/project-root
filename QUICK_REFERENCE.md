# VoiceBot Quick Reference Guide

## 🚀 Quick Start (5 minutes)

### 1. Setup
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.template .env
```

### 2. Configure `.env`
```env
FINNHUB_API_KEY=your_key
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_id
```

### 3. Run
```bash
python system_launcher.py
```

### 4. Access
- API: http://127.0.0.1:5000
- Dashboard: http://127.0.0.1:8501

---

## 📋 Common Commands

| Task | Command |
|------|---------|
| Start All | `python system_launcher.py` |
| Start API Only | `python system_launcher.py --api-only` |
| Start Dashboard | `python system_launcher.py --dashboard-only` |
| Check Health | `python system_launcher.py --health` |
| Test Integration | `python test_system_integration.py` |
| Initialize DB | `python -c "from database import init_db; init_db()"` |
| View Logs | `tail -f logs/voicebot.log` |
| Check Config | `python -c "from system_config import get_config; c=get_config(); print(c.get_all())"` |

---

## 🔌 API Quick Reference

### Authentication
```bash
# Login - get token
curl -X POST http://127.0.0.1:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"user","password":"pass"}'

# Use token in headers
curl -H "Authorization: Bearer TOKEN" \
  http://127.0.0.1:5000/api/signals/today
```

### Signals
```bash
# Get today's signals
curl -H "Authorization: Bearer TOKEN" \
  http://127.0.0.1:5000/api/signals/today

# Get signal history (last 30 days)
curl -H "Authorization: Bearer TOKEN" \
  http://127.0.0.1:5000/api/signals/history?days=30
```

### Trades
```bash
# List open trades
curl -H "Authorization: Bearer TOKEN" \
  http://127.0.0.1:5000/api/trades?status=OPEN

# Create trade
curl -X POST http://127.0.0.1:5000/api/trades \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"RELIANCE.NS","quantity":10,"entry_price":2500}'
```

### Portfolio
```bash
# Get portfolio summary
curl -H "Authorization: Bearer TOKEN" \
  http://127.0.0.1:5000/api/portfolio

# Get holdings
curl -H "Authorization: Bearer TOKEN" \
  http://127.0.0.1:5000/api/holdings
```

---

## 🗂️ Project Structure

```
project-root/
├── system_launcher.py          # Main entry point
├── run_scheduler.py            # Scheduler runner
├── database.py                 # SQLAlchemy models
├── system_config.py            # Configuration management
├── system_logger.py            # Logging system
├── system_health.py            # Health monitoring
├── system_orchestration.py     # Service orchestration
├── manage_migrations.py        # Database migrations
│
├── app_api.py                  # Flask API
├── app.py                      # Streamlit dashboard
│
├── .env.template               # Configuration template
├── requirements.txt            # Dependencies
├── Dockerfile                  # Container image
├── docker-compose.yml          # Container orchestration
│
├── modules/                    # Trading logic
│   ├── scheduler.py
│   ├── data_fetch.py
│   ├── predictor.py
│   ├── google_sheets.py
│   ├── telegram_signal_bot.py
│   └── ...
│
├── logs/                       # Log files
│   ├── voicebot.log
│   └── errors.log
│
├── data/                       # Data files
│   └── voicebot.db            # SQLite database
│
└── docs/                       # Documentation
    ├── SYSTEM_SETUP_GUIDE.md
    ├── API_REFERENCE.md
    └── DATABASE_GUIDE.md
```

---

## ⚙️ Configuration Key Settings

| Setting | Purpose | Values |
|---------|---------|--------|
| `ENVIRONMENT` | Deployment mode | `development`, `staging`, `production` |
| `DB_TYPE` | Database | `sqlite`, `postgresql` |
| `ENABLE_API` | Start API server | `true`, `false` |
| `ENABLE_DASHBOARD` | Start dashboard | `true`, `false` |
| `ENABLE_SCHEDULER` | Start scheduler | `true`, `false` |
| `ENABLE_LIVE_TRADING` | Use real trades | `true`, `false` |
| `ENABLE_TELEGRAM_NOTIFICATIONS` | Send Telegram alerts | `true`, `false` |
| `LOG_LEVEL` | Logging verbosity | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

Full list in `.env.template`

---

## 🧪 Testing

### Integration Test
```bash
python test_system_integration.py
```

Validates:
- ✓ Configuration loading
- ✓ Database connectivity
- ✓ Logging system
- ✓ Health monitoring
- ✓ All models & services

### Manual Test - Create User
```python
from database import SessionLocal, User
from werkzeug.security import generate_password_hash

session = SessionLocal()
user = User(
    username="testuser",
    email="test@example.com",
    password_hash=generate_password_hash("password")
)
session.add(user)
session.commit()
print("✓ User created")
```

### Health Check
```bash
python system_launcher.py --health
```

---

## 🐛 Troubleshooting Quick Fixes

| Issue | Fix |
|-------|-----|
| Port 5000 in use | `lsof -i :5000` then `kill -9 PID` |
| Database locked (SQLite) | Delete `data/voicebot.db` and reinit |
| Import errors | `pip install -r requirements.txt --force-reinstall` |
| Telegram not sending | Verify token in `.env` and check internet |
| Dashboard not loading | Check port 8501 not in use, clear cache |
| Scheduler not running | `grep ENABLE_SCHEDULER .env` to verify |

---

## 📚 Important Files

| File | Purpose |
|------|---------|
| `database.py` | All ORM models and DB operations |
| `system_config.py` | Configuration management |
| `system_logger.py` | Logging setup |
| `system_health.py` | Health monitoring |
| `system_launcher.py` | Main entry point |
| `app_api.py` | REST API server |
| `app.py` | Web dashboard |
| `.env` | Configuration (never commit!) |
| `logs/voicebot.log` | Main application log |

---

## 🔐 Security Checklist

- [ ] Changed `JWT_SECRET` in `.env` (production only)
- [ ] Strong database password set
- [ ] API keys not in version control (in `.env`)
- [ ] `.env` added to `.gitignore`
- [ ] HTTPS enabled (reverse proxy)
- [ ] API CORS restricted to known domains
- [ ] Database backups enabled
- [ ] Logs reviewed regularly

---

## 📞 For More Help

- **Full Setup**: See `SYSTEM_SETUP_GUIDE.md`
- **API Docs**: See `API_REFERENCE.md`
- **Database**: See `DATABASE.md`
- **Logs**: `tail -f logs/voicebot.log`
- **Config**: Check `.env` and `system_config.py`

---

**Last Updated**: April 3, 2026
