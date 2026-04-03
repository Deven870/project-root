# VoiceBot Integrated System - Complete Setup Guide

## 🚀 System Overview

VoiceBot is a fully integrated trading system with three main components:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│           🚀 VOICEBOT TRADING SYSTEM v1.0                  │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   REST API   │  │  Dashboard   │  │  Scheduler   │     │
│  │   (Flask)    │  │ (Streamlit)  │  │(APScheduler) │     │
│  │  Port 5000   │  │ Port 8501    │  │  Background  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         ↓                ↓                   ↓              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            SQLAlchemy ORM Database                   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │   Users     │  │   Signals   │  │   Trades    │  │  │
│  │  ├─────────────┤  ├─────────────┤  ├─────────────┤  │  │
│  │  │ API Keys    │  │ Confidence  │  │   Status    │  │  │
│  │  │ Portfolios  │  │ Metadata    │  │   PnL       │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Telegram   │  │Google Sheets │  │  Finnhub     │     │
│  │  Alerts      │  │  Sync        │  │  Live Data   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

- Python 3.10+
- PostgreSQL 13+ (optional, SQLite included for development)
- Git
- pip or conda
- Docker & Docker Compose (optional, for containerized deployment)

## 🔧 Installation

### Step 1: Clone & Setup

```bash
cd /path/to/project-root
git clone <repo-url>
cd voicebot

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment

```bash
# Copy template
cp .env.template .env

# Edit .env with your values
# IMPORTANT: Update API keys, credentials, etc.
nano .env  # or use your editor
```

### Step 4: Setup Credentials

#### Google Sheets (Optional)
1. Create Google Cloud project
2. Enable Sheets API
3. Create Service Account
4. Download JSON credentials → `service_account.json`

#### Telegram Bot (Optional)
1. Message @BotFather on Telegram
2. Create new bot → Get `TELEGRAM_BOT_TOKEN`
3. Forward message from bot to get `TELEGRAM_CHAT_ID`

### Step 5: Initialize Database

```bash
# For SQLite (development)
python -c \"from database import init_db; init_db()\"

# For PostgreSQL, update .env first, then:
python manage_migrations.py init
python manage_migrations.py upgrade
```

## 🚀 Running the System

### Option 1: All Components (Recommended)

```bash
python system_launcher.py
```

This starts:
- Flask API (http://127.0.0.1:5000)
- Streamlit Dashboard (http://127.0.0.1:8501)
- APScheduler (background jobs)

### Option 2: Individual Components

```bash
# API Only
python system_launcher.py --api-only

# Dashboard Only
python system_launcher.py --dashboard-only

# Scheduler Only
python system_launcher.py --scheduler-only

# Check System Health
python system_launcher.py --health
```

### Option 3: Docker Deployment

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📍 Access Points

Once running, access:

- **API**: http://127.0.0.1:5000
- **Dashboard**: http://127.0.0.1:8501
- **API Health**: http://127.0.0.1:5000/api/health
- **API Docs**: http://127.0.0.1:5000/api/docs (when available)

## 🔑 API Authentication

All API endpoints require JWT token (except /api/auth/login, /api/auth/register):

```bash
# Get token
curl -X POST http://127.0.0.1:5000/api/auth/login \\
  -H \"Content-Type: application/json\" \\
  -d '{\"username\": \"user\", \"password\": \"pass\"}'

# Use token (add to headers)
curl -H \"Authorization: Bearer YOUR_TOKEN\" \\
  http://127.0.0.1:5000/api/signals/today
```

## 📊 System Architecture

### Database Models

```
User
├── id (PK)
├── username
├── email
├── password_hash
└── Relationships:
    ├── signals (1:N)
    ├── trades (1:N)
    ├── portfolios (1:N)
    └── api_keys (1:N)

Signal
├── id (PK)
├── user_id (FK)
├── symbol
├── signal_type (BUY/SELL/HOLD)
├── confidence
├── metadata (JSON)
└── trades (1:N)

Trade
├── id (PK)
├── user_id (FK)
├── symbol
├── quantity
├── entry_price
├── exit_price
├── pnl
├── status (OPEN/CLOSED/STOPPED)

Portfolio
├── id (PK)
├── user_id (FK)
├── cash_balance
├── total_value
└── holdings (1:N)
```

### Component Responsibilities

**API Server (Flask)**
- User authentication & authorization
- Signal retrieval and generation
- Trade management
- Portfolio tracking
- Real-time data endpoints
- Health status reporting

**Dashboard (Streamlit)**
- User-friendly interface
- Performance visualization
- Trade history
- Signal live feeds
- Portfolio analysis
- Configuration management

**Scheduler (APScheduler)**
- Morning market scans
- Continuous signal generation
- Open trade monitoring
- PnL calculations
- Stop loss/target price checks
- Daily EOD reports

**Database (SQLAlchemy + PostgreSQL/SQLite)**
- Persistent data storage
- User management
- Signal history
- Trade records
- System logs
- Configuration cache

### Logging System

Logs are written to:
- **Console**: Colored, real-time output
- **logs/voicebot.log**: Full application logs (rotating)
- **logs/errors.log**: Errors only (JSON format)
- **Database**: Critical events (if enabled)

Access logs:
```bash
tail -f logs/voicebot.log        # Follow main log
tail -f logs/errors.log         # Follow errors
```

### Health Monitoring

Check system health:
```bash
python system_launcher.py --health
```

Returns status of:
- Database connectivity
- API server
- Scheduler status
- Telegram availability
- Google Sheets access
- Finnhub API connectivity

## ⚙️ Configuration

### Environment Variables

Key configurations in `.env`:

```
# Server
API_HOST=0.0.0.0
API_PORT=5000
DASHBOARD_PORT=8501

# Database
DB_TYPE=sqlite  # or postgresql
DB_PATH=data/voicebot.db

# Trading
ENABLE_LIVE_TRADING=false
ENABLE_BACKTESTING=true
DEFAULT_RISK_PERCENTAGE=2.0

# Features
ENABLE_SCHEDULER=true
ENABLE_API=true
ENABLE_DASHBOARD=true
```

Full list in `.env.template`

## 🧪 Testing & Validation

### Test Configuration
```bash
python -c \"from system_config import Config; c=Config(); print(c.get_all())\"
```

### Test Database
```bash
python -c \"from database import get_db_session; session=get_db_session(); print('✓ DB OK')\"
```

### Test Logging
```bash
python -c \"from system_logger import get_logger; logger=get_logger('test'); logger.info('✓ Logging OK')\"
```

### Run Unit Tests
```bash
pytest tests/ -v
```

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Find & kill process on port
lsof -i :5000
kill -9 <PID>

# Or use different port
python system_launcher.py --api-port 5001
```

### Database Connection Failed
```bash
# Check configuration
cat .env | grep DB_

# For PostgreSQL
psql -h localhost -U postgres -d voicebot

# For SQLite
sqlite3 data/voicebot.db \".tables\"
```

### Scheduler Not Running
```bash
# Check if enabled
grep ENABLE_SCHEDULER .env

# Start manually
python run_scheduler.py
```

### Telegram Not Sending
```bash
# Verify credentials
grep TELEGRAM .env

# Test connection
python -c \"from modules.telegram_signal_bot import TelegramBot; print('✓ Telegram OK')\"
```

## 📈 Common Operations

### Create New User
```bash
python -c \"
from database import SessionLocal, User
from werkzeug.security import generate_password_hash
session = SessionLocal()
user = User(username='newuser', email='test@test.com', password_hash=generate_password_hash('pass123'))
session.add(user)
session.commit()
print('✓ User created')
\"
```

### Export Trades
```bash
python -c \"
from database import SessionLocal, Trade
session = SessionLocal()
trades = session.query(Trade).all()
for t in trades:
    print(f'{t.symbol}: {t.pnl}')
\"
```

### Reset Database (Development Only)
```bash
python manage_migrations.py reset
```

## 🔐 Security Best Practices

1. **Never commit `.env` to version control**
   ```bash
   echo \".env\" >> .gitignore
   ```

2. **Change JWT secret in production**
   ```env
   JWT_SECRET=long-random-secret-key-here
   ```

3. **Use strong database passwords**
   ```env
   DB_PASSWORD=complex-password-123!@#
   ```

4. **Enable HTTPS in production** (use reverse proxy like nginx)

5. **Restrict API access**
   ```env
   CORS_ORIGINS=https://yourdomain.com
   ```

## 📚 API Endpoints

### Authentication
- `POST /api/auth/register` - Create account
- `POST /api/auth/login` - Login
- `POST /api/auth/logout` - Logout

### Signals
- `GET /api/signals/today` - Today's signals
- `GET /api/signals/history` - Signal history
- `POST /api/signals/generate` - Generate signal

### Trades
- `GET /api/trades` - List trades
- `POST /api/trades` - Create trade
- `PUT /api/trades/{id}` - Update trade

### Portfolio
- `GET /api/portfolio` - Portfolio summary
- `GET /api/holdings` - Current holdings
- `POST /api/holdings` - Add holding

### System
- `GET /api/health` - System health
- `GET /api/status` - Component status

Full API documentation: See `API_REFERENCE.md`

## 🔄 Deployment Pipeline

### Local Development
```bash
python system_launcher.py
```

### Staging
```bash
export ENVIRONMENT=staging
docker-compose -f docker-compose.staging.yml up
```

### Production
```bash
export ENVIRONMENT=production
docker-compose -f docker-compose.prod.yml up -d
# Monitor with: docker-compose logs -f
```

## 📞 Support & Documentation

- **Architecture**: See `SYSTEM_ARCHITECTURE.md`
- **API Guide**: See `API_REFERENCE.md`
- **Database Guide**: See `DATABASE.md`
- **Troubleshooting**: See docs folder
- **Development**: See `CONTRIBUTING.md`

## ✅ Checklist

After setup, verify:

- [ ] Python 3.10+ installed
- [ ] Dependencies installed (`pip list | grep -i sqlalchemy`)
- [ ] `.env` file created and configured
- [ ] Database initialized (`data/voicebot.db` exists or PostgreSQL running)
- [ ] System launched without errors (`python system_launcher.py`)
- [ ] API responding (`curl http://127.0.0.1:5000/api/health`)
- [ ] Dashboard accessible (`open http://127.0.0.1:8501`)
- [ ] Health check passing (`python system_launcher.py --health`)
- [ ] Logs being written (`tail -f logs/voicebot.log`)

## 🎉 Next Steps

1. Create user account via dashboard or API
2. Generate API key for programmatic access
3. Configure trading parameters
4. Connect to Telegram for alerts
5. Run backtesting on historical data
6. Start live monitoring

---

**Last Updated**: April 3, 2026
**System Version**: 1.0
**Status**: Production Ready
