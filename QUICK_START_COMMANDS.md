# 🚀 QUICK START COMMANDS - DIGITRADER v4.0

## ONE-LINER START (Recommended)

```bash
cd c:\Users\DEVENDER\OneDrive\Desktop\voicbot\project-root && streamlit run app.py
```

**Then open:** http://localhost:8501

---

## ALL AVAILABLE COMMANDS

### 1. START DASHBOARD
```bash
streamlit run app.py
```
- Opens interactive web dashboard
- Real-time stock analysis
- Trading signals display
- Navigate to: http://localhost:8501

### 2. START API SERVER
```bash
python app_api.py
```
- RESTful API endpoints
- JSON responses
- Programmatic access
- Base URL: http://localhost:5000

### 3. EXECUTE TRADES
```bash
python execute_daily_trades.py
```
- Process all NSE stocks
- Generate trading signals
- Execute trades (paper or live)
- Log results

### 4. RUN 24/7 SCHEDULER
```bash
python run_scheduler.py
```
- Continuous operation
- Auto-refresh every 5 minutes
- Scheduled trade execution
- Background monitoring

### 5. QUICK VERIFICATION
```bash
python final_verification.py
```
- Test all systems
- Verify data pipeline
- Check signal generation
- Validate error handling

### 6. TEST API ENDPOINTS
```bash
python test_api.py
```
- Verify API functionality
- Test all endpoints
- Validate responses
- Performance check

---

## WORKFLOW EXAMPLES

### Option A: Interactive Monitoring
```bash
# Terminal 1: Start Dashboard
streamlit run app.py

# Terminal 2: Monitor Logs
Get-Content trading_system.log -Tail 20 -Wait
```

### Option B: Automated Trading (24/7)
```bash
# Terminal 1: Start API Server
python app_api.py

# Terminal 2: Run Scheduler
python run_scheduler.py

# Terminal 3: Monitor Logs
Get-Content trading_system.log -Tail 20 -Wait
```

### Option C: One-Time Analysis
```bash
python execute_daily_trades.py
# Output to: trading_results.csv
```

---

## STOPPING COMMANDS

### Stop Dashboard (Streamlit)
```bash
# Press Ctrl+C in terminal
# Or kill process:
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Stop API Server
```bash
# Press Ctrl+C in terminal
# Or kill process:
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Stop Scheduler
```bash
# Press Ctrl+C in terminal
```

---

## MONITORING COMMANDS

### View Live Logs
```bash
Get-Content trading_system.log -Tail 50 -Wait
```

### Check Process Status
```bash
Get-Process | Where-Object {$_.ProcessName -like "*python*"}
```

### Check Port Usage
```bash
netstat -ano | findstr :8501
netstat -ano | findstr :5000
```

### View Recent Trading Results
```bash
Get-Content trading_results.csv -Tail 20
```

---

## TROUBLESHOOTING COMMANDS

### Verify Installation
```bash
python -c "import streamlit; print('✅ Streamlit OK')"
python -c "import pandas; print('✅ Pandas OK')"
python -c "import yfinance; print('✅ YFinance OK')"
```

### Check Python Version
```bash
python --version
# Expected: 3.9+
```

### Reinstall Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### Clear Cache (if issues)
```bash
# Delete Streamlit cache
Remove-Item -Recurse -Force ~/.streamlit/cache*

# Restart dashboard
streamlit run app.py
```

---

## ENVIRONMENT SETUP

### First Time Setup
```bash
# Navigate to project
cd c:\Users\DEVENDER\OneDrive\Desktop\voicbot\project-root

# Create .env file (if needed)
# Add your API keys:
# OPENAI_API_KEY=your_key
# FINNHUB_API_KEY=your_key

# Install/update packages
pip install -r requirements.txt

# Run verification
python final_verification.py

# Start dashboard
streamlit run app.py
```

---

## CHEAT SHEET

| Task | Command |
|------|---------|
| Start Dashboard | `streamlit run app.py` |
| Start API | `python app_api.py` |
| Run Trades | `python execute_daily_trades.py` |
| 24/7 Scheduler | `python run_scheduler.py` |
| Verify System | `python final_verification.py` |
| View Logs | `Get-Content trading_system.log -Tail 50 -Wait` |
| Stop Dashboard | `Ctrl+C` or `taskkill /PID <PID> /F` |
| Stop API | `Ctrl+C` or `taskkill /PID <PID> /F` |

---

## DASHBOARD URL

**Once Started:** http://localhost:8501

### What You'll See
- ✅ NSE stock list (86 stocks)
- ✅ Buy/Sell/Hold signals
- ✅ Confidence scores
- ✅ Technical analysis
- ✅ Market metrics
- ✅ Real-time updates

---

## API ENDPOINTS

**Base URL:** http://localhost:5000 (after starting app_api.py)

### Endpoints
- `GET /api/stocks` - List all NSE stocks
- `POST /api/analyze` - Analyze single stock
- `GET /api/signals` - Get signals for all stocks
- `POST /api/trade` - Execute trade

### Example Request
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"stock": "RELIANCE.NS"}'
```

---

## EXPECTED OUTPUT

### Dashboard Should Show
```
DigiTrader v4.0
━━━━━━━━━━━━━━━━━━━━━━━━━

Stock Analysis
Select Stock: RELIANCE.NS

Signal: ⚪ HOLD
Score: 0.048
Confidence: 73.0%
Data Quality: 🟢 EXCELLENT

Technical Analysis
Market Metrics
Trading History
```

### Console Should Show
```
✅ NSE Stock List: 86 stocks loaded
✅ Utils Module: OK
✅ Precision Analyzer: Initialized
✅ RELIANCE.NS: Current Price ₹X
✅ Signal Generated: HOLD (73% confidence)
```

---

## IMPORTANT NOTES

- ✅ Dashboard auto-refreshes every 5 minutes
- ✅ API is RESTful and JSON-based
- ✅ Logs are stored in `trading_system.log`
- ✅ Trading results saved to `trading_results.csv`
- ✅ Database: SQLite at `data/trades.db`
- ✅ System runs 24/7 if scheduler is active
- ✅ Paper trading is enabled by default
- ✅ No real trades executed without configuration

---

## READY TO START?

### Quickest Way to See It Working:

```bash
cd c:\Users\DEVENDER\OneDrive\Desktop\voicbot\project-root
streamlit run app.py
```

**Then open:** http://localhost:8501

**That's it! Your DigiTrader dashboard is live!**

---

*Last Updated: April 3, 2025*  
*Version: 4.0 - Production Ready*
