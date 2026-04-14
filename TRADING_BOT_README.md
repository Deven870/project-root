# 🤖 AUTOMATED TRADING BOT - COMPLETE DOCUMENTATION

**Status:** ✅ **PRODUCTION READY**  
**Version:** 5.0 (Complete)  
**Last Updated:** April 14, 2026

---

## 🚀 QUICK START (2 minutes)

### Prerequisites
- Python 3.8+
- API server running (port 8000)
- Live predictions service active

### Launch Trading Bot
```bash
python run_trading_bot.py
```

Expected output:
```
🤖 NSEIQ TRADING BOT v1.0

📋 BOT CONFIGURATION:
   Initial Capital: ₹300,000
   Min Confidence: 75%
   Signal Filter: STRONG_BUY
   Risk per Trade: 8%
   Daily Loss Limit: 7%

🚀 Start trading bot? (y/n): y

✅ Starting trading bot...
📌 Press Ctrl+C to stop

[Bot listening for signals...]
```

The bot will then:
1. ✅ Connect to live predictions API
2. ✅ Listen for STRONG_BUY signals (>75% confidence)
3. ✅ Place trades automatically
4. ✅ Monitor exits (target/SL)
5. ✅ Display status every 30 seconds
6. ✅ Export stats on exit (Ctrl+C)

---

## 📚 COMPLETE DOCUMENTATION SUITE

### 1. **TRADING_BOT_SETUP.md** 👈 START HERE
**What it covers:**
- 5-minute quick start guide
- System components overview
- How the bot works (step-by-step)
- Real-time example output
- Safety features explained
- Next steps and roadmap

**When to use:** First time setup, understanding the system

---

### 2. **TRADING_BOT_API.md** 🔌
**What it covers:**
- 10+ REST API endpoints
- 2 WebSocket endpoints
- Complete endpoint reference
- Python client examples
- cURL examples for testing
- Integration patterns

**When to use:** Programmatic integration, calling bot from other code

**Key endpoints:**
```
GET  /api/v1/live/predictions          → All live signals
GET  /api/v1/bot/status                → Bot current status
GET  /api/v1/bot/positions             → Open trades
GET  /api/v1/bot/account/stats         → P&L & metrics
POST /api/v1/bot/positions/{id}/close  → Manual close
WS   /ws/predictions                   → Real-time stream
```

---

### 3. **TRADING_BOT_TROUBLESHOOTING.md** 🐛
**What it covers:**
- 6 common issues & solutions
- Debugging techniques
- Log file inspection
- Data analysis examples
- Performance optimization
- Advanced debugging

**When to use:** Something isn't working, need to diagnose

**Common issues covered:**
- Connection refused
- No trades placed
- Trades not closing
- Daily loss limit hit
- Insufficient capital
- High losses/low accuracy

---

### 4. **TRADING_BOT_IMPLEMENTATION.md** 📊
**What it covers:**
- Complete system architecture
- File structure (8 files)
- How everything connects
- Risk management details
- Deployment path (paper → live)
- Success metrics

**When to use:** Understanding full system, planning upgrades

---

### 5. **README.md** (Project Root)
**What it covers:**
- Project overview
- Setup instructions
- Features list
- Architecture
- Dependencies

**When to use:** General project information

---

### 6. **START_API.md**
**What it covers:**
- API server startup
- Configuration
- Health checks
- Logs

**When to use:** Setting up or troubleshooting API server

---

## 🎯 WHICH DOCUMENT TO READ?

```
🟢 NEW TO TRADING BOT?
   └─→ Read: TRADING_BOT_SETUP.md
       Then: TRADING_BOT_API.md

🟡 SOMETHING ISN'T WORKING?
   └─→ Read: TRADING_BOT_TROUBLESHOOTING.md

🔵 INTEGRATING WITH OTHER CODE?
   └─→ Read: TRADING_BOT_API.md

🟣 WANT FULL TECHNICAL DEEP-DIVE?
   └─→ Read: TRADING_BOT_IMPLEMENTATION.md

🔴 DEPLOYING TO PRODUCTION?
   └─→ Read: TRADING_BOT_IMPLEMENTATION.md (Deployment Path)
       Then: TRADING_BOT_TROUBLESHOOTING.md (Debugging)
```

---

## 📊 SYSTEM ARCHITECTURE AT A GLANCE

```
LIVE PREDICTIONS (every 60 seconds)
     ↓↓↓
Live Prediction Service (15 stocks)
     ├─→ WebSocket Broadcasting
     ├─→ Google Sheets Auto-Logging
     └─→ Dashboard Live Feed
          ↓↓↓
    TRADING BOT MAIN LOOP
     ├─ Filter: Confidence > 75%? Signal = STRONG_BUY?
     ├─ Validate: Risk checks (8% risk, ₹21k daily limit)
     ├─ Execute: Place trade via Paper Trading Engine
     ├─ Monitor: Check Exit conditions every 60s
     │   ├─ If target hit: CLOSE (PROFIT)
     │   └─ If SL hit: CLOSE (LOSS)
     └─ Report: Export CSV/JSON, Display Stats
```

---

## 🔧 CONFIGURATION AT A GLANCE

### Your Settings (Pre-configured)
```
Capital: ₹300,000
Risk per Trade: 8% (₹24,000 max per trade)
Daily Loss Limit: 7% (₹21,000 stop loss for day)
Min Confidence: 75% (only 75%+ predictions)
Signal Filter: STRONG_BUY only
Max Positions: 4 trades open simultaneously
Trading Mode: Paper (simulated, safe)
```

### Position Sizing Example
```
Entry: ₹2,850 | SL: ₹2,798 | Target: ₹2,950
Risk = 2,850 - 2,798 = ₹52/share
Quantity = (300,000 × 8%) ÷ 52 = 461 shares
Capital Used = 461 × 2,850 = ₹1,313,850 (adjusted to available)
Expected Profit if Target Hit: ~₹4,700 (100×2,850 ÷ 52)
```

---

## 📈 REAL-TIME MONITORING

The bot displays status every 30 seconds:

```
📊 BOT STATUS - 14:30:45
Status: RUNNING ✅
Signals: 5 | Trades Placed: 2 | Closed: 1
Daily P&L: ₹8,500 📈
Open Positions: 2
Capital: ₹308,500 (₹47,800 deployed)
Win Rate: 100%
```

---

## 🚦 WHEN TO USE THIS BOT

### ✅ Good Use Cases
- Testing trading signals before live trading
- Learning automated trading strategies
- Validating prediction accuracy
- Backtesting entry/exit logic
- Building confidence before live capital

### ⚠️ Limitations
- Paper trading only (no real shares)
- Limited to 15 monitored stocks
- Requires live prediction service
- Manual capital adjustment needed
- No broker integration (yet)

---

## 🐍 PYTHON INTEGRATION EXAMPLE

```python
import requests
import asyncio

# Get live predictions
resp = requests.get("http://localhost:8000/api/v1/live/predictions")
predictions = resp.json()

# Check bot status
resp = requests.get("http://localhost:8000/api/v1/bot/status")
bot_status = resp.json()

print(f"Bot Status: {bot_status['status']}")
print(f"Daily P&L: ₹{bot_status['account']['total_pnl']}")

# Get open positions
resp = requests.get("http://localhost:8000/api/v1/bot/positions")
positions = resp.json()

for pos in positions["positions"]:
    pnl_pct = pos["unrealized_pnl_pct"]
    print(f"{pos['stock']}: {pnl_pct:+.2f}%")

# Close a position manually
resp = requests.post(
    "http://localhost:8000/api/v1/bot/positions/POS001/close",
    params={"reason": "MANUAL"}
)

if resp.json()["success"]:
    print("Position closed successfully")
```

---

## 🔍 DEBUGGING QUICK REFERENCE

**Bot not connecting:**
```bash
curl http://localhost:8000/health
# Should return: {"status":"ok"}
```

**Check live predictions:**
```bash
curl http://localhost:8000/api/v1/live/predictions | head -20
```

**View bot status:**
```bash
curl http://localhost:8000/api/v1/bot/status | jq .
```

**View current positions:**
```bash
curl http://localhost:8000/api/v1/bot/positions | jq '.positions[]'
```

**Check logs in real-time:**
```bash
tail -f trading_bot_*.log
```

---

## 📊 PERFORMANCE TRACKING

### Expected Results (1-2 weeks)

| Metric | Target | Check |
|--------|--------|-------|
| Trades Placed | 10+ | `grep "TRADE OPENED" trading_bot_*.log \| wc -l` |
| Win Rate | 60%+ | Dashboard or export JSON |
| P&L | +₹10,000+ | `account_stats_*.json` |
| Max 1-Trade Loss | <₹24,000 | Any trade P&L |
| Daily Limit Respected | 100% | Check daily P&L < ₹21,000 |

---

## 🔐 SAFETY FEATURES

### Built-In Protections
✅ **No Real Money** - Paper trading only  
✅ **Daily Loss Limit** - Stops after ₹21,000 loss  
✅ **Position Size Limit** - 8% risk per trade max  
✅ **Max Concurrent Trades** - Only 4 open at once  
✅ **Capital Protection** - Won't trade beyond available funds  
✅ **Risk/Reward Validation** - Better than 1:1 required  

### When To Pause
- ⚠️ Win rate drops below 40%
- ⚠️ Daily loss hits ₹21,000
- ⚠️ Unexpected large losses
- ⚠️ API connection issues

---

## 📚 FILE REFERENCE

### Configuration Files
```
run_trading_bot.py          ← Main launcher (edit config here)
backend/app/main.py         ← FastAPI server
backend/app/config.py       ← Server config
```

### Service Files
```
backend/app/services/trading_bot.py              ← Main bot logic
backend/app/services/paper_trading_engine.py    ← Account engine
backend/app/services/risk_manager.py             ← Risk validation
backend/app/services/live_prediction_service.py ← Predictions
```

### Output Files (Auto-generated)
```
trades_20260414_143045.csv          ← Trade history
account_stats_20260414_143045.json  ← Statistics
trading_bot_20260414_143045.log     ← Execution log
```

---

## 🎯 NEXT STEPS

### This Week
1. Run bot during market hours (9:15 AM - 3:30 PM IST)
2. Monitor real-time status every 30 seconds
3. Verify trades are placed correctly
4. Check target/SL exits are working

### Next Week
1. Analyze trade data (win rate, P&L)
2. Review prediction accuracy
3. Fine-tune parameters if needed
4. Plan for live trading (if results good)

### Live Trading (When Ready)
1. Get broker API credentials
2. Integrate with broker SDK
3. Start with small capital
4. Monitor closely
5. Scale gradually

---

## ❓ FAQ

**Q: How often are predictions updated?**  
A: Every 60 seconds during market hours (9:15 AM - 3:30 PM IST)

**Q: Can I change the configuration?**  
A: Yes! Edit `run_trading_bot.py` and update the config dictionary

**Q: What if I want different capital/risk?**  
A: Edit `run_trading_bot.py` lines 20-30 with new values

**Q: How do I know if it's working?**  
A: Run during market hours and check console output every 30 seconds

**Q: What happens if bot crashes?**  
A: Positions remain tracked in database; restart bot to continue

**Q: Can I use real money?**  
A: No, this is paper trading (simulated). Requires broker integration for real trades

**Q: Where are the trade logs?**  
A: Exported to `trades_*.csv` and `account_stats_*.json` when bot stops

---

## 🆘 SUPPORT

### For Different Issues

**API not working:**
- See: `START_API.md`

**Bot won't start:**
- See: `TRADING_BOT_TROUBLESHOOTING.md` → Issue 1

**No trades being placed:**
- See: `TRADING_BOT_TROUBLESHOOTING.md` → Issue 2

**Need API documentation:**
- See: `TRADING_BOT_API.md`

**Want full technical details:**
- See: `TRADING_BOT_IMPLEMENTATION.md`

---

## 🎉 YOU'RE READY!

Your trading bot is fully set up and ready to use. Simply execute:

```bash
python run_trading_bot.py
```

And watch automated trading happen in real-time! 📈🚀

---

**Questions? Check the documentation suite above.**  
**Issues? See TRADING_BOT_TROUBLESHOOTING.md**  
**API integration? See TRADING_BOT_API.md**

---

**Version:** 5.0 | **Status:** ✅ Production Ready | **Built:** April 14, 2026
