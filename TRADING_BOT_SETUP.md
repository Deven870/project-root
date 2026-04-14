# 🤖 TRADING BOT SETUP GUIDE - April 14, 2026

## ✅ Complete Automated Trading System Ready

Your trading bot is now **fully integrated** with the live predictions system!

---

## 🚀 QUICK START (5 Minutes)

### Prerequisites
Make sure these are running first:
```powershell
# Terminal 1 - API Server (if not already running)
python -m uvicorn backend.app.main:app --port 8000 --reload

# Terminal 2 - Dashboard (Optional, for monitoring)
streamlit run dashboard.py
```

### Start Trading Bot
```powershell
# Terminal 3 - Trading Bot
python run_trading_bot.py
```

**What happens:**
1. Bot initializes with ₹300,000 capital
2. Shows configuration summary
3. Asks for confirmation (type 'y')
4. Starts listening for STRONG BUY signals from live predictions
5. Auto-places trades when conditions met
6. Monitors exits (target hit or stop loss)
7. Prints status every 30 seconds

---

## 🎯 BOT CONFIGURATION

### Your Settings
- **Capital:** ₹300,000
- **Risk per Trade:** 8% (₹24,000)
- **Daily Loss Limit:** 7% (₹21,000)
- **Min Confidence:** 75%
- **Signals:** STRONG BUY only
- **Max Open Positions:** 4

---

## 📊 SYSTEM COMPONENTS

### 1. **Paper Trading Engine** 
- `backend/app/services/paper_trading_engine.py`
- Simulates trading account
- Tracks all trades and P&L
- No real money involved (safe for testing)

### 2. **Risk Manager**
- `backend/app/services/risk_manager.py`
- Calculates position size based on risk
- Validates trades before execution
- Enforces daily loss limits

### 3. **Trading Bot**
- `backend/app/services/trading_bot.py`
- Listens to live predictions
- Filters by confidence & signal
- Places and manages trades

### 4. **Bot Launcher**
- `run_trading_bot.py`
- User-friendly interface
- Real-time status monitoring
- Auto-exports reports

---

## 🔄 HOW IT WORKS

```
Live Predictions (Every 60s)
        ↓
Trading Bot Filters:
  ├─ Confidence > 75%?
  ├─ Signal = STRONG BUY?
  ├─ Stock not already open?
  └─ Capital available?
        ↓
Risk Manager Validates:
  ├─ Entry < Stop Loss?
  ├─ Target > Entry?
  ├─ RR Ratio > 1:1?
  ├─ Daily loss not exceeded?
  └─ Position limit not reached?
        ↓
✅ TRADE PLACED
  ├─ Entry: Prediction Current Price
  ├─ Target: Prediction Target Price
  ├─ SL: Prediction Stop Loss
  └─ Qty: Calculated from 8% risk
        ↓
MONITOR EXITS
  ├─ Price hits Target? → CLOSED (PROFIT)
  ├─ Price hits SL? → CLOSED (LOSS)
  └─ Every 60s: Check conditions
```

---

## 📈 REAL-TIME EXAMPLE

While bot running, you'll see:

```
🤖 NSEIQ TRADING BOT v1.0

📋 BOT CONFIGURATION:
   Capital: ₹300,000
   Min Confidence: 75%
   Signal Filter: STRONG_BUY
   Risk per Trade: 8%
   Daily Loss Limit: 7%
   Max Open Positions: 4

🚀 Start trading bot? (y/n): y

✅ Starting trading bot...
📌 Press Ctrl+C to stop

[Bot listening... waiting for signals...]

======================================================================
📊 BOT STATUS - 14:30:45
======================================================================
Status: RUNNING
Signals Received: 3
Trades Placed: 2
Trades Closed: 0
Daily P&L: ₹0
Open Positions: 2
Current Capital: ₹252,000
Capital Deployed: ₹48,000
Win Rate: 0.0%

Open Positions (2):
  • RELIANCE: ₹2,850.00 → ₹2,950.00 (SL: ₹2,798.50)
  • TCS: ₹3,245.00 → ₹3,350.00 (SL: ₹3,145.00)
======================================================================
```

---

## 🎯 UNDERSTANDING BOT DECISIONS

### When Bot PLACES a trade:
```
✅ TRADE OPENED: TCS
   Entry: ₹3,245.00 (Qty: 5)
   Target: ₹3,350.00 | SL: ₹3,145.00
   Capital Used: ₹16,225
   Remaining: ₹235,775
```

Why this position size?
- Risk amount = ₹300,000 × 8% = ₹24,000
- Risk per share = ₹3,245 - ₹3,145 = ₹100
- Quantity = ₹24,000 ÷ ₹100 = 240... but need ₹24,000 capital
- Actual: Adjusted to available capital & risk

### When Bot CLOSES a trade:
```
✅ PROFIT TRADE CLOSED: RELIANCE
   Exit: ₹2,920.00 | Reason: TARGET_HIT
   P&L: ₹7,000 (+2.44%)
   Account Balance: ₹259,000
```

---

## 📊 MONITORING THE BOT

### Option 1: Console Output (Real-time)
```powersh
python run_trading_bot.py
```
Prints status every 30 seconds

### Option 2: Dashboard (If running)
```
http://localhost:8501 → 🔴 Live Feed tab
Shows predictions being processed
```

### Option 3: Log Files
```
trading_bot_20260414_143045.log
```
Complete record of all bot actions

---

## 📁 OUTPUT FILES

When you stop the bot (Ctrl+C), it exports:

### 1. **trades_20260414_143045.csv**
All trades in spreadsheet format:
```
trade_id, timestamp_entry, stock, entry_price, target_price, stop_loss, 
exit_price, pnl, pnl_percent, exit_reason
```

### 2. **account_stats_20260414_143045.json**
Account statistics:
```json
{
  "account_name": "Trading Bot",
  "initial_capital": 300000,
  "current_capital": 315000,
  "total_pnl": 15000,
  "pnl_percent": 5.0,
  "winning_trades": 3,
  "losing_trades": 1,
  "win_rate": 75.0,
  ...
}
```

### 3. **trading_bot_20260414_143045.log**
Complete bot log:
```
2026-04-14 14:30:45 - INFO - ✅ TRADE OPENED: RELIANCE
2026-04-14 14:31:00 - INFO - ✅ PROFIT TRADE CLOSED: RELIANCE
[...]
```

---

## 🛡️ SAFETY FEATURES

### Risk Management Active:
✅ **8% Risk per Trade** - Won't risk more than ₹24,000 per trade  
✅ **7% Daily Loss Limit** - Stops trading if ₹21,000 lost in a day  
✅ **Position Limiting** - Max 4 open trades simultaneously  
✅ **Capital Management** - Won't trade if capital insufficient  
✅ **RR Ratio Validation** - Ensures reward > risk always  

### Paper Trading Safe:
✅ **NO REAL MONEY** - All simulated  
✅ **NO BROKER APIs** - No live order execution  
✅ **Safe Testing** - Perfect for learning & validation  

---

## ⚠️ IMPORTANT NOTES

### What Bot Does:
- ✅ Listens to live predictions every 60 seconds
- ✅ Filters by your criteria (confidence, signal)
- ✅ Calculates position size based on risk
- ✅ Places simulated trades
- ✅ Exits when target/SL hit
- ✅ Tracks P&L and statistics

### What Bot Doesn't Do:
- ❌ Place REAL trades (paper trading only)
- ❌ Connect to brokers (use your own later)
- ❌ Make predictions (uses live predictions API)
- ❌ Guarantee profits (backtest before using)

### Market Hours:
- Bot runs 24/7 (async)
- Places trades only when live predictions available (9:15 AM - 3:30 PM IST)
- Outside hours: Uses cached prices from previous update

---

## 🚀 NEXT STEPS

### 1. **This Week**: Monitor and Test
- Run bot during market hours
- Verify signals and entries are correct
- Check exits (target hits, SL hits)
- Analyze P&L and accuracy

### 2. **Next Week**: Optimize Settings
- Adjust min confidence if too strict
- Fine-tune risk percentage
- Test with different time horizons
- Measure win rate & accuracy

### 3. **Production**: Live Integration
- Get broker API credentials (Zerodha, Angel, etc.)
- Replace paper trading with live orders
- Start with small capital
- Monitor closely

---

## 🐛 TROUBLESHOOTING

### Problem: "Connection refused"
```
❌ Could not fetch predictions from API
```
**Solution:**
- Ensure API server is running (port 8000)
- Check: `curl http://localhost:8000/health`

### Problem: "No trades placed"
```
Bot running but no positions opening
```
**Possible reasons:**
- No STRONG BUY signals being generated
- Market closed (9:15 AM - 3:30 PM IST only)
- Check confidence > 75% filter
- All current stocks already have positions

### Problem: "Insufficient capital"
```
❌ Capital needed > available
```
**Solution:**
- Increase initial capital in config
- Close existing positions
- Reduce risk per trade %

---

## 📞 SUPPORT

### Check Bot Logs
```bash
tail -f trading_bot_*.log      # Watch in real-time
grep "ERROR" trading_bot_*.log # Find errors only
```

### Debug Status
```python
from backend.app.services.trading_bot import get_trading_bot
bot = get_trading_bot()
print(bot.get_bot_status())       # Current status
print(bot.get_positions())        # Open positions
```

### Manual Trade Management
```python
# Close a position manually
result = bot.close_position("RELIANCE", reason="MANUAL")
print(result)
```

---

## 📊 SUCCESS METRICS

After running for 1-2 weeks, check:

| Metric | Target | Your Result |
|--------|--------|------------|
| Trades Placed | 10+ | ___ |
| Win Rate | 60%+ | ___ |
| P&L | +₹10,000+ | ___ |
| Max Drawdown | < 7% | ___ |
| Avg Trade | Positive | ___ |

If targets met → Ready for live trading!
If not → Adjust settings and retest

---

## 🎉 YOU'RE READY!

Your trading bot is now:
- ✅ Fully configured with your settings
- ✅ Connected to live predictions
- ✅ Risk management enabled
- ✅ Paper trading safe
- ✅ Ready to trade!

### Quick Launch:
```bash
python run_trading_bot.py
```

### Monitor:
Check console every 30s for status updates

### Stop:
Press `Ctrl+C` to stop and export reports

---

**Good luck with automated trading! 🚀📈**

For detailed code documentation, see:
- `backend/app/services/trading_bot.py` - Bot logic
- `backend/app/services/paper_trading_engine.py` - Account engine
- `backend/app/services/risk_manager.py` - Risk management
