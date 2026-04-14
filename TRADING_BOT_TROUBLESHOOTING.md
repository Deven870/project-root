# 🐛 TRADING BOT TROUBLESHOOTING & DEBUG GUIDE

Complete guide for diagnosing and fixing trading bot issues.

---

## 🔴 COMMON ISSUES & SOLUTIONS

### Issue 1: "Connection refused" when starting bot
```
❌ ERROR: Could not connect to API at http://localhost:8000
```

**Cause:** API server not running

**Solutions:**

1. **Check if server is running:**
   ```bash
   curl http://localhost:8000/health
   ```
   Should return: `{"status":"ok"}`

2. **Start API server:**
   ```bash
   # Terminal 1
   cd c:\Users\DEVENDER\OneDrive\Desktop\voicbot\project-root
   python -m uvicorn backend.app.main:app --port 8000 --reload
   ```

3. **Check port 8000 is available:**
   ```bash
   # PowerShell
   netstat -ano | findstr :8000
   
   # If something is using it, kill it:
   taskkill /PID <PID> /F
   ```

4. **Verify API is accessible:**
   ```bash
   curl -v http://localhost:8000/api/v1/live/status
   ```

---

### Issue 2: Bot runs but no trades placed
```
🤖 BOT RUNNING... but no positions opening
Signals: 5 | Trades Placed: 0
```

**Possible causes & solutions:**

**A) No STRONG BUY signals generated**
```python
# Check what signals are being generated
import requests

response = requests.get("http://localhost:8000/api/v1/live/predictions")
preds = response.json()

for symbol, pred in preds["predictions"].items():
    if pred["confidence"] > 0.75:  # Your filter
        signal = pred["signal"]
        conf = pred["confidence"]
        print(f"{symbol}: {signal} @ {conf*100:.1f}%")

# If mostly BUY, not STRONG_BUY, adjust filter in run_trading_bot.py
# Or wait for stronger signals (live prediction engine needs time)
```

**B) Market closed (outside 9:15 AM - 3:30 PM IST)**
```python
# Check market status
response = requests.get("http://localhost:8000/api/v1/live/status")
status = response.json()

print(f"Market Status: {status['market_status']}")
print(f"Next Update: {status['next_update']}")

# Bot places trades only during market hours
# Retry during 9:15 AM - 3:30 PM IST
```

**C) Confidence filter too strict**
```python
# Current: confidence > 0.75 (75%)
# Try: confidence > 0.60 (60%) to catch more signals

# Edit: run_trading_bot.py, line ~85
MIN_CONFIDENCE = 0.60  # Changed from 0.75

# Then restart bot
```

**D) All desired stocks already have open positions**
```python
# Check open positions
response = requests.get("http://localhost:8000/api/v1/bot/positions")
positions = response.json()

print(f"Open Positions: {positions['total_open']}")
for pos in positions["positions"]:
    print(f"  - {pos['stock']}")

# Bot won't open duplicate positions
# Close one or increase max_positions limit
```

**E) Insufficient capital**
```python
# Check available capital
response = requests.get("http://localhost:8000/api/v1/bot/account/stats")
stats = response.json()

print(f"Capital Available: ₹{stats['cash_available']:,}")
print(f"Capital Deployed: ₹{stats['total_deployed']:,}")
print(f"Min Risk per Trade: ₹{stats['limits']['max_risk_per_trade']:,}")

# If deployed > available, no new trades possible
# Either close positions or increase initial capital
```

---

### Issue 3: Trades not closing at target/SL
```
Position: RELIANCE
Entry: ₹2,850 | Target: ₹2,950 | SL: ₹2,798
Current: ₹2,955 | Status: STILL OPEN ❌
```

**Possible causes:**

**A) Bot not running (checker stopped)**
```python
# Check bot status
response = requests.get("http://localhost:8000/api/v1/bot/status")
status = response.json()

if status["status"] != "RUNNING":
    print("⚠️ Bot is not running!")
    # Restart: python run_trading_bot.py
```

**B) Exit check not happening**
```python
# Edit trading_bot.py, check_exits() method
# Ensure it's being called in main_loop

async def main_loop(self):
    while self.is_running:
        predictions = await self.get_live_predictions()
        
        # THIS MUST BE HERE
        await self.check_exits(predictions)  # ← Check if missing
        
        filtered = self.filter_predictions(predictions)
        for stock, pred in filtered.items():
            await self.process_signal(stock, pred)
```

**C) Price data stale**
```python
# If market closed or predictions not updating:
response = requests.get("http://localhost:8000/api/v1/live/status")
status = response.json()

print(f"Last Update: {status['last_update']}")
print(f"Next Update: {status['next_update']}")

# Market hours only: 9:15 AM - 3:30 PM IST
# Outside hours: prices from last update
```

**D) Manual close needed**
```python
# Manually close position
import requests

response = requests.post(
    "http://localhost:8000/api/v1/bot/positions/POS001/close",
    params={
        "exit_price": 2955.00,
        "reason": "MANUAL_TARGET"
    }
)

print(response.json())
```

---

### Issue 4: Daily loss limit hit (trading stopped)
```
❌ DAILY LOSS LIMIT EXCEEDED
Daily Loss: ₹21,500 | Limit: ₹21,000
No new trades will be accepted until tomorrow
```

**Understanding the limit:**
- **Daily Loss Limit:** 7% of ₹300,000 = ₹21,000
- **Once hit:** No more trades today
- **Resets:** Every trading day (next market open)

**Solutions:**

**A) It's normal**
- Risk limit is working as intended
- Stop trading for the day
- Review what went wrong

**B) Adjust settings for tomorrow:**
```python
# Edit run_trading_bot.py
"daily_loss_limit": 0.10,  # Increase from 0.07 (7%) to 10%
"risk_per_trade": 0.06,    # Decrease from 0.08 (8%) to 6%
```

**C) Check trades that caused loss:**
```python
response = requests.get("http://localhost:8000/api/v1/bot/export/json")
data = response.json()

losing_trades = [t for t in data["trades"] if t["pnl"] < 0]
for trade in losing_trades:
    print(f"{trade['stock']}: {trade['pnl']} ({trade['exit_reason']})")
```

---

### Issue 5: High P&L losses instead of small profits
```
Result after 1 hour:
Daily P&L: ₹-15,000  (Loss!)
Win Rate: 25% (should be higher)
Avg Loss: ₹-6,000 vs Avg Win: ₹2,000
```

**Diagnosis:**

**A) Wrong signal filter**
```python
# Check that ONLY STRONG_BUY are trading
response = requests.get("http://localhost:8000/api/v1/live/predictions")
preds = response.json()

# See what signals are available
signals_distribution = {}
for symbol, pred in preds["predictions"].items():
    sig = pred["signal"]
    signals_distribution[sig] = signals_distribution.get(sig, 0) + 1

print("Signals:", signals_distribution)
# Should show mostly: STRONG_BUY, BUY, HOLD

# Trading BUY signals too? Should be STRONG_BUY only
```

**B) Predictions not accurate at entry**
```python
# Check prediction accuracy at entry time
# Export trades and compare predictions with actual price moves

response = requests.get("http://localhost:8000/api/v1/bot/export/json")
trades = response.json()

for trade in trades["trades"][:5]:  # Check first 5
    actual_move = (trade["exit_price"] - trade["entry_price"]) / trade["entry_price"]
    predicted_move = (trade["target_price"] - trade["entry_price"]) / trade["entry_price"]
    
    print(f"{trade['stock']}:")
    print(f"  Predicted: {predicted_move*100:+.2f}% | Actual: {actual_move*100:+.2f}%")
    print(f"  Accuracy: {'✅' if (predicted_move > 0) == (actual_move > 0) else '❌'}")

# If low accuracy, predictions need tuning (not bot issue)
```

**C) Wrong position sizing**
```python
# Check if position size is too large
# Formula: qty = risk_amount / (entry - stop_loss)
# A large SL = small qty = small profits but also small losses

status = requests.get("http://localhost:8000/api/v1/bot/status").json()
print(f"Capital: ₹{status['account']['current_capital']:,}")
print(f"Expected Risk per Trade: ₹{status['account']['current_capital'] * 0.08:,.0f}")

# If first trade has much smaller P&L than expected, SL might be too wide
```

**D) Exiting too early**
```python
# Example: Target = +3%, SL = -1%
# RR ratio = 3:1 (good)
# But if exiting at -1%, average loss is larger

# Check average wins vs losses
stats = requests.get("http://localhost:8000/api/v1/bot/account/stats").json()
print(f"Avg Win: ₹{stats['performance']['avg_win']:,}")
print(f"Avg Loss: ₹{stats['performance']['avg_loss']:,}")
print(f"Ratio: 1:{abs(stats['performance']['avg_loss'] / stats['performance']['avg_win']):.2f}")

# Should be 1:1 or better (1:2 preferred)
```

---

### Issue 6: "Insufficient capital" error despite capital available
```
❌ Capital needed (₹45,000) > available (₹42,000)
Trade not placed
```

**Reason:** Position sizing calculated more capital needed than available

**Solutions:**

**A) Check actual capital:**
```python
response = requests.get("http://localhost:8000/api/v1/bot/positions")
positions = response.json()

print(f"Deployed: ₹{positions['capital_deployed']:,}")
print(f"Available: ₹{positions['capital_available']:,}")
print(f"Total: ₹{positions['capital_deployed'] + positions['capital_available']:,}")

# If total < ₹300,000, losses have reduced capital
```

**B) Close existing position:**
```python
response = requests.get("http://localhost:8000/api/v1/bot/positions")
positions = response.json()

# Close smallest position
if positions["positions"]:
    smallest = min(positions["positions"], key=lambda x: x["entry_value"])
    close_resp = requests.post(
        f"http://localhost:8000/api/v1/bot/positions/{smallest['position_id']}/close"
    )
    print(f"Closed: {smallest['stock']}")
```

**C) Reduce risk per trade:**
```python
# Edit run_trading_bot.py
"risk_per_trade": 0.04,  # Reduced from 0.08 (8%) to 4%

# Restart bot
# Position size will be smaller, capital preserved
```

---

## 🔧 DEBUGGING & DIAGNOSTICS

### Enable Debug Mode
```python
# Edit run_trading_bot.py at top

import logging

# Set to DEBUG for verbose output
logging.basicConfig(
    level=logging.DEBUG,  # Was: INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now run
python run_trading_bot.py
```

### Check Log Files
```bash
# Find latest log
ls -t trading_bot_*.log | head -1

# Watch in real-time
tail -f trading_bot_2026-04-14_*.log

# Find errors only
grep "ERROR\|EXCEPTION" trading_bot_*.log

# Find all trades
grep "TRADE OPENED\|TRADE CLOSED" trading_bot_*.log
```

### Validate Predictions
```python
# Check if predictions are working
import requests

url = "http://localhost:8000/api/v1/live/predictions"
response = requests.get(url)

if response.status_code != 200:
    print(f"❌ Error: {response.status_code}")
    print(response.text)
else:
    data = response.json()
    print(f"✅ Predictions received")
    print(f"Stocks: {len(data['predictions'])}")
    print(f"Last update: {data['timestamp']}")
    
    # Show samples
    for symbol, pred in list(data["predictions"].items())[:3]:
        print(f"{symbol}: {pred['signal']} @ {pred['confidence']*100:.0f}%")
```

### Trace Single Trade
```python
# When trade is placed, trace through manually

import requests

# 1. Get prediction
pred_resp = requests.get("http://localhost:8000/api/v1/live/predictions/RELIANCE")
pred = pred_resp.json()
print(f"Prediction: {pred['signal']} @ {pred['confidence']*100:.1f}%")

# 2. Filter check
if pred["signal"] == "STRONG_BUY" and pred["confidence"] > 0.75:
    print("✅ Passes filter")
else:
    print("❌ Fails filter - won't trade")
    return

# 3. Risk validation
capital = 300000
risk_per_share = pred["current_price"] - pred["stop_loss"]
qty = int((capital * 0.08) / risk_per_share)
capital_used = qty * pred["current_price"]

print(f"Capital used: ₹{capital_used:,}")
print(f"Entry: ₹{pred['current_price']}")
print(f"SL: ₹{pred['stop_loss']} (Risk: ₹{risk_per_share:.2f}/share)")
print(f"Target: ₹{pred['target_price']}")

# Should show valid trade parameters
```

### Real-Time Status Check
```python
# Quick script to monitor bot continuously

import requests
import time
from datetime import datetime

while True:
    resp = requests.get("http://localhost:8000/api/v1/bot/status")
    status = resp.json()
    
    print(f"\n{datetime.now().strftime('%H:%M:%S')} - {status['status']}")
    print(f"Signals: {status['signals_received']} | Trades: {status['trades_placed']}")
    print(f"P&L: ₹{status['account']['total_pnl']:,} | Capital: ₹{status['account']['current_capital']:,}")
    
    time.sleep(30)
```

---

## 🔍 DATABASE QUERIES & DATA INSPECTION

### View All Trades
```python
response = requests.get("http://localhost:8000/api/v1/bot/trades?limit=100")
trades = response.json()

print(f"Total: {trades['total_trades']}")
for trade in trades["trades"]:
    status = "✅" if trade["status"] == "CLOSED" else "⌛"
    pnl = f"₹{trade.get('pnl', 0):,}" if trade["status"] == "CLOSED" else "PNL: TBD"
    print(f"{status} {trade['stock']:8} @ {trade['entry_price']:7.2f} → {pnl}")
```

### Find Losing Trades
```python
response = requests.get("http://localhost:8000/api/v1/bot/export/json")
data = response.json()

losing = [t for t in data["trades"] if t.get("pnl", 0) < 0]
losing.sort(key=lambda x: x.get("pnl", 0))

print(f"Losing Trades: {len(losing)}")
for trade in losing[:5]:
    print(f"{trade['stock']:8} | Entry: {trade['entry_price']:7.2f} | Exit: {trade['exit_price']:7.2f} | P&L: ₹{trade.get('pnl', 0):,} | Reason: {trade.get('exit_reason', 'N/A')}")
```

### Identify Best Signals
```python
response = requests.get("http://localhost:8000/api/v1/bot/export/json")
data = response.json()

# Group by stock
by_stock = {}
for trade in data["trades"]:
    stock = trade["stock"]
    if stock not in by_stock:
        by_stock[stock] = {"wins": 0, "losses": 0, "pnl": 0}
    
    by_stock[stock]["pnl"] += trade.get("pnl", 0)
    if trade.get("pnl", 0) > 0:
        by_stock[stock]["wins"] += 1
    else:
        by_stock[stock]["losses"] += 1

# Sort by P&L
for stock, stats in sorted(by_stock.items(), key=lambda x: x[1]["pnl"], reverse=True):
    winrate = stats["wins"] / (stats["wins"] + stats["losses"]) if stats["wins"] + stats["losses"] > 0 else 0
    print(f"{stock:8} | P&L: ₹{stats['pnl']:+7,} | W/L: {stats['wins']}/{stats['losses']} ({winrate*100:.0f}%)")
```

---

## 🚀 PERFORMANCE OPTIMIZATION

### Speed up execution
```python
# Edit trading_bot.py - reduce loop interval if market is very fast

async def main_loop(self):
    while self.is_running:
        # ... code ...
        await asyncio.sleep(30)  # Changed from 60 to 30 seconds for faster exits
```

### Reduce API calls
```python
# Cache predictions locally instead of fetching every time
from functools import lru_cache
import time

last_fetch = 0
cached_predictions = None

async def get_live_predictions(self):
    global last_fetch, cached_predictions
    
    now = time.time()
    if now - last_fetch < 10:  # Use cache for 10 seconds
        return cached_predictions
    
    cached_predictions = await self.get_from_api()
    last_fetch = now
    return cached_predictions
```

---

## 📞 ADVANCED DEBUGGING

### Python Debugger
```python
# Add breakpoint in trading_bot.py

import pdb

async def process_signal(self, stock, prediction):
    pdb.set_trace()  # ← Execution stops here
    # Now use: p stock, p prediction, c (continue), n (next), etc.
```

### Detailed Event Logging
```python
import logging

# Create detailed logger
logger = logging.getLogger("trading_bot.detailed")
logger.setLevel(logging.DEBUG)

handler = logging.FileHandler("bot_detailed.log")
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'))
logger.addHandler(handler)

# Use throughout code
logger.debug(f"Processing trade: {stock} at {entry_price}")
logger.debug(f"Filters: confidence={conf > 0.75}, signal={signal == 'STRONG_BUY'}")
```

### Test in Isolated Environment
```python
# Create minimal test case

async def test_single_trade():
    from backend.app.services.trading_bot import TradingBot
    
    bot = TradingBot(initial_capital=100000)
    
    # Mock a prediction
    prediction = {
        "symbol": "TEST",
        "current_price": 100,
        "target_price": 110,
        "stop_loss": 95,
        "signal": "STRONG_BUY",
        "confidence": 0.9
    }
    
    # Process
    result = await bot.process_signal("TEST", prediction)
    print(result)

# Run: asyncio.run(test_single_trade())
```

---

## 📋 DEBUGGING CHECKLIST

- [ ] API server running on port 8000
- [ ] Live predictions service started (check `/api/v1/live/status`)
- [ ] Bot configuration file exists (`run_trading_bot.py`)
- [ ] Initial capital set correctly (₹300,000)
- [ ] Signal filter is "STRONG_BUY" (or your preference)
- [ ] Confidence threshold set (>75%)
- [ ] Market hours check (9:15 AM - 3:30 PM IST)
- [ ] No capital issues (capital available > position size needed)
- [ ] Bot loop is running every 60 seconds
- [ ] Exits checking for target and SL hit
- [ ] Daily loss limit not exceeded
- [ ] Max positions limit not exceeded
- [ ] P&L calculations are accurate
- [ ] Trades exporting to CSV/JSON

---

## 📚 FURTHER RESOURCES

For more details, see:
- `backend/app/services/trading_bot.py` - Main bot logic
- `backend/app/services/paper_trading_engine.py` - Account tracking
- `backend/app/services/risk_manager.py` - Risk validation
- `trading_bot_*.log` - Execution logs
- `trades_*.csv` - Trade exports
- `account_stats_*.json` - Statistics
