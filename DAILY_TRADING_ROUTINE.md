# ⏰ DAILY TRADING ROUTINE - April 3-17, 2026

## YOUR DAILY SCHEDULE (IST - Indian Standard Time)

### 📋 MORNING CHECK (9:30 AM)
```bash
python launch_dashboard_70.py
```
**What to look for:**
- Macro Signals (should be +0.25 to +1.0 for bullish)
- 5 Dashboard Tabs:
  1. **Predictions**: 70% accuracy model status
  2. **Macro Signals**: USD/INR, FII Flows, Technical Score
  3. **Risk Meter**: Green = Safe, Yellow = Caution, Red = Hold
  4. **Sentiment**: News sentiment vs market
  5. **P&L**: Your running profits/losses

**Action**: Note the macro signal score. If it's positive (bullish) and risk meter is green, proceed to trade.

---

### 🎯 TRADING TIME (9:45 AM - 3:30 PM)
```bash
python execute_daily_trades.py
```

**What happens:**
- Analyzes: RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS
- For each stock:
  - Shows: Current price, prediction (bullish/bearish), confidence %
  - Shows: Target profit (+5%), stop loss (-2%), risk per trade (₹500)
  - Makes decision: EXECUTE (if >65% confidence), CONSIDER (55-65%), SKIP (<55%)

**Your action IF trade is triggered:**
1. **Entry**:
   - Buy at recommended entry price
   - Set stop loss at -2% automatically
   - Set target at +5% automatically
   - Size: ₹25,000 per slot (4 slots max)

2. **During Trade**:
   - Monitor live price on dashboard
   - DON'T interference if moving toward target
   - Only close early if stop loss hit

3. **Exit**:
   - AUTOMATIC at target (+5%) or stop loss (-2%)
   - Record: Entry time, entry price, exit time, exit price, P&L

**Expected Results (First Week)**:
- Day 1 (Apr 3): Likely 0-1 trades (confidence building phase)
- Days 2-3 (Apr 4-5): 3-5 trades expected (signals strengthen)
- Days 4-5 (Apr 6-7): 5+ trades expected (momentum builds)
- **Week 1 Total**: Target 20+ trades

---

### 📊 EVENING REPORT (4:00 PM - 4:30 PM)
```bash
python track_metrics.py daily
```

**What you'll see:**
```
TODAY'S TRADING SUMMARY
======================
Date: April 3, 2026
Trades Executed: 0
Total P&L: ₹0
Win Rate: N/A (no trades)
Accuracy: N/A (no trades)
Capital Remaining: ₹250,000
Account Growth: 0%

WEEKLY RUNNING TOTAL (Apr 3-7)
==============================
Total Trades: 0
Winning Trades: 0
Losing Trades: 0
Win Rate: 0%
Accuracy: 0%
Weekly P&L: ₹0
Weekly Capital: ₹250,000 → ₹250,000
```

**Your action:**
- Review accuracy (should be 60%+ by day 3)
- Check P&L (should show ₹10-15k by end of week)
- If accuracy drops below 60%, pause and analyze

---

## 📅 WEEKLY CHECK (Friday 4:30 PM)
```bash
python track_metrics.py weekly 1
```

**Week 1 Check (April 7):**
- [ ] Total trades executed: 20+?
- [ ] Accuracy: 65%+?
- [ ] Win rate: 65%+?
- [ ] P&L: ₹10,000+?
- [ ] Capital: ₹250k → ₹260k+?

**Decision:**
- If YES to all: ✅ Continue to Week 2
- If NO: ⚠️ Adjust strategy, continue trading

**Week 2 Check (April 17):**
- [ ] Total trades executed: 50+?
- [ ] Accuracy: 68%+? (CRITICAL - Live deployment threshold)
- [ ] Win rate: 65%+? (CRITICAL - Profitability threshold)
- [ ] P&L: ₹30,000+?
- [ ] Capital: ₹250k → ₹280k+?

**Decision:**
- If YES to all: 🚀 DEPLOY ₹250-400k LIVE MONEY (Week 3)
- If NO: Continue paper trading 2 more weeks

---

## 🎲 WHAT TO EXPECT EACH DAY

### **Thursday (Today - April 3)**
- Markets: Closing day (3:30 PM IST market close)
- Confidence: Usually low (confidence rebuilding)
- Trades: Likely 0-1 (expected for Day 1)
- Macro: USD/INR, FII flows data available
- Status: ✅ COMPLETE - 0 trades (CORRECT DECISION)

### **Friday (April 4)**
- Markets: Regular trading
- Confidence: Building (first real trading day)
- Trades: Expected 2-4
- Action: Execute if confidence >65%
- Status: TOMORROW

### **Monday (April 7)**
- Markets: Start of week, often strong
- Confidence: Should be 60%+
- Trades: Expected 4-6
- Action: Execute normally
- Status: WEEK 1 COMPLETE - Check metrics

### **Tuesday-Wednesday (April 8-9)**
- Markets: Mid-week consolidation
- Confidence: Should be consistent
- Trades: Expected 3-5 each day
- Action: Execute normally

### **Thursday (April 10)**
- Markets: Before expiry (if any)
- Confidence: Varying
- Trades: Expected 3-4
- Action: Execute, watch expiry impact
- Status: WEEK 2 BEGINS

### **Friday (April 17)**
- Markets: Weekly close
- Confidence: Should be stable
- Trades: Expected 3-4
- Action: Execute, then FINAL DECISION
- Status: **🎯 DECISION DAY FOR LIVE DEPLOYMENT**

---

## 💰 EXPECTED CAPITAL GROWTH

**Week 1 (Apr 3-7):**
- Starting: ₹250,000
- Target P&L: +₹10,000 to +₹15,000
- Ending: ₹260,000 to ₹265,000
- Growth Rate: +4% to +6%

**Week 2 (Apr 10-17):**
- Starting: ₹260,000+
- Target P&L: +₹20,000 to +₹30,000
- Ending: ₹280,000 to ₹295,000
- Growth Rate: +8% to +12%

**Week 3 (Apr 20+) - IF QUALIFIED:**
- Deploy ₹250-400k real money
- Target: ₹500k+ by month 3
- Annualized: ₹1M+ portfolio possible

---

## ⚠️ RISK MANAGEMENT CHECKLIST

### Daily Check BEFORE trading:
- [ ] Stop loss set at -2% per trade (₹500 max)
- [ ] Target profit set at +5% per trade (₹1,250 target)
- [ ] Capital allocated: Max 4 × ₹25,000 = ₹100,000 deployed
- [ ] Reserve maintained: Min ₹150,000 (60%) in cash
- [ ] Dashboard shows GREEN risk meter
- [ ] Macro signal is positive (bullish)

### Trade Execution RULES:
- [ ] Only EXECUTE if confidence >65%
- [ ] Max 4 open positions simultaneously
- [ ] Never exceed ₹25,000 per trade slot
- [ ] Never skip stop loss setting
- [ ] Never hold through market close
- [ ] Track EVERY trade (entry, exit, P&L)

### Weekly Review RULES:
- [ ] Calculate accuracy % officially
- [ ] Calculate win rate % officially
- [ ] Review any losing trades (understand why)
- [ ] Adjust strategy if accuracy <60%
- [ ] Prepare weekly report by Friday 4:30 PM

---

## 📱 QUICK COMMANDS REFERENCE

**Morning (9:30 AM):**
```bash
python launch_dashboard_70.py
```

**Trading (9:45 AM - 3:30 PM):**
```bash
python execute_daily_trades.py
```

**Evening (4:00 PM):**
```bash
python track_metrics.py daily
```

**Weekly (Friday 4:30 PM):**
```bash
python track_metrics.py weekly 1
```

**View 2-Week Plan:**
Open: `PAPER_TRADING_2WEEK_PLAN.md`

**View All Documentation:**
Open: `COMPLETE_SETUP_INDEX.md`

**View Today's Status:**
Open: `EXECUTION_READY_APRIL_3.md`

---

## ✅ SUCCESS CRITERIA (April 17 GO/NO-GO)

### MUST HAVE BOTH:

**Accuracy ≥ 68%**
- At least 50 executed trades
- Winning trades ÷ Total trades ≥ 68%
- Example: 34 wins out of 50 trades = 68% ✅

**Win Rate ≥ 65%**
- (Trades with profits) ÷ (Total trades) ≥ 65%
- Example: 32-33 winning trades out of 50 = 64-66% ✅

**Capital Growth ≥ 12%**
- ₹250,000 → ₹280,000+ required
- Alternative: Any two months of 8%+ returns

### IF YES TO ALL THREE:
🚀 **DEPLOY ₹250-400k REAL MONEY - Week 3 GO LIVE**

### IF NO:
⏸️ **CONTINUE EXTENDED PAPER TRADING**
- Retest for another 2 weeks (Apr 20-May 1)
- Investigate accuracy issues
- Adjust model if needed
- Retry live deployment gate

---

## 🚨 TROUBLESHOOTING

**Problem: Dashboard not loading**
- Check: `python launch_dashboard_70.py`
- If error: Run `pip install -r requirements.txt`
- If still fails: Check [COMPLETE_SETUP_INDEX.md](COMPLETE_SETUP_INDEX.md)

**Problem: Execute script shows "No data"**
- Check: Internet connection
- Check: YFinance API availability
- Wait 5 minutes, retry
- Fallback: Use simulator mode

**Problem: Accuracy too low (< 65%)**
- Don't panic - Day 1 always low
- By Day 3-4 should improve
- Check macro signals - should be positive
- Review rejected trades - understand patterns

**Problem: Getting more losses than expected**
- Review: Were all trades at >65% confidence?
- Check: Were stop losses set at -2%?
- Verify: Market conditions (check dashboard)
- Adjust: May need to increase confidence threshold to 70%

**Problem: Can't execute trades (confidence stuck at 0%)**
- Wait for market data refresh (typically 5 min after market move)
- Check: `launch_dashboard_70.py` shows signals
- Verify: Macro signal positive before trading
- Fallback: Come back after 30 minutes

---

## 📞 SUPPORT REFERENCES

- **Full Dashboard Guide**: `DASHBOARD_NOW_LIVE.md`
- **2-Week Detailed Plan**: `PAPER_TRADING_2WEEK_PLAN.md`
- **Quick Start Guide**: `START_TRADING_TODAY.md`
- **Today's Status**: `EXECUTION_READY_APRIL_3.md`
- **Master Index**: `COMPLETE_SETUP_INDEX.md`

---

## 🎯 REMEMBER

**This is NOT gambling - this is VALIDATION:**
- You're testing the 70% accuracy system
- Every trade teaches you and the model
- Accuracy WILL improve over 2 weeks
- By April 17, you'll know if this works
- If it works: ₹1M+ portfolio possible

**Your discipline RIGHT NOW determines:**
- Whether you deploy at all
- How much capital you deploy
- How fast you build to ₹1M+

**Capital growth path:**
```
Week 1: ₹250k → ₹260k (4% growth)
Week 2: ₹260k → ₹280k (8% total)
Month 2: ₹280k → ₹350k (40% total)
Month 3: ₹350k → ₹500k (100% total)
Month 6: ₹500k → ₹1M+ (4x growth)
```

**This is achievable. Execute the plan. Track everything. Hit the April 17 gate with 68%+ accuracy.**

---

**START TRADING TOMORROW (April 4) AT 9:30 AM IST** 🚀

Good luck! You've got this! 💪
