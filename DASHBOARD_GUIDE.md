# 🚀 NSEIQ v5.0 - Quick Start Guide

**System Status: ✅ PRODUCTION READY**

---

## 🎯 Live Services

| Service | URL | Purpose |
|---------|-----|---------|
| **Interactive Dashboard** | http://localhost:8501 | User & Admin Interface |
| **Swagger API Docs** | http://localhost:8000/docs | API Testing & Documentation |
| **REST API** | http://localhost:8000 | Backend Services |
| **ReDoc** | http://localhost:8000/redoc | Alternative API Docs |

---

## 📊 Dashboard Tabs

### 1️⃣ Live Dashboard (Home)
- Real-time portfolio value & P&L
- Top market performers
- Current open positions  
- Portfolio allocation charts
- Daily P&L tracking

### 2️⃣ Stock Prediction
- 6-layer analysis engine
- Select stock → Mode → Sector
- Confidence & Signal Strength gauges
- Technical & Fundamental analysis
- Price targets (Conservative/Base/Bull)
- Risk/Reward analysis
- 30-day candl estick chart

### 3️⃣ Portfolio Builder
- Set total capital
- Choose risk profile (Conservative/Moderate/Aggressive)
- Select investment horizon
- Auto-generate diversified portfolio
- View allocations with % breakdown
- Risk management rules applied
- Rebalancing schedule

### 4️⃣ Analytics & Backtest
- Backtest any strategy on historical data
- Equity curve visualization
- Performance metrics (Return, Sharpe, Win Rate, etc.)
- Trade statistics & analysis
- Risk-adjusted returns

### 5️⃣ Trade Journal
- Complete trade history
- Filter by stock, status, date
- Entry/Exit prices & P&L
- Setup documentation
- Performance summary

### 6️⃣ Admin Panel
- System health monitoring
- CPU/Memory/API response time gauges
- Request analytics & logs
- Configuration settings
- System information & about

---

## 🔮 How to Generate a Prediction

1. **Open Dashboard**: http://localhost:8501
2. **Click**: 🔮 Stock Prediction
3. **Select**:
   - Stock: TCS, INFY, RELIANCE, etc.
   - Mode: INTRADAY, SWING, POSITIONAL, LONGTERM
   - Sector: Technology, Finance, Energy, etc.
4. **Click**: 🚀 Generate Prediction
5. **View**:
   - Confidence Score (0-100%)
   - Signal Strength gauge
   - Expected Accuracy
   - Technical analysis results
   - Fundamental analysis metrics
   - Price targets & probabilities
   - Risk/Reward ratio
   - Risk factors
   - 30-day chart

---

## 💼 Build a Portfolio

1. **Open Dashboard**: http://localhost:8501
2. **Click**: 💼 Portfolio Builder
3. **Set Parameters**:
   - Total Capital: ₹50,000 - ₹10,000,000+
   - Risk Profile: Conservative/Moderate/Aggressive
   - Horizon: Intraday/Swing/Positional/LongTerm
   - # of Stocks: 1-10
4. **Select Stocks**: Choose from 30+ NSE stocks
5. **Click**: 🔨 Build Portfolio
6. **Review**:
   - Portfolio summary (capital deployed, cash reserve)
   - Stock allocations with quantities
   - % of portfolio per holding
   - Risk management rules applied
   - Rebalancing schedule

---

## 📊 Backtest a Strategy

1. **Open Dashboard**: http://localhost:8501
2. **Click**: 📊 Analytics & Backtest
3. **Configure**:
   - Start Date: Choose from calendar
   - End Date: Until today
   - Test Stock: Select any NSE stock
   - Strategy: 6-Layer NSEIQ / SMA / RSI
4. **Click**: ▶️ Run Backtest
5. **View Results**:
   - Total Return % & amount
   - Annualized Return
   - Sharpe Ratio
   - Win Rate
   - Equity curve chart
   - Trade statistics (total, wins, losses, avg P&L)

---

## 📖 View Trade Journal

1. **Open Dashboard**: http://localhost:8501
2. **Click**: 📈 Trade Journal
3. **Filters** (optional):
   - Filter by stock
   - Filter by status (Open/Closed/Profit/Loss)
   - Date range
4. **View**:
   - All trades with entry/exit prices
   - Quantity & P&L per trade
   - Setup name & strategy
   - Performance summary (total trades, win rate, avg R:R, monthly return)

---

## ⚙️ Admin Monitoring

1. **Open Dashboard**: http://localhost:8501
2. **Click**: ⚙️ Admin Panel
3. **View**:
   - **System Monitor**: CPU, RAM, API response time
   - **API Logs**: Recent API requests & responses
   - **Settings**: Configure API host, ports, thresholds
   - **About**: System info & report generation

---

## 🔌 API Endpoints

### Predictions
```
POST /api/v1/nseiq/predict
{
  "ticker": "RELIANCE",
  "mode": "SWING",
  "sector": "Energy"
}
```

### Portfolio Generation
```
POST /api/v1/nseiq/portfolio
{
  "total_capital": 250000,
  "risk_profile": "MODERATE",
  "horizon": "SWING",
  "candidate_stocks": [...]
}
```

### Health Check
```
GET /health
GET /api/v1/health
```

### Other Endpoints
- `GET /api/v1/nseiq/portfolio/status` - Current holdings
- `POST /api/v1/nseiq/backtest` - Run backtest
- `GET /api/v1/nseiq/sheets/summary` - Trading summary
- `POST /api/v1/nseiq/log-trade` - Manual trade logging
- `POST /api/v1/nseiq/alert` - Fire alerts

---

## 🛠️ Configuration

**API Settings** (in .env):
```
NSEIQ_API_HOST=localhost
NSEIQ_API_PORT=8000
NSE_API_KEY=your_api_key
FINNHUB_API_KEY=your_key
GOOGLE_SHEETS_ID=your_sheets_id
```

**Dashboard Settings** (in Admin Panel):
- Min Confidence Threshold: 0-100%
- Max Open Positions: 1-20
- API Host/Port
- Other configurations

---

## 📞 Support

- **API Docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501
- **Logs**: Check terminal output
- **Issues**: Review Admin Panel → API Logs

---

## ✅ Features Completed

- ✅ 6-layer prediction engine
- ✅ Portfolio optimization
- ✅ Google Sheets logging
- ✅ FastAPI backend (15 endpoints)
- ✅ Streamlit dashboard
- ✅ Real-time monitoring
- ✅ Backtesting framework
- ✅ Trade journal tracking
- ✅ Risk management rules
- ✅ Production deployment

---

## 🎯 Next Steps

1. 📊 **Test Dashboard**: Navigate all tabs
2. 🔮 **Generate Predictions**: Try different stocks
3. 💼 **Build Portfolios**: Experiment with different risk profiles
4. 📈 **Run Backtests**: Test strategies on historical data
5. 🔌 **Use API**: Integrate with external systems
6. 📱 **Deploy**: Put on production server

---

**NSEIQ v5.0 - Institutional NSE Stock Intelligence System**  
Build: April 11, 2026 | Status: Production Ready ✅
