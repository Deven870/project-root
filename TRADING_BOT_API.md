# 🔌 TRADING BOT API REFERENCE

Complete API documentation for the trading bot integration with live predictions.

---

## 📌 BASE URL
```
http://localhost:8000
```

---

## 🔴 LIVE PREDICTIONS ENDPOINTS

### 1. Get Live Predictions (Real-Time)
```http
GET /api/v1/live/predictions
```

**Description:** Fetch latest predictions for all monitored stocks (updated every 60 seconds)

**Response:**
```json
{
  "timestamp": "2026-04-14T14:30:45.123456",
  "predictions": {
    "RELIANCE": {
      "symbol": "RELIANCE",
      "current_price": 2850.45,
      "target_price": 2950.00,
      "stop_loss": 2798.50,
      "signal": "STRONG_BUY",
      "confidence": 0.85,
      "layers": {
        "technical": 0.88,
        "fundamental": 0.82,
        "sentiment": 0.85,
        "macro": 0.80,
        "options": 0.87,
        "insider": 0.83
      },
      "timestamp": "2026-04-14T14:30:20.456789"
    },
    "TCS": {
      "symbol": "TCS",
      "current_price": 3245.10,
      "target_price": 3350.00,
      "stop_loss": 3145.00,
      "signal": "BUY",
      "confidence": 0.72,
      ...
    },
    ...
  },
  "market_status": "OPEN",
  "next_update": "2026-04-14T14:31:45.000000"
}
```

**Usage:**
```python
import requests

response = requests.get("http://localhost:8000/api/v1/live/predictions")
data = response.json()

# Filter for STRONG BUY with 75%+ confidence
strong_signals = {
    symbol: pred
    for symbol, pred in data["predictions"].items()
    if pred["signal"] == "STRONG_BUY" and pred["confidence"] > 0.75
}

print(f"Trading signals: {len(strong_signals)}")
```

---

### 2. Get Single Stock Prediction
```http
GET /api/v1/live/predictions/{symbol}
```

**Parameters:**
- `symbol` (string): Stock symbol (e.g., "RELIANCE", "TCS")

**Response:**
```json
{
  "symbol": "RELIANCE",
  "current_price": 2850.45,
  "target_price": 2950.00,
  "stop_loss": 2798.50,
  "signal": "STRONG_BUY",
  "confidence": 0.85,
  "layers": {...},
  "timestamp": "2026-04-14T14:30:20.456789"
}
```

**Usage:**
```python
response = requests.get("http://localhost:8000/api/v1/live/predictions/RELIANCE")
reliance_pred = response.json()

print(f"RELIANCE: {reliance_pred['signal']} @ {reliance_pred['confidence']*100:.1f}%")
```

---

### 3. Get Live Service Status
```http
GET /api/v1/live/status
```

**Response:**
```json
{
  "service_status": "RUNNING",
  "is_market_open": true,
  "last_update": "2026-04-14T14:30:45.123456",
  "next_update": "2026-04-14T14:31:45.000000",
  "stocks_monitored": 15,
  "update_frequency": "60s",
  "uptime_seconds": 3600
}
```

**Usage:**
```python
response = requests.get("http://localhost:8000/api/v1/live/status")
status = response.json()

if status["service_status"] == "RUNNING":
    print("✅ Live predictions are live!")
else:
    print("⚠️ Service not running")
```

---

### 4. Force Refresh Predictions
```http
POST /api/v1/live/refresh
```

**Description:** Force immediate update (normally 60-second interval)

**Response:**
```json
{
  "success": true,
  "message": "Predictions refreshed",
  "updated_at": "2026-04-14T14:30:55.789123",
  "stocks_updated": 15
}
```

**Usage:**
```python
response = requests.post("http://localhost:8000/api/v1/live/refresh")
if response.json()["success"]:
    print(f"Updated {response.json()['stocks_updated']} stocks")
```

---

## 🤖 TRADING BOT CONTROL ENDPOINTS

### 5. Get Bot Status
```http
GET /api/v1/bot/status
```

**Response:**
```json
{
  "bot_id": "nseiq_trader_001",
  "status": "RUNNING",
  "uptime": 3600,
  "signals_received": 5,
  "trades_placed": 3,
  "trades_closed": 1,
  "account": {
    "initial_capital": 300000,
    "current_capital": 285000,
    "total_pnl": -15000,
    "daily_pnl": -12000,
    "open_positions": 2,
    "max_positions": 4
  },
  "stats": {
    "win_rate": 33.33,
    "avg_win": 8000,
    "avg_loss": -12000,
    "total_trades": 3
  }
}
```

---

### 6. Get Open Positions
```http
GET /api/v1/bot/positions
```

**Response:**
```json
{
  "total_open": 2,
  "capital_deployed": 48000,
  "capital_available": 237000,
  "positions": [
    {
      "position_id": "POS001",
      "stock": "RELIANCE",
      "entry_price": 2850.00,
      "current_price": 2875.50,
      "target_price": 2950.00,
      "stop_loss": 2798.50,
      "quantity": 10,
      "entry_value": 28500,
      "current_value": 28755,
      "unrealized_pnl": 255,
      "unrealized_pnl_pct": 0.89,
      "entry_time": "2026-04-14T14:15:30.123456",
      "elapsed_time": "15m 25s"
    },
    {
      "position_id": "POS002",
      "stock": "TCS",
      "entry_price": 3245.00,
      "current_price": 3210.00,
      "target_price": 3350.00,
      "stop_loss": 3145.00,
      "quantity": 5,
      "entry_value": 16225,
      "current_value": 16050,
      "unrealized_pnl": -175,
      "unrealized_pnl_pct": -1.08,
      "entry_time": "2026-04-14T14:20:15.654321",
      "elapsed_time": "10m 30s"
    }
  ]
}
```

**Usage:**
```python
response = requests.get("http://localhost:8000/api/v1/bot/positions")
positions = response.json()

print(f"Open Positions: {positions['total_open']}")
print(f"Capital Deployed: ₹{positions['capital_deployed']:,}")
print(f"Capital Available: ₹{positions['capital_available']:,}")

for pos in positions["positions"]:
    pnl_color = "📈" if pos["unrealized_pnl"] > 0 else "📉"
    print(f"{pnl_color} {pos['stock']}: {pos['unrealized_pnl_pct']:+.2f}% | Target: {pos['target_price']}")
```

---

### 7. Close Position Manually
```http
POST /api/v1/bot/positions/{position_id}/close
```

**Parameters:**
- `position_id` (string, URL): Position ID to close
- `exit_price` (float, query): Exit price (uses current price if not provided)
- `reason` (string, query): Reason for close (e.g., "MANUAL", "INVALID_SIGNAL")

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/bot/positions/POS001/close?exit_price=2875.50&reason=MANUAL"
```

**Response:**
```json
{
  "success": true,
  "position_id": "POS001",
  "stock": "RELIANCE",
  "exit_price": 2875.50,
  "entry_price": 2850.00,
  "pnl": 255,
  "pnl_percent": 0.89,
  "reason": "MANUAL",
  "closed_at": "2026-04-14T14:35:00.123456"
}
```

**Usage:**
```python
response = requests.post(
    "http://localhost:8000/api/v1/bot/positions/POS001/close",
    params={
        "exit_price": 2875.50,
        "reason": "MANUAL_STOP"
    }
)

result = response.json()
if result["success"]:
    print(f"✅ Closed: {result['stock']} | P&L: ₹{result['pnl']}")
```

---

### 8. Get Trade History
```http
GET /api/v1/bot/trades
```

**Query Parameters:**
- `limit` (int): Max trades to return (default: 50)
- `offset` (int): Offset for pagination (default: 0)
- `status` (string): Filter by status ("OPEN", "CLOSED", "ALL") (default: "ALL")

**Response:**
```json
{
  "total_trades": 25,
  "trades": [
    {
      "trade_id": "TRD001",
      "stock": "RELIANCE",
      "entry_time": "2026-04-14T14:15:30.123456",
      "entry_price": 2850.00,
      "current_price": 2875.50,
      "target_price": 2950.00,
      "stop_loss": 2798.50,
      "quantity": 10,
      "signal": "STRONG_BUY",
      "confidence": 0.85,
      "status": "OPEN",
      "entry_pnl": 255,
      "entry_pnl_pct": 0.89,
      "reason_for_entry": "STRONG_BUY signal with 85% confidence"
    },
    {
      "trade_id": "TRD002",
      "stock": "INFY",
      "entry_time": "2026-04-14T14:10:00.456789",
      "exit_time": "2026-04-14T14:28:00.789123",
      "entry_price": 1890.00,
      "exit_price": 1920.50,
      "target_price": 1950.00,
      "stop_loss": 1850.00,
      "quantity": 15,
      "signal": "STRONG_BUY",
      "confidence": 0.78,
      "status": "CLOSED",
      "pnl": 456,
      "pnl_percent": 1.56,
      "exit_reason": "TARGET_HIT",
      "hold_time": "18m"
    }
  ]
}
```

---

### 9. Get Account Statistics
```http
GET /api/v1/bot/account/stats
```

**Response:**
```json
{
  "account_name": "NSEIQ Trading Bot",
  "period": "2026-04-14",
  "initial_capital": 300000,
  "current_capital": 285000,
  "account_balance": 285000,
  "total_deployed": 48000,
  "cash_available": 237000,
  "total_pnl": -15000,
  "total_pnl_percent": -5.0,
  "daily_pnl": -12000,
  "daily_pnl_percent": -4.0,
  "trades": {
    "total": 3,
    "open": 2,
    "closed": 1,
    "winning": 1,
    "losing": 0
  },
  "performance": {
    "win_rate": 100.0,
    "avg_win": 456,
    "avg_loss": 0,
    "largest_win": 456,
    "largest_loss": 0,
    "profit_factor": "∞"
  },
  "limits": {
    "daily_loss_limit": 21000,
    "daily_loss_remaining": 9000,
    "max_positions": 4,
    "positions_remaining": 2,
    "max_risk_per_trade": 24000
  },
  "last_updated": "2026-04-14T14:35:45.123456"
}
```

---

### 10. Export Account Data
```http
GET /api/v1/bot/export/{format}
```

**Parameters:**
- `format` (string): Export format - "csv" or "json"

**Request:**
```bash
# Export as CSV
curl "http://localhost:8000/api/v1/bot/export/csv" > trades.csv

# Export as JSON
curl "http://localhost:8000/api/v1/bot/export/json" > account_stats.json
```

**CSV Format:**
```
trade_id,stock,entry_time,entry_price,exit_time,exit_price,quantity,pnl,pnl_percent,exit_reason
TRD001,RELIANCE,2026-04-14T14:15:30.123456,2850.00,2026-04-14T14:28:00.789123,2875.50,10,255,0.89,TARGET_HIT
TRD002,INFY,2026-04-14T14:10:00.456789,,1890.00,,1920.00,15,,,OPEN
```

**JSON Format:**
```json
{
  "export_date": "2026-04-14T14:35:45.123456",
  "account_stats": {...},
  "trades": [...],
  "positions": [...]
}
```

---

## 🔌 WEBSOCKET ENDPOINTS

### 11. Real-Time Predictions Stream
```
ws://localhost:8000/ws/predictions
```

**Description:** Subscribe to real-time prediction updates

**Connection:**
```python
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    print(f"Update: {data['symbol']} - {data['signal']} @ {data['confidence']*100:.1f}%")

def on_error(ws, error):
    print(f"Error: {error}")

ws = websocket.WebSocketApp(
    "ws://localhost:8000/ws/predictions",
    on_message=on_message,
    on_error=on_error
)
ws.run_forever()
```

**Message Format:**
```json
{
  "symbol": "RELIANCE",
  "current_price": 2850.45,
  "target_price": 2950.00,
  "stop_loss": 2798.50,
  "signal": "STRONG_BUY",
  "confidence": 0.85,
  "timestamp": "2026-04-14T14:30:45.123456"
}
```

---

### 12. Single Stock Stream
```
ws://localhost:8000/ws/stock/{symbol}
```

**Parameters:**
- `symbol` (string): Stock symbol (e.g., "RELIANCE")

**Usage:**
```python
ws = websocket.WebSocketApp("ws://localhost:8000/ws/stock/RELIANCE")
ws.run_forever()
```

---

## 📊 AUTHENTICATION & RATE LIMITS

### Authentication
Currently **no authentication** required (local development).

For production, add API keys:
```python
headers = {
    "Authorization": f"Bearer {your_api_key}",
    "Content-Type": "application/json"
}

response = requests.get(
    "http://localhost:8000/api/v1/live/predictions",
    headers=headers
)
```

### Rate Limits
- **Live Predictions**: 60 requests/minute
- **Bot Status**: 120 requests/minute
- **WebSocket**: Unlimited

---

## 🐍 PYTHON CLIENT EXAMPLE

Complete example using all bot endpoints:

```python
import requests
import json
from datetime import datetime

class TradingBotClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    # PREDICTIONS
    def get_live_predictions(self):
        """Get all live predictions"""
        response = requests.get(f"{self.base_url}/api/v1/live/predictions")
        return response.json()
    
    def get_prediction(self, symbol):
        """Get prediction for single stock"""
        response = requests.get(f"{self.base_url}/api/v1/live/predictions/{symbol}")
        return response.json()
    
    def get_live_status(self):
        """Get service status"""
        response = requests.get(f"{self.base_url}/api/v1/live/status")
        return response.json()
    
    # BOT OPERATIONS
    def get_bot_status(self):
        """Get bot status"""
        response = requests.get(f"{self.base_url}/api/v1/bot/status")
        return response.json()
    
    def get_positions(self):
        """Get open positions"""
        response = requests.get(f"{self.base_url}/api/v1/bot/positions")
        return response.json()
    
    def close_position(self, position_id, exit_price=None, reason="MANUAL"):
        """Close a position"""
        params = {"reason": reason}
        if exit_price:
            params["exit_price"] = exit_price
        
        response = requests.post(
            f"{self.base_url}/api/v1/bot/positions/{position_id}/close",
            params=params
        )
        return response.json()
    
    def get_account_stats(self):
        """Get account statistics"""
        response = requests.get(f"{self.base_url}/api/v1/bot/account/stats")
        return response.json()
    
    def export_trades(self, format="csv"):
        """Export trades"""
        response = requests.get(f"{self.base_url}/api/v1/bot/export/{format}")
        return response.text if format == "csv" else response.json()

# USAGE
if __name__ == "__main__":
    client = TradingBotClient()
    
    # Check predictions
    print("=== LIVE PREDICTIONS ===")
    preds = client.get_live_predictions()
    for symbol, pred in list(preds["predictions"].items())[:3]:
        print(f"{symbol}: {pred['signal']} @ {pred['confidence']*100:.1f}%")
    
    # Check bot status
    print("\n=== BOT STATUS ===")
    status = client.get_bot_status()
    print(f"Status: {status['status']}")
    print(f"Trades Placed: {status['signals_received']}")
    print(f"Capital: ₹{status['account']['current_capital']:,}")
    
    # Check positions
    print("\n=== OPEN POSITIONS ===")
    positions = client.get_positions()
    for pos in positions["positions"]:
        print(f"{pos['stock']}: ₹{pos['current_value']:,} | P&L: {pos['unrealized_pnl_pct']:+.2f}%")
    
    # Export trades
    print("\n=== EXPORTING ===")
    trades = client.export_trades("json")
    print(f"Total Trades: {len(trades['trades'])}")
```

---

## 🔄 INTEGRATION WORKFLOW

### Example: Continuous Monitoring Loop
```python
import asyncio
import requests
from datetime import datetime

async def monitor_bot():
    """Monitor bot continuously"""
    client = TradingBotClient()
    
    while True:
        try:
            # Get current status
            status = client.get_bot_status()
            positions = client.get_positions()
            
            # Display
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}]")
            print(f"Bot: {status['status']} | Signals: {status['signals_received']}")
            print(f"Open Positions: {positions['total_open']}")
            print(f"Capital: ₹{status['account']['current_capital']:,}")
            
            # Check for exit conditions
            for pos in positions["positions"]:
                pnl_pct = pos["unrealized_pnl_pct"]
                print(f"  {pos['stock']}: {pnl_pct:+.2f}%", end="")
                
                # Auto-close if hits target or SL
                if pnl_pct >= 2.5:  # Target hit example
                    result = client.close_position(pos["position_id"], reason="TARGET_HIT")
                    print(" → CLOSED (TARGET)", result)
                elif pnl_pct <= -1.5:  # SL hit example
                    result = client.close_position(pos["position_id"], reason="SL_HIT")
                    print(" → CLOSED (SL)", result)
                else:
                    print()
            
            # Sleep and repeat
            await asyncio.sleep(60)
        
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(5)

# Run
asyncio.run(monitor_bot())
```

---

## ✅ TESTING ENDPOINTS

### cURL Examples

**Get predictions:**
```bash
curl -X GET "http://localhost:8000/api/v1/live/predictions"
```

**Get bot status:**
```bash
curl -X GET "http://localhost:8000/api/v1/bot/status" | jq .
```

**Get positions:**
```bash
curl -X GET "http://localhost:8000/api/v1/bot/positions" | jq '.positions[]'
```

**Close position:**
```bash
curl -X POST "http://localhost:8000/api/v1/bot/positions/POS001/close?reason=MANUAL"
```

**Export trades:**
```bash
curl -X GET "http://localhost:8000/api/v1/bot/export/csv" > trades.csv
```

---

## 📚 FURTHER INTEGRATION

### Connect to Broker APIs
Once tested with paper trading, integrate broker APIs:

```python
from zerodha_broker_api import ZerobhaClient  # Example

class LiveTradingBot(TradingBot):
    def __init__(self, *args, broker_client: ZerobhaClient, **kwargs):
        super().__init__(*args, **kwargs)
        self.broker = broker_client
    
    async def place_trade(self, stock, entry_price, target, sl, qty, **kwargs):
        # Place real order through broker
        order = self.broker.place_order(
            symbol=stock,
            price=entry_price,
            quantity=qty,
            order_type="LIMIT",
            transaction_type="BUY"
        )
        
        # Log to paper engine for tracking
        self.account.place_trade(stock, entry_price, target, sl, qty, **kwargs)
        
        return order
```

---

**For more details, see:**
- `backend/app/services/trading_bot.py` - Bot implementation
- `backend/app/services/paper_trading_engine.py` - Account engine
- `backend/app/main.py` - FastAPI routes
