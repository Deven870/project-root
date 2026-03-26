# Google Sheets Integration Setup Guide

## Overview
This guide explains how to set up Google Sheets integration for real-time data updates.

## Architecture
- **5 Tabs**: Live Signals, My Trades, P&L Dashboard, News Feed, Config
- **Batching**: All updates batched into 1 API call per refresh cycle
- **API Quota**: Google free tier = 300 requests/minute

## Setup Steps

### 1. Create a Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable the **Google Sheets API** and **Google Drive API**

### 2. Create a Service Account
1. In Google Cloud Console, go to **Service Accounts**
2. Click **Create Service Account**
3. Fill in the details (e.g., name: "voicbot-sheets")
4. Click **Create and Continue**
5. Grant the role: **Editor** (for read/write access)
6. Click **Continue**
7. Click **Done**

### 3. Generate Service Account Key
1. Click on the newly created service account
2. Go to the **Keys** tab
3. Click **Add Key** → **Create new key**
4. Choose **JSON** format
5. Click **Create** - this downloads the JSON file
6. Save the JSON file as `service_account.json` in your project root

### 4. Create Google Sheet
1. Go to [Google Sheets](https://sheets.google.com)
2. Create a new spreadsheet (name it something like "VoicBot Trading Data")
3. Share the sheet with the service account email (found in the JSON file)
   - The email looks like: `voicbot-sheets@project-id.iam.gserviceaccount.com`
   - Grant **Editor** permissions

### 5. Configure .env
Update `.env` with:
```env
SHEETS_ID=<your-spreadsheet-id>
SERVICE_ACCOUNT_FILE=service_account.json
```

Get the spreadsheet ID from the URL:
```
https://docs.google.com/spreadsheets/d/{SHEETS_ID}/edit
                                         ^^^^^^^^
```

### 6. Test the Integration
```python
from modules.google_sheets import get_authorized_client, get_worksheet

# Test authentication
client = get_authorized_client()

# Test worksheet creation/access
ws = get_worksheet("Live Signals")
print("✓ Integration working!")
```

## Usage

### Update News Feed (from sentiment_engine)
```python
from modules.sentiment_engine import get_news_with_sentiment

news_items = get_news_with_sentiment("RELIANCE.NS", from_days=7, push_to_sheets=True)
```

### Update P&L Dashboard (from backtester)
```python
from modules.google_sheets import update_pnl_dashboard

pnl_data = {
    "total_trades": 42,
    "winners": 28,
    "losers": 14,
    "win_rate": 66.67,
    "total_pnl": 4250,
    "avg_win": 150,
    "avg_loss": -85,
    "profit_factor": 1.65
}
update_pnl_dashboard(pnl_data)
```

### Batch Update All Tabs (Efficient!)
```python
from modules.google_sheets import batch_update_all_tabs

batch_update_all_tabs(
    live_signals=[...],
    trades=[...],
    pnl=pnl_data,
    news=news_items,
    config={...}
)
```

### Add Trade to My Trades
```python
from modules.google_sheets import add_trade

add_trade({
    "entry_time": "2024-01-15 09:30",
    "entry_price": 2650.50,
    "exit_time": "2024-01-15 14:45",
    "exit_price": 2670.25,
    "qty": 10,
    "profit_loss": 198.50,
    "status": "CLOSED"
})
```

### Update Live Signals
```python
from modules.google_sheets import update_live_signals

update_live_signals([
    {
        "timestamp": "2024-01-15 10:30",
        "stock": "RELIANCE",
        "signal": "BUY",
        "strength": 0.85,
        "confidence": 0.92
    }
])
```

## API Call Optimization

### ✓ GOOD (Batched)
```python
# Single call per refresh cycle
batch_update_all_tabs(live_signals=..., trades=..., pnl=..., news=..., config=...)
```

### ✗ BAD (Un-batched)
```python
# Multiple individual calls = quota burnout
update_live_signals([...])
update_pnl_dashboard({...})
update_news_feed([...])
```

## Troubleshooting

### "ServiceAccountCredentials" error
- Ensure `service_account.json` exists in project root
- Verify file is valid JSON
- Check service account email in JSON

### "Unauthorized" error
- Service account may not have access to the spreadsheet
- Make sure the sheet is shared with the service account email
- Grant **Editor** permissions (not just Viewer)

### "Spreadsheet not found"
- Verify `SHEETS_ID` is correct (copy from URL)
- Check `SHEETS_ID` in `.env` matches actual spreadsheet

### Quota exceeded (429 errors)
- You're making too many cell-by-cell updates
- Use batching! Call one update function per cycle
- Avoid updating the same cell multiple times per minute

## API Quota Reference
- **Free Tier**: 300 requests/minute
- **Recommended Cycle**: 1 request/tab per refresh cycle = 5 requests/min max (safe)
- **Safe Update Frequency**: Once per 2-5 seconds minimum

## File Structure
```
project-root/
├── service_account.json      ← Google Sheets credentials
├── .env                       ← Contains SHEETS_ID, SERVICE_ACCOUNT_FILE
├── modules/
│   ├── google_sheets.py       ← Main integration module
│   ├── sentiment_engine.py    ← Integrated for News Feed
│   └── backtester.py          ← Will integrate for P&L Dashboard
└── ...
```
