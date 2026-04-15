"""
═══════════════════════════════════════════════════════════════════════════════
NSEIQ Trading Bot → Google Sheets Sync
═══════════════════════════════════════════════════════════════════════════════
Pushes live trading bot data (positions, trades, stats) to Google Sheets
"""

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import json
import os
from pathlib import Path

# Configuration
SHEET_ID = "1RuJHwfu2xfAYbSNMc05yzbz1M15kex6pL4MkyyiOxVw"
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

# Service account JSON path
CREDS_PATH = Path.home() / '.config' / 'gspread' / 'service_account.json'

def setup_sheets_client():
    """Authenticate and return Google Sheets client"""
    if not CREDS_PATH.exists():
        print(f"""
❌ Google Sheets credentials not found at: {CREDS_PATH}

📋 SETUP INSTRUCTIONS:
1. Go to: https://console.cloud.google.com/
2. Create new project: "NSEIQ Trading Bot"
3. Enable APIs:
   - Google Sheets API
   - Google Drive API
4. Create Service Account:
   - Service Account → Create Service Account
   - Create Key → JSON format
   - Download JSON file
5. Save file to: {CREDS_PATH}
6. Share your Google Sheet with the service account email:
   - Open the downloaded JSON
   - Copy "client_email"
   - Share the sheet with that email
7. Run this script again

Alternative: Use user authentication by creating credentials interactively
        """)
        return None
    
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(str(CREDS_PATH), SCOPES)
        client = gspread.authorize(creds)
        print(f"✅ Authenticated with Google Sheets")
        return client
    except Exception as e:
        print(f"❌ Auth error: {e}")
        return None

def get_bot_data_from_file():
    """Read bot data from local files/logs"""
    # For now, return sample data structure that matches our bot
    return {
        "status": {
            "state": "RUNNING",
            "uptime_seconds": 3600,
            "trades_executed": 1,
            "positions_open": 1,
            "last_signal_time": datetime.now().isoformat()
        },
        "account": {
            "initial_capital": 300000,
            "current_capital": 300000,
            "deployed_capital": 0,
            "available_capital": 300000,
            "total_pnl": 0,
            "pnl_percent": 0
        },
        "positions": [
            {
                "ticker": "M&M",
                "quantity": 480,
                "entry_price": 450,
                "entry_value": 216000,
                "current_price": 450,
                "current_value": 216000,
                "pnl": 0,
                "pnl_percent": 0,
                "target_price": 530,
                "stop_loss": 400,
                "entry_time": "2026-04-15 11:56:00",
                "status": "OPEN"
            }
        ],
        "trades": [
            {
                "trade_id": "BOT_20260415_001",
                "ticker": "M&M",
                "type": "BUY",
                "quantity": 480,
                "entry_price": 450,
                "entry_value": 216000,
                "exit_price": None,
                "exit_value": None,
                "pnl": None,
                "pnl_percent": None,
                "entry_time": "2026-04-15 11:56:00",
                "exit_time": None,
                "status": "OPEN",
                "signal": "STRONG_BUY",
                "confidence": 80.5
            }
        ]
    }

def insert_positions_to_sheets(client, positions):
    """Insert current positions into Google Sheets"""
    try:
        sheet = client.open_by_key(SHEET_ID)
        
        # Get or create "Open Positions" worksheet
        try:
            ws = sheet.worksheet("Open Positions")
            ws.clear()
        except gspread.exceptions.WorksheetNotFound:
            ws = sheet.add_worksheet(title="Open Positions", rows=100, cols=15)
        
        # Headers
        headers = [
            "Timestamp",
            "Ticker",
            "Quantity",
            "Entry Price",
            "Entry Value",
            "Current Price",
            "Current Value",
            "P&L",
            "P&L %",
            "Target Price",
            "Stop Loss",
            "Entry Time",
            "Status"
        ]
        
        ws.append_row(headers, table_range="A1")
        
        # Data rows
        for pos in positions:
            row = [
                datetime.now().isoformat(),
                pos.get("ticker"),
                pos.get("quantity"),
                pos.get("entry_price"),
                pos.get("entry_value"),
                pos.get("current_price"),
                pos.get("current_value"),
                pos.get("pnl"),
                pos.get("pnl_percent"),
                pos.get("target_price"),
                pos.get("stop_loss"),
                pos.get("entry_time"),
                pos.get("status")
            ]
            ws.append_row(row, table_range="A2")
        
        print(f"✅ Inserted {len(positions)} positions to Google Sheets")
        return True
    except Exception as e:
        print(f"❌ Error inserting positions: {e}")
        return False

def insert_trades_to_sheets(client, trades):
    """Insert trades into Google Sheets"""
    try:
        sheet = client.open_by_key(SHEET_ID)
        
        # Get or create "Trades" worksheet
        try:
            ws = sheet.worksheet("Trades")
        except gspread.exceptions.WorksheetNotFound:
            ws = sheet.add_worksheet(title="Trades", rows=500, cols=18)
        
        # Headers
        headers = [
            "Timestamp",
            "Trade ID",
            "Ticker",
            "Type",
            "Quantity",
            "Entry Price",
            "Entry Value",
            "Exit Price",
            "Exit Value",
            "P&L",
            "P&L %",
            "Entry Time",
            "Exit Time",
            "Status",
            "Signal",
            "Confidence",
            "Duration"
        ]
        
        # Check if headers exist
        first_row = ws.row_values(1)
        if not first_row or first_row[0] != "Timestamp":
            ws.insert_row(headers, index=1, table_range="A1")
        
        # Add new trades
        for trade in trades:
            existing = ws.findall(trade.get("trade_id", ""))
            
            if not existing:  # New trade
                row = [
                    datetime.now().isoformat(),
                    trade.get("trade_id"),
                    trade.get("ticker"),
                    trade.get("type"),
                    trade.get("quantity"),
                    trade.get("entry_price"),
                    trade.get("entry_value"),
                    trade.get("exit_price"),
                    trade.get("exit_value"),
                    trade.get("pnl"),
                    trade.get("pnl_percent"),
                    trade.get("entry_time"),
                    trade.get("exit_time"),
                    trade.get("status"),
                    trade.get("signal"),
                    trade.get("confidence"),
                    ""  # Duration (calculated)
                ]
                ws.append_row(row, table_range="A2")
        
        print(f"✅ Inserted {len(trades)} trades to Google Sheets")
        return True
    except Exception as e:
        print(f"❌ Error inserting trades: {e}")
        return False

def insert_stats_to_sheets(client, stats):
    """Insert account statistics summary"""
    try:
        sheet = client.open_by_key(SHEET_ID)
        
        # Get or create "Daily Stats" worksheet
        try:
            ws = sheet.worksheet("Daily Stats")
        except gspread.exceptions.WorksheetNotFound:
            ws = sheet.add_worksheet(title="Daily Stats", rows=100, cols=12)
        
        # Headers
        headers = [
            "Date",
            "Time",
            "Initial Capital",
            "Current Capital",
            "Deployed Capital",
            "Available Capital",
            "Total P&L",
            "P&L %",
            "Positions Open",
            "Trades Executed",
            "Win Rate",
            "Status"
        ]
        
        first_row = ws.row_values(1)
        if not first_row or first_row[0] != "Date":
            ws.insert_row(headers, index=1, table_range="A1")
        
        # Summary row
        summary_row = [
            datetime.now().strftime("%Y-%m-%d"),
            datetime.now().strftime("%H:%M:%S"),
            stats.get("account", {}).get("initial_capital"),
            stats.get("account", {}).get("current_capital"),
            stats.get("account", {}).get("deployed_capital"),
            stats.get("account", {}).get("available_capital"),
            stats.get("account", {}).get("total_pnl"),
            stats.get("account", {}).get("pnl_percent"),
            stats.get("status", {}).get("positions_open"),
            stats.get("status", {}).get("trades_executed"),
            "N/A",
            stats.get("status", {}).get("state")
        ]
        
        ws.append_row(summary_row, table_range="A2")
        
        print(f"✅ Inserted daily stats to Google Sheets")
        return True
    except Exception as e:
        print(f"❌ Error inserting stats: {e}")
        return False

def main():
    print("""
╔═════════════════════════════════════════════════════════════════════════════╗
║              🤖 NSEIQ Trading Bot → Google Sheets Sync                      ║
║                    Pushing Live Bot Data to Excel                           ║
╚═════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Setup Google Sheets client
    client = setup_sheets_client()
    if not client:
        print("\n⚠️  Cannot proceed without Google Sheets authentication")
        return False
    
    # Get bot data
    print("\n📊 Fetching trading bot data...")
    bot_data = get_bot_data_from_file()
    
    print(f"\n📤 Syncing to Google Sheets (ID: {SHEET_ID[:20]}...)")
    
    # Insert data
    insert_stats_to_sheets(client, bot_data)
    insert_positions_to_sheets(client, bot_data.get("positions", []))
    insert_trades_to_sheets(client, bot_data.get("trades", []))
    
    print("\n✅ Sync completed!")
    print(f"📊 Open sheet: https://docs.google.com/spreadsheets/d/{SHEET_ID}")
    
    return True

if __name__ == "__main__":
    main()
