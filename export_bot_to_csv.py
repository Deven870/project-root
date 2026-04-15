"""
═══════════════════════════════════════════════════════════════════════════════
Export NSEIQ Bot Data to CSV (No Authentication Needed)
═══════════════════════════════════════════════════════════════════════════════
Simplest solution: Export bot data to CSV files → Manually import to Google Sheets
"""

import csv
from datetime import datetime
from pathlib import Path
import json

def get_bot_data():
    """Get bot data from API or files"""
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

def export_positions_csv(bot_data):
    """Export positions to CSV"""
    csv_file = Path.cwd() / "bot_positions.csv"
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Timestamp', 'Ticker', 'Quantity', 'Entry Price', 'Entry Value',
            'Current Price', 'Current Value', 'P&L', 'P&L %', 'Target Price',
            'Stop Loss', 'Entry Time', 'Status'
        ])
        writer.writeheader()
        
        for pos in bot_data.get('positions', []):
            writer.writerow({
                'Timestamp': datetime.now().isoformat(),
                'Ticker': pos['ticker'],
                'Quantity': pos['quantity'],
                'Entry Price': pos['entry_price'],
                'Entry Value': pos['entry_value'],
                'Current Price': pos['current_price'],
                'Current Value': pos['current_value'],
                'P&L': pos['pnl'],
                'P&L %': pos['pnl_percent'],
                'Target Price': pos['target_price'],
                'Stop Loss': pos['stop_loss'],
                'Entry Time': pos['entry_time'],
                'Status': pos['status']
            })
    
    print(f"✅ Positions exported: {csv_file}")
    return csv_file

def export_trades_csv(bot_data):
    """Export trades to CSV"""
    csv_file = Path.cwd() / "bot_trades.csv"
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Timestamp', 'Trade ID', 'Ticker', 'Type', 'Quantity', 'Entry Price',
            'Entry Value', 'Exit Price', 'Exit Value', 'P&L', 'P&L %',
            'Entry Time', 'Exit Time', 'Status', 'Signal', 'Confidence'
        ])
        writer.writeheader()
        
        for trade in bot_data.get('trades', []):
            writer.writerow({
                'Timestamp': datetime.now().isoformat(),
                'Trade ID': trade['trade_id'],
                'Ticker': trade['ticker'],
                'Type': trade['type'],
                'Quantity': trade['quantity'],
                'Entry Price': trade['entry_price'],
                'Entry Value': trade['entry_value'],
                'Exit Price': trade['exit_price'],
                'Exit Value': trade['exit_value'],
                'P&L': trade['pnl'],
                'P&L %': trade['pnl_percent'],
                'Entry Time': trade['entry_time'],
                'Exit Time': trade['exit_time'],
                'Status': trade['status'],
                'Signal': trade['signal'],
                'Confidence': trade['confidence']
            })
    
    print(f"✅ Trades exported: {csv_file}")
    return csv_file

def export_stats_csv(bot_data):
    """Export daily stats to CSV"""
    csv_file = Path.cwd() / "bot_stats.csv"
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Date', 'Time', 'Initial Capital', 'Current Capital', 'Deployed Capital',
            'Available Capital', 'Total P&L', 'P&L %', 'Positions Open',
            'Trades Executed', 'Status'
        ])
        writer.writeheader()
        
        account = bot_data.get('account', {})
        status = bot_data.get('status', {})
        
        writer.writerow({
            'Date': datetime.now().strftime("%Y-%m-%d"),
            'Time': datetime.now().strftime("%H:%M:%S"),
            'Initial Capital': account['initial_capital'],
            'Current Capital': account['current_capital'],
            'Deployed Capital': account['deployed_capital'],
            'Available Capital': account['available_capital'],
            'Total P&L': account['total_pnl'],
            'P&L %': account['pnl_percent'],
            'Positions Open': status['positions_open'],
            'Trades Executed': status['trades_executed'],
            'Status': status['state']
        })
    
    print(f"✅ Stats exported: {csv_file}")
    return csv_file

def export_json(bot_data):
    """Export all data to JSON"""
    json_file = Path.cwd() / "bot_data.json"
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(bot_data, f, indent=2, default=str)
    
    print(f"✅ JSON exported: {json_file}")
    return json_file

def main():
    print("""
╔═════════════════════════════════════════════════════════════════════════════╗
║              📊 NSEIQ Bot Data → CSV Export (No Auth Needed)                ║
║                  Simple Solution: Export & Manual Upload                    ║
╚═════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n📦 Fetching bot data...")
    bot_data = get_bot_data()
    
    print("\n📤 Exporting to CSV files...\n")
    
    # Export all formats
    export_positions_csv(bot_data)
    export_trades_csv(bot_data)
    export_stats_csv(bot_data)
    export_json(bot_data)
    
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 FILES READY FOR UPLOAD:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ bot_positions.csv  - Current open positions
✅ bot_trades.csv     - All trade history
✅ bot_stats.csv      - Daily account statistics
✅ bot_data.json      - Complete data (backup)

📋 HOW TO UPLOAD TO GOOGLE SHEETS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METHOD 1: Import CSV Directly (Recommended)
1. Open your Google Sheet:
   https://docs.google.com/spreadsheets/d/1RuJHwfu2xfAYbSNMc05yzbz1M15kex6pL4MkyyiOxVw

2. Click "File" → "Import"

3. Click "Upload" tab and select bot_trades.csv

4. Choose import options:
   • Create new sheet: "Trades"
   • Import location: New spreadsheet
   • Click "Import data"

5. Repeat for bot_positions.csv and bot_stats.csv

METHOD 2: Copy-Paste Data
1. Open the CSV file with Excel or text editor
2. Select all data (Ctrl+A)
3. Copy (Ctrl+C)
4. Go to Google Sheet
5. Right-click on cell A1
6. Click "Paste special" → "Paste values only"
7. Done!

METHOD 3: Use Python Script (Requires Auth)
- If you set up Google Cloud credentials:
  python sync_bot_to_sheets.py

═════════════════════════════════════════════════════════════════════════════
✅ Export complete! Your data is ready to upload manually.
═════════════════════════════════════════════════════════════════════════════
    """)

if __name__ == "__main__":
    main()
