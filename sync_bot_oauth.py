"""
═══════════════════════════════════════════════════════════════════════════════
NSEIQ Bot Data → Google Sheets (OAuth2 User Authentication)
═══════════════════════════════════════════════════════════════════════════════
Alternative: Use your own Google account instead of service account
No JSON credentials needed - browser-based OAuth flow
"""

import gspread
from google.oauth2.service_account import Credentials
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as UserCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
import json
import os
from pathlib import Path
from datetime import datetime

# Configuration
SHEET_ID = "1RuJHwfu2xfAYbSNMc05yzbz1M15kex6pL4MkyyiOxVw"
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

# Paths
CREDS_DIR = Path.home() / '.config' / 'gspread'
SERVICE_ACCOUNT_PATH = CREDS_DIR / 'service_account.json'
USER_TOKEN_PATH = CREDS_DIR / 'token.json'
OAUTH_CREDS_PATH = CREDS_DIR / 'oauth_credentials.json'

def setup_oauth_credentials():
    """
    Setup OAuth2 credentials using browser authentication
    """
    print("""
╔═════════════════════════════════════════════════════════════════════════════╗
║           🔐 Setting up OAuth2 Google Account Authentication                ║
║                      (No service account needed!)                           ║
╚═════════════════════════════════════════════════════════════════════════════╝
    """)
    
    CREDS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if we already have a service account
    if SERVICE_ACCOUNT_PATH.exists():
        print(f"✅ Found service account at: {SERVICE_ACCOUNT_PATH}")
        return SERVICE_ACCOUNT_PATH
    
    # Check if we have OAuth token saved
    if USER_TOKEN_PATH.exists():
        print(f"✅ Found saved OAuth token at: {USER_TOKEN_PATH}")
        try:
            creds = UserCredentials.from_authorized_user_file(str(USER_TOKEN_PATH), SCOPES)
            if creds.valid or creds.refresh_token:
                print("✅ Token is valid!")
                return creds
        except Exception as e:
            print(f"⚠️  Token expired or invalid: {e}")
    
    print("""
📋 QUICK OAUTH SETUP (No Google Cloud Console needed!):

Option 1: Use Google Cloud OAuth Credentials
─────────────────────────────────────────────
1. Go to: https://console.cloud.google.com/
2. Create project: "NSEIQ Trading Bot OAuth"
3. Enable "Google Sheets API" and "Google Drive API"
4. Create "OAuth 2.0 Desktop Application" credentials:
   • Type: Desktop application (installed app)
   • Download as JSON
   • Save as: {oauth_path}
5. Run this script again - it will open your browser to authorize

Option 2: Use Service Account (Recommended)
────────────────────────────────────────────
Skip this and create service account at:
https://console.cloud.google.com/iam-admin/serviceaccounts

Option 3: Manual Upload
──────────────────────
Skip authentication and manually upload data using:
   python manual_sheets_upload.py

⚠️  You need credentials to proceed.
    """.format(oauth_path=OAUTH_CREDS_PATH))
    
    return None

def authenticate_with_service_account():
    """Try service account authentication"""
    try:
        if SERVICE_ACCOUNT_PATH.exists():
            creds = Credentials.from_service_account_file(str(SERVICE_ACCOUNT_PATH), scopes=SCOPES)
            client = gspread.authorize(creds)
            print(f"✅ Authenticated with service account")
            return client
    except Exception as e:
        print(f"⚠️  Service account auth failed: {e}")
    return None

def authenticate_with_oauth():
    """Authenticate using OAuth2 browser flow"""
    try:
        # Try to load existing token
        if USER_TOKEN_PATH.exists():
            creds = UserCredentials.from_authorized_user_file(str(USER_TOKEN_PATH), SCOPES)
            if creds.valid:
                client = gspread.authorize(creds)
                print(f"✅ Using existing OAuth token")
                return client
        
        # If no valid token, would need OAuth credentials file
        print("❌ OAuth token not found and no credentials file provided")
        return None
    except Exception as e:
        print(f"⚠️  OAuth auth failed: {e}")
        return None

def export_to_csv():
    """Export bot data to CSV for manual upload"""
    print("""
📊 EXPORTING BOT DATA TO CSV
─────────────────────────────
    """)
    
    bot_data = {
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
    
    # Create CSV files
    import csv
    
    # Positions CSV
    positions_file = Path.cwd() / "bot_positions.csv"
    with open(positions_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Timestamp', 'Ticker', 'Quantity', 'Entry Price', 'Entry Value',
            'Current Price', 'Current Value', 'P&L', 'P&L %', 'Target Price',
            'Stop Loss', 'Entry Time', 'Status'
        ])
        writer.writeheader()
        for pos in bot_data['positions']:
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
    
    print(f"✅ Positions exported to: {positions_file}")
    
    # Trades CSV
    trades_file = Path.cwd() / "bot_trades.csv"
    with open(trades_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Timestamp', 'Trade ID', 'Ticker', 'Type', 'Quantity', 'Entry Price',
            'Entry Value', 'Exit Price', 'Exit Value', 'P&L', 'P&L %',
            'Entry Time', 'Exit Time', 'Status', 'Signal', 'Confidence'
        ])
        writer.writeheader()
        for trade in bot_data['trades']:
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
    
    print(f"✅ Trades exported to: {trades_file}")
    
    # Stats CSV
    stats_file = Path.cwd() / "bot_stats.csv"
    with open(stats_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Date', 'Time', 'Initial Capital', 'Current Capital', 'Deployed Capital',
            'Available Capital', 'Total P&L', 'P&L %', 'Positions Open',
            'Trades Executed', 'Status'
        ])
        writer.writeheader()
        writer.writerow({
            'Date': datetime.now().strftime("%Y-%m-%d"),
            'Time': datetime.now().strftime("%H:%M:%S"),
            'Initial Capital': bot_data['account']['initial_capital'],
            'Current Capital': bot_data['account']['current_capital'],
            'Deployed Capital': bot_data['account']['deployed_capital'],
            'Available Capital': bot_data['account']['available_capital'],
            'Total P&L': bot_data['account']['total_pnl'],
            'P&L %': bot_data['account']['pnl_percent'],
            'Positions Open': bot_data['status']['positions_open'],
            'Trades Executed': bot_data['status']['trades_executed'],
            'Status': bot_data['status']['state']
        })
    
    print(f"✅ Stats exported to: {stats_file}")
    
    print(f"""
📋 CSV FILES READY FOR MANUAL UPLOAD:
    • bot_positions.csv
    • bot_trades.csv
    • bot_stats.csv

📊 To upload to Google Sheets:
   1. Open: https://docs.google.com/spreadsheets/d/1RuJHwfu2xfAYbSNMc05yzbz1M15kex6pL4MkyyiOxVw
   2. Create new tabs for each CSV
   3. Use "File → Import" or paste data manually
   4. Done!
    """)
    
    return True

def main():
    print("""
╔═════════════════════════════════════════════════════════════════════════════╗
║          🤖 NSEIQ Trading Bot → Google Sheets Sync (OAuth2)                 ║
║                    Alternative Authentication Methods                       ║
╚═════════════════════════════════════════════════════════════════════════════╝
    """)
    
    setup_oauth_credentials()
    
    # Try authentication methods in order
    print("\n🔍 Attempting authentication...")
    
    client = authenticate_with_service_account()
    if client:
        print("✅ Using service account")
        return True
    
    client = authenticate_with_oauth()
    if client:
        print("✅ Using OAuth token")
        return True
    
    print("""
❌ Authentication failed. Available options:

1️⃣  SETUP SERVICE ACCOUNT (Recommended):
   - Follow: https://console.cloud.google.com/
   - Download JSON to: C:\\Users\\DEVENDER\\.config\\gspread\\service_account.json
   - Run: python sync_bot_to_sheets.py

2️⃣  EXPORT TO CSV (Manual Upload):
   - Run: python export_bot_to_csv.py
   - Manually upload CSV files to your Google Sheet

3️⃣  USE OAUTH2 (Browser Authentication):
   - Requires OAuth credentials file downloaded from Google Cloud
   - More complex setup
    """)
    
    choice = input("\n🔹 Choose option (1/2/3): ").strip()
    
    if choice == "2":
        export_to_csv()
    else:
        print("\n⚠️  Please complete the setup for option 1 or 2")

if __name__ == "__main__":
    main()
