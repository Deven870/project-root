#!/usr/bin/env python
"""
╔════════════════════════════════════════════════════════════════════════════╗
║                   GOOGLE SHEETS SETUP GUIDE                               ║
║                  Auto-Logging Trades to Your Google Sheet                 ║
╚════════════════════════════════════════════════════════════════════════════╝

Your Google Sheet:
https://docs.google.com/spreadsheets/d/1RuJHwfu2xfAYbSNMc05yzbz1M15kex6pL4MkyyiOxVw

The bot is ready to auto-log ALL trades to this sheet!

However, you need to authenticate with Google. There are 2 ways:

═════════════════════════════════════════════════════════════════════════════
OPTION 1: Service Account (Recommended - No Popups)
═════════════════════════════════════════════════════════════════════════════

1. Go to: https://console.cloud.google.com/

2. Create a new project:
   - Click "Select a Project" > "NEW PROJECT"
   - Name it: "NSEIQ Trading Bot"
   - Click "CREATE"

3. Enable Google Sheets API:
   - In APIs & Services > Library
   - Search for "Google Sheets API"
   - Click it > "ENABLE"

4. Create Service Account:
   - Go to APIs & Services > Credentials
   - Click "CREATE CREDENTIALS" > "Service Account"
   - Fill details:
     - Service account name: nseiq-trading-bot
   - Click "CREATE AND CONTINUE"
   - Grant role: Editor
   - Click "CONTINUE" > "DONE"

5. Create Key:
   - Go to Service Accounts
   - Click the service account you just created
   - Go to "Keys" tab
   - "Add Key" > "Create new key"
   - Choose "JSON" > "CREATE"
   - Save the JSON file

6. Share Google Sheet:
   - Copy the "client_email" from the JSON file
   - Open your Google Sheet
   - Click "Share" > Paste email > "Share"
   - Make sure to give EDITOR permissions

7. Place JSON File:
   - Create directory: ~/.config/gspread/
   - Move the JSON file there
   - Name it: service_account.json
   
   On Windows:
   C:\\Users\\<YourName>\\.config\\gspread\\service_account.json
   
   On Linux/Mac:
   ~/.config/gspread/service_account.json

═════════════════════════════════════════════════════════════════════════════
OPTION 2: OAuth2 (Manual - Easier Setup)
═════════════════════════════════════════════════════════════════════════════

Simply Google sign in when prompted. The system will automatically handle auth.
(Requires manual interaction for first run)

═════════════════════════════════════════════════════════════════════════════
VERIFY SETUP
═════════════════════════════════════════════════════════════════════════════

Run this to verify the setup works:
  $ python -c "from gspread import authorize; print('✅ gspread working')"

═════════════════════════════════════════════════════════════════════════════
WHAT GETS LOGGED
═════════════════════════════════════════════════════════════════════════════

✅ Sheet 1: "Trades"
   - Every trade (open/closed)
   - Entry price, target, stop loss
   - Volume, capital used
   - Risk/Reward ratio
   - Exit price, P&L, Return %

✅ Sheet 2: "Daily Stats"
   - Daily summary
   - Total trades, wins, losses
   - Win rate %
   - Daily P&L
   - Drawdown %

✅ Sheet 3: "Open Positions"
   - Real-time open positions
   - Current price vs entry
   - Unrealized P&L
   - Days held

═════════════════════════════════════════════════════════════════════════════
TROUBLESHOOTING
═════════════════════════════════════════════════════════════════════════════

❌ "No module named 'gspread'"
   → Install: pip install gspread oauth2client

❌ "permission denied" error
   → Make sure service account email has EDITOR access to the sheet

❌ "Unable to authenticate"
   → Check that JSON file is in ~/.config/gspread/service_account.json

❌ "Spreadsheet not found"
   → Verify the sheet ID is correct (check URL)
   → Make sure it's shared with service account email

═════════════════════════════════════════════════════════════════════════════

Your trading bot is ready to log to Google Sheets! 🎉

Once set up, every trade will automatically appear in your sheet.

"""

print(__doc__)

# Try to import and test
try:
    import gspread
    print("✅ gspread is installed")
except ImportError:
    print("⚠️  gspread not installed. Install with:")
    print("   pip install gspread oauth2client")

try:
    import oauth2client
    print("✅ oauth2client is installed")
except ImportError:
    print("⚠️  oauth2client not installed. Install with:")
    print("   pip install gspread oauth2client")

import os
creds_path = os.path.expanduser("~/.config/gspread/service_account.json")
if os.path.exists(creds_path):
    print(f"✅ Service account credentials found at: {creds_path}")
else:
    print(f"⚠️  Service account not found at: {creds_path}")
    print("    Follow the steps above to set up authentication")
