"""
═══════════════════════════════════════════════════════════════════════════════
Google Sheets API Setup Guide - NSEIQ Trading Bot
═══════════════════════════════════════════════════════════════════════════════
Step-by-step instructions to enable Google Sheets integration
"""

SETUP_GUIDE = """
╔═════════════════════════════════════════════════════════════════════════════╗
║           📊 SETTING UP GOOGLE SHEETS FOR NSEIQ TRADING BOT                 ║
║                    Complete Step-by-Step Instructions                       ║
╚═════════════════════════════════════════════════════════════════════════════╝

STEP 1: CREATE GOOGLE CLOUD PROJECT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Go to Google Cloud Console:
   👉 https://console.cloud.google.com/

2. Click "Create Project" (top blue button)

3. Enter project name:
   • Name: "NSEIQ Trading Bot"
   • Click CREATE

4. Wait for project creation (1-2 minutes)


STEP 2: ENABLE REQUIRED APIs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. In the top search bar, search: "Google Sheets API"
   
2. Click "Google Sheets API" from results

3. Click the blue "ENABLE" button

4. Go back and search: "Google Drive API"

5. Click "Google Drive API" from results

6. Click the blue "ENABLE" button


STEP 3: CREATE SERVICE ACCOUNT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. In the left sidebar, go to:
   • "Service Accounts" (under "IAM and Admin")

2. Click "Create Service Account"

3. Fill in the form:
   • Service account name: "nseiq-trading-bot"
   • Service account ID: (auto-filled)
   • Description: "Trading bot data sync to Google Sheets"
   • Click CREATE AND CONTINUE

4. Grant roles (Optional - can skip):
   • Click CONTINUE (no roles needed for service account)

5. Click CREATE KEY:
   • Key type: JSON
   • Click CREATE
   
6. A JSON file will download automatically
   ✅ SAVE THIS FILE - you need it!


STEP 4: PLACE JSON FILE IN CORRECT LOCATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Create folder:
   📁 C:\\Users\\DEVENDER\\.config\\gspread
   
   If folder doesn't exist:
   • Open File Explorer
   • Go to: C:\\Users\\DEVENDER
   • Right-click → New → Folder
   • Name it ".config"
   • Inside .config, create new folder "gspread"

2. Move the downloaded JSON file:
   • Rename it to: service_account.json
   • Move to: C:\\Users\\DEVENDER\\.config\\gspread\\

   ✅ Final path should be:
      C:\\Users\\DEVENDER\\.config\\gspread\\service_account.json


STEP 5: SHARE GOOGLE SHEET WITH SERVICE ACCOUNT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Open the downloaded JSON file with Notepad:
   • Find the line: "client_email": "..."
   • Copy the email address (looks like: xxx@xxx.iam.gserviceaccount.com)

2. Open your Google Sheet:
   👉 https://docs.google.com/spreadsheets/d/1RuJHwfu2xfAYbSNMc05yzbz1M15kex6pL4MkyyiOxVw

3. Click "Share" (top right button)

4. Paste the service account email in the share field

5. Uncheck "Notify people"

6. Click "Share"

✅ Now the bot can read/write to your sheet!


STEP 6: RUN THE SYNC SCRIPT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Open PowerShell in project folder

2. Run:
   python sync_bot_to_sheets.py

3. Expected output:
   ✅ Authenticated with Google Sheets
   ✅ Inserted X positions to Google Sheets
   ✅ Inserted X trades to Google Sheets
   ✅ Inserted daily stats to Google Sheets
   ✅ Sync completed!


STEP 7: VERIFY IN GOOGLE SHEETS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Open your Google Sheet

2. Tabs should now show:
   📑 Daily Stats - Account summary data
   📑 Open Positions - Current open trades
   📑 Trades - All trade history

3. Data should be populated automatically!


TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ "File not found" error:
   ➡️  Make sure JSON is at: C:\\Users\\DEVENDER\\.config\\gspread\\service_account.json

❌ "Permission denied" error:
   ➡️  Share the Google Sheet with the service account email

❌ "Invalid credentials" error:
   ➡️  Download a NEW JSON key and try again

✅ All working? Next step: Run sync script on schedule!


NEXT: AUTOMATE DAILY SYNC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Once credentials are set up, you can schedule daily syncs using:
   • Windows Task Scheduler
   • Cron job (Linux/Mac)
   • Run manually anytime with: python sync_bot_to_sheets.py

═══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(SETUP_GUIDE)
