#!/usr/bin/env python
"""Quick Google Sheets Setup Test and Guide"""

import os
import sys

print("\n" + "="*80)
print("🔐 GOOGLE SHEETS AUTHENTICATION TEST")
print("="*80 + "\n")

# Test imports
print("📦 Checking dependencies...\n")

try:
    import gspread
    print("✅ gspread installed")
except ImportError:
    print("❌ gspread NOT installed")
    print('   Fix: pip install gspread oauth2client')
    sys.exit(1)

try:
    import oauth2client
    print("✅ oauth2client installed")
except ImportError:
    print("❌ oauth2client NOT installed")
    print('   Fix: pip install oauth2client')
    sys.exit(1)

# Check credentials
print("\n📋 Checking credentials...\n")

creds_path = os.path.expanduser("~/.config/gspread/service_account.json")

if os.path.exists(creds_path):
    print(f"✅ Service account found at: {creds_path}")
    
    # Try to authenticate
    try:
        from oauth2client.service_account import ServiceAccountCredentials
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
        gc = gspread.authorize(creds)
        print("✅ Authentication successful!")
        
        # Try to access the sheet
        try:
            sheet_id = "1RuJHwfu2xfAYbSNMc05yzbz1M15kex6pL4MkyyiOxVw"
            sheet = gc.open_by_key(sheet_id)
            print(f"✅ Google Sheet accessible!")
            print(f"   Sheet Name: {sheet.title}")
            print(f"   Worksheets: {[ws.title for ws in sheet.worksheets()]}")
            
            print("\n✅✅✅ ALL SYSTEMS GO! Your trades will auto-log to the sheet! ✅✅✅\n")
            
        except Exception as e:
            print(f"❌ Cannot access sheet: {e}")
            print("   Make sure:")
            print("   1. Sheet is shared with the service account email")
            print("   2. Service account has EDITOR permissions")
            
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        
else:
    print(f"❌ Credentials NOT found at: {creds_path}")
    print("\n📚 Setup Instructions:")
    print("="*80)
    print("""
1. Go to: https://console.cloud.google.com/

2. Create a new project named "NSEIQ Trading Bot"

3. Enable Google Sheets API:
   - APIs & Services > Library
   - Search "Google Sheets API"
   - Click ENABLE

4. Create Service Account:
   - APIs & Services > Credentials
   - CREATE CREDENTIALS > Service Account
   - Name: nseiq-trading-bot
   - Role: Editor
   - CREATE & CONTINUE

5. Generate JSON Key:
   - Go to Service Accounts
   - Click your service account
   - Keys tab > Add Key > Create new key
   - JSON > CREATE
   - Save the JSON file

6. Move credentials file:
   - Create folder: ~/.config/gspread/
   - Move JSON file there
   - Rename to: service_account.json
   
   Windows path:
   C:\\Users\\<YourUsername>\\.config\\gspread\\service_account.json

7. Share your Google Sheet:
   - Open: https://docs.google.com/spreadsheets/d/1RuJHwfu2xfAYbSNMc05yzbz1M15kex6pL4MkyyiOxVw
   - Click Share
   - Copy "client_email" from the JSON file
   - Paste email in "Share with people..."
   - Give EDITOR permission
   - Share

8. Test again:
   - python test_sheets_auth.py
    """)
    print("="*80)

print("\n")
