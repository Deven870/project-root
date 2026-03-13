# SHEETS_SETUP.md

# Google Sheets Integration Setup Guide

## What This Does

This integration lets you:
- 📊 Track all your stock searches in real-time
- 💰 Log investments with automatic P&L tracking
- 📈 View daily portfolio analysis
- 📋 Maintain a full portfolio record
- ☁️ Access everything from anywhere (cloud-based)

## Step 1: Set Up Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project:
   - Click "Select a Project" at the top
   - Click "NEW PROJECT"
   - Name it "Digitrader" (or your choice)
   - Click "CREATE"

3. Enable Google Sheets API:
   - In the search bar, search for "Google Sheets API"
   - Click on it and press "ENABLE"

4. Enable Google Drive API:
   - In the search bar, search for "Google Drive API"
   - Click on it and press "ENABLE"

## Step 2: Create Service Account

1. In the left sidebar, click "Service Accounts"
2. Click "CREATE SERVICE ACCOUNT"
3. Fill in details:
   - **Service account name**: `digitrader-sheets`
   - **Service account ID**: (auto-filled)
   - **Description**: "For accessing investment tracking sheets"
   - Click "CREATE AND CONTINUE"

4. Grant permissions:
   - **Basic** role: `Editor`
   - Click "CONTINUE"

5. Create key:
   - Skip the "Grant users access" section
   - Click "CREATE KEY"
   - Choose JSON format
   - Click "CREATE"
   - A JSON file will download automatically

## Step 3: Save Credentials

1. The downloaded JSON file contains your credentials
2. **Rename it to** `google_credentials.json`
3. **Move it** to your project root folder:
   ```
   project-root/
   ├── google_credentials.json  ← Put it here
   ├── app.py
   ├── modules/
   └── ...
   ```

## Step 4: Use in App

1. Open the Streamlit app: `streamlit run app.py`
2. Go to the **📊 Tracking Dashboard** page
3. Click **"📋 Setup Sheets & Start Tracking"**
4. Choose:
   - **Create New Sheet** - (recommended for first time)
   - **Or enter existing Sheet URL**

5. The app will:
   - Create a new Google Sheet automatically
   - Share the URL with you
   - Create 4 worksheets:
     - **Searches**: Log every stock you analyze
     - **Investments**: Track your investments
     - **Daily Analysis**: Daily portfolio snapshots
     - **Portfolio**: Current holdings

## Step 5: Start Tracking

### From Trading Dashboard
- After analyzing a stock, click **"📊 Log to Sheets"**
- This saves your search, prediction, and sentiment to the Sheets

### From Portfolio Page
- Enter investment details
- Click **"💾 Log Investment"**
- Tracks entry price, P&L, and returns

### Daily Analysis
- Each day, click **"📈 Log Today's Analysis"**
- Automatically calculates portfolio value, P&L, best/worst performers

## Real-Time Features

✅ **Auto-Sync**: All data updates instantly in Google Sheets  
✅ **Cloud-Backed**: Access from any device, anytime  
✅ **Shareable**: Share the Sheet URL with advisors/friends  
✅ **Mobile-Friendly**: View on phone via Google Sheets app  
✅ **Automatic Charts**: Google Sheets can create pivot tables & charts  

## Sharing Your Sheet

1. Open your Google Sheet (URL shown in app)
2. Click **Share** (top right)
3. Add email addresses of people you want to share with
4. Set permissions (View / Comment / Edit)
5. Send

## Troubleshooting

### ❌ "Credentials not found"
- Make sure `google_credentials.json` is in the project root
- Restart the Streamlit app

### ❌ "Permission denied"
- Go back to Google Cloud Console
- Check that the Service Account has `Editor` role
- Regenerate the JSON key if needed

### ❌ "Sheet URL not working"
- Make sure you copied the full URL from the browser
- Should start with `https://docs.google.com/spreadsheets/`

## Accessing Your Sheets Without App

1. You can access your sheet anytime at the URL
2. Google Sheets lets you:
   - Create charts & graphs
   - Build pivot tables
   - Use formulas (=SUM, =AVERAGE, etc.)
   - Download as Excel
   - Export as PDF

## Example Formulas in Google Sheets

Once data is logging, you can add formulas:

- **Total P&L**: `=SUM(G:G)` in Daily Analysis
- **Average Return**: `=AVERAGE(H:H)` in Investments
- **Best Day**: `=MAX(E:E)` in Daily Analysis
- **Best Stock**: `=INDEX(A:A,MATCH(MAX(H:H),H:H,0))` in Portfolio

---

**Need help?** Check the app's **"💡 Help"** section or review this guide in SHEETS_SETUP.md
