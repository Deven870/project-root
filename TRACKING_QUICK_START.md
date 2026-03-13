# TRACKING_QUICK_START.md

# 📊 Real-Time Investment Tracking Quick Start

## What You Can Do

✅ **Log Every Stock Search** - Track your analysis decisions  
✅ **Record Investments** - Maintain entry prices and amounts  
✅ **Daily P&L Tracking** - Monitor portfolio performance daily  
✅ **Real-Time Sync** - Access from anywhere via Google Sheets  
✅ **Historical Analysis** - Review past searches and investments  
✅ **Cloud-Based** - No local files, always backed up  

---

## 3-Minute Setup

### 1. **Get Google Credentials** (5 minutes, one-time)

Do this FIRST:
1. Go to https://console.cloud.google.com/
2. Create a new project (click "Select a Project" → "NEW PROJECT")  
3. Search for "Google Sheets API" and click ENABLE
4. Search for "Google Drive API" and click ENABLE
5. Click "Service Accounts" in left menu
6. Click "CREATE SERVICE ACCOUNT"
   - Name: `digitrader-sheets`
   - Click "CREATE AND CONTINUE"
7. Don't assign roles yet, just continue
8. At the bottom, click "CREATE KEY" → JSON → "CREATE"
9. Copy the downloaded file to your project root as `google_credentials.json`

**That's it for Google!**

### 2. **Start Using in the App**

Open Streamlit: `streamlit run app.py`

1. Go to **"📊 Tracking Dashboard"** page (new tab!)
2. Expand **"⚙️ Setup Sheets & Start Tracking"**
3. Click **"🆕 Create New Investment Tracker Sheet"**
4. ✅ Done! You'll get a link to your Google Sheet

---

## Daily Workflow

### Morning: Review Previous Day
- Open the **"📊 Tracking Dashboard"**
- Go to **"📜 History"** tab
- See your recent searches and P&L

### During Market: Log Searches
- Go to **"📊 Trading Dashboard"** (homepage)
- Find a stock you like
- See the prediction
- Click **"📊 Log to Sheets"** (we'll add this button!)
- Data saved automatically to your sheet

### Evening: Log Your Investments
- Go back to **Tracking Dashboard**
- Click **"💰 Log Investment"** tab
- Enter stock, amount, entry price
- Click "Log Investment"
- Automatically tracked!

### End of Day: Daily Analysis
- Go to **"📈 Daily Analysis"** tab
- Enter portfolio value
- Add notes on best/worst performers
- Click "Log Daily Analysis"
- Creates a historical snapshot

---

## What Gets Tracked

### 📊 Searches Sheet
Every stock you analyze:
```
Date | Symbol | Trend | Confidence | Current Price | Predicted Price | Expected Return | Sentiment
```

### 💰 Investments Sheet  
Every purchase you make:
```
Date | Symbol | Amount | Entry Price | Current Price | P&L | Return % | Horizon | Status
```

### 📈 Daily Analysis Sheet
Portfolio snapshots each day:
```
Date | Total Invested | Current Value | P&L | Return % | Best Stock | Worst Stock | Notes
```

### 📋 Portfolio Sheet
Your active holdings:
```
Symbol | Shares | Entry Price | Current Price | Investment | Value | P&L | Return % | Last Updated
```

---

## Cool Things You Can Do

### In Your Google Sheet:
1. **Create Charts**: Select data → Insert → Chart
2. **Pivot Tables**: Data → Pivot Table
3. **Formulas**: 
   - `=SUM(F:F)` = Total P&L
   - `=AVERAGE(H:H)` = Average Return %
   - `=MAX(H:H)` = Best Return
   - `=COUNTIF(C:C,"Bullish")` = Bullish Signals

4. **Share with Others**:
   - Click Share (top right)
   - Add emails
   - They can see your portfolio in real-time!

5. **Mobile Access**:
   - Open in Google Sheets mobile app
   - View portfolio anytime

6. **Export**:
   - File → Download as Excel
   - File → Download as PDF

---

## Troubleshooting

### "Credentials not found"
- Make sure `google_credentials.json` is in the project root folder
- Restart the app: `streamlit run app.py`

### "Permission denied"
- Go back to Google Cloud Console
- Check service account has "Editor" role

### "Sheet URL not working"
- Copy the FULL URL from the browser
- Should start with `https://docs.google.com/spreadsheets/`

### Sheets not updating?
- Refresh the page
- Or restart the Streamlit app

---

## Example Investment Tracking Scenario

**Monday 9:15 AM (Market Open)**
1. Analyze RELIANCE.NS on Trading Dashboard
2. Bullish signal, 78% confidence
3. Click "Log to Sheets" → Saved to Searches sheet

**Monday 11:30 AM**
1. Decide to invest ₹5,000 in RELIANCE at ₹1,350
2. Go to Tracking Dashboard → "💰 Log Investment"
3. Fill in: RELIANCE.NS | ₹5,000 | ₹1,350
4. Click "Log Investment" → Saved

**Monday 3:30 PM (Market Close)**
1. Portfolio value: ₹5,100 (up ₹100)
2. Go to "📈 Daily Analysis" tab
3. Enter: Invested: ₹5,000 | Current: ₹5,100
4. Click "Log Daily Analysis" → Saved

**Result**: Over time, your Google Sheet fills up with:
- Every search decision
- Every investment with entry/exit prices
- Daily performance snapshots
- Historical comparison data

Now you can:
- See patterns in your trading
- Evaluate prediction accuracy
- Calculate real returns
- Share results with advisors

---

## Next: Add Real-Time Updates

Soon we'll add:
- ⏰ Automatic daily portfolio updates
- 📧 Email alerts for big moves
- 📱 SMS notifications
- 🔔 Price alerts
- 📊 Advanced analytics & reports

---

**Ready to start?** 
Open Streamlit and go to **📊 Tracking Dashboard**! 🚀

For full setup instructions, see **SHEETS_SETUP.md**
