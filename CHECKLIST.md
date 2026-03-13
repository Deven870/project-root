# 📋 SETUP CHECKLIST - Real-Time Investment Tracker

## ✅ Implementation Checklist

### Phase 1: Google Cloud Setup (First Time - 5 minutes)

- [ ] Go to https://console.cloud.google.com/
- [ ] Create new project named "Digitrader"
- [ ] Search for "Google Sheets API" → Click ENABLE
- [ ] Search for "Google Drive API" → Click ENABLE
- [ ] Click "Service Accounts" in left menu
- [ ] Click "CREATE SERVICE ACCOUNT"
  - [ ] Name: `digitrader-sheets`
  - [ ] Click "CREATE AND CONTINUE"
- [ ] Don't assign roles, just continue
- [ ] Click "CREATE KEY" → JSON format → CREATE
- [ ] Downloaded file: rename to `google_credentials.json`
- [ ] Move to project root folder
- [ ] Verify: File exists at `project-root/google_credentials.json`

### Phase 2: App Setup

- [ ] Install packages: 
  ```bash
  pip install gspread google-auth-oauthlib
  ```
- [ ] Run app: `streamlit run app.py`
- [ ] Navigate to: **📊 Tracking Dashboard** (new tab)
- [ ] Expand: **"⚙️ Setup Sheets & Start Tracking"**
- [ ] Click: **"🆕 Create New Investment Tracker Sheet"**
- [ ] Copy and save the Google Sheet URL from success message
- [ ] Test: All 4 worksheets appeared in your Google Sheet
  - [ ] Searches
  - [ ] Investments
  - [ ] Daily Analysis
  - [ ] Portfolio

### Phase 3: First Tracking

- [ ] Go to **📊 Trading Dashboard** (home page)
- [ ] Select a stock (e.g., RELIANCE.NS)
- [ ] Click "💾 Log to Sheets" button
- [ ] Verify: Data appears in your Google Sheet "Searches" tab
- [ ] Go back to **📊 Tracking Dashboard**
- [ ] Go to **"💰 Log Investment"** tab
- [ ] Log a test investment
- [ ] Verify: Data appears in "Investments" tab
- [ ] Go to **"📈 Daily Analysis"** tab
- [ ] Log a daily snapshot
- [ ] Verify: Data appears in "Daily Analysis" tab

### Phase 4: Ongoing Usage

- [ ] Daily: Log stock searches
- [ ] After investment: Log in Investment tab
- [ ] End of day: Log Daily Analysis
- [ ] Weekly: Review portfolio holdings
- [ ] Monthly: Export data & analyze trends

---

## 🎯 Quick Commands

```bash
# Start the app
streamlit run app.py

# Install missing packages
pip install gspread google-auth-oauthlib google-auth-httplib2

# Check credentials file exists
ls google_credentials.json  # Unix/Mac
dir google_credentials.json # Windows

# Verify Python syntax
python -m py_compile modules/sheets_tracker.py
```

---

## 📖 Documentation Reference

- **For detailed setup**: Read `SHEETS_SETUP.md`
- **For quick reference**: Read `TRACKING_QUICK_START.md`
- **For overview**: Read `TRACKER_README.md`
- **For this checklist**: You're reading it! ✓

---

## 🆘 If Something Goes Wrong

### ❌ "Credentials not found"
- [ ] Check file exists: `google_credentials.json`
- [ ] Location: Project root folder
- [ ] Filename spelling: exactly `google_credentials.json`
- [ ] Restart app: `Ctrl+C` then `streamlit run app.py`

### ❌ "Permission denied"
- [ ] Go to Google Cloud Console
- [ ] Click Service Accounts
- [ ] Find your account
- [ ] Check role is "Editor"
- [ ] Regenerate key if needed

### ❌ "Sheet URL not working"
- [ ] Copy full URL from browser address bar
- [ ] Should start with: `https://docs.google.com/spreadsheets/`
- [ ] Contains: `/d/` followed by sheet ID
- [ ] Ends with: `/edit` or `/edit#gid=0`

### ❌ "gspread import error"
- [ ] Install: `pip install gspread`
- [ ] Also install: `pip install google-auth-oauthlib`

### ❌ "Data not showing up"
- [ ] Refresh Google Sheet (F5 or Cmd+R)
- [ ] Check tabs exist (Searches, Investments, etc.)
- [ ] Restart Streamlit app
- [ ] Check app console for error messages

---

## 🎉 Success Indicators

When everything is working:

✅ Created Google Sheet called "Investment Tracker - Digitrader"  
✅ Sheet has 4 worksheets (Searches, Investments, Daily Analysis, Portfolio)  
✅ Can log searches from Trading Dashboard  
✅ Can log investments from Tracking Dashboard  
✅ Can view data immediately in Google Sheet  
✅ Can access sheet from Google Sheets app on mobile  

---

## 📱 Mobile Setup (Optional)

- [ ] Download: Google Sheets app (free)
- [ ] Open: Your sheet URL in the app
- [ ] Can view: Portfolio from anywhere
- [ ] Can see: Real-time updates
- [ ] Can share: URL with others

---

## 🔗 Important URLs

- Google Cloud Console: https://console.cloud.google.com/
- Your Sheet: [Will be shown after clicking "Create Sheet"]
- Google Sheets App: https://play.google.com/store/apps/details?id=com.google.android.apps.docs.editors.sheets (Android) / App Store (iOS)

---

## 📝 Environment Variables (Optional)

If you want to save your sheet URL in `.env`:

```env
SHEETS_URL=https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit#gid=0
```

This way app remembers your sheet across restarts.

---

## 🎯 Next Milestones

**Week 1**: Get tracking working, log first trades  
**Week 2**: Build portfolio, track 5-10 stocks  
**Month 1**: Analyze patterns, review prediction accuracy  
**Ongoing**: Share reports, improve strategies  

---

## 💪 You're All Set!

Once you complete Phase 1 & 2, you have a **production-ready real-time tracking system**.

**Start with**: `streamlit run app.py` → **📊 Tracking Dashboard** 🚀

---

**Questions?** See SHEETS_SETUP.md for detailed troubleshooting!

Good luck tracking! 📊💰
