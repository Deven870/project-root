# 🚀 PRODUCTION DEPLOYMENT GUIDE
**Digitrader - Smart Trading Assistant**

**Version:** 1.0  
**Last Updated:** March 21, 2026  
**Status:** ✅ Ready for Production Deployment

---

## 📋 Pre-Deployment Checklist

### ✅ Step 1: Environment Setup (30 minutes)

- [ ] **Install Python 3.9+**
  ```bash
  python --version  # Should be 3.9 or higher
  ```

- [ ] **Install Dependencies**
  ```bash
  pip install -r requirements.txt
  ```

- [ ] **Create Virtual Environment** (Recommended)
  ```bash
  python -m venv .venv
  # Windows
  .venv\Scripts\activate
  # Mac/Linux
  source .venv/bin/activate
  ```

### ✅ Step 2: Configuration Setup (15 minutes)

- [ ] **Copy .env.example to .env**
  ```bash
  cp .env.example .env
  # or on Windows
  copy .env.example .env
  ```

- [ ] **Fill in .env with Your Values**
  ```bash
  STOCK_SYMBOL=RELIANCE.NS              # (or your preferred stock)
  NEWS_API_KEY=your_key_here            # Get from https://newsapi.org/
  ENABLE_SENTIMENT_ANALYSIS=true        # Optional
  LOG_LEVEL=INFO                        # INFO for production
  DEBUG_MODE=false                      # IMPORTANT: Set to false
  ```

### ✅ Step 3: Google Sheets Integration (Optional but Recommended)

- [ ] **Create Google Cloud Project**
  - Go to https://console.cloud.google.com/
  - Click "Select a Project" → "NEW PROJECT"
  - Name: `Digitrader` or your choice
  - Click CREATE

- [ ] **Enable Required APIs**
  - In Cloud Console, search for "Google Sheets API" → ENABLE
  - Search for "Google Drive API" → ENABLE

- [ ] **Create Service Account**
  - Click "Service Accounts" in left menu
  - Click "CREATE SERVICE ACCOUNT"
  - Name: `digitrader-sheets`
  - Click "CREATE AND CONTINUE"
  - Skip role assignment, click CONTINUE
  - Click "CREATE KEY" → Select JSON → CREATE
  - Downloaded file: Rename to `google_credentials.json`

- [ ] **Save Credentials File**
  ```bash
  # Move google_credentials.json to project root
  mv ~/Downloads/service-account-key.json ./google_credentials.json
  ```

- [ ] **Update .env with Sheet URL** (Optional)
  ```bash
  SHEETS_URL=https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit
  GOOGLE_CREDENTIALS_PATH=./google_credentials.json
  ```

### ✅ Step 4: Validate Configuration

- [ ] **Run Configuration Validation**
  ```bash
  python modules/config_validator.py
  ```
  Expected output: `✅ All validations passed - Ready for deployment!`

- [ ] **Check All Required Files Exist**
  ```bash
  # Windows
  dir requirements.txt
  dir app.py
  dir modules/
  dir config.py
  dir .env
  
  # Mac/Linux
  ls requirements.txt
  ls app.py
  ls modules/
  ls config.py
  ls .env
  ```

---

## 🧪 Testing Before Deployment

### ✅ Step 5: Local Testing

- [ ] **Test Dashboard Locally**
  ```bash
  streamlit run app.py
  ```
  - Navigate to: http://localhost:8501
  - Check all tabs load without errors
  - Test stock search functionality
  - Verify P&L tracking (if sheets configured)

- [ ] **Test Backtesting Module**
  ```bash
  python run_live_backtest.py
  ```
  - Should complete without errors
  - Check output files in `results/`

- [ ] **Test Data Fetching**
  ```bash
  python quick_accuracy_test.py
  ```
  - Should fetch real data or use fallback
  - Verify accuracy percentages

- [ ] **Check Logs**
  ```bash
  # Should see logs in:
  cat logs/digitrader.log
  ```

---

## 🌐 Cloud Deployment Options

### Option A: Streamlit Cloud (Easiest - Free for public apps)

**Pros:** Simple, free, auto-scaling  
**Cons:** Public by default, limited resources

**Steps:**
1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Click "New app" → Connect GitHub repo
4. Select main branch and `app.py`
5. Add secrets in Streamlit Cloud dashboard:
   ```
   NEWS_API_KEY = your_key
   STOCK_SYMBOL = RELIANCE.NS
   ```
6. Deploy!

**Set Environment Variables in Streamlit Cloud:**
```
# .streamlit/secrets.toml (in Streamlit Cloud)
STOCK_SYMBOL = "RELIANCE.NS"
NEWS_API_KEY = "your_newsapi_key"
GOOGLE_CREDENTIALS_PATH = "./google_credentials.json"
LOG_LEVEL = "INFO"
DEBUG_MODE = false
```

### Option B: Docker + Cloud Run (GCP)

**Pros:** Scalable, isolated, reliable  
**Cons:** Requires Docker knowledge

**Create Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_CLIENT_SHOWERRORDETAILS=false

CMD ["streamlit", "run", "app.py"]
```

**Deploy to Cloud Run:**
```bash
gcloud run deploy digitrader \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Option C: AWS EC2 + Gunicorn

**Pros:** Full control, scalable  
**Cons:** More setup required

**Create systemd service file:**
```bash
# /etc/systemd/system/digitrader.service
[Unit]
Description=Digitrader Streamlit App
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/project-root
Environment="PATH=/home/ec2-user/project-root/.venv/bin"
ExecStart=/home/ec2-user/project-root/.venv/bin/streamlit run app.py --server.port 80
Restart=always

[Install]
WantedBy=multi-user.target
```

**Start service:**
```bash
sudo systemctl start digitrader
sudo systemctl enable digitrader
```

### Option D: Railway.app (Simple & Fast)

**Pros:** Very simple setup, good free tier  
**Cons:** Limited by platform capabilities

1. Go to https://railway.app
2. Create new project
3. Connect your GitHub repo
4. Add environment variables
5. Deploy!

---

## 🔒 Production Security Checklist

### ✅ Security Configuration

- [ ] **Set DEBUG_MODE = false in .env**
  ```bash
  DEBUG_MODE=false
  ```

- [ ] **Set STREAMLIT_CLIENT_SHOWERRORDETAILS = false**
  ```bash
  STREAMLIT_CLIENT_SHOWERRORDETAILS=false
  ```

- [ ] **Never commit .env file to Git**
  ```bash
  echo ".env" >> .gitignore
  echo "google_credentials.json" >> .gitignore
  git add .gitignore
  ```

- [ ] **Use Environment Variables for Secrets**
  - Never hardcode API keys
  - Always load from .env or environment

- [ ] **Restrict Google Credentials Permissions**
  - Only enable Sheets API
  - Only enable Drive API (if needed)
  - Disable other APIs

- [ ] **Use HTTPS Only**
  - Most cloud platforms handle this automatically
  - Test with `https://your-app-url.com`

- [ ] **Set Up Rate Limiting** (for API calls)
  - NewsAPI has rate limits (check documentation)
  - Implement caching where possible

---

## 📊 Monitoring & Maintenance

### ✅ Set Up Monitoring

- [ ] **Check Application Logs**
  ```bash
  tail -f logs/digitrader.log
  ```

- [ ] **Monitor Trade Execution** (if enabled)
  ```bash
  tail -f logs/trades.log
  ```

- [ ] **Set Up Error Alerts**
  - Consider email notifications for critical errors
  - Set up Slack webhook for alerts

- [ ] **Monitor Resource Usage**
  - CPU usage (should be < 50%)
  - Memory usage (should be < 2GB)
  - Disk space (ensure grows < 100MB/month)

### ✅ Regular Maintenance

- [ ] **Daily**: Check application logs for errors
- [ ] **Weekly**: Validate data accuracy on 2-3 stocks
- [ ] **Monthly**: 
  - Review backtest results
  - Update any deprecated dependencies
  - Clean up old log files

- [ ] **Quarterly**:
  - Review ML model accuracy
  - Test disaster recovery procedures
  - Update security patches

---

## 🐛 Troubleshooting Deployment

### Issue: "Credentials not found"
**Solution:**
```bash
# Check if google_credentials.json exists
ls google_credentials.json

# If missing, download from Google Cloud
# Then restart application
```

### Issue: "NEWS_API_KEY not configured"
**Solution:**
```bash
# Get free key from:
# https://newsapi.org/register

# Add to .env
NEWS_API_KEY=your_api_key_here

# Restart application
```

### Issue: Application runs slowly
**Solution:**
- Check LOG_LEVEL setting (set to WARNING in production)
- Increase instance size (if on cloud)
- Enable caching for API calls

### Issue: "Memory exceeded" error
**Solution:**
- Reduce DEFAULT_LOOKBACK_DAYS in .env
- Clear old logs: `rm logs/*.log.*`
- Restart application

---

## ✅ Final Deployment Checklist

Before going live:

- [ ] All tests pass locally
- [ ] Configuration validated (run `python modules/config_validator.py`)
- [ ] .env file configured with real credentials
- [ ] google_credentials.json placed in project root
- [ ] .env added to .gitignore
- [ ] DEBUG_MODE = false
- [ ] LOG_LEVEL = INFO (or WARNING)
- [ ] All required Python packages installed
- [ ] Code pushed to Git (if using Streamlit Cloud or similar)
- [ ] Secrets configured on cloud platform
- [ ] HTTPS enabled
- [ ] Test the live application with real browser
- [ ] Monitor logs for first 24 hours
- [ ] Set up automated backups

---

## 📞 Support & Documentation

- **Dashboard Guide:** See [TRACKER_README.md](TRACKER_README.md)
- **Google Sheets Setup:** See [SHEETS_SETUP.md](SHEETS_SETUP.md)
- **Quick Start:** See [TRACKING_QUICK_START.md](TRACKING_QUICK_START.md)
- **Accuracy Report:** See [DEPLOYMENT_REPORT.md](DEPLOYMENT_REPORT.md)

---

## 🎉 Deployment Complete!

Once deployed:
1. Share the URL with users
2. Set up monitoring alerts
3. Schedule regular maintenance tasks
4. Collect user feedback
5. Plan feature improvements

---

**Questions?** Check the documentation files or review the code comments.

**Ready to deploy?** Start with **Step 1** above! 🚀
