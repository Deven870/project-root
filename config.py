"""
Application Configuration
Loads and provides access to environment variables and settings.
All settings are validated at startup via config_validator.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ===== Stock Market Settings =====
STOCK_SYMBOL = os.getenv("STOCK_SYMBOL", "RELIANCE.NS")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
DEFAULT_LOOKBACK_DAYS = int(os.getenv("DEFAULT_LOOKBACK_DAYS", "365"))

# ===== Watchlist & Portfolio =====
WATCHLIST = os.getenv("WATCHLIST", "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ITC.NS").split(",")
STARTING_CAPITAL = float(os.getenv("STARTING_CAPITAL", "100000"))
MAX_RISK_PCT = float(os.getenv("MAX_RISK_PCT", "0.02"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.65"))
VIX_THRESHOLD = float(os.getenv("VIX_THRESHOLD", "20"))
MIN_POSITIVE_SENTIMENT = float(os.getenv("MIN_POSITIVE_SENTIMENT", "0.20"))

# ===== Google Sheets Settings =====
GOOGLE_SHEETS_ID = os.getenv("GOOGLE_SHEETS_ID", "")
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
SHEETS_URL = os.getenv("SHEETS_URL", "")

# ===== Telegram & Email Alerts =====
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")

# ===== Angel One SmartAPI (FIX 10) =====
ANGEL_API_KEY = os.getenv("ANGEL_API_KEY", "")
ANGEL_CLIENT_ID = os.getenv("ANGEL_CLIENT_ID", "")
ANGEL_MPIN = os.getenv("ANGEL_MPIN", "")
ANGEL_TOTP_KEY = os.getenv("ANGEL_TOTP_KEY", "")

# ===== Excel Tracking (FIX 11) =====
EXCEL_TRACKER_PATH = os.getenv("EXCEL_TRACKER_PATH", "Digitrader_PaperTrading.xlsx")

# ===== Feature Flags =====
ENABLE_SENTIMENT_ANALYSIS = os.getenv("ENABLE_SENTIMENT_ANALYSIS", "true").lower() == "true"
ENABLE_BACKTESTING = os.getenv("ENABLE_BACKTESTING", "true").lower() == "true"
ENABLE_LIVE_TRADING = os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true"
USE_SYNTHETIC_DATA = os.getenv("USE_SYNTHETIC_DATA", "false").lower() == "true"

# ===== Logging Settings =====
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/digitrader.log")

# ===== Debug Settings =====
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# ===== Broker Integration (Future) =====
ZERODHA_API_KEY = os.getenv("ZERODHA_API_KEY", "")
ZERODHA_SECRET = os.getenv("ZERODHA_SECRET", "")
BROKER_TYPE = os.getenv("BROKER_TYPE", "")


# =========================================
# Configuration Validation
# =========================================

def validate_config():
    """
    Validate that all required configuration is present.
    Prints warnings for missing optional configs.
    """
    errors = []
    warnings = []
    
    # Required keys
    required_keys = {
        "NEWS_API_KEY": ("NewsAPI key", NEWS_API_KEY),
        "FINNHUB_API_KEY": ("Finnhub API key", FINNHUB_API_KEY),
    }
    
    # Optional but strongly recommended
    optional_keys = {
        "TELEGRAM_BOT_TOKEN": ("Telegram bot token (for alerts)", TELEGRAM_BOT_TOKEN),
        "TELEGRAM_CHAT_ID": ("Telegram chat ID (for alerts)", TELEGRAM_CHAT_ID),
        "ANGEL_API_KEY": ("Angel One API key (for live prices)", ANGEL_API_KEY),
        "EXCEL_TRACKER_PATH": ("Excel tracker path", EXCEL_TRACKER_PATH),
    }
    
    # Check required
    print("\n" + "="*60)
    print("DIGITRADER CONFIGURATION VALIDATION")
    print("="*60)
    
    print("\n📋 Required Configuration:")
    for key, (desc, value) in required_keys.items():
        if value:
            print(f"  ✓ {desc}")
        else:
            errors.append(f"Missing required: {desc} (set {key} in .env)")
            print(f"  ✗ {desc} - MISSING")
    
    # Check optional
    print("\n📌 Optional Configuration:")
    for key, (desc, value) in optional_keys.items():
        if value:
            print(f"  ✓ {desc}")
        else:
            warnings.append(f"Missing optional: {desc} (set {key} in .env)")
            print(f"  ⚠ {desc} - NOT SET")
    
    # Print summary
    print("\n" + "-"*60)
    if errors:
        print(f"⚠️  ERRORS ({len(errors)}):")
        for error in errors:
            print(f"   • {error}")
        print("\nPlease fix these errors before running the app.")
        return False
    
    if warnings:
        print(f"ℹ️  WARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"   • {warning}")
        print("\nThe app will run but with limited functionality.")
    
    print("\n✅ Configuration is valid!")
    print("="*60 + "\n")
    return True


# Run validation on module load
if __name__ != "__main__":
    validate_config()
