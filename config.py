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

# ===== Google Sheets Settings =====
GOOGLE_SHEETS_ID = os.getenv("GOOGLE_SHEETS_ID", "")
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
SHEETS_URL = os.getenv("SHEETS_URL", "")

# ===== Telegram & Email Alerts =====
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")

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
