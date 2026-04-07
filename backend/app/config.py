"""
Configuration management for DigiTrader API
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ===== APP SETTINGS =====
APP_NAME = "DigiTrader API v5"
APP_VERSION = "5.0.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# ===== DATABASE =====
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./digitrader.db"
)

# ===== REDIS =====
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# ===== CELERY =====
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# ===== CORS =====
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8501",
    "http://127.0.0.1:3000",
]

# ===== API KEYS =====
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "K5T3L5U9N6QFQLXB")
FINNHUB_KEY = os.getenv("FINNHUB_KEY", "d78dqqhr01qhel7vjal0")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "92a2bc8ddf5f4a6c916643ed8257a621")
GEMINI_KEY = os.getenv("GEMINI_KEY", "AIzaSyDLcz9qeozO9tFNE1fv0mVVZNe-tFj2s-U")

# ===== CACHE SETTINGS =====
PRICE_CACHE_TTL = 60  # 1 minute
SENTIMENT_CACHE_TTL = 300  # 5 minutes
ANALYSIS_CACHE_TTL = 180  # 3 minutes

# ===== WORKERS =====
CELERY_WORKER_COUNT = int(os.getenv("CELERY_WORKER_COUNT", "5"))
ANALYSIS_TIMEOUT = 30  # seconds

# ===== NSE STOCKS =====
NIFTY_50_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HDFC.NS", "BAJAJFINSV.NS", "BHARTIARTL.NS", "LT.NS", "ITC.NS",
    "SBIN.NS", "MARUTI.NS", "LOWADECOM.NS", "SUNPHARMA.NS", "WIPRO.NS",
    "ASIANPAINT.NS", "AXISBANK.NS", "DMARUTI.NS", "NTPC.NS", "POWERGRID.NS",
    "ONGC.NS", "COALINDIA.NS", "BAJAJ-AUTO.NS", "TITAN.NS", "M&MFIN.NS",
    "BPCL.NS", "MULTPL.NS", "KOTAKBANK.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "EICHERMOT.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "GAIL.NS", "GMRINFRA.NS",
    "ADANIGREEN.NS", "ADANIENT.NS", "GODREJCP.NS", "NESTLEIND.NS", "PHARMAIND.NS",
    "SHREECEM.NS", "TECHM.NS", "TATAPOWER.NS", "TATAMOTORS.NS", "INDIGO.NS",
    "RELINFRA.NS", "APOLLOHOSP.NS", "CIPLA.NS", "DRREDDY.NS", "LUPIN.NS"
]
