import os
from dotenv import load_dotenv

load_dotenv()

STOCK_SYMBOL = os.getenv("STOCK_SYMBOL", "RELIANCE.NS")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# Google Sheets Configuration
SHEETS_ID = os.getenv("SHEETS_ID", "")  # Google Sheets spreadsheet ID
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "service_account.json")  # Path to service account JSON
