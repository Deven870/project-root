import os
from dotenv import load_dotenv

load_dotenv()

STOCK_SYMBOL = os.getenv("STOCK_SYMBOL", "RELIANCE.NS")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
