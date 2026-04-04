"""
Alpha Vantage API Client
Real-time data fetching for NSE stocks
API Key: K5T3L5U9N6QFQLXB
"""

import requests
import pandas as pd
from datetime import datetime
import time
from functools import lru_cache

API_KEY = "K5T3L5U9N6QFQLXB"
BASE_URL = "https://www.alphavantage.co/query"

# Rate limiting: Alpha Vantage free tier = 5 API calls/min
RATE_LIMIT = 0.2  # 5 calls per second = 1 call per 0.2 seconds
last_request_time = 0


def rate_limited_request(func):
    """Decorator for rate limiting API calls"""
    def wrapper(*args, **kwargs):
        global last_request_time
        elapsed = time.time() - last_request_time
        if elapsed < RATE_LIMIT:
            time.sleep(RATE_LIMIT - elapsed)
        last_request_time = time.time()
        return func(*args, **kwargs)
    return wrapper


@rate_limited_request
def get_quote(symbol):
    """Get latest price quote for a stock"""
    try:
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": API_KEY
        }
        response = requests.get(BASE_URL, params=params, timeout=5)
        data = response.json()
        
        if "Global Quote" in data and data["Global Quote"].get("05. price"):
            quote = data["Global Quote"]
            return {
                "symbol": symbol,
                "price": float(quote.get("05. price", 0)),
                "change": float(quote.get("09. change", 0)),
                "change_pct": float(quote.get("10. change percent", "0").replace("%", "")),
                "volume": int(quote.get("06. volume", 0)),
                "timestamp": datetime.now().isoformat()
            }
        return None
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return None


@rate_limited_request
def get_daily_data(symbol, outputsize="full"):
    """Get daily OHLCV data for technical analysis"""
    try:
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": API_KEY,
            "outputsize": outputsize
        }
        response = requests.get(BASE_URL, params=params, timeout=5)
        data = response.json()
        
        if "Time Series (Daily)" in data:
            ts = data["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(ts, orient='index')
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Convert to float
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = df[col].astype(float)
            df['Volume'] = df['Volume'].astype(int)
            
            return df
        return None
    except Exception as e:
        print(f"❌ Error fetching daily data for {symbol}: {e}")
        return None


@rate_limited_request
def get_intraday_data(symbol, interval="15min"):
    """Get intraday data for quick trading signals"""
    try:
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "apikey": API_KEY
        }
        response = requests.get(BASE_URL, params=params, timeout=5)
        data = response.json()
        
        key = f"Time Series ({interval})"
        if key in data:
            ts = data[key]
            df = pd.DataFrame.from_dict(ts, orient='index')
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = df[col].astype(float)
            df['Volume'] = df['Volume'].astype(int)
            
            return df
        return None
    except Exception as e:
        print(f"❌ Error fetching intraday data for {symbol}: {e}")
        return None


def calculate_rsi(prices, period=14):
    """Calculate RSI (Relative Strength Index)"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    except:
        return None


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    try:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        histogram = macd - macd_signal
        return {
            "macd": macd.iloc[-1],
            "signal": macd_signal.iloc[-1],
            "histogram": histogram.iloc[-1]
        }
    except:
        return None


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    try:
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=std_dev).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return {
            "upper": upper.iloc[-1],
            "middle": sma.iloc[-1],
            "lower": lower.iloc[-1]
        }
    except:
        return None


def get_technical_indicators(symbol):
    """Get comprehensive technical indicators for a stock"""
    try:
        df = get_daily_data(symbol, outputsize="compact")
        if df is None or df.empty:
            return None
        
        prices = df['Close']
        
        # Calculate indicators
        rsi = calculate_rsi(prices)
        macd = calculate_macd(prices)
        bb = calculate_bollinger_bands(prices)
        
        # Latest price and volatility
        latest_price = prices.iloc[-1]
        volatility = prices.pct_change().std() * 100
        
        return {
            "symbol": symbol,
            "price": latest_price,
            "rsi": rsi,
            "macd": macd,
            "bollinger_bands": bb,
            "volatility": volatility,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"❌ Error calculating indicators for {symbol}: {e}")
        return None


def get_batch_quotes(symbols):
    """Get quotes for multiple symbols (respecting rate limit)"""
    results = []
    for symbol in symbols:
        quote = get_quote(symbol)
        if quote:
            results.append(quote)
    return results


def test_connection():
    """Test API connection"""
    try:
        quote = get_quote("RELIANCE.NS")
        if quote:
            return True, f"✅ Connected. RELIANCE.NS: ₹{quote['price']}"
        else:
            return False, "❌ No data returned"
    except Exception as e:
        return False, f"❌ Connection failed: {e}"
