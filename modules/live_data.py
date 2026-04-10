"""
FIX 10: Add Angel One SmartAPI for live price data
Provides real-time NSE tick data when market is open.
"""

import os
from datetime import datetime, time
from dotenv import load_dotenv

load_dotenv()

# Angel One API credentials
ANGEL_API_KEY = os.getenv("ANGEL_API_KEY", "")
ANGEL_CLIENT_ID = os.getenv("ANGEL_CLIENT_ID", "")
ANGEL_MPIN = os.getenv("ANGEL_MPIN", "")
ANGEL_TOTP_KEY = os.getenv("ANGEL_TOTP_KEY", "")

# Symbol token mapping (NSE stock -> Angel One token)
SYMBOL_TOKENS = {
    "RELIANCE": "2885",
    "RELIANCE.NS": "2885",
    "TCS": "3456",
    "TCS.NS": "3456",
    "HDFCBANK": "1270",
    "HDFCBANK.NS": "1270",
    "INFY": "9124",
    "INFY.NS": "9124",
    "ICICIBANK": "5228",
    "ICICIBANK.NS": "5228",
    "WIPRO": "11537",
    "WIPRO.NS": "11537",
    "BAJFINANCE": "1506",
    "BAJFINANCE.NS": "1506",
    "SBIN": "18973",
    "SBIN.NS": "18973",
    "MARUTI": "32735",
    "MARUTI.NS": "32735",
    "TATAMOTORS": "3499",
    "TATAMOTORS.NS": "3499",
    "ITC": "3002",
    "ITC.NS": "3002",
}


def is_market_open():
    """Check if NSE market is currently open (9:15-15:30 IST)."""
    import pytz
    tz = pytz.timezone("Asia/Kolkata")
    now = datetime.now(tz)
    
    # Market closed on weekends
    if now.weekday() >= 5:
        return False
    
    market_open = time(9, 15)
    market_close = time(15, 30)
    
    return market_open <= now.time() <= market_close


def get_angel_connection():
    """
    Establish connection to Angel One SmartAPI.
    
    Returns
    -------
    SmartConnect object or None if connection fails
    """
    try:
        if not all([ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_MPIN, ANGEL_TOTP_KEY]):
            print("Warning: Angel One credentials not fully configured in .env")
            return None
        
        try:
            from SmartApi import SmartConnect
            import pyotp
        except ImportError:
            print("Warning: SmartApi package not installed. Install via: pip install smartapi-python pyotp")
            return None
        
        # Generate TOTP token
        totp = pyotp.TOTP(ANGEL_TOTP_KEY).now()
        
        # Connect
        obj = SmartConnect(api_key=ANGEL_API_KEY)
        data = obj.generateSession(ANGEL_CLIENT_ID, ANGEL_MPIN, totp)
        
        if data.get("status"):
            print(f"✓ Angel One connection successful")
            return obj
        else:
            print(f"✗ Angel One connection failed: {data.get('message', 'Unknown error')}")
            return None
    
    except Exception as e:
        print(f"Angel One connection error: {e}")
        return None


def get_live_price(symbol):
    """
    Fetch current LTP (Last Traded Price) for a symbol.
    
    Parameters
    ----------
    symbol : str
        Stock symbol (e.g., "RELIANCE.NS" or "RELIANCE")
        
    Returns
    -------
    float
        Current price, or 0.0 if fetch fails
    """
    if not is_market_open():
        print(f"Warning: Market is currently closed. Cannot fetch live price for {symbol}")
        return 0.0
    
    try:
        # Get symbol token
        sym_clean = symbol.replace(".NS", "").upper()
        symbol_token = SYMBOL_TOKENS.get(symbol) or SYMBOL_TOKENS.get(sym_clean)
        
        if not symbol_token:
            print(f"Warning: Symbol token not found for {symbol}")
            return 0.0
        
        # Get connection
        conn = get_angel_connection()
        if not conn:
            return 0.0
        
        # Fetch LTP
        ltp_data = conn.ltpData("NSE", symbol_token, symbol_token)
        
        if ltp_data.get("status"):
            ltp = float(ltp_data.get("data", {}).get("ltp", 0.0))
            print(f"✓ Live price for {symbol}: ₹{ltp:,.2f}")
            return ltp
        else:
            print(f"✗ Could not fetch LTP for {symbol}: {ltp_data.get('message', 'Unknown')}")
            return 0.0
    
    except Exception as e:
        print(f"Error fetching live price: {e}")
        return 0.0


def get_live_ohlcv_1min(symbol, from_dt, to_dt):
    """
    Fetch 1-minute OHLCV candles for intraday use.
    
    Parameters
    ----------
    symbol : str
        Stock symbol
    from_dt : datetime
        Start datetime
    to_dt : datetime
        End datetime
        
    Returns
    -------
    pd.DataFrame
        OHLCV data with 1-minute candles
    """
    try:
        import pandas as pd
        
        if not is_market_open():
            print("Market is closed, cannot fetch live 1-min candles")
            return pd.DataFrame()
        
        sym_clean = symbol.replace(".NS", "").upper()
        symbol_token = SYMBOL_TOKENS.get(symbol) or SYMBOL_TOKENS.get(sym_clean)
        
        if not symbol_token:
            print(f"Symbol token not found for {symbol}")
            return pd.DataFrame()
        
        conn = get_angel_connection()
        if not conn:
            return pd.DataFrame()
        
        # Fetch candles
        candles = conn.getCandleData(
            "NSE",
            symbol_token,
            "1minute",
            from_dt.isoformat(),
            to_dt.isoformat()
        )
        
        if candles.get("status"):
            data = candles.get("data", {})
            candle_list = data.get("candles", [])
            
            if not candle_list:
                print(f"No candle data available for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(candle_list, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df.columns = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df.set_index("Timestamp", inplace=True)
            
            print(f"✓ Fetched {len(df)} 1-minute candles for {symbol}")
            return df
        else:
            print(f"Error fetching candles: {candles.get('message', 'Unknown')}")
            return pd.DataFrame()
    
    except Exception as e:
        print(f"Error in get_live_ohlcv_1min: {e}")
        return pd.DataFrame()


def fetch_live_or_cached_price(symbol, use_cache_fallback=True):
    """
    Fetch live price during market hours, use yfinance fallback outside market hours.
    
    Parameters
    ----------
    symbol : str
        Stock symbol
    use_cache_fallback : bool
        If True and Angel One fails, fallback to yfinance
        
    Returns
    -------
    float
        Current price
    """
    if is_market_open():
        price = get_live_price(symbol)
        if price > 0:
            return price
    
    # Fallback to yfinance
    if use_cache_fallback:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol if symbol.endswith('.NS') else f"{symbol}.NS")
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception as e:
            print(f"YFinance fallback error: {e}")
    
    return 0.0
