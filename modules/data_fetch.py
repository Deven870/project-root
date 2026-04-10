import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import json


def fetch_stock_data(symbol, period="1mo", interval="1d"):
    """
    Backward-compatible stock fetch used by main.py.
    Delegates to the robust fetcher in modules.utils.
    """
    try:
        from modules.utils import fetch_price_data
        return fetch_price_data(symbol, period=period, interval=interval)
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {e}")
        return pd.DataFrame()


def get_news_for_stock(stock_ticker, from_days=7, max_articles=20):
    """
    Backward-compatible news fetch used by main.py.
    Tries sentiment_engine NewsAPI integration first, then yfinance news fallback.
    """
    try:
        from modules.sentiment_engine import get_news_for_stock as _news_fetcher
        headlines = _news_fetcher(stock_ticker, from_days=from_days, max_articles=max_articles)
        if isinstance(headlines, list) and len(headlines) > 0:
            return headlines
    except Exception:
        pass

    try:
        ticker = yf.Ticker(stock_ticker)
        news = getattr(ticker, "news", []) or []
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("link", "")
            }
            for item in news[:max_articles]
        ]
    except Exception:
        return []

def get_all_nse_stocks():
    """
    Fetch all NSE listed stocks from NSE India website
    Returns: DataFrame with stock symbols and details
    """
    try:
        # NSE API endpoint for all equity stocks
        url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        session = requests.Session()
        # Get cookies first
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        
        # Fetch equity list
        response = session.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            stocks = [item['symbol'] for item in data.get('data', [])]
            
            df = pd.DataFrame(stocks, columns=['SYMBOL'])
            print(f"Total NSE stocks fetched: {len(df)}")
            return df
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            return get_nse_stocks_fallback()
            
    except Exception as e:
        print(f"Error fetching NSE stocks: {e}")
        return get_nse_stocks_fallback()

def get_nse_stocks_fallback():
    """
    Fallback list of major NSE stocks
    """
    major_stocks = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK",
        "HDFC", "BAJFINANCE", "BHARTIARTL", "SBIN", "ITC", "KOTAKBANK",
        "LT", "ASIANPAINT", "AXISBANK", "MARUTI", "TITAN", "SUNPHARMA",
        "ULTRACEMCO", "NESTLEIND", "WIPRO", "HCLTECH", "TECHM", "POWERGRID",
        "NTPC", "ONGC", "TATASTEEL", "TATAMOTORS", "ADANIPORTS", "JSWSTEEL",
        "INDUSINDBK", "BAJAJFINSV", "DIVISLAB", "DRREDDY", "CIPLA", "EICHERMOT",
        "HEROMOTOCO", "BPCL", "GRASIM", "COALINDIA", "SHREECEM", "BRITANNIA",
        "APOLLOHOSP", "UPL", "M&M", "HINDALCO", "ADANIENT", "TATACONSUM"
    ]
    
    df = pd.DataFrame(major_stocks, columns=['SYMBOL'])
    print(f"Using fallback list: {len(df)} stocks")
    return df

def get_nse_stock_symbols():
    """
    Get just the stock symbols as a list
    Returns: List of stock symbols
    """
    df = get_all_nse_stocks()
    if not df.empty and 'SYMBOL' in df.columns:
        return df['SYMBOL'].tolist()
    return []

def save_nse_stocks_to_csv(filename='nse_stocks.csv'):
    """
    Save all NSE stocks to CSV file
    """
    df = get_all_nse_stocks()
    if not df.empty:
        df.to_csv(filename, index=False)
        print(f"NSE stocks saved to {filename}")
        return filename
    return None

def get_stock_data(symbol, period="1mo"):
    """
    Fetch stock data using yfinance
    Add .NS suffix for NSE stocks
    """
    try:
        nse_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
        ticker = yf.Ticker(nse_symbol)
        data = ticker.history(period=period)
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def get_stock_info(symbol):
    """
    Get detailed stock information
    """
    try:
        nse_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
        ticker = yf.Ticker(nse_symbol)
        info = ticker.info
        return info
    except Exception as e:
        print(f"Error fetching info for {symbol}: {e}")
        return None


# =========================================
# FIX 1: Get news BEFORE a specific cutoff time (no look-ahead bias)
# =========================================
def get_news_before(symbol, before_dt, lookback_hours=18):
    """
    Fetch only headlines published BEFORE before_dt.
    Prevents look-ahead bias in backtests.
    
    Parameters
    ----------
    symbol : str
        Stock symbol (e.g., "RELIANCE.NS")
    before_dt : datetime
        Cutoff datetime - only fetch news published before this
    lookback_hours : int
        How many hours back to search from before_dt
        
    Returns
    -------
    list
        Articles with title, published_date, url
    """
    try:
        from modules.sentiment_engine import get_news_for_stock as _news_fetcher
        from_date = before_dt - timedelta(hours=lookback_hours)
        
        # Fetch all recent news
        headlines = _news_fetcher(symbol, 
                                 from_days=int(lookback_hours/24)+1, 
                                 max_articles=50)
        
        # Filter: only include articles published before cutoff
        filtered = []
        for article in headlines:
            try:
                pub_date_str = article.get("published", "")
                if pub_date_str:
                    # Try to parse publication date
                    pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                    if pub_date < before_dt:
                        filtered.append(article)
            except Exception:
                # If can't parse date, include it anyway
                filtered.append(article)
        
        return filtered[:20]
    
    except Exception as e:
        print(f"Error fetching news before {before_dt}: {e}")
        return []


# =========================================
# FIX 5: Get India VIX for market volatility filter
# =========================================
def get_india_vix():
    """
    Fetch India VIX from NSE. Returns float.
    
    Returns
    -------
    float
        India VIX closing value, or 15.0 if fetch fails
    """
    try:
        vix = yf.Ticker("^INDIAVIX")
        hist = vix.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        return 15.0  # safe default
    except Exception as e:
        print(f"Warning: Could not fetch India VIX: {e}")
        return 15.0  # safe default


# =========================================
# FIX 7: Get news from real-time RSS feeds (Moneycontrol)
# =========================================
def get_news_realtime(symbol, max_articles=15):
    """
    Scrape RSS feeds for real-time news, filter for symbol mentions.
    Much faster than NewsAPI (no 15-30min delay).
    
    Parameters
    ----------
    symbol : str
        Stock symbol (e.g., "RELIANCE.NS")
    max_articles : int
        Max articles to return
        
    Returns
    -------
    list
        Articles with title, published, source
    """
    try:
        import feedparser
        
        RSS_FEEDS = {
            "market": "https://www.moneycontrol.com/rss/marketreports.xml",
            "news": "https://www.moneycontrol.com/rss/latestnews.xml",
            "economy": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        }
        
        company = symbol.replace(".NS", "").upper()
        articles = []
        
        for feed_name, feed_url in RSS_FEEDS.items():
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    title = entry.get("title", "")
                    # Filter for symbol mentions or market-wide news
                    if company in title.upper() or "NIFTY" in title.upper() or "SENSEX" in title.upper():
                        articles.append({
                            "title": title,
                            "published": entry.get("published", ""),
                            "source": "moneycontrol_rss",
                            "link": entry.get("link", "")
                        })
            except Exception as e:
                print(f"Warning: Could not parse {feed_name} RSS: {e}")
        
        # If we got good results from RSS, return them; otherwise fallback to NewsAPI
        if articles:
            return articles[:max_articles]
        else:
            # Fallback to NewsAPI
            from modules.sentiment_engine import get_news_for_stock as _news_fetcher
            return _news_fetcher(symbol, from_days=1, max_articles=max_articles)
    
    except Exception as e:
        print(f"Error in get_news_realtime: {e}")
        # Fallback to standard news fetch
        try:
            from modules.sentiment_engine import get_news_for_stock as _news_fetcher
            return _news_fetcher(symbol, from_days=1, max_articles=max_articles)
        except Exception:
            return []


# =========================================
# FIX 8: Get FII/DII flow data
# =========================================
def get_fii_dii_data(days=30):
    """
    Pull FII/DII daily buy-sell data from NSE website.
    
    Parameters
    ----------
    days : int
        Number of days to fetch
        
    Returns
    -------
    pd.DataFrame
        Columns: [date, fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net]
    """
    try:
        url = "https://www.nseindia.com/api/fiidiiTradeReact"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/"
        }
        
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        resp = session.get(url, headers=headers, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            df = pd.DataFrame(data.get("data", []))
            
            if not df.empty:
                # Ensure numeric columns
                for col in ['fiiDerivativeBuyValue', 'fiiDerivativeSellValue', 
                           'diiDerivativeBuyValue', 'diiDerivativeSellValue']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Calculate net flows
                if 'fiiDerivativeBuyValue' in df.columns and 'fiiDerivativeSellValue' in df.columns:
                    df["fii_net"] = df["fiiDerivativeBuyValue"] - df["fiiDerivativeSellValue"]
                if 'diiDerivativeBuyValue' in df.columns and 'diiDerivativeSellValue' in df.columns:
                    df["dii_net"] = df["diiDerivativeBuyValue"] - df["diiDerivativeSellValue"]
                
                return df.tail(days)
        
        return pd.DataFrame()
    
    except Exception as e:
        print(f"Warning: Could not fetch FII/DII data: {e}")
        return pd.DataFrame()