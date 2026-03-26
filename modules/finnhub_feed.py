"""
Finnhub API integration for real-time stock data and news.
Supports both NSE (India) and US stock markets.
"""
import finnhub
import os
from datetime import datetime, time
import pytz
from dotenv import load_dotenv

load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# Initialize Finnhub client
try:
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
except Exception as e:
    print(f"Warning: Finnhub client initialization failed: {e}")
    finnhub_client = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _normalize_symbol(symbol):
    """
    Convert NSE symbols to Finnhub format.
    
    Examples:
    - "RELIANCE.NS" → "NSE:RELIANCE"
    - "RELIANCE" → "NSE:RELIANCE"
    - "AAPL" → "AAPL" (unchanged, US symbols)
    - "AAPL.US" → "AAPL" (remove .US suffix)
    """
    if not symbol:
        return ""
    
    # Remove trailing .US suffix if present
    if symbol.endswith(".US"):
        symbol = symbol[:-3]
    
    # Convert NSE format
    if symbol.endswith(".NS"):
        base_symbol = symbol[:-3]
        return f"NSE:{base_symbol}"
    elif ":" in symbol:
        # Already in exchange:symbol format
        return symbol
    elif symbol.isupper() and len(symbol) <= 5:
        # Check if it looks like a US symbol
        if symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD"]:
            return symbol
        # Check if it could be NSE (no dots, all caps, short)
        if not any(c.isdigit() for c in symbol):
            # Assume it's NSE
            return f"NSE:{symbol}"
    
    return symbol


def is_market_open(market="NSE"):
    """
    Check if market is currently open.
    
    Parameters
    ----------
    market : str
        "NSE" for India (9:15-15:30 IST, Mon-Fri)
        "US" for USA (9:30-16:00 EST, Mon-Fri)
    
    Returns
    -------
    bool
        True if market is open, False otherwise
    """
    try:
        if market.upper() == "NSE":
            # NSE: 9:15 AM to 3:30 PM IST, Monday-Friday
            tz = pytz.timezone("Asia/Kolkata")
            now = datetime.now(tz)
            market_open = time(9, 15)
            market_close = time(15, 30)
            is_weekday = now.weekday() < 5  # Monday=0, Friday=4
            return is_weekday and market_open <= now.time() <= market_close
        
        elif market.upper() == "US":
            # US: 9:30 AM to 4:00 PM EST, Monday-Friday
            tz = pytz.timezone("America/New_York")
            now = datetime.now(tz)
            market_open = time(9, 30)
            market_close = time(16, 0)
            is_weekday = now.weekday() < 5  # Monday=0, Friday=4
            return is_weekday and market_open <= now.time() <= market_close
        
        else:
            print(f"Unknown market: {market}")
            return False
    
    except Exception as e:
        print(f"Error checking market status: {e}")
        return False


# ============================================================================
# COMPANY NEWS
# ============================================================================

def get_company_news(symbol, from_date=None, to_date=None):
    """
    Fetch company news for a given symbol.
    
    Parameters
    ----------
    symbol : str
        Stock symbol (e.g., "RELIANCE.NS", "AAPL", "MSFT")
    from_date : str, optional
        Start date in "YYYY-MM-DD" format
    to_date : str, optional
        End date in "YYYY-MM-DD" format
    
    Returns
    -------
    list of dict
        Each dict has: headline, summary, url, datetime, source
        Returns empty list on error
    """
    try:
        if not finnhub_client:
            print("Finnhub client not initialized")
            return []
        
        # Normalize symbol to Finnhub format
        normalized_symbol = _normalize_symbol(symbol)
        
        # If dates not provided, use today
        if not from_date or not to_date:
            today = datetime.now().strftime("%Y-%m-%d")
            from_date = from_date or today
            to_date = to_date or today
        
        # Fetch news
        news_data = finnhub_client.company_news(normalized_symbol, _from=from_date, to=to_date)
        
        if not isinstance(news_data, list):
            print(f"Unexpected response format for {symbol}")
            return []
        
        # Format response
        articles = []
        for item in news_data:
            articles.append({
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
                "datetime": item.get("datetime", ""),
                "source": item.get("source", ""),
                "symbol": item.get("symbol", normalized_symbol)
            })
        
        return articles
    
    except Exception as e:
        print(f"Error fetching company news for {symbol}: {e}")
        return []


# ============================================================================
# REAL-TIME QUOTE
# ============================================================================

def get_realtime_quote(symbol):
    """
    Fetch real-time quote data for a symbol.
    
    Parameters
    ----------
    symbol : str
        Stock symbol (e.g., "RELIANCE.NS", "AAPL")
    
    Returns
    -------
    dict with keys: price, open, high, low, previous_close, change_pct, timestamp
        Returns empty dict on error
    """
    try:
        if not finnhub_client:
            print("Finnhub client not initialized")
            return {}
        
        normalized_symbol = _normalize_symbol(symbol)
        quote = finnhub_client.quote(normalized_symbol)
        
        if not quote or "c" not in quote:
            print(f"No quote data for {symbol}")
            return {}
        
        return {
            "symbol": symbol,
            "price": quote.get("c", 0),  # Current price
            "open": quote.get("o", 0),   # Open price
            "high": quote.get("h", 0),   # High price
            "low": quote.get("l", 0),    # Low price
            "previous_close": quote.get("pc", 0),  # Previous close
            "change_pct": round((quote.get("c", 0) - quote.get("pc", 0)) / quote.get("pc", 1) * 100, 2),
            "timestamp": quote.get("t", 0)
        }
    
    except Exception as e:
        print(f"Error fetching real-time quote for {symbol}: {e}")
        return {}


# ============================================================================
# MARKET NEWS
# ============================================================================

def get_market_news(category="general", min_id=0):
    """
    Fetch general market news.
    
    Parameters
    ----------
    category : str
        News category (e.g., "general", "forex", "merger", "earnings")
    min_id : int
        Pagination parameter (news item ID)
    
    Returns
    -------
    list of dict
        Each dict has: id, category, datetime, headline, image, url, source
        Returns empty list on error
    """
    try:
        if not finnhub_client:
            print("Finnhub client not initialized")
            return []
        
        news_data = finnhub_client.general_news(category, min_id=min_id)
        
        if not isinstance(news_data, list):
            print(f"Unexpected response format for market news")
            return []
        
        articles = []
        for item in news_data:
            articles.append({
                "id": item.get("id", 0),
                "category": item.get("category", category),
                "datetime": item.get("datetime", ""),
                "headline": item.get("headline", ""),
                "image": item.get("image", ""),
                "url": item.get("url", ""),
                "source": item.get("source", "")
            })
        
        return articles
    
    except Exception as e:
        print(f"Error fetching market news: {e}")
        return []


# ============================================================================
# EARNINGS CALENDAR
# ============================================================================

def get_earnings_calendar(symbol):
    """
    Fetch earnings calendar information for a symbol.
    
    Parameters
    ----------
    symbol : str
        Stock symbol (e.g., "AAPL", "TSLA")
    
    Returns
    -------
    dict with: symbol, report_date, eps_estimate, eps_actual, revenue_estimate, revenue_actual
        Returns empty dict on error
    """
    try:
        if not finnhub_client:
            print("Finnhub client not initialized")
            return {}
        
        normalized_symbol = _normalize_symbol(symbol)
        
        # Remove exchange prefix if present for earnings calendar
        if ":" in normalized_symbol:
            normalized_symbol = normalized_symbol.split(":")[-1]
        
        earnings = finnhub_client.earnings_calendar(symbol=normalized_symbol)
        
        if not earnings or "earningsCalendar" not in earnings:
            print(f"No earnings data for {symbol}")
            return {}
        
        # Get most recent/upcoming earnings
        for item in sorted(earnings["earningsCalendar"], 
                          key=lambda x: x.get("reportDate", ""), 
                          reverse=True):
            return {
                "symbol": symbol,
                "report_date": item.get("reportDate", ""),
                "eps_estimate": item.get("epsEstimate", None),
                "eps_actual": item.get("epsActual", None),
                "revenue_estimate": item.get("revenueEstimate", None),
                "revenue_actual": item.get("revenueActual", None),
                "surprise_pct": item.get("surprisePercent", None)
            }
        
        return {}
    
    except Exception as e:
        print(f"Error fetching earnings for {symbol}: {e}")
        return {}


# ============================================================================
# COMPANY PROFILE
# ============================================================================

def get_company_profile(symbol):
    """
    Fetch company profile information.
    
    Parameters
    ----------
    symbol : str
        Stock symbol
    
    Returns
    -------
    dict with company information
        Returns empty dict on error
    """
    try:
        if not finnhub_client:
            print("Finnhub client not initialized")
            return {}
        
        normalized_symbol = _normalize_symbol(symbol)
        
        # Remove exchange prefix if present
        if ":" in normalized_symbol:
            normalized_symbol = normalized_symbol.split(":")[-1]
        
        profile = finnhub_client.company_profile2(symbol=normalized_symbol)
        
        if not profile:
            print(f"No profile data for {symbol}")
            return {}
        
        return {
            "symbol": symbol,
            "name": profile.get("name", ""),
            "industry": profile.get("finnhubIndustry", ""),
            "country": profile.get("country", ""),
            "market_cap": profile.get("marketCapitalization", None),
            "ipo_date": profile.get("ipoDate", ""),
            "website": profile.get("weburl", ""),
            "description": profile.get("description", "")
        }
    
    except Exception as e:
        print(f"Error fetching company profile for {symbol}: {e}")
        return {}


# ============================================================================
# TEST/MAIN
# ============================================================================

if __name__ == "__main__":
    print("Testing Finnhub Feed module...\n")
    
    # Test market status
    print("Market Status:")
    print(f"  NSE Open: {is_market_open('NSE')}")
    print(f"  US Open: {is_market_open('US')}")
    
    # Test news
    print("\nCompany News (RELIANCE.NS):")
    news = get_company_news("RELIANCE.NS")
    if news:
        print(f"  Found {len(news)} articles")
        if news:
            print(f"  Latest: {news[0].get('headline')}")
    
    # Test quote
    print("\nReal-time Quote (RELIANCE.NS):")
    quote = get_realtime_quote("RELIANCE.NS")
    if quote:
        print(f"  Price: {quote.get('price')}")
        print(f"  Change: {quote.get('change_pct')}%")
    
    # Test market news
    print("\nMarket News:")
    market_news = get_market_news("general")
    if market_news:
        print(f"  Found {len(market_news)} articles")
        if market_news:
            print(f"  Latest: {market_news[0].get('headline')}")
    
    print("\n✓ Tests complete")
