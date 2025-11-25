import yfinance as yf
import requests
from datetime import datetime, timedelta

NEWS_API_KEY = "fbba6b423b7e426a810eb007ad444242"  

def fetch_stock_data(ticker, period="7d", interval="1d"):
    """
    Fetch historical stock data using yfinance
    period: '7d', '1mo', etc.
    interval: '1d', '1h', '5m'
    """
    try:
        df = yf.download(ticker, period=period, interval=interval)
        if df.empty:
            raise ValueError("No data fetched for ticker.")
        return df
    except Exception as e:
        raise ValueError(f"Error fetching stock data: {e}")

def get_news_for_stock(stock_ticker, from_days=7, max_articles=20):
    """
    Fetch recent news for a stock using NewsAPI
    """
    today = datetime.now()
    from_date = today - timedelta(days=from_days)
    
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={stock_ticker} AND (NSE OR stock OR market)&"
        f"from={from_date.strftime('%Y-%m-%d')}&"
        f"sortBy=publishedAt&"
        f"pageSize={max_articles}&"
        f"apiKey={NEWS_API_KEY}"
    )

    response = requests.get(url)
    data = response.json()

    if data.get("status") != "ok":
        return []

    headlines = [{"title": a["title"], "url": a["url"]} for a in data["articles"]]
    return headlines
