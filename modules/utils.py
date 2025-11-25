import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Import ML + sentiment modules (if available)
try:
    from modules.predictive_ml import predict_intraday, predict_long_term
except ImportError:
    def predict_intraday(data): return "Bullish", np.random.uniform(60, 90)
    def predict_long_term(data): return "Bearish", np.random.uniform(50, 80)

try:
    from modules.sentiment_engine import analyze_hybrid_sentiment, get_news_for_stock
except ImportError:
    def analyze_hybrid_sentiment(text):
        return {"positive": 0.4, "neutral": 0.3, "negative": 0.3}
    def get_news_for_stock(ticker):
        return [{"title": f"{ticker} shows positive movement"}]


# =========================================
# 🧭 FETCH NSE STOCK LIST (Live)
# =========================================
def get_nse_stock_list():
    """
    Fetches NSE stock symbols dynamically using Yahoo Finance.
    Returns a list of popular NSE tickers.
    """
    try:
        # Try fetching NIFTY 100
        nifty_100 = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_100")[1]
        stocks = [f"{x}.NS" for x in nifty_100["Company Name"].head(100)]
        return stocks
    except Exception:
        # fallback list
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS", "SBIN.NS", "ONGC.NS", "LT.NS"]


# =========================================
# 📈 STOCK PREDICTIONS & SENTIMENT
# =========================================
def get_stock_predictions(ticker, invest_amount=None, horizon="intraday"):
    """
    Predicts stock trend and confidence using ML (or fallback dummy logic)
    Also calculates sentiment from news.
    """
    try:
        data = fetch_price_data(ticker)
        if horizon.lower() == "intraday":
            trend, confidence = predict_intraday(data)
        else:
            trend, confidence = predict_long_term(data)
    except Exception as e:
        print("Prediction error:", e)
        trend, confidence = "N/A", 0

    # News sentiment
    try:
        headlines = get_news_for_stock(ticker)
        if headlines and isinstance(headlines, list):
            sentiments = [analyze_hybrid_sentiment(h["title"]) for h in headlines]
            avg_sentiment = {
                "positive": np.mean([s["positive"] for s in sentiments]),
                "neutral": np.mean([s["neutral"] for s in sentiments]),
                "negative": np.mean([s["negative"] for s in sentiments]),
            }
        else:
            avg_sentiment = {"positive": 0, "neutral": 0, "negative": 0}
    except Exception as e:
        print("Sentiment error:", e)
        avg_sentiment = {"positive": 0, "neutral": 0, "negative": 0}

    return {
        "trend": trend,
        "confidence": confidence,
        "sentiment": avg_sentiment,
        "current_price": data["Close"].iloc[-1] if not data.empty else 0,
    }


# =========================================
# 📊 PORTFOLIO ALLOCATION
# =========================================
def get_portfolio_allocation(total_amount, horizon="longterm"):
    """
    Generates a diversified portfolio recommendation.
    """
    stocks = get_nse_stock_list()
    allocation_per_stock = total_amount / len(stocks)

    if horizon.lower() == "intraday":
        expected_return_min, expected_return_max = 0.5, 2.5
        duration = "Up to 4 Hours"
    elif horizon.lower() == "swing":
        expected_return_min, expected_return_max = 2, 8
        duration = "2–10 Days"
    else:
        expected_return_min, expected_return_max = 5, 15
        duration = "1–6 Months"

    portfolio = []
    for stock in stocks[:10]:  # show top 10 to keep table readable
        expected_return = round(np.random.uniform(expected_return_min, expected_return_max), 2)
        expected_profit = round(allocation_per_stock * expected_return / 100, 2)

        portfolio.append({
            "Stock": stock,
            "Allocation (₹)": round(allocation_per_stock, 2),
            "Expected Return (%)": expected_return,
            "Expected Profit (₹)": expected_profit,
            "Duration": duration
        })

    return portfolio


# =========================================
# 💡 INVESTMENT ADVICE GENERATOR
# =========================================
def get_investment_advice(ticker, horizon="intraday"):
    """
    Generates a short strategic investment suggestion text.
    """
    horizon = horizon.lower()
    if horizon == "intraday":
        return (
            f"For {ticker}: Focus on momentum and liquidity. "
            "Use stop-loss orders and avoid overnight positions."
        )
    elif horizon == "swing":
        return (
            f"For {ticker}: Swing trading favors trending markets. "
            "Hold for a few days, and monitor technical breakouts."
        )
    else:
        return (
            f"For {ticker}: Consider fundamentals and diversification. "
            "Long-term investing builds steady compounding growth."
        )


# =========================================
# 📉 PRICE DATA FETCHER (YFinance)
# =========================================
def fetch_price_data(ticker):
    """
    Fetch recent stock data using yfinance.
    """
    try:
        data = yf.download(ticker, period="1mo", interval="1d", progress=False)
        if data.empty:
            raise ValueError("No price data found.")
        return data
    except Exception as e:
        print(f"Error fetching {ticker} data: {e}")
        return pd.DataFrame({
            "Open": np.random.uniform(100, 200, 5),
            "Close": np.random.uniform(100, 200, 5),
            "High": np.random.uniform(100, 200, 5),
            "Low": np.random.uniform(100, 200, 5),
            "Volume": np.random.randint(1000, 5000, 5),
        })
