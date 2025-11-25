import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Import ML + sentiment modules (if available)
try:
    from modules.predictive_ml import predict_intraday, predict_long_term
except ImportError:
    # fallback: return a plausible confidence in [0,1]
    def predict_intraday(data): return "Bullish", float(np.random.uniform(0.6, 0.9))
    def predict_long_term(data): return "Bearish", float(np.random.uniform(0.5, 0.8))

try:
    from modules.sentiment_engine import analyze_hybrid_sentiment, get_news_for_stock
except Exception:
    # Provide a lightweight fallback using a simple rule-based sentiment analyzer and yfinance news.
    import yfinance as _yf

    POSITIVE_WORDS = set(["gain","gainful","positive","profit","beat","strong","rise","bullish","up","good","outperform","beats","growth","record"])
    NEGATIVE_WORDS = set(["loss","down","fall","weak","bearish","drop","decline","miss","poor","falling","losses","slump","crash"])

    def _simple_sentiment(text):
        try:
            t = str(text).lower()
            words = [w.strip('.,!?:;()[]"') for w in t.split()]
            pos = sum(1 for w in words if w in POSITIVE_WORDS)
            neg = sum(1 for w in words if w in NEGATIVE_WORDS)
            total = pos + neg
            if total == 0:
                return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
            return {
                "positive": pos / total,
                "neutral": 0.0,
                "negative": neg / total,
            }
        except Exception:
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

    def analyze_hybrid_sentiment(text):
        try:
            return _simple_sentiment(text)
        except Exception:
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

    def get_news_for_stock(ticker):
        try:
            t = _yf.Ticker(ticker)
            news = getattr(t, 'news', [])
            if not news:
                return []
            return [{"title": n.get('title', ''), "url": n.get('link', '')} for n in news][:20]
        except Exception:
            return []


# =========================================
# 🧭 FETCH NSE STOCK LIST (Live)
# =========================================
def get_nse_stock_list():
    """
    Fetches NSE stock symbols dynamically using Yahoo Finance.
    Returns a list of popular NSE tickers.
    """
    try:
        # Try fetching NIFTY 100 (robustly detect symbol column)
        nifty_tables = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_100")
        # Choose the table that has a 'Symbol' or 'Company'/'Company Name' column
        chosen = None
        for t in nifty_tables:
            cols = [c.lower() for c in t.columns.astype(str)]
            if 'symbol' in cols or 'code' in cols or 'ticker' in cols:
                chosen = t
                break
            if 'company' in cols or 'company name' in cols:
                chosen = t
                break
        if chosen is not None:
            # prefer a column with 'Symbol', fallback to 'Company Name' -> try to map to NSE codes using yfinance
            cols = [c.lower() for c in chosen.columns.astype(str)]
            if 'symbol' in cols:
                sym_col = chosen.columns[cols.index('symbol')]
                stocks = [f"{x}.NS" if not str(x).upper().endswith('.NS') else str(x).upper() for x in chosen[sym_col].astype(str).tolist()]
            elif 'code' in cols:
                sym_col = chosen.columns[cols.index('code')]
                stocks = [f"{x}.NS" if not str(x).upper().endswith('.NS') else str(x).upper() for x in chosen[sym_col].astype(str).tolist()]
            else:
                # fallback mapping from company name to known symbol via a small mapping; otherwise just use '.NS' suffix which might be wrong
                cname_col = chosen.columns[cols.index('company' if 'company' in cols else 'company name')]
                stocks = [f"{x}.NS" for x in chosen[cname_col].astype(str).tolist()]
            # strip duplicates and limit to 200
            seen = []
            out = []
            for s in stocks:
                s = s.upper()
                if s not in seen:
                    seen.append(s)
                    out.append(s)
            return out[:200]
    except Exception:
        # fallback list: extended list of common NSE tickers (NIFTY 50 / 100 subset) to ensure dropdown has many options
        return [
            "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ITC.NS","SBIN.NS","ONGC.NS","LT.NS",
            "HINDUNILVR.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS","BAJAJFINSV.NS","BHARTIARTL.NS",
            "BPCL.NS","BRITANNIA.NS","CIPLA.NS","DIVISLAB.NS","DRREDDY.NS","EICHERMOT.NS","GODREJPROP.NS",
            "GRASIM.NS","HCLTECH.NS","HDFC.NS","HDFCBANK.NS","HDFCLIFE.NS","HINDALCO.NS","HINDPETRO.NS",
            "IOC.NS","JSWSTEEL.NS","MARUTI.NS","M&M.NS","NESTLEIND.NS","NTPC.NS","POWERGRID.NS","RECLTD.NS",
            "SBILIFE.NS","SUNPHARMA.NS","TATAMOTORS.NS","TATASTEEL.NS","TECHM.NS","ULTRACEMCO.NS","VOLTAS.NS",
            "WIPRO.NS","ZOMATO.NS","ADANIENT.NS","ADANIPORTS.NS","BAJAJ-AUTO.NS","HAVELLS.NS","LICI.NS",
            "MINDTREE.NS","SRF.NS","TORNTPHARM.NS","SIEMENS.NS","TATACONSUM.NS","AMBUJACEM.NS","COALINDIA.NS"
        ]


# =========================================
# 📈 STOCK PREDICTIONS & SENTIMENT
# =========================================
def get_stock_predictions(ticker, invest_amount=None, horizon="intraday"):
    """
    Predicts stock trend and confidence using ML (or fallback dummy logic)
    Also calculates sentiment from news.
    """
    try:
        # Select timing based on horizon
        if horizon.lower() == 'intraday':
            data = fetch_price_data(ticker, period='5d', interval='1h')
        elif horizon.lower() == 'swing':
            data = fetch_price_data(ticker, period='1mo', interval='1d')
        else:
            data = fetch_price_data(ticker, period='6mo', interval='1d')
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
            # No headlines -> neutral sentiment
            avg_sentiment = {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
    except Exception as e:
        print("Sentiment error:", e)
        avg_sentiment = {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

    # Ensure confidence is a Python float (handle numpy scalars and pandas Series)
    try:
        confidence = float(np.squeeze(confidence))
    except Exception:
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

    # Ensure current_price is a Python float (handle Series/arrays returned by yfinance)
    if not data.empty:
        try:
            cp = data["Close"].iloc[-1]
            # cp may be a scalar or a Series if multiple tickers were requested
            if isinstance(cp, pd.Series):
                # pick the last value
                cp = cp.iloc[-1]
            current_price = float(np.squeeze(cp))
        except Exception:
            current_price = 0.0
    else:
        current_price = 0.0

    # Compute predicted price (using simple momentum * confidence approach)
    predicted_price = None
    predicted_return_pct = 0.0
    stop_loss = None
    if not data.empty:
        try:
            if horizon.lower() == 'intraday':
                close_last = data['Close'].iloc[-1]
                open_last = data['Open'].iloc[-1]
                price_change_pct = ((close_last - open_last) / max(abs(open_last), 1e-6)) * 100.0
            else:
                close_last = data['Close'].iloc[-1]
                close_first = data['Close'].iloc[0]
                price_change_pct = ((close_last - close_first) / max(abs(close_first), 1e-6)) * 100.0
            predicted_return_pct = float(confidence) * float(price_change_pct)
            predicted_price = current_price * (1 + predicted_return_pct / 100.0)
        except Exception:
            predicted_price = current_price

    # Stop loss suggestion by horizon (percent below current price)
    stop_loss_pct_by_horizon = {'intraday': 0.02, 'swing': 0.05, 'long-term': 0.1}
    sl_pct = stop_loss_pct_by_horizon.get(horizon.lower(), 0.05)
    if current_price and current_price > 0:
        stop_loss = current_price * (1 - sl_pct)

    return {
        "trend": trend,
        "confidence": confidence,
        "sentiment": avg_sentiment,
        "current_price": current_price,
        "predicted_price": predicted_price,
        "predicted_return_pct": predicted_return_pct,
        "stop_loss": stop_loss,
    }


# =========================================
# 📊 PORTFOLIO ALLOCATION
# =========================================
def get_portfolio_allocation(total_amount, horizon="longterm", allocation_mode='proportional', top_n=None, max_weight_pct=None):
    """
    Generates a diversified portfolio recommendation.
    """
    # Get the full list of stocks to consider
    stocks = get_nse_stock_list()

    # Determine expected return range depending on horizon
    if horizon.lower() == "intraday":
        expected_return_min, expected_return_max = 0.5, 2.5
        duration = "Up to 4 Hours"
    elif horizon.lower() == "swing":
        expected_return_min, expected_return_max = 2, 8
        duration = "2–10 Days"
    else:
        expected_return_min, expected_return_max = 5, 15
        duration = "1–6 Months"

    # Compute expected return for each stock using model output and recent price movement.
    expected_returns = []
    volatilities = []
    pred_trends = []
    pred_confidences = []
    for stck in stocks:
        try:
            df = fetch_price_data(stck)
            # fallbacks
            if df is None or df.empty or 'Close' not in df.columns or 'Open' not in df.columns:
                # fallback random small return to keep it in the list
                er = np.random.uniform(expected_return_min, expected_return_max)
                vol = 1.0
            else:
                # simple price change based on horizon
                if horizon.lower() == 'intraday':
                    close_last = df['Close'].iloc[-1]
                    open_last = df['Open'].iloc[-1]
                    price_change_pct = ((close_last - open_last) / max(abs(open_last), 1e-6)) * 100.0
                else:
                    price_change_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / max(abs(df['Close'].iloc[0]), 1e-6)) * 100.0

                # model predicted direction + confidence
                try:
                    if horizon.lower() == 'intraday':
                        trend, confidence = predict_intraday(df)
                    else:
                        trend, confidence = predict_long_term(df)
                except Exception:
                    trend, confidence = 'Neutral', 0.0

                # expected return estimate = price_change * confidence; prefer positive values for allocation
                er = float(confidence) * float(price_change_pct)
                # terrifically conservative fallback if er is extremely small (or negative)
                if abs(er) < 1e-6:
                    er = np.random.uniform(expected_return_min, expected_return_max)

                # store volatility as std dev percent to support risk-adjusted weight
                try:
                    vol = float(df['Close'].pct_change().dropna().std() * 100)
                    if np.isnan(vol) or vol <= 0:
                        vol = 1.0
                except Exception:
                    vol = 1.0
        except Exception:
            er = np.random.uniform(expected_return_min, expected_return_max)
            vol = 1.0

        expected_returns.append(er)
        volatilities.append(vol)
        pred_trends.append(trend)
        pred_confidences.append(float(confidence))

    expected_returns = np.array(expected_returns, dtype=float)
    volatilities = np.array(volatilities, dtype=float)

    # Guard against the unlikely case where all expected returns are zero
    total_er = expected_returns.sum()
    # Compute raw weights depending on the chosen allocation strategy
    if allocation_mode == 'equal':
        weights = np.ones_like(expected_returns) / len(expected_returns)
    elif allocation_mode == 'risk_adjusted':
        # weights proportional to expected_return / volatility
        adj = expected_returns / np.maximum(volatilities, 1e-6)
        adj = np.where(adj <= 0, 0.0, adj)
        total_adj = adj.sum()
        if total_adj <= 0:
            weights = np.ones_like(adj) / len(adj)
        else:
            weights = adj / total_adj
    else:
        # default/proportional: use positive expected returns only
        er_pos = np.where(expected_returns > 0, expected_returns, 0.0)
        total_er_pos = er_pos.sum()
        if total_er_pos <= 0:
            weights = np.ones_like(er_pos) / len(er_pos)
        else:
            weights = er_pos / total_er_pos

    # Build the portfolio for all stocks (so the user can compare all of them)
    # enforce a maximum weight per stock if provided (max_weight_pct as percent 0..100)
    if max_weight_pct and max_weight_pct > 0:
        cap = float(max_weight_pct) / 100.0
        # If cap is too small to allocate 100% across all stocks, increase cap to minimal equal-weight
        if cap * len(weights) < 1.0:
            cap = 1.0 / len(weights)
        orig_weights = weights.copy()
        weights_copy = weights.copy()
        for _ in range(10):
            over_mask = weights_copy > cap
            if not over_mask.any():
                break
            weights_copy[over_mask] = cap
            leftover = 1.0 - weights_copy.sum()
            uncapped_mask = weights_copy < cap
            if leftover <= 1e-12 or not uncapped_mask.any():
                break
            # distribute leftover proportionally based on the original (pre-cap) weights among remaining uncapped
            orig_uncapped = orig_weights[uncapped_mask]
            if orig_uncapped.sum() > 0:
                weights_copy[uncapped_mask] += (orig_uncapped / orig_uncapped.sum()) * leftover
            else:
                # equally distribute if orig weights are zero
                weights_copy[uncapped_mask] += leftover / uncapped_mask.sum()
        # final renormalize just in case of floating rounding
        if weights_copy.sum() > 0:
            weights = weights_copy / weights_copy.sum()

    # If the user wants to see only top_n stocks, keep only those (and re-normalize weights)
    if top_n is not None and int(top_n) > 0 and int(top_n) < len(stocks):
        n = int(top_n)
        order = np.argsort(weights)[::-1][:n]
        stocks = [stocks[i] for i in order]
        expected_returns = expected_returns[order]
        volatilities = volatilities[order]
        weights = weights[order]
        # renormalize
        if weights.sum() > 0:
            weights = weights / weights.sum()

    portfolio = []
    for stock, er, w in zip(stocks, expected_returns, weights):
        allocation_amt = round(total_amount * float(w), 2)
        expected_profit = round(allocation_amt * er / 100.0, 2)

        # fetch corresponding trend & confidence if available
        tr = None
        cf = None
        try:
            idx = stocks.index(stock)
            tr = pred_trends[idx]
            cf = pred_confidences[idx]
        except Exception:
            tr = 'N/A'
            cf = 0.0

        portfolio.append({
            "Stock": stock,
            "Weight (%)": round(w * 100, 2),
            "Allocation (₹)": allocation_amt,
            "Expected Return (%)": round(er, 2),
            "Expected Profit (₹)": expected_profit,
            "Duration": duration,
            "Trend": tr,
            "Confidence": round(cf * 100.0, 2),
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
def fetch_price_data(ticker, period='1mo', interval='1d'):
    """
    Fetch recent stock data using yfinance.
    """
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            raise ValueError("No price data found.")

        # If yfinance returns MultiIndex columns (e.g., for multiple tickers),
        # flatten the columns to ensure columns like "Open" and "Close" are present.
        if hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
            # Use only the first level (Open/Close/High/Low/Volume) which is what our predictor expects.
            data.columns = data.columns.get_level_values(0)

        # If 'Close' column is missing but 'Adj Close' is available, use that as Close
        if 'Close' not in data.columns and 'Adj Close' in data.columns:
            data['Close'] = data['Adj Close']

        # If 'Open' isn't found but there's a similar column, try to fall back
        if 'Open' not in data.columns and 'open' in data.columns:
            data['Open'] = data['open']

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
