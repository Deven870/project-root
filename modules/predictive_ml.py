# modules/predictive_ml.py

import pandas as pd

def predict_intraday(data: pd.DataFrame):
    """
    Simple predictive logic for intraday trend.
    Returns trend ('Bullish', 'Bearish', 'Neutral') and confidence (0-1)
    """
    if data is None or data.empty:
        return "No data", 0.0

    required_cols = {"Open", "Close"}
    if not required_cols.issubset(data.columns):
        return "No valid columns", 0.0

    close_last = data["Close"].tail(1).values[0]
    open_last = data["Open"].tail(1).values[0]

    if close_last > open_last:
        trend = "Bullish"
    elif close_last < open_last:
        trend = "Bearish"
    else:
        trend = "Neutral"

    price_diff = abs(close_last - open_last)
    avg_price = (close_last + open_last) / 2
    confidence = min(round(price_diff / avg_price, 2), 1.0)

    return trend, confidence


def predict_long_term(data: pd.DataFrame):
    """
    Simple placeholder for long-term trend prediction.
    Returns trend and confidence
    """
    if data is None or data.empty:
        return "No data", 0.0

    required_cols = {"Open", "Close"}
    if not required_cols.issubset(data.columns):
        return "No valid columns", 0.0

    avg_open = data["Open"].mean()
    avg_close = data["Close"].mean()

    if avg_close > avg_open:
        trend = "Bullish"
    elif avg_close < avg_open:
        trend = "Bearish"
    else:
        trend = "Neutral"

    confidence = min(round(abs(avg_close - avg_open) / avg_open, 2), 1.0)
    return trend, confidence
