import numpy as np

def simple_trend_prediction(prices):
    """A very basic trend prediction based on price change"""
    if len(prices) < 2:
        return "Not enough data", 0.0

    change = prices[-1] - prices[0]
    confidence = abs(change) / max(prices[0], 1)

    if change > 0:
        trend = "Bullish"
    elif change < 0:
        trend = "Bearish"
    else:
        trend = "Neutral"

    return trend, round(float(confidence), 2)
