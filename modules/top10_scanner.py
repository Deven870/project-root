import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from modules.utils import get_stock_predictions
from modules.sheets_live import get_worksheet, log_signal

logger = logging.getLogger(__name__)

DEFAULT_WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS", "SBIN.NS",
    "WIPRO.NS", "AXISBANK.NS", "MARUTI.NS", "TATASTEEL.NS", "AAPL", "GOOGL",
    "META", "MSFT", "NVDA", "PAYTM.NS", "ZOMATO.NS"
]


def load_watchlist():
    """
    Read watchlist from Google Sheets Config tab key WATCHLIST.
    Fallbacks to DEFAULT_WATCHLIST when unavailable.
    """
    try:
        ws = get_worksheet("Config")
        rows = ws.get_all_records()
        for row in rows:
            key = str(row.get("Key", "")).strip().upper()
            if key == "WATCHLIST":
                value = str(row.get("Value", "")).strip()
                if value:
                    items = [x.strip() for x in value.split(",") if x.strip()]
                    if items:
                        return items
    except Exception as e:
        logger.warning("Falling back to default watchlist: %s", e)
    return DEFAULT_WATCHLIST


def scan_single_stock(symbol, horizon, invest_amount=10000):
    """
    Scan one stock and normalize output into a single dict.
    Returns None if prediction fails.
    """
    try:
        result = get_stock_predictions(symbol, invest_amount=invest_amount, horizon=horizon)
        if not isinstance(result, dict):
            return None

        trend = str(result.get("trend", "")).strip()
        confidence = float(result.get("confidence", 0.0) or 0.0)
        current_price = float(result.get("current_price", 0.0) or 0.0)
        predicted_price = float(result.get("predicted_price", 0.0) or 0.0)
        expected_return_pct = float(result.get("predicted_return_pct", 0.0) or 0.0)
        stop_loss = float(result.get("stop_loss", 0.0) or 0.0)

        sentiment = result.get("sentiment", {}) or {}
        sentiment_score = float(sentiment.get("positive", 0.0) - sentiment.get("negative", 0.0))

        return {
            "symbol": symbol,
            "trend": trend,
            "confidence": confidence,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "expected_return_pct": expected_return_pct,
            "stop_loss": stop_loss,
            "sentiment_score": sentiment_score,
        }
    except Exception as e:
        logger.warning("Scan failed for %s: %s", symbol, e)
        return None


def _scan_many(watchlist, horizon, invest_amount):
    scanned = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(scan_single_stock, sym, horizon, invest_amount): sym
            for sym in watchlist
        }
        for future in as_completed(futures):
            item = future.result()
            if item:
                scanned.append(item)
    return scanned


def _to_dataframe(items):
    rows = []
    for idx, item in enumerate(items, start=1):
        rows.append({
            "Rank": idx,
            "Stock": item["symbol"],
            "Trend": item["trend"],
            "Confidence": item["confidence"],
            "CurrentPrice": item["current_price"],
            "PredictedPrice": item["predicted_price"],
            "ExpectedReturn": item["expected_return_pct"],
            "StopLoss": item["stop_loss"],
            "SentimentScore": item["sentiment_score"],
            "CompositeScore": item["composite_score"],
        })
    return pd.DataFrame(rows, columns=[
        "Rank", "Stock", "Trend", "Confidence", "CurrentPrice", "PredictedPrice",
        "ExpectedReturn", "StopLoss", "SentimentScore", "CompositeScore"
    ])


def get_top10(horizon="Intraday", invest_amount=10000, min_confidence=0.65):
    """
    Scan watchlist concurrently and return top 10 bullish opportunities.
    Ranking score: confidence * expected_return_pct.
    """
    watchlist = load_watchlist()
    scanned = _scan_many(watchlist, horizon, invest_amount)

    filtered = []
    for item in scanned:
        trend = item["trend"].lower()
        if "bull" in trend and item["confidence"] >= min_confidence:
            item["composite_score"] = item["confidence"] * item["expected_return_pct"]
            filtered.append(item)

    ranked = sorted(filtered, key=lambda x: x["composite_score"], reverse=True)[:10]

    for item in ranked:
        try:
            log_signal(
                symbol=item["symbol"],
                trend=item["trend"],
                confidence=item["confidence"],
                current_price=item["current_price"],
                predicted_price=item["predicted_price"],
                expected_return=item["expected_return_pct"],
                stop_loss=item["stop_loss"],
                horizon=horizon,
                sentiment_score=item["sentiment_score"],
            )
        except Exception as e:
            logger.warning("Signal log failed for %s: %s", item["symbol"], e)

    return _to_dataframe(ranked)


def get_top10_bearish(horizon="Intraday", min_confidence=0.65):
    """
    Scan watchlist concurrently and return top 10 bearish opportunities.
    """
    watchlist = load_watchlist()
    scanned = _scan_many(watchlist, horizon, invest_amount=10000)

    filtered = []
    for item in scanned:
        trend = item["trend"].lower()
        if "bear" in trend and item["confidence"] >= min_confidence:
            # For shorts, absolute return magnitude is a better signal strength proxy.
            item["composite_score"] = item["confidence"] * abs(item["expected_return_pct"])
            filtered.append(item)

    ranked = sorted(filtered, key=lambda x: x["composite_score"], reverse=True)[:10]

    for item in ranked:
        try:
            log_signal(
                symbol=item["symbol"],
                trend=item["trend"],
                confidence=item["confidence"],
                current_price=item["current_price"],
                predicted_price=item["predicted_price"],
                expected_return=item["expected_return_pct"],
                stop_loss=item["stop_loss"],
                horizon=horizon,
                sentiment_score=item["sentiment_score"],
            )
        except Exception as e:
            logger.warning("Signal log failed for %s: %s", item["symbol"], e)

    return _to_dataframe(ranked)
