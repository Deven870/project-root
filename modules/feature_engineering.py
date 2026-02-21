# modules/feature_engineering.py
"""
Technical indicator feature engineering for stock prediction.
Generates a rich feature set from OHLCV data for ML models.
"""

import pandas as pd
import numpy as np


# =========================================
# Core Technical Indicators
# =========================================

def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window, min_periods=1).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD, Signal line, and Histogram."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    """Bollinger Bands (upper, middle, lower)."""
    middle = sma(series, window)
    std = series.rolling(window=window, min_periods=1).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=1).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (volume * direction).cumsum()


def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
    """Stochastic Oscillator %K and %D."""
    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d = k.rolling(window=d_period, min_periods=1).mean()
    return k, d


# =========================================
# Feature Builder
# =========================================

def build_features(df: pd.DataFrame, sentiment_score: float = 0.0) -> pd.DataFrame:
    """
    Build a full feature set from OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Open, High, Low, Close, Volume
    sentiment_score : float
        Hybrid sentiment score (positive - negative) in [-1, 1]

    Returns
    -------
    pd.DataFrame
        DataFrame with all features, NaNs dropped.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    feat = pd.DataFrame(index=df.index)

    close = df["Close"].squeeze() if hasattr(df["Close"], "squeeze") else df["Close"]
    open_ = df["Open"].squeeze() if hasattr(df["Open"], "squeeze") else df["Open"]
    high = df["High"].squeeze() if hasattr(df["High"], "squeeze") else df["High"]
    low = df["Low"].squeeze() if hasattr(df["Low"], "squeeze") else df["Low"]
    volume = df["Volume"].squeeze() if hasattr(df["Volume"], "squeeze") else df["Volume"]

    # --- Price Features ---
    feat["close"] = close
    feat["open"] = open_
    feat["high"] = high
    feat["low"] = low
    feat["volume"] = volume

    # Returns
    feat["return_1d"] = close.pct_change(1)
    feat["return_3d"] = close.pct_change(3)
    feat["return_5d"] = close.pct_change(5)

    # Lag features
    for lag in [1, 2, 3, 5]:
        feat[f"close_lag_{lag}"] = close.shift(lag)
        feat[f"return_lag_{lag}"] = feat["return_1d"].shift(lag)

    # --- Moving Averages ---
    feat["sma_5"] = sma(close, 5)
    feat["sma_10"] = sma(close, 10)
    feat["sma_20"] = sma(close, 20)
    feat["ema_12"] = ema(close, 12)
    feat["ema_26"] = ema(close, 26)

    # Price relative to MAs
    feat["close_to_sma5"] = close / feat["sma_5"] - 1
    feat["close_to_sma20"] = close / feat["sma_20"] - 1
    feat["sma5_to_sma20"] = feat["sma_5"] / feat["sma_20"] - 1

    # --- RSI ---
    feat["rsi_14"] = rsi(close, 14)
    feat["rsi_7"] = rsi(close, 7)

    # --- MACD ---
    macd_line, signal_line, macd_hist = macd(close)
    feat["macd"] = macd_line
    feat["macd_signal"] = signal_line
    feat["macd_hist"] = macd_hist

    # --- Bollinger Bands ---
    bb_upper, bb_middle, bb_lower = bollinger_bands(close)
    feat["bb_upper"] = bb_upper
    feat["bb_lower"] = bb_lower
    feat["bb_width"] = (bb_upper - bb_lower) / (bb_middle + 1e-10)
    feat["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # --- ATR ---
    feat["atr_14"] = atr(high, low, close, 14)
    feat["atr_pct"] = feat["atr_14"] / (close + 1e-10)

    # --- Volume ---
    feat["volume_sma_10"] = sma(volume, 10)
    feat["volume_ratio"] = volume / (feat["volume_sma_10"] + 1e-10)
    feat["obv"] = obv(close, volume)

    # --- Stochastic Oscillator ---
    stoch_k, stoch_d = stochastic_oscillator(high, low, close)
    feat["stoch_k"] = stoch_k
    feat["stoch_d"] = stoch_d

    # --- Candlestick Features ---
    feat["body_size"] = (close - open_).abs() / (close + 1e-10)
    feat["upper_shadow"] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / (close + 1e-10)
    feat["lower_shadow"] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / (close + 1e-10)

    # --- Volatility ---
    feat["volatility_5"] = close.rolling(5, min_periods=1).std() / (close + 1e-10)
    feat["volatility_20"] = close.rolling(20, min_periods=1).std() / (close + 1e-10)

    # --- Sentiment (constant for the batch, varies per prediction call) ---
    feat["sentiment_score"] = sentiment_score

    # --- Target: next-day return and direction ---
    feat["target_return"] = close.pct_change(1).shift(-1)  # next day's return
    feat["target_direction"] = (feat["target_return"] > 0).astype(int)  # 1=Bullish, 0=Bearish

    # Drop rows with NaN
    feat = feat.dropna()

    return feat


def get_feature_columns():
    """Return the list of feature column names (excludes targets and raw OHLCV)."""
    return [
        "return_1d", "return_3d", "return_5d",
        "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_5",
        "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
        "sma_5", "sma_10", "sma_20", "ema_12", "ema_26",
        "close_to_sma5", "close_to_sma20", "sma5_to_sma20",
        "rsi_14", "rsi_7",
        "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_lower", "bb_width", "bb_position",
        "atr_14", "atr_pct",
        "volume_sma_10", "volume_ratio", "obv",
        "stoch_k", "stoch_d",
        "body_size", "upper_shadow", "lower_shadow",
        "volatility_5", "volatility_20",
        "sentiment_score",
    ]
