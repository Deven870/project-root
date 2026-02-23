# modules/baseline_models.py
"""
Baseline models for research paper comparison.
Implements traditional trading strategies and simple models.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


# =========================================
# Buy and Hold Strategy
# =========================================

def buy_and_hold_baseline(prices: pd.Series) -> Dict:
    """
    Buy and Hold baseline - simplest strategy.
    Buy at first price, sell at last price.
    
    Args:
        prices: Series of stock prices
    
    Returns:
        Dictionary with return, signals, and equity curve
    """
    if len(prices) < 2:
        return {
            "strategy": "Buy & Hold",
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "signals": np.ones(len(prices)),
            "equity_curve": np.array([1.0] * len(prices))
        }
    
    # Always hold (signal = 1)
    signals = np.ones(len(prices))
    
    # Calculate returns
    returns = prices.pct_change().fillna(0).values
    cum_returns = (1 + returns).cumprod()
    
    total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    
    # Sharpe ratio
    if np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
    else:
        sharpe = 0.0
    
    # Max drawdown
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    max_dd = abs(np.min(drawdown)) * 100 if len(drawdown) > 0 else 0.0
    
    return {
        "strategy": "Buy & Hold",
        "total_return_pct": float(total_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": float(max_dd),
        "signals": signals,
        "equity_curve": cum_returns,
        "n_trades": 0
    }


# =========================================
# Moving Average Crossover (MACD-based)
# =========================================

def moving_average_crossover(prices: pd.Series, fast: int = 12, slow: int = 26) -> Dict:
    """
    Moving Average Crossover Strategy (similar to MACD).
    Buy when fast MA crosses above slow MA, sell when below.
    
    Args:
        prices: Series of stock prices
        fast: Fast MA period
        slow: Slow MA period
    
    Returns:
        Dictionary with strategy performance
    """
    if len(prices) < slow + 5:
        return buy_and_hold_baseline(prices)  # Fallback
    
    # Calculate moving averages
    ma_fast = prices.rolling(window=fast).mean()
    ma_slow = prices.rolling(window=slow).mean()
    
    # Generate signals: 1 (buy/hold) when fast > slow, 0 (sell) otherwise
    signals = (ma_fast > ma_slow).astype(int).values
    
    # Calculate strategy returns
    returns = prices.pct_change().fillna(0).values
    strategy_returns = signals[:-1] * returns[1:]  # Shift to avoid lookahead
    strategy_returns = np.concatenate([[0], strategy_returns])
    
    cum_returns = (1 + strategy_returns).cumprod()
    
    total_return = (cum_returns[-1] - 1) * 100
    
    # Sharpe ratio
    if np.std(strategy_returns) > 0:
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Max drawdown
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    max_dd = abs(np.min(drawdown)) * 100 if len(drawdown) > 0 else 0.0
    
    # Count trades (signal changes)
    n_trades = np.sum(np.diff(signals) != 0)
    
    return {
        "strategy": "MA Crossover",
        "total_return_pct": float(total_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": float(max_dd),
        "signals": signals,
        "equity_curve": cum_returns,
        "n_trades": int(n_trades)
    }


# =========================================
# RSI-based Strategy
# =========================================

def rsi_strategy(prices: pd.Series, period: int = 14, 
                 oversold: int = 30, overbought: int = 70) -> Dict:
    """
    RSI-based trading strategy.
    Buy when RSI < oversold, sell when RSI > overbought.
    
    Args:
        prices: Series of stock prices
        period: RSI period
        oversold: Oversold threshold (default 30)
        overbought: Overbought threshold (default 70)
    
    Returns:
        Dictionary with strategy performance
    """
    if len(prices) < period + 5:
        return buy_and_hold_baseline(prices)
    
    # Calculate RSI
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # Neutral RSI for NaN values
    
    # Generate signals
    # Hold (1) when RSI is in normal range or oversold (bullish)
    # Sell (0) when overbought (bearish)
    signals = np.where(rsi < oversold, 1,  # Buy on oversold
                      np.where(rsi > overbought, 0,  # Sell on overbought
                              1))  # Hold otherwise (default bullish)
    
    # Calculate returns
    returns = prices.pct_change().fillna(0).values
    strategy_returns = signals[:-1] * returns[1:]
    strategy_returns = np.concatenate([[0], strategy_returns])
    
    cum_returns = (1 + strategy_returns).cumprod()
    
    total_return = (cum_returns[-1] - 1) * 100
    
    # Sharpe ratio
    if np.std(strategy_returns) > 0:
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Max drawdown
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    max_dd = abs(np.min(drawdown)) * 100 if len(drawdown) > 0 else 0.0
    
    # Count trades
    n_trades = np.sum(np.diff(signals) != 0)
    
    return {
        "strategy": "RSI Strategy",
        "total_return_pct": float(total_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": float(max_dd),
        "signals": signals,
        "equity_curve": cum_returns,
        "n_trades": int(n_trades)
    }


# =========================================
# Momentum Strategy
# =========================================

def momentum_strategy(prices: pd.Series, lookback: int = 20) -> Dict:
    """
    Momentum strategy: Buy if price > price N days ago, else sell.
    
    Args:
        prices: Series of stock prices
        lookback: Lookback period for momentum
    
    Returns:
        Dictionary with strategy performance
    """
    if len(prices) < lookback + 5:
        return buy_and_hold_baseline(prices)
    
    # Calculate momentum: current price vs N-day ago price
    momentum = prices.pct_change(periods=lookback).fillna(0)
    
    # Signals: 1 if positive momentum, 0 if negative
    signals = (momentum > 0).astype(int).values
    
    # Calculate returns
    returns = prices.pct_change().fillna(0).values
    strategy_returns = signals[:-1] * returns[1:]
    strategy_returns = np.concatenate([[0], strategy_returns])
    
    cum_returns = (1 + strategy_returns).cumprod()
    
    total_return = (cum_returns[-1] - 1) * 100
    
    # Sharpe ratio
    if np.std(strategy_returns) > 0:
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Max drawdown
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    max_dd = abs(np.min(drawdown)) * 100 if len(drawdown) > 0 else 0.0
    
    # Count trades
    n_trades = np.sum(np.diff(signals) != 0)
    
    return {
        "strategy": "Momentum",
        "total_return_pct": float(total_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": float(max_dd),
        "signals": signals,
        "equity_curve": cum_returns,
        "n_trades": int(n_trades)
    }


# =========================================
# Mean Reversion Strategy
# =========================================

def mean_reversion_strategy(prices: pd.Series, window: int = 20, threshold: float = 1.5) -> Dict:
    """
    Mean reversion strategy using Bollinger Bands concept.
    Buy when price is below lower band, sell when above upper band.
    
    Args:
        prices: Series of stock prices
        window: Moving average window
        threshold: Number of standard deviations for bands
    
    Returns:
        Dictionary with strategy performance
    """
    if len(prices) < window + 5:
        return buy_and_hold_baseline(prices)
    
    # Calculate moving average and standard deviation
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    
    upper_band = ma + (threshold * std)
    lower_band = ma - (threshold * std)
    
    # Signals: Buy (1) when below lower band, Sell (0) when above upper band
    # Hold current position otherwise
    signals = np.ones(len(prices))  # Default hold
    signals = np.where(prices < lower_band, 1,  # Buy oversold
                      np.where(prices > upper_band, 0,  # Sell overbought
                              signals))
    
    # Fill forward to maintain position
    signals = pd.Series(signals).fillna(method='ffill').fillna(1).values
    
    # Calculate returns
    returns = prices.pct_change().fillna(0).values
    strategy_returns = signals[:-1] * returns[1:]
    strategy_returns = np.concatenate([[0], strategy_returns])
    
    cum_returns = (1 + strategy_returns).cumprod()
    
    total_return = (cum_returns[-1] - 1) * 100
    
    # Sharpe ratio
    if np.std(strategy_returns) > 0:
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Max drawdown
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    max_dd = abs(np.min(drawdown)) * 100 if len(drawdown) > 0 else 0.0
    
    # Count trades
    n_trades = np.sum(np.diff(signals) != 0)
    
    return {
        "strategy": "Mean Reversion",
        "total_return_pct": float(total_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": float(max_dd),
        "signals": signals,
        "equity_curve": cum_returns,
        "n_trades": int(n_trades)
    }


# =========================================
# Random Trading (Lower Bound)
# =========================================

def random_trading_baseline(prices: pd.Series, n_simulations: int = 100) -> Dict:
    """
    Random trading baseline - purely random buy/sell signals.
    Average performance over multiple random simulations.
    
    Args:
        prices: Series of stock prices
        n_simulations: Number of random simulations to average
    
    Returns:
        Dictionary with average random performance
    """
    returns = prices.pct_change().fillna(0).values
    
    all_results = []
    
    for _ in range(n_simulations):
        # Random signals (0 or 1 with equal probability)
        signals = np.random.randint(0, 2, size=len(prices))
        
        strategy_returns = signals[:-1] * returns[1:]
        strategy_returns = np.concatenate([[0], strategy_returns])
        cum_returns = (1 + strategy_returns).cumprod()
        
        total_return = (cum_returns[-1] - 1) * 100
        
        if np.std(strategy_returns) > 0:
            sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_dd = abs(np.min(drawdown)) * 100 if len(drawdown) > 0 else 0.0
        
        all_results.append({
            "return": total_return,
            "sharpe": sharpe,
            "max_dd": max_dd
        })
    
    # Average across simulations
    avg_return = np.mean([r["return"] for r in all_results])
    avg_sharpe = np.mean([r["sharpe"] for r in all_results])
    avg_max_dd = np.mean([r["max_dd"] for r in all_results])
    
    return {
        "strategy": "Random Trading",
        "total_return_pct": float(avg_return),
        "sharpe_ratio": float(avg_sharpe),
        "max_drawdown_pct": float(avg_max_dd),
        "signals": None,
        "equity_curve": None,
        "n_trades": len(prices) // 2  # Approximate
    }


# =========================================
# Run All Baselines
# =========================================

def run_all_baselines(prices: pd.Series) -> pd.DataFrame:
    """
    Run all baseline strategies on given price series.
    
    Args:
        prices: Series of stock prices
    
    Returns:
        DataFrame with all baseline results
    """
    baselines = [
        buy_and_hold_baseline(prices),
        moving_average_crossover(prices),
        rsi_strategy(prices),
        momentum_strategy(prices),
        mean_reversion_strategy(prices),
        random_trading_baseline(prices)
    ]
    
    results = []
    for baseline in baselines:
        results.append({
            "Strategy": baseline["strategy"],
            "Total Return (%)": baseline["total_return_pct"],
            "Sharpe Ratio": baseline["sharpe_ratio"],
            "Max Drawdown (%)": baseline["max_drawdown_pct"],
            "# Trades": baseline.get("n_trades", 0)
        })
    
    return pd.DataFrame(results)
