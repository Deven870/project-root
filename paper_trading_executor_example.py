"""
Example: Paper Trading Executor for 70% Accuracy System
Shows how to execute trades and log them
"""

import yfinance as yf
from datetime import datetime
from modules.paper_trading_framework import PaperTradingAccount, PaperTradeExecutor
from modules.prediction_70_integration import predict_composite
from modules.feature_engineering import build_features
import json
from pathlib import Path

# Load paper trading account
account = PaperTradingAccount(initial_capital=10000)
executor = PaperTradeExecutor(account)

# Example ticker
ticker = "RELIANCE.NS"

# Fetch data
df = yf.download(ticker, period="100d", progress=False)

# Build features
features = build_features(df)

# Get prediction from 70% system (composite for 70% accuracy)
trend, confidence, signal, regime = predict_composite(features, df['Close'].values, ticker)

print(f"\nExample Trade Execution:")
print(f"Ticker: {ticker}")
print(f"Prediction: {'Bullish' if trend == 1 else 'Bearish'}")
print(f"Confidence: {confidence:.0%}")
print(f"Signal: {signal}")

# Log trade
if confidence > 0.65:
    current_price = float(df['Close'].iloc[-1])
    
    # Execute prediction
    executor.execute_prediction(
        prediction={'trend': trend, 'confidence': confidence, 'signal': signal},
        ticker=ticker,
        current_price=current_price,
        features=features
    )
    
    # Print account status
    stats = account.get_stats()
    print(f"\nAccount Status:")
    print(f"Value: ${stats['account_value']:,.0f}")
    print(f"Positions: {stats['trades_open']}")
    print(f"P&L: ${stats['total_pnl']:+,.0f}")
else:
    print(f"Confidence too low ({confidence:.0%}), skipping trade")

print("\n" + "="*60)
print("Use this script as template for daily trading")
print("="*60)
