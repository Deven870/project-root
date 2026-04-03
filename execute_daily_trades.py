#!/usr/bin/env python
"""
Daily Paper Trading Executor - April 3, 2026
Execute and track trades using the 70% accuracy system
All amounts in Indian Rupees (₹)
"""

import json
from datetime import datetime
from pathlib import Path
import yfinance as yf
from modules.feature_engineering import build_features
from modules.paper_trading_framework import PaperTradingManager
from modules.prediction_70_integration import get_router
from modules.macro_signals import get_macro_signals

def get_prediction(ticker: str):
    """Get 70% composite prediction for a stock"""
    try:
        # Fetch data
        df = yf.download(ticker, period="100d", progress=False)
        
        if df is None or len(df) < 50:
            return None, None, None, None, None
        
        # Build features
        features = build_features(df)
        current_price = float(df['Close'].iloc[-1])
        
        # Get router
        router = get_router()
        
        # Get composite prediction (70% accuracy target)
        try:
            trend, confidence, signal, regime = router.predict_composite(
                features, df['Close'].values, ticker
            )
        except:
            # Fallback to swing if composite fails
            trend, confidence = router.predict_swing(
                features, df['Close'].values, ticker
            )
            signal = "medium"
            regime = "ranging"
        
        return trend, confidence, signal, regime, current_price
    
    except Exception as e:
        print(f"⚠️ Error getting prediction for {ticker}: {e}")
        return None, None, None, None, None


def execute_daily_trade(ticker: str, capital_per_trade: float = 25000):
    """
    Execute a single trade based on 70% system prediction
    
    Args:
        ticker: Stock ticker (e.g., "RELIANCE.NS")
        capital_per_trade: Capital allocated per trade in ₹ (default: ₹25,000)
    """
    
    print(f"\n{'='*70}")
    print(f"📊 DAILY TRADE EXECUTION: {ticker}")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    # Get prediction
    trend, confidence, signal, regime, current_price = get_prediction(ticker)
    
    if trend is None:
        print(f"❌ Could not get prediction for {ticker}")
        return None
    
    # Get macro signals
    try:
        macro = get_macro_signals()
        macro_signal = macro.get_composite_macro_signal()
        macro_boost = macro_signal['composite_signal']
    except:
        macro_boost = 0.0
    
    print(f"\n📈 PREDICTION DATA:")
    print(f"  Current Price: ₹{current_price:,.2f}")
    print(f"  Trend: {'🔼 BULLISH' if trend == 1 else '🔽 BEARISH'}")
    print(f"  Confidence: {confidence:.0%}")
    print(f"  Signal Strength: {signal}")
    print(f"  Regime: {regime}")
    print(f"  Macro Boost: {macro_boost:+.2f}")
    
    print(f"\n💼 TRADE PARAMETERS:")
    print(f"  Capital Allocated: ₹{capital_per_trade:,.0f}")
    print(f"  Entry Price: ₹{current_price:,.2f}")
    print(f"  Target (+5%): ₹{current_price*1.05:,.2f}")
    print(f"  Stop Loss (-2%): ₹{current_price*0.98:,.2f}")
    print(f"  Risk per Trade: ₹{capital_per_trade * 0.02:,.0f}")
    print(f"  Potential Profit: ₹{capital_per_trade * 0.05:,.0f}")
    
    # Decision logic
    decision = "SKIP"
    reason = ""
    
    if confidence > 0.65 and signal != "weak":
        decision = "EXECUTE"
        reason = f"Strong signal ({confidence:.0%} confidence)"
    elif confidence > 0.55 and signal == "medium":
        decision = "CONSIDER"
        reason = f"Medium signal ({confidence:.0%} confidence)"
    else:
        decision = "SKIP"
        reason = f"Weak signal ({confidence:.0%} confidence, {signal} strength)"
    
    print(f"\n🎯 DECISION: {decision}")
    print(f"   Reason: {reason}")
    
    # If executing, log the trade
    if decision == "EXECUTE":
        trade_log = {
            "date": datetime.now().isoformat(),
            "ticker": ticker,
            "action": "BUY",
            "entry_price": current_price,
            "target_price": current_price * 1.05,
            "stop_loss": current_price * 0.98,
            "capital_allocated": capital_per_trade,
            "confidence": confidence,
            "signal": signal,
            "regime": regime,
            "macro_boost": macro_boost,
            "status": "OPEN"
        }
        
        print(f"\n✅ TRADE LOGGED:")
        print(f"  Entry: ₹{current_price:,.2f}")
        print(f"  Shares: {capital_per_trade / current_price:.0f}")
        
        return trade_log
    
    return None


def main():
    """Main trading function"""
    
    print("\n" + "="*70)
    print("🎯 70% ACCURACY SYSTEM - DAILY PAPER TRADING EXECUTION")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%A, %B %d, %Y')}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S IST')}")
    
    # Stocks to analyze today
    stocks_to_trade = [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
    ]
    
    print(f"\n📋 Analyzing {len(stocks_to_trade)} stocks for trading opportunities...")
    
    trades_executed = []
    
    for ticker in stocks_to_trade:
        trade = execute_daily_trade(ticker, capital_per_trade=25000)
        if trade:
            trades_executed.append(trade)
    
    # Summary
    print(f"\n\n{'='*70}")
    print("📊 DAILY SUMMARY")
    print(f"{'='*70}")
    print(f"Total Stocks Analyzed: {len(stocks_to_trade)}")
    print(f"Trades Executed: {len(trades_executed)}")
    print(f"Total Capital Deployed: ₹{sum(t['capital_allocated'] for t in trades_executed):,.0f}")
    
    if trades_executed:
        print(f"\n✅ Trade Details:")
        for i, trade in enumerate(trades_executed, 1):
            print(f"\n  Trade #{i}:")
            print(f"    Stock: {trade['ticker']}")
            print(f"    Entry: ₹{trade['entry_price']:,.2f}")
            print(f"    Target: ₹{trade['target_price']:,.2f}")
            print(f"    Stop: ₹{trade['stop_loss']:,.2f}")
            print(f"    Capital: ₹{trade['capital_allocated']:,.0f}")
            print(f"    Confidence: {trade['confidence']:.0%}")
    else:
        print(f"\n⚠️  No high-confidence signals today. Waiting for better opportunities.")
    
    # Save trade log
    log_file = Path("paper_trading_logs") / f"trades_{datetime.now().strftime('%Y%m%d')}.json"
    log_file.parent.mkdir(exist_ok=True)
    
    with open(log_file, 'w') as f:
        json.dump({
            "date": datetime.now().isoformat(),
            "stocks_analyzed": stocks_to_trade,
            "trades_executed": trades_executed,
            "summary": {
                "total_analyzed": len(stocks_to_trade),
                "total_executed": len(trades_executed),
                "total_capital": sum(t['capital_allocated'] for t in trades_executed)
            }
        }, f, indent=2)
    
    print(f"\n💾 Trade log saved: {log_file}")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
