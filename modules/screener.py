"""
Real-time Stock Screener
Fast screening for 20-30 NSE stocks with technical + fundamental metrics
"""

import pandas as pd
from modules.alpha_vantage_client import get_technical_indicators, get_quote
import streamlit as st

# Top NSE stocks to screen
DEFAULT_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HDFC.NS", "WIPRO.NS", "AXIS.NS", "MARUTI.NS", "BAJAJFINSV.NS",
    "LT.NS", "SUNPHARMA.NS", "NESTLEIND.NS", "SBIN.NS", "INDUSINDBK.NS",
    "BAJAJFINSV.NS", "POWERGRID.NS", "NIFTY50.NS", "TECHM.NS", "HCLTECH.NS"
]


@st.cache_data(ttl=300)  # Cache for 5 minutes
def screen_stocks(symbols=None, min_rsi=30, max_rsi=70):
    """
    Screen stocks based on technical criteria
    
    Args:
        symbols: List of stock symbols to screen
        min_rsi: Minimum RSI (oversold)
        max_rsi: Maximum RSI (overbought)
    
    Returns:
        DataFrame with screened stocks and metrics
    """
    if symbols is None:
        symbols = DEFAULT_STOCKS
    
    screened_stocks = []
    progress_bar = st.progress(0)
    
    for idx, symbol in enumerate(symbols):
        try:
            # Get quote
            quote = get_quote(symbol)
            if not quote:
                progress_bar.progress((idx + 1) / len(symbols))
                continue
            
            # Get technical indicators
            indicators = get_technical_indicators(symbol)
            if not indicators:
                progress_bar.progress((idx + 1) / len(symbols))
                continue
            
            # Calculate signals
            rsi = indicators.get('rsi', None)
            macd = indicators.get('macd', {})
            bb = indicators.get('bollinger_bands', {})
            
            # Signal determination
            signal = "NEUTRAL"
            if rsi is not None:
                if rsi < 30:
                    signal = "OVERSOLD (BUY)"
                elif rsi > 70:
                    signal = "OVERBOUGHT (SELL)"
                elif rsi < 40:
                    signal = "WEAK"
                elif rsi > 60:
                    signal = "STRONG"
            
            # MACD signal
            macd_signal = "NEUTRAL"
            if macd and macd.get('histogram', 0) > 0:
                macd_signal = "BULLISH"
            elif macd and macd.get('histogram', 0) < 0:
                macd_signal = "BEARISH"
            
            screened_stocks.append({
                'Symbol': symbol,
                'Price': quote['price'],
                'Change %': quote['change_pct'],
                'RSI': round(rsi, 2) if rsi else "N/A",
                'Signal': signal,
                'MACD': macd_signal,
                'Volatility %': round(indicators['volatility'], 2),
                'Volume': quote['volume'],
                'Action': 'BUY' if 'BUY' in signal else ('SELL' if 'SELL' in signal else 'HOLD')
            })
        
        except Exception as e:
            print(f"❌ Error screening {symbol}: {e}")
        
        progress_bar.progress((idx + 1) / len(symbols))
    
    if screened_stocks:
        df = pd.DataFrame(screened_stocks)
        return df.sort_values('RSI', ascending=False)
    
    return pd.DataFrame()


def get_sector_performance(symbols=None):
    """Get performance by sector"""
    if symbols is None:
        symbols = DEFAULT_STOCKS
    
    sectors = {
        'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS'],
        'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'AXIS.NS', 'SBIN.NS', 'INDUSINDBK.NS'],
        'Finance': ['HDFC.NS', 'BAJAJFINSV.NS'],
        'Auto': ['MARUTI.NS'],
        'Pharma': ['SUNPHARMA.NS'],
        'Energy': ['RELIANCE.NS', 'POWERGRID.NS'],
        'Utilities': ['NESTLE.NS', 'LT.NS']
    }
    
    sector_performance = []
    
    for sector, sector_stocks in sectors.items():
        sector_changes = []
        for stock in sector_stocks:
            if stock in symbols:
                quote = get_quote(stock)
                if quote:
                    sector_changes.append(quote['change_pct'])
        
        if sector_changes:
            avg_change = sum(sector_changes) / len(sector_changes)
            sector_performance.append({
                'Sector': sector,
                'Avg Change %': round(avg_change, 2),
                'Stocks': len(sector_changes),
                'Status': '🟢 UP' if avg_change > 0 else '🔴 DOWN'
            })
    
    return pd.DataFrame(sector_performance).sort_values('Avg Change %', ascending=False)


def get_top_movers(symbols=None, limit=10):
    """Get top gainers and losers"""
    if symbols is None:
        symbols = DEFAULT_STOCKS
    
    quotes = []
    for symbol in symbols:
        quote = get_quote(symbol)
        if quote:
            quotes.append(quote)
    
    df = pd.DataFrame(quotes)
    
    gainers = df.nlargest(limit // 2, 'change_pct')[['symbol', 'price', 'change_pct']]
    losers = df.nsmallest(limit // 2, 'change_pct')[['symbol', 'price', 'change_pct']]
    
    return gainers, losers


def calculate_correlation_matrix(symbols=None):
    """Calculate correlation between stocks"""
    if symbols is None:
        symbols = DEFAULT_STOCKS[:10]  # Top 10 for performance
    
    try:
        prices = {}
        for symbol in symbols:
            # Get last 60 days of data
            quote = get_quote(symbol)
            if quote:
                prices[symbol] = quote['price']
        
        if prices:
            df = pd.DataFrame([prices])
            return df.corr()
    
    except Exception as e:
        print(f"❌ Error calculating correlation: {e}")
    
    return None
