import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import json

def get_all_nse_stocks():
    """
    Fetch all NSE listed stocks from NSE India website
    Returns: DataFrame with stock symbols and details
    """
    try:
        # NSE API endpoint for all equity stocks
        url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        session = requests.Session()
        # Get cookies first
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        
        # Fetch equity list
        response = session.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            stocks = [item['symbol'] for item in data.get('data', [])]
            
            df = pd.DataFrame(stocks, columns=['SYMBOL'])
            print(f"Total NSE stocks fetched: {len(df)}")
            return df
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            return get_nse_stocks_fallback()
            
    except Exception as e:
        print(f"Error fetching NSE stocks: {e}")
        return get_nse_stocks_fallback()

def get_nse_stocks_fallback():
    """
    Fallback list of major NSE stocks
    """
    major_stocks = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK",
        "HDFC", "BAJFINANCE", "BHARTIARTL", "SBIN", "ITC", "KOTAKBANK",
        "LT", "ASIANPAINT", "AXISBANK", "MARUTI", "TITAN", "SUNPHARMA",
        "ULTRACEMCO", "NESTLEIND", "WIPRO", "HCLTECH", "TECHM", "POWERGRID",
        "NTPC", "ONGC", "TATASTEEL", "TATAMOTORS", "ADANIPORTS", "JSWSTEEL",
        "INDUSINDBK", "BAJAJFINSV", "DIVISLAB", "DRREDDY", "CIPLA", "EICHERMOT",
        "HEROMOTOCO", "BPCL", "GRASIM", "COALINDIA", "SHREECEM", "BRITANNIA",
        "APOLLOHOSP", "UPL", "M&M", "HINDALCO", "ADANIENT", "TATACONSUM"
    ]
    
    df = pd.DataFrame(major_stocks, columns=['SYMBOL'])
    print(f"Using fallback list: {len(df)} stocks")
    return df

def get_nse_stock_symbols():
    """
    Get just the stock symbols as a list
    Returns: List of stock symbols
    """
    df = get_all_nse_stocks()
    if not df.empty and 'SYMBOL' in df.columns:
        return df['SYMBOL'].tolist()
    return []

def save_nse_stocks_to_csv(filename='nse_stocks.csv'):
    """
    Save all NSE stocks to CSV file
    """
    df = get_all_nse_stocks()
    if not df.empty:
        df.to_csv(filename, index=False)
        print(f"NSE stocks saved to {filename}")
        return filename
    return None

def get_stock_data(symbol, period="1mo"):
    """
    Fetch stock data using yfinance
    Add .NS suffix for NSE stocks
    """
    try:
        nse_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
        ticker = yf.Ticker(nse_symbol)
        data = ticker.history(period=period)
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def get_stock_info(symbol):
    """
    Get detailed stock information
    """
    try:
        nse_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
        ticker = yf.Ticker(nse_symbol)
        info = ticker.info
        return info
    except Exception as e:
        print(f"Error fetching info for {symbol}: {e}")
        return None