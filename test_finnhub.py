#!/usr/bin/env python3
"""Test Finnhub API directly"""
import requests

FINNHUB_KEY = "d78dqqhr01qhel7vjal0"

print("Testing Finnhub API...")

try:
    params = {"symbol": "INFY", "token": FINNHUB_KEY}
    response = requests.get("https://finnhub.io/api/v1/company-news", params=params, timeout=5)
    news = response.json()
    
    print(f"Response type: {type(news)}")
    print(f"Response length: {len(news) if isinstance(news, (list, dict)) else 'N/A'}")
    
    if isinstance(news, list):
        print(f"Is list: Yes, items: {len(news)}")
        if news:
            print(f"First item keys: {list(news[0].keys())}")
    elif isinstance(news, dict):
        print(f"Is dict: Yes, keys: {list(news.keys())}")
        if "data" in news:
            print(f"Has 'data' key: Yes, type: {type(news['data'])}")
    
    # Also try with .NS suffix
    print("\nTrying with .NS suffix...")
    params = {"symbol": "INFY.NS", "token": FINNHUB_KEY}
    response = requests.get("https://finnhub.io/api/v1/company-news", params=params, timeout=5)
    news = response.json()
    print(f"Response type: {type(news)}, Length: {len(news) if isinstance(news, (list, dict)) else 'N/A'}")
    
except Exception as e:
    print(f"Error: {e}")
