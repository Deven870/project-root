"""
Advanced Multi-API Integration Layer
Combines Finnhub + Alpha Vantage + NewsAPI + Gemini for superior predictions
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import json
from functools import lru_cache
import time

# API Keys
FINNHUB_KEY = "d78dqqhr01qhel7vjal0d78dqqhr01qhel7vjalg"
ALPHA_VANTAGE_KEY = "K5T3L5U9N6QFQLXB"
NEWSAPI_KEY = "92a2bc8ddf5f4a6c916643ed8257a621"
GEMINI_KEY = "AIzaSyDLcz9qeozO9tFNE1fv0mVVZNe-tFj2s-U"

# Base URLs
FINNHUB_URL = "https://finnhub.io/api/v1"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
NEWSAPI_URL = "https://newsapi.org/v2"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"


class AdvancedDataClient:
    """Unified API client for all data sources"""
    
    def __init__(self):
        self.rate_limit_time = 0
        self.last_requests = {}
    
    # ============ FINNHUB INTEGRATION ============
    def get_finnhub_quote(self, symbol):
        """Get real-time stock quote from Finnhub"""
        try:
            params = {
                "symbol": symbol,
                "token": FINNHUB_KEY
            }
            response = requests.get(f"{FINNHUB_URL}/quote", params=params, timeout=5)
            data = response.json()
            
            if data:
                return {
                    "symbol": symbol,
                    "price": data.get("c", 0),
                    "high": data.get("h", 0),
                    "low": data.get("l", 0),
                    "open": data.get("o", 0),
                    "previous_close": data.get("pc", 0),
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            print(f"❌ Finnhub quote error for {symbol}: {e}")
        return None
    
    def get_finnhub_sentiment(self, symbol):
        """Get sentiment analysis from Finnhub (company news + sentiment)"""
        try:
            params = {
                "symbol": symbol,
                "token": FINNHUB_KEY
            }
            response = requests.get(f"{FINNHUB_URL}/company-news", params=params, timeout=5)
            news_items = response.json()
            
            if news_items:
                sentiments = []
                for item in news_items[:20]:  # Last 20 news items
                    # Extract sentiment from headline analysis
                    headline = item.get("headline", "")
                    sentiment = self._analyze_text_sentiment(headline)
                    sentiments.append(sentiment)
                
                avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
                
                return {
                    "symbol": symbol,
                    "sentiment_score": avg_sentiment,
                    "news_count": len(news_items),
                    "interpretation": self._interpret_sentiment(avg_sentiment),
                    "source": "finnhub",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            print(f"❌ Finnhub sentiment error: {e}")
        return None
    
    def get_finnhub_company_profile(self, symbol):
        """Get company fundamentals from Finnhub"""
        try:
            params = {
                "symbol": symbol,
                "token": FINNHUB_KEY
            }
            response = requests.get(f"{FINNHUB_URL}/company-basic-financials", params=params, timeout=5)
            data = response.json()
            
            if data and "series" in data:
                metrics = data["series"].get("quarterly", [])
                if metrics:
                    latest = metrics[-1]  # Most recent quarter
                    return {
                        "symbol": symbol,
                        "pe_ratio": latest.get("roic", {}).get("v"),
                        "revenue_growth": latest.get("totalRevenue"),
                        "profit_margin": latest.get("netMargin"),
                        "roe": latest.get("roe")
                    }
        except Exception as e:
            print(f"❌ Finnhub profile error: {e}")
        return None
    
    # ============ NEWSAPI INTEGRATION ============
    def get_news_sentiment_bulk(self, symbol, limit=30):
        """Get news articles from NewsAPI with detailed sentiment analysis"""
        try:
            # Map NSE symbols to company names for better news matching
            company_names = {
                "RELIANCE.NS": "Reliance Industries",
                "TCS.NS": "Tata Consultancy Services",
                "INFY.NS": "Infosys",
                "HDFCBANK.NS": "HDFC Bank",
                "ICICIBANK.NS": "ICICI Bank"
            }
            
            query = company_names.get(symbol, symbol)
            
            params = {
                "q": query,
                "sortBy": "publishedAt",
                "limit": limit,
                "apiKey": NEWSAPI_KEY
            }
            
            response = requests.get(f"{NEWSAPI_URL}/everything", params=params, timeout=5)
            articles = response.json().get("articles", [])
            
            if articles:
                sentiments = []
                for article in articles:
                    title = article.get("title", "")
                    description = article.get("description", "")
                    text = f"{title} {description}"
                    
                    sentiment = self._analyze_text_sentiment(text)
                    sentiments.append({
                        "title": title,
                        "sentiment": sentiment,
                        "published_at": article.get("publishedAt"),
                        "source": article.get("source", {}).get("name")
                    })
                
                avg_sentiment = sum(s["sentiment"] for s in sentiments) / len(sentiments) if sentiments else 0
                
                return {
                    "symbol": symbol,
                    "news_sentiment": avg_sentiment,
                    "articles_analyzed": len(articles),
                    "recent_articles": sentiments[:5],
                    "interpretation": self._interpret_sentiment(avg_sentiment),
                    "source": "newsapi",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            print(f"❌ NewsAPI error: {e}")
        return None
    
    # ============ GEMINI AI ANALYSIS ============
    def get_gemini_prediction(self, symbol, technical_data, sentiment_data, news_data):
        """Use Gemini AI to generate advanced trading predictions"""
        try:
            prompt = f"""
            Analyze this stock data and provide a trading recommendation:
            
            Symbol: {symbol}
            
            Technical Data:
            - Price: {technical_data.get('price', 'N/A')}
            - RSI: {technical_data.get('rsi', 'N/A')}
            - MACD: {technical_data.get('macd', 'N/A')}
            - Bollinger Bands: {technical_data.get('bollinger', 'N/A')}
            
            Sentiment Data:
            - Finnhub Sentiment: {sentiment_data.get('finnhub_sentiment', 'N/A')}
            - News Sentiment: {sentiment_data.get('news_sentiment', 'N/A')}
            - Recent News: {news_data}
            
            Provide:
            1. Trading recommendation (BUY/SELL/HOLD)
            2. Confidence level (0-100%)
            3. Target price
            4. Risk level
            5. Key factors influencing the decision
            
            Format as JSON.
            """
            
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            }
            
            response = requests.post(
                f"{GEMINI_URL}?key={GEMINI_KEY}",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    text_response = result["candidates"][0]["content"]["parts"][0]["text"]
                    
                    # Parse JSON from response
                    try:
                        # Extract JSON from response (it might be wrapped in markdown)
                        import re
                        json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
                        if json_match:
                            prediction = json.loads(json_match.group())
                            prediction["source"] = "gemini"
                            prediction["timestamp"] = datetime.now().isoformat()
                            return prediction
                    except:
                        return {
                            "raw_response": text_response,
                            "source": "gemini",
                            "timestamp": datetime.now().isoformat()
                        }
        except Exception as e:
            print(f"❌ Gemini AI error: {e}")
        
        return None
    
    # ============ SENTIMENT ANALYSIS ============
    @staticmethod
    def _analyze_text_sentiment(text):
        """Simple sentiment analysis on text (-1 to +1)"""
        positive_words = ['bullish', 'up', 'gain', 'rise', 'surge', 'rally', 'strong', 'beat', 'outperform', 'buy']
        negative_words = ['bearish', 'down', 'loss', 'fall', 'decline', 'slump', 'weak', 'miss', 'underperform', 'sell']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0
        
        return (pos_count - neg_count) / total
    
    @staticmethod
    def _interpret_sentiment(score):
        """Convert sentiment score to readable text"""
        if score > 0.6:
            return "🟢 VERY BULLISH"
        elif score > 0.2:
            return "🟢 BULLISH"
        elif score > -0.2:
            return "⚪ NEUTRAL"
        elif score > -0.6:
            return "🔴 BEARISH"
        else:
            return "🔴 VERY BEARISH"
    
    # ============ CONSOLIDATED ANALYSIS ============
    def get_comprehensive_analysis(self, symbol, technical_data):
        """Get all analyses combined for final prediction"""
        try:
            print(f"🔍 Analyzing {symbol}...")
            
            # Fetch from all sources
            finnhub_quote = self.get_finnhub_quote(symbol)
            finnhub_sentiment = self.get_finnhub_sentiment(symbol)
            finnhub_profile = self.get_finnhub_company_profile(symbol)
            news_sentiment = self.get_news_sentiment_bulk(symbol)
            
            # Compile sentiment data
            sentiment_data = {
                "finnhub_sentiment": finnhub_sentiment.get("sentiment_score", 0) if finnhub_sentiment else 0,
                "news_sentiment": news_sentiment.get("news_sentiment", 0) if news_sentiment else 0,
                "finnhub_interpretation": finnhub_sentiment.get("interpretation", "N/A") if finnhub_sentiment else "N/A",
                "news_interpretation": news_sentiment.get("interpretation", "N/A") if news_sentiment else "N/A"
            }
            
            # Get Gemini AI prediction
            gemini_prediction = self.get_gemini_prediction(
                symbol,
                technical_data,
                sentiment_data,
                news_sentiment.get("recent_articles", []) if news_sentiment else []
            )
            
            # Compile final analysis
            analysis = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "price_data": finnhub_quote,
                "technical_data": technical_data,
                "sentiment_analysis": sentiment_data,
                "fundamental_data": finnhub_profile,
                "news_analysis": {
                    "sentiment": news_sentiment.get("news_sentiment") if news_sentiment else 0,
                    "articles_count": news_sentiment.get("articles_analyzed", 0) if news_sentiment else 0,
                    "recent": news_sentiment.get("recent_articles", []) if news_sentiment else []
                },
                "gemini_prediction": gemini_prediction,
                "overall_score": self._calculate_overall_score(sentiment_data, gemini_prediction)
            }
            
            return analysis
        
        except Exception as e:
            print(f"❌ Comprehensive analysis error: {e}")
            return None
    
    @staticmethod
    def _calculate_overall_score(sentiment_data, gemini_prediction):
        """Calculate weighted overall prediction score"""
        scores = []
        
        # Finnhub sentiment weight: 30%
        if sentiment_data.get("finnhub_sentiment"):
            scores.append(sentiment_data["finnhub_sentiment"] * 0.3)
        
        # News sentiment weight: 30%
        if sentiment_data.get("news_sentiment"):
            scores.append(sentiment_data["news_sentiment"] * 0.3)
        
        # Gemini AI weight: 40% (most advanced)
        if gemini_prediction:
            confidence = gemini_prediction.get("confidence_level", 50) / 100  # Convert to 0-1
            scores.append(confidence * 0.4)
        
        return sum(scores) / len(scores) if scores else 0


# ============ USAGE EXAMPLE ============
def demonstrate():
    """Demonstrate the advanced analysis system"""
    client = AdvancedDataClient()
    
    symbol = "RELIANCE.NS"
    
    # Get technical data (from Alpha Vantage)
    technical_data = {
        "price": 2500,
        "rsi": 65,
        "macd": "bullish",
        "bollinger": "upper band"
    }
    
    # Get comprehensive analysis
    analysis = client.get_comprehensive_analysis(symbol, technical_data)
    
    if analysis:
        print(f"\n✅ Analysis for {symbol}:")
        print(json.dumps(analysis, indent=2))
        return analysis
    else:
        print(f"❌ Failed to analyze {symbol}")
        return None
