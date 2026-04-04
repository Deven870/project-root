"""
Sentiment Analysis Module
Analyze market sentiment, economic impact, and news
"""

import pandas as pd
from datetime import datetime, timedelta
import random  # Placeholder until real news API

class SentimentAnalyzer:
    """Analyze sentiment for trading decisions"""
    
    def __init__(self):
        self.sentiment_cache = {}
    
    def get_stock_sentiment(self, symbol):
        """
        Get sentiment for a stock from multiple sources
        Returns: -1 (bearish) to +1 (bullish)
        """
        try:
            # For now, generate based on technical indicators
            # TODO: Integrate with NewsAPI or Finnhub sentiment
            
            sentiment_score = random.uniform(-0.5, 1.0)  # Placeholder
            
            return {
                "symbol": symbol,
                "sentiment": sentiment_score,
                "interpretation": self._interpret_sentiment(sentiment_score),
                "sources": ["technical", "news", "social"],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error analyzing sentiment for {symbol}: {e}")
            return None
    
    @staticmethod
    def _interpret_sentiment(score):
        """Interpret sentiment score as text"""
        if score > 0.5:
            return "🟢 VERY BULLISH"
        elif score > 0.2:
            return "🟡 BULLISH"
        elif score > -0.2:
            return "⚪ NEUTRAL"
        elif score > -0.5:
            return "🔴 BEARISH"
        else:
            return "🔴 VERY BEARISH"
    
    def get_economic_calendar_impact(self):
        """
        Get today's economic calendar events
        Returns: List of events and their impact on markets
        """
        events = [
            {
                "event": "RBI Policy Rate Decision",
                "time": "14:30",
                "impact": "HIGH",
                "previous": "6.5%",
                "forecast": "6.5%",
                "affected_symbols": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"],
                "direction": "BEARISH"  # If rate increases = bearish for stocks
            },
            {
                "event": "Inflation Data Release",
                "time": "12:00",
                "impact": "HIGH",
                "previous": "5.7%",
                "forecast": "5.5%",
                "affected_symbols": ["SBIN.NS", "RELIANCE.NS"],
                "direction": "BULLISH"  # Lower inflation = bullish
            },
            {
                "event": "Corporate Earnings",
                "time": "16:00",
                "impact": "MEDIUM",
                "companies": ["TCS.NS", "INFY.NS"],
                "direction": "MIXED"
            }
        ]
        return events
    
    def get_news_sentiment(self, symbol, limit=5):
        """
        Get recent news and sentiment for a stock
        TODO: Integrate with NewsAPI (newsapi.org)
        """
        # Placeholder - should call actual news API
        news_items = [
            {
                "symbol": symbol,
                "title": f"Technical Analysis: {symbol} breakout expected",
                "sentiment": 0.7,
                "source": "Trading Blog",
                "date": datetime.now() - timedelta(hours=2)
            },
            {
                "symbol": symbol,
                "title": f"Buy signal detected in {symbol}",
                "sentiment": 0.6,
                "source": "Technical Analysis",
                "date": datetime.now() - timedelta(hours=4)
            }
        ]
        return news_items[:limit]
    
    def generate_combined_sentiment(self, symbol, technical_score=0, news_score=0, 
                                   economic_impact=0):
        """
        Combine multiple sentiment sources for final trading decision
        """
        weights = {
            "technical": 0.4,
            "news": 0.35,
            "economic": 0.25
        }
        
        combined = (
            technical_score * weights["technical"] +
            news_score * weights["news"] +
            economic_impact * weights["economic"]
        )
        
        # Normalize to -1 to +1
        combined = max(-1, min(1, combined))
        
        return {
            "symbol": symbol,
            "combined_sentiment": combined,
            "interpretation": self._interpret_sentiment(combined),
            "weights": weights,
            "components": {
                "technical": technical_score,
                "news": news_score,
                "economic": economic_impact
            },
            "recommendation": self._get_trading_recommendation(combined),
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def _get_trading_recommendation(sentiment_score):
        """Get trading recommendation based on sentiment"""
        if sentiment_score > 0.6:
            return "🟢 STRONG BUY"
        elif sentiment_score > 0.2:
            return "🟡 BUY"
        elif sentiment_score > -0.2:
            return "⚪ HOLD"
        elif sentiment_score > -0.6:
            return "🔴 SELL"
        else:
            return "🔴 STRONG SELL"
    
    def get_sentiment_for_portfolio(self, symbols):
        """Get sentiment for entire portfolio"""
        sentiments = []
        for symbol in symbols:
            sentiment = self.get_stock_sentiment(symbol)
            if sentiment:
                sentiments.append(sentiment)
        
        return pd.DataFrame(sentiments) if sentiments else pd.DataFrame()


def analyze_ml_prediction_accuracy(actual_prices, predicted_prices):
    """
    Compare ML predictions vs actual prices
    Calculate accuracy metrics
    """
    try:
        df = pd.DataFrame({
            'Actual': actual_prices,
            'Predicted': predicted_prices
        })
        
        df['Error'] = df['Actual'] - df['Predicted']
        df['Error %'] = (df['Error'] / df['Actual']) * 100
        df['Correct Direction'] = (
            (df['Actual'] > df['Actual'].shift(1)) == 
            (df['Predicted'] > df['Predicted'].shift(1))
        ).astype(int)
        
        accuracy = df['Correct Direction'].mean() * 100
        mae = df['Error %'].abs().mean()
        
        return {
            "accuracy": round(accuracy, 2),
            "mae": round(mae, 2),
            "details": df.to_dict('records')
        }
    except Exception as e:
        print(f"❌ Error analyzing ML accuracy: {e}")
        return None
