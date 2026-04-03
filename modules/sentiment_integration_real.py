"""
Real Sentiment Integration with APIs
Connects to real news sources and social media for sentiment analysis
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from modules.sentiment_engine import analyze_finbert
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class RealSentimentIntegrator:
    """Integrates real sentiment data from multiple sources"""

    def __init__(self):
        self.cache = {}
        self.cache_timestamp = {}
        self.cache_duration = 3600  # 1 hour

        # API configuration (provide your own keys)
        self.newsapi_key = self._load_api_key('newsapi_key')
        self.finnhub_key = self._load_api_key('finnhub_key')

    def _load_api_key(self, key_name):
        """Load API key from config file"""
        try:
            config_file = Path('config_api_keys.json')
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    return config.get(key_name, '')
        except Exception as e:
            logger.warning(f"Could not load {key_name}: {e}")
        return ''

    def _is_cache_valid(self, key):
        """Check if cache is still valid"""
        if key not in self.cache_timestamp:
            return False
        age = (datetime.now() - self.cache_timestamp[key]).total_seconds()
        return age < self.cache_duration

    def get_newsapi_sentiment(self, ticker, company_name, days=7):
        """
        Get sentiment from NewsAPI
        Requires: NewsAPI key from https://newsapi.org
        
        Args:
            ticker: Stock ticker
            company_name: Company name for search
            days: Days to look back
            
        Returns:
            dict: {
                'sentiment_score': -1 to +1,
                'articles': count,
                'source': 'NewsAPI'
            }
        """
        try:
            if self._is_cache_valid(f'newsapi_{ticker}'):
                return self.cache[f'newsapi_{ticker}']

            if not self.newsapi_key:
                logger.debug("NewsAPI key not configured")
                return self._get_fallback_sentiment()

            # Fetch articles
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f"{company_name} OR {ticker}",
                'from': (datetime.now() - timedelta(days=days)).isoformat(),
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': self.newsapi_key
            }

            response = requests.get(url, params=params, timeout=5)

            if response.status_code != 200:
                logger.warning(f"NewsAPI error: {response.status_code}")
                return self._get_fallback_sentiment()

            articles = response.json().get('articles', [])

            if not articles:
                logger.debug(f"No articles found for {ticker}")
                return self._get_fallback_sentiment()

            # Sentiment analysis on headlines
            sentiments = []
            for article in articles[:20]:  # Top 20 recent
                headline = article.get('title', '')
                sentiment = analyze_finbert(headline)
                
                # Convert to -1 to +1 score
                score = sentiment.get('positive', 0) - sentiment.get('negative', 0)
                sentiments.append(score)

            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

            result = {
                'sentiment_score': float(avg_sentiment),
                'articles': len(articles),
                'source': 'NewsAPI',
                'strength': self._sentiment_strength(avg_sentiment)
            }

            self.cache[f'newsapi_{ticker}'] = result
            self.cache_timestamp[f'newsapi_{ticker}'] = datetime.now()

            logger.info(f"{ticker}: NewsAPI sentiment {avg_sentiment:+.2f} ({len(articles)} articles)")
            return result

        except Exception as e:
            logger.warning(f"NewsAPI sentiment failed: {e}")
            return self._get_fallback_sentiment()

    def get_finnhub_sentiment(self, ticker, days=7):
        """
        Get sentiment from Finnhub
        Requires: Finnhub API key from https://finnhub.io
        
        Args:
            ticker: Stock ticker
            days: Days to look back
            
        Returns:
            dict: Sentiment data
        """
        try:
            if self._is_cache_valid(f'finnhub_{ticker}'):
                return self.cache[f'finnhub_{ticker}']

            if not self.finnhub_key:
                logger.debug("Finnhub key not configured")
                return self._get_fallback_sentiment()

            # Fetch company news
            url = f"https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': ticker,
                'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                'to': datetime.now().strftime('%Y-%m-%d'),
                'token': self.finnhub_key
            }

            response = requests.get(url, params=params, timeout=5)

            if response.status_code != 200:
                logger.warning(f"Finnhub error: {response.status_code}")
                return self._get_fallback_sentiment()

            articles = response.json()

            if not articles:
                return self._get_fallback_sentiment()

            # Analyze sentiment
            sentiments = []
            for article in articles[:15]:
                headline = article.get('headline', '')
                sentiment = analyze_finbert(headline)
                score = sentiment.get('positive', 0) - sentiment.get('negative', 0)
                sentiments.append(score)

            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

            result = {
                'sentiment_score': float(avg_sentiment),
                'articles': len(articles),
                'source': 'Finnhub',
                'strength': self._sentiment_strength(avg_sentiment)
            }

            self.cache[f'finnhub_{ticker}'] = result
            self.cache_timestamp[f'finnhub_{ticker}'] = datetime.now()

            logger.info(f"{ticker}: Finnhub sentiment {avg_sentiment:+.2f}")
            return result

        except Exception as e:
            logger.warning(f"Finnhub sentiment failed: {e}")
            return self._get_fallback_sentiment()

    def get_twitter_sentiment(self, ticker):
        """
        Get Twitter sentiment
        Note: Requires Twitter API v2 credentials and elevated access
        
        This is a placeholder; implement with tweepy or similar
        """
        logger.info("Twitter sentiment tracking - placeholder")
        return self._get_fallback_sentiment()

    def get_reddit_sentiment(self, subreddit_keywords):
        """
        Get Reddit sentiment
        Note: Requires PRAW (Python Reddit API Wrapper)
        
        This is a placeholder; implement with praw library
        """
        logger.info("Reddit sentiment tracking - placeholder")
        return self._get_fallback_sentiment()

    def _sentiment_strength(self, score):
        """Categorize sentiment strength"""
        abs_score = abs(score)
        if abs_score > 0.6:
            return 'strong'
        elif abs_score > 0.3:
            return 'moderate'
        else:
            return 'weak'

    def _get_fallback_sentiment(self):
        """Return neutral sentiment when APIs unavailable"""
        return {
            'sentiment_score': 0.0,
            'articles': 0,
            'source': 'fallback',
            'strength': 'weak'
        }

    def get_composite_sentiment(self, ticker, company_name):
        """
        Get composite sentiment from all available sources
        
        Returns:
            dict: {
                'composite_score': -1 to +1,
                'sources': list of source results,
                'confidence': 0-1,
                'recommendation': 'bullish' | 'neutral' | 'bearish'
            }
        """
        sources = []

        # Try NewsAPI
        if self.newsapi_key:
            newsapi = self.get_newsapi_sentiment(ticker, company_name)
            sources.append(newsapi)

        # Try Finnhub
        if self.finnhub_key:
            finnhub = self.get_finnhub_sentiment(ticker)
            sources.append(finnhub)

        # Always include fallback
        if not sources:
            sources.append(self._get_fallback_sentiment())

        # Average scores
        scores = [s['sentiment_score'] for s in sources]
        composite_score = sum(scores) / len(scores) if scores else 0

        # Confidence based on number of sources
        confidence = min(len(sources) * 0.3, 0.9)

        # Recommendation
        if composite_score > 0.3:
            recommendation = 'bullish'
        elif composite_score < -0.3:
            recommendation = 'bearish'
        else:
            recommendation = 'neutral'

        result = {
            'composite_score': float(composite_score),
            'sources': sources,
            'confidence': float(confidence),
            'recommendation': recommendation,
            'num_sources': len(sources)
        }

        logger.info(f"{ticker}: Composite sentiment {composite_score:+.2f} ({recommendation})")
        return result


class SentimentBooster:
    """Boosts prediction confidence based on sentiment"""

    def __init__(self):
        self.integrator = RealSentimentIntegrator()

    def boost_prediction(self, prediction, ticker, company_name):
        """
        Boost prediction confidence with sentiment data
        
        Args:
            prediction: dict with 'trend' and 'confidence'
            ticker: Stock ticker
            company_name: Company name
            
        Returns:
            dict: Boosted prediction
        """
        sentiment = self.integrator.get_composite_sentiment(ticker, company_name)

        trend = prediction.get('trend', 0)
        base_confidence = prediction.get('confidence', 0.5)

        # Sentiment alignment boost
        if (trend == 1 and sentiment['composite_score'] > 0.2) or \
           (trend == 0 and sentiment['composite_score'] < -0.2):
            # Sentiment aligns with prediction
            boost = sentiment['confidence'] * 0.15  # Up to 15% boost
        else:
            # Sentiment contradicts prediction
            boost = -sentiment['confidence'] * 0.08  # Up to -8% penalty

        boosted_confidence = max(0, min(1, base_confidence + boost))

        return {
            'trend': trend,
            'base_confidence': base_confidence,
            'sentiment_score': sentiment['composite_score'],
            'boosted_confidence': boosted_confidence,
            'boost_applied': boost,
            'sentiment_recommendation': sentiment['recommendation']
        }


# Setup API keys configuration file
def create_api_keys_template():
    """Create template for API keys configuration"""
    template = {
        "newsapi_key": "your_newsapi_key_here",
        "finnhub_key": "your_finnhub_key_here",
        "twitter_bearer_token": "your_twitter_bearer_token_here",
        "reddit_client_id": "your_reddit_client_id_here",
        "reddit_client_secret": "your_reddit_client_secret_here"
    }

    config_file = Path('config_api_keys.json')
    if not config_file.exists():
        with open(config_file, 'w') as f:
            json.dump(template, f, indent=2)
        logger.info(f"Created API keys template at {config_file}")
        logger.info("Please fill in your API keys and keep the file secure!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create template
    create_api_keys_template()

    # Test without real API keys (will use fallback)
    print("\n" + "=" * 60)
    print("SENTIMENT INTEGRATION TEST")
    print("=" * 60)

    integrator = RealSentimentIntegrator()

    # Test composite sentiment (will be neutral without API keys)
    sentiment = integrator.get_composite_sentiment(
        ticker="RELIANCE",
        company_name="Reliance Industries"
    )

    print(f"\nComposite Sentiment: {sentiment['composite_score']:+.2f}")
    print(f"Recommendation: {sentiment['recommendation']}")
    print(f"Confidence: {sentiment['confidence']:.0%}")
    print(f"Sources: {sentiment['num_sources']}")

    # Test sentiment booster
    print("\n" + "=" * 60)
    print("SENTIMENT BOOST EXAMPLE")
    print("=" * 60)

    booster = SentimentBooster()
    prediction = {'trend': 1, 'confidence': 0.70}
    boosted = booster.boost_prediction(prediction, "RELIANCE", "Reliance Industries")

    print(f"\nBase confidence: {boosted['base_confidence']:.0%}")
    print(f"Sentiment score: {boosted['sentiment_score']:+.2f}")
    print(f"Boosted confidence: {boosted['boosted_confidence']:.0%}")
    print(f"Boost applied: {boosted['boost_applied']:+.1%}")
