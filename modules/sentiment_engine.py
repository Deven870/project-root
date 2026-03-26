import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
try:
    from textblob import TextBlob
    _TEXTBLOB_AVAILABLE = True
except Exception:
    # TextBlob may not be installed; fall back to simpler analysis
    TextBlob = None
    _TEXTBLOB_AVAILABLE = False
import requests
from datetime import datetime, timedelta

# ------------------------------
# FinBERT for financial sentiment
# ------------------------------
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer_finbert = None
model_finbert = None
labels = ["negative", "neutral", "positive"]

def analyze_finbert(text):
    global tokenizer_finbert, model_finbert
    if tokenizer_finbert is None or model_finbert is None:
        try:
            tokenizer_finbert = AutoTokenizer.from_pretrained(finbert_model_name)
            model_finbert = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)
        except Exception as e:
            print(f"Warning: FinBERT load failed: {e}")
            return {"negative": 0.0, "neutral": 1.0, "positive": 0.0}
    try:
        inputs = tokenizer_finbert(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model_finbert(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
        return dict(zip(labels, probs.tolist()[0]))
    except Exception as e:
        print(f"FinBERT analysis error: {e}")
        return {"negative": 0.0, "neutral": 1.0, "positive": 0.0}

def analyze_general_sentiment(text):
    if _TEXTBLOB_AVAILABLE:
        try:
            blob = TextBlob(text)
            p = blob.sentiment.polarity
        except Exception:
            p = 0.0
    else:
        # Very small rule-based fallback
        t = str(text).lower()
        pos = any(w in t for w in ["gain","profit","rise","up","strong","good","positive","outperform"])
        neg = any(w in t for w in ["loss","drop","down","weak","fail","negative","decline","miss"])
        if pos and not neg:
            p = 0.5
        elif neg and not pos:
            p = -0.5
        else:
            p = 0.0
    if p > 0.05:
        return {"negative": 0, "neutral": 0, "positive": 1}
    elif p < -0.05:
        return {"negative": 1, "neutral": 0, "positive": 0}
    else:
        return {"negative": 0, "neutral": 1, "positive": 0}

def analyze_hybrid_sentiment(text):
    fin = analyze_finbert(text)
    gen = analyze_general_sentiment(text)
    try:
        hybrid = {k: round(0.7*fin.get(k, 0) + 0.3*gen.get(k, 0), 4) for k in labels}
    except Exception:
        # If fin or gen returns a scalar, fallback to combining as proportions
        hybrid = {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
    return hybrid

# ------------------------------
# News fetching (from Finnhub via finnhub_feed module)
# ------------------------------
import os
from dotenv import load_dotenv
from modules.finnhub_feed import get_company_news as fetch_company_news
load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

def get_news_for_stock(stock_ticker, from_days=7, max_articles=20):
    """
    Fetch news for stock using Finnhub API via finnhub_feed module.
    
    Parameters
    ----------
    stock_ticker : str
        Stock symbol (e.g., "RELIANCE.NS", "AAPL")
    from_days : int
        Number of days back to fetch news from
    max_articles : int
        Maximum number of articles to return
    
    Returns
    -------
    list
        List of dicts with keys: headline, summary, url, source
    """
    try:
        today = datetime.now()
        from_date = today - timedelta(days=from_days)
        to_date = today
        
        # Fetch news using finnhub_feed module
        articles = fetch_company_news(
            stock_ticker,
            from_date=from_date.strftime('%Y-%m-%d'),
            to_date=to_date.strftime('%Y-%m-%d')
        )
        
        # Extract relevant fields and limit to max_articles
        headlines = []
        for article in articles[:max_articles]:
            headlines.append({
                "headline": article.get("headline", ""),
                "summary": article.get("summary", ""),
                "url": article.get("url", ""),
                "source": article.get("source", "")
            })
        
        return headlines
    
    except Exception as e:
        print(f"Error fetching news for {stock_ticker}: {e}")
        return []

def get_news_with_sentiment(stock_ticker, from_days=7, max_articles=20, push_to_sheets=True):
    """
    Fetch news and analyze sentiment, optionally pushing to Google Sheets.
    Returns list of news items with sentiment analysis.
    
    The sentiment analysis uses the hybrid approach (FinBERT + TextBlob).
    """
    try:
        headlines = get_news_for_stock(stock_ticker, from_days, max_articles)
        news_items = []
        
        for article in headlines:
            headline_text = article.get("headline", "")
            url = article.get("url", "")
            
            # Analyze sentiment of headline and summary combined
            text_to_analyze = f"{headline_text} {article.get('summary', '')}"
            sentiment_scores = analyze_hybrid_sentiment(text_to_analyze)
            
            # Determine sentiment label
            if sentiment_scores.get("positive", 0) > 0.5:
                sentiment = "POSITIVE"
                score = sentiment_scores["positive"]
            elif sentiment_scores.get("negative", 0) > 0.5:
                sentiment = "NEGATIVE"
                score = sentiment_scores["negative"]
            else:
                sentiment = "NEUTRAL"
                score = sentiment_scores["neutral"]
            
            news_items.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "title": headline_text,
                "url": url,
                "sentiment": sentiment,
                "sentiment_score": round(score, 4),
                "source": article.get("source", "")
            })
        
        # Push to Google Sheets if enabled
        if push_to_sheets and len(news_items) > 0:
            try:
                from modules.google_sheets import update_news_feed
                update_news_feed(news_items)
            except Exception as e:
                print(f"Warning: Could not push to Google Sheets: {e}")
        
        return news_items
    except Exception as e:
        print(f"Error in get_news_with_sentiment: {e}")
        return []
