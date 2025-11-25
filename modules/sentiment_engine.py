import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
import requests
from datetime import datetime, timedelta

# ------------------------------
# FinBERT for financial sentiment
# ------------------------------
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer_finbert = AutoTokenizer.from_pretrained(finbert_model_name)
model_finbert = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)
labels = ["negative", "neutral", "positive"]

def analyze_finbert(text):
    inputs = tokenizer_finbert(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model_finbert(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
    return dict(zip(labels, probs.tolist()[0]))

def analyze_general_sentiment(text):
    blob = TextBlob(text)
    p = blob.sentiment.polarity
    if p > 0.05:
        return {"negative": 0, "neutral": 0, "positive": 1}
    elif p < -0.05:
        return {"negative": 1, "neutral": 0, "positive": 0}
    else:
        return {"negative": 0, "neutral": 1, "positive": 0}

def analyze_hybrid_sentiment(text):
    fin = analyze_finbert(text)
    gen = analyze_general_sentiment(text)
    hybrid = {k: round(0.7*fin[k] + 0.3*gen[k], 4) for k in labels}
    return hybrid

# ------------------------------
# News fetching (from NewsAPI)
# ------------------------------
NEWS_API_KEY = "fbba6b423b7e426a810eb007ad444242"

def get_news_for_stock(stock_ticker, from_days=7, max_articles=20):
    today = datetime.now()
    from_date = today - timedelta(days=from_days)
    
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={stock_ticker} AND (NSE OR stock OR market)&"
        f"from={from_date.strftime('%Y-%m-%d')}&"
        f"sortBy=publishedAt&"
        f"pageSize={max_articles}&"
        f"apiKey={NEWS_API_KEY}"
    )
    
    response = requests.get(url)
    data = response.json()
    
    if data.get("status") != "ok":
        print(f"Error fetching news for {stock_ticker}: {data}")
        return []
    
    headlines = [{"title": article["title"], "url": article["url"]} for article in data.get("articles", [])]
    return headlines
