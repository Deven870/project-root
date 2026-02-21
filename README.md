# Smart Trading Assistant (Digitrader)

A Streamlit-based stock trading assistant for NSE (National Stock Exchange) stocks. Provides real-time predictions, sentiment analysis, and portfolio allocation strategies.

## Features

- **Trading Dashboard** — Select any NSE stock and get:
  - Trend prediction (Bullish / Bearish / Neutral) with confidence score
  - Current and predicted prices with expected return %
  - Stop-loss suggestions based on investment horizon
  - Hybrid sentiment analysis (FinBERT + TextBlob) from news headlines
  - Interactive price chart with Plotly

- **Portfolio Suggestions** — Enter an investment amount and get:
  - Diversified portfolio allocation across NSE stocks
  - Three allocation modes: Proportional, Equal, Risk-adjusted
  - Configurable max weight cap and top-N filtering
  - Expected profit and return calculations

## Tech Stack

| Component        | Technology                        |
|-----------------|-----------------------------------|
| UI              | Streamlit                         |
| Market Data     | yfinance                          |
| Sentiment       | FinBERT (transformers) + TextBlob |
| News            | NewsAPI                           |
| ML Predictions  | scikit-learn, pandas, numpy       |
| Charts          | Plotly                            |

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd project-root
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the project root:
   ```env
   NEWS_API_KEY=your_newsapi_key_here
   STOCK_SYMBOL=RELIANCE.NS
   ```
   Get a free API key from [newsapi.org](https://newsapi.org/).

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
project-root/
├── app.py                  # Streamlit UI (main entry point)
├── main.py                 # CLI script for quick analysis
├── config.py               # Loads environment variables
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
├── .gitignore
└── modules/
    ├── __init__.py
    ├── data_fetch.py       # Stock data & news fetching
    ├── predictive_ml.py    # Trend prediction logic
    ├── predictor.py        # Simple trend predictor
    ├── sentiment_engine.py # FinBERT + TextBlob sentiment
    └── utils.py            # Core business logic & helpers
```

## Usage

### Web Dashboard
```bash
streamlit run app.py
```

### CLI Quick Analysis
```bash
python main.py
```

## Disclaimer

This tool is for **educational purposes only**. Stock market predictions are inherently uncertain. Do not use this as financial advice. Always consult a qualified financial advisor before making investment decisions.