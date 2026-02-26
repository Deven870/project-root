# Smart Trading Assistant (Digitrader)

A Streamlit-based stock trading assistant for NSE (National Stock Exchange) stocks. This project provides data ingestion, feature engineering, predictive ML, hybrid sentiment analysis, backtesting, and portfolio allocation in one end-to-end pipeline.

## What This Project Does

1. **Collects data**
   - Pulls historical prices via `yfinance`.
   - Retrieves news headlines via NewsAPI.

2. **Builds features**
   - Technical indicators and time-series features.
   - Sentiment features from FinBERT + TextBlob.

3. **Trains models**
   - Tree-based and sequence models for trend and price movement.
   - Supports multiple horizons (intraday and longer-term).

4. **Evaluates strategies**
   - Walk-forward validation and backtesting.
   - Trading metrics like Sharpe ratio and max drawdown.
   - Baseline strategies for comparison.

5. **Delivers outputs**
   - Interactive Streamlit dashboard.
   - CLI for quick analysis and experiments.
   - Portfolio allocation suggestions.

## Features

- **Trading Dashboard**
  - Trend prediction (Bullish / Bearish / Neutral) with confidence.
  - Current and predicted prices with expected return percentage.
  - Stop-loss suggestions based on investment horizon.
  - Hybrid sentiment analysis from news headlines.
  - Interactive Plotly charting.

- **Portfolio Suggestions**
  - Diversified allocation across NSE stocks.
  - Allocation modes: Proportional, Equal, Risk-adjusted.
  - Configurable max weight cap and top-N filtering.
  - Expected profit and return calculations.

- **Research Pipeline**
  - Reproducible experiments with statistical testing.
  - Baseline comparisons and ablation studies.
  - Result export for paper-ready tables and plots.

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

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd project-root
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the project root:
   ```env
   NEWS_API_KEY=your_newsapi_key_here
   STOCK_SYMBOL=RELIANCE.NS
   ```
   Get a free API key from [newsapi.org](https://newsapi.org/).

5. **Run the app**
   ```bash
   streamlit run app.py
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

### Full Experiments
```bash
python run_experiments.py
```

## Project Structure

```
project-root/
├── app.py                  # Streamlit UI (main entry point)
├── main.py                 # CLI script for quick analysis
├── run_experiments.py      # Reproducible research experiments
├── config.py               # Loads environment variables
├── requirements.txt        # Python dependencies
├── RESEARCH_QUICKSTART.md  # How to run research experiments
├── RESEARCH_READINESS.md   # Research gaps and roadmap
└── modules/
    ├── __init__.py
    ├── backtester.py           # Backtesting and trading simulation
    ├── baseline_models.py      # Baseline strategies
    ├── data_fetch.py           # Stock data and news fetching
    ├── feature_engineering.py  # Technical indicators and features
    ├── predictive_ml.py        # Model training and evaluation
    ├── predictor.py            # Lightweight trend predictor
    ├── sentiment_engine.py     # FinBERT + TextBlob sentiment
    ├── statistical_tests.py    # Significance testing and CIs
    └── utils.py                # Core business logic and helpers
```

## Research Notes

- For a step-by-step research workflow, see [RESEARCH_QUICKSTART.md](RESEARCH_QUICKSTART.md).
- For readiness gaps and publication checklist, see [RESEARCH_READINESS.md](RESEARCH_READINESS.md).

## Disclaimer

This tool is for educational purposes only. Stock market predictions are inherently uncertain. Do not use this as financial advice. Always consult a qualified financial advisor before making investment decisions.