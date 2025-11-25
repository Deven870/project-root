from modules.data_fetch import fetch_stock_data, fetch_headlines
from modules.predictor import simple_trend_prediction
from modules.utils import print_analysis
from config import STOCK_SYMBOL

# ===== Intraday (short-term) =====
intraday_data = fetch_stock_data(STOCK_SYMBOL, period="5d", interval="1h")
intraday_prices = intraday_data['Close'].values
intraday_trend, intraday_conf = simple_trend_prediction(intraday_prices)

# ===== Long Term =====
long_term_data = fetch_stock_data(STOCK_SYMBOL, period="6mo", interval="1d")
long_term_prices = long_term_data['Close'].values
long_term_trend, long_term_conf = simple_trend_prediction(long_term_prices)

# ===== Print Trend Results =====
print_analysis("Intraday", intraday_trend, intraday_conf)
print_analysis("Long Term", long_term_trend, long_term_conf)

# ===== Latest Headlines =====
headlines = fetch_headlines(STOCK_SYMBOL)
print("\nLatest Headlines:")
for i, h in enumerate(headlines, 1):
    print(f"{i}. {h}")
