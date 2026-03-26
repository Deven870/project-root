"""
Quick accuracy test with synthetic data - allows immediate testing and improvement
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Generate realistic synthetic stock data
np.random.seed(42)

def generate_stock_data(n_samples=1000):
    """Generate realistic synthetic OHLC data with technical indicators"""
    
    # Generate price series
    returns = np.random.normal(0.0005, 0.02, n_samples)
    prices = 2500 * np.exp(np.cumsum(returns))
    
    # OHLC
    opens = prices + np.random.normal(0, prices * 0.005, n_samples)
    closes = prices + np.random.normal(0, prices * 0.005, n_samples)
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.01, n_samples)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.01, n_samples)))
    volumes = np.random.uniform(1e6, 5e6, n_samples)
    
    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    })
    
    # Technical indicators
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['MACD'] = calculate_macd(df['Close'])
    df['ATR'] = calculate_atr(df, 14)
    df['BB_upper'], df['BB_lower'] = calculate_bollinger(df['Close'], 20)
    
    # Returns and momentum
    df['Returns_1d'] = df['Close'].pct_change()
    df['Returns_5d'] = df['Close'].pct_change(5)
    df['Momentum'] = df['Close'].diff(10)
    df['Volatility'] = df['Returns_1d'].rolling(10).std()
    
    # Fill NaN
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    # Create target: 1 if price goes up >0.5% in next day, 0 otherwise
    df['Target'] = (df['Close'].shift(-1) > df['Close'] * 1.005).astype(int)
    df = df.dropna()
    
    return df

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    return macd

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = ranges.rolling(period).mean()
    return atr

def calculate_bollinger(prices, period=20):
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    return upper, lower

# Generate data
print("=" * 60)
print("ACCURACY TESTING & IMPROVEMENT - QUICK EVALUATION")
print("=" * 60)

data = generate_stock_data(2000)
feature_cols = ['Open', 'High', 'Low', 'Volume', 'SMA_5', 'SMA_20', 'RSI', 
                'MACD', 'ATR', 'Returns_1d', 'Returns_5d', 'Momentum', 'Volatility', 'BB_upper', 'BB_lower']

X = data[feature_cols].values
y = data['Target'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test multiple models
models = {
    'Random Forest (baseline)': RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42),
    'Random Forest (tuned)': RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, 
                                                     min_samples_leaf=2, random_state=42),
    'Gradient Boosting (baseline)': GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42),
    'Gradient Boosting (tuned)': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=7, 
                                                            subsample=0.8, random_state=42),
}

print("\nCURRENT ACCURACY RATINGS:\n")
results = {}

for name, model in models.items():
    if 'scaled' in name.lower():
        model.fit(X_train_scaled, y_train)
        accuracy = model.score(X_test_scaled, y_test)
    else:
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
    
    results[name] = accuracy
    rating = "✓ ACCEPTABLE" if accuracy >= 0.60 else "✗ POOR" if accuracy < 0.55 else "△ MARGINAL"
    print(f"  {name:40s}  {accuracy*100:6.2f}%   [{rating}]")

avg_accuracy = np.mean(list(results.values()))
print(f"\nAverage Accuracy: {avg_accuracy*100:.2f}%")

# Assessment
print("\n" + "=" * 60)
if avg_accuracy >= 0.65:
    print("✓ ACCURACY IS SATISFACTORY FOR DEPLOYMENT")
    print("  Your models are ready for real-time trading (65%+ accuracy)")
elif avg_accuracy >= 0.60:
    print("△ ACCURACY IS MARGINAL - IMPROVEMENTS RECOMMENDED")
    print("  Consider tuning hyperparameters or adding better features")
else:
    print("✗ ACCURACY IS POOR - MAJOR IMPROVEMENTS NEEDED")
    print("  Models need significant optimization before deployment")

print("=" * 60)

# Improvement recommendations
print("\nTO BOOST ACCURACY BEFORE DEPLOYMENT:\n")
improvements = [
    "1. Add Volume-based indicators (Volume SMA, Money Flow)",
    "2. Include sector/relative strength features", 
    "3. Add time-of-day patterns (for intraday trading)",
    "4. Optimize feature selection (remove correlated features)",
    "5. Use ensemble methods combining multiple strategies",
    "6. Add news sentiment scores (if NewsAPI working)",
    "7. Implement walk-forward validation to prevent overfitting",
    "8. Use threshold optimization for precision/recall trade-off",
]

for imp in improvements:
    print(f"  {imp}")

print("\n" + "=" * 60)
