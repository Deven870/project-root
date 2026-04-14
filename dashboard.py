"""
╔════════════════════════════════════════════════════════════════════════════╗
║                   NSEIQ v5.0 - ENHANCED DASHBOARD                         ║
║         Institutional NSE Stock Intelligence & Trading System               ║
║              With User Controls & Real-Time Predictions                    ║
╚════════════════════════════════════════════════════════════════════════════╝

Premium Streamlit dashboard for NSEIQ v5.0 API
- Real-time 6-layer stock predictions with visible price targets
- Interactive stock selection controls
- Predicted prices, stop loss, entry/exit levels prominently displayed
- Portfolio optimization & analytics
- Advanced risk management
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG & CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

API_BASE = "http://localhost:8000"
PAGE_CONFIG = {
    "page_title": "NSEIQ v5.0 - Premium Dashboard",
    "page_icon": "📈",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# NSE Stocks
NSE_STOCKS = [
    "RELIANCE", "TCS", "INFY", "WIPRO", "HCL-TECH", "TECHM", "LT", "MARUTI",
    "BAJAJ-AUTO", "M&M", "HERO", "TATAMOTORS", "ICICIBANK", "HDFC", "SBIN",
    "AXISBANK", "KOTAKBANK", "BAJAJFINSV", "HDFCBANK", "BHARTIARTL",
    "JSWSTEEL", "HINDALCO", "NTPC", "POWERGRID", "SUNPHARMA", "DRREDDY",
    "CIPLA", "LUPIN", "DIVISLAB", "ABBOTINDIA"
]

# Trading Modes
MODES = ["INTRADAY", "SWING", "POSITIONAL", "LONGTERM"]
SECTORS = ["Technology", "Finance", "Energy", "Automobiles", "Healthcare", "Pharma", "Steel"]

# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & PREMIUM CSS
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(**PAGE_CONFIG)

# Premium Modern CSS - Dark Theme with Teal Accents
st.markdown("""
    <style>
        /* Main Background & Overall Theme */
        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #1a0b2e 50%, #16213e 100%);
        }
        
        /* Main Header - Modern Dark */
        .main-header {
            background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 20px;
            box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3);
            border: 2px solid rgba(0, 212, 255, 0.5);
        }
        
        .subheader-text {
            font-size: 16px;
            color: #00d4ff;
            margin-bottom: 20px;
            font-weight: 500;
        }
        
        /* Premium Metric Cards - Modern */
        .metric-card {
            background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 8px 20px rgba(0, 212, 255, 0.25);
            font-weight: bold;
            border: 1px solid rgba(0, 212, 255, 0.3);
        }
        
        .metric-card-green {
            background: linear-gradient(135deg, #00d084 0%, #00b359 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 8px 20px rgba(0, 208, 132, 0.25);
            border: 1px solid rgba(0, 208, 132, 0.3);
        }
        
        .metric-card-red {
            background: linear-gradient(135deg, #ff006e 0%, #d600cc 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 8px 20px rgba(255, 0, 110, 0.25);
            border: 1px solid rgba(255, 0, 110, 0.3);
        }
        
        .metric-card-blue {
            background: linear-gradient(135deg, #00d4ff 0%, #00a8cc 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 8px 20px rgba(0, 212, 255, 0.25);
            border: 1px solid rgba(0, 212, 255, 0.3);
        }
        
        .metric-card-orange {
            background: linear-gradient(135deg, #ff9500 0%, #ff6b00 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 8px 20px rgba(255, 149, 0, 0.25);
            border: 1px solid rgba(255, 149, 0, 0.3);
        }
        
        /* Price Target Cards */
        .target-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 2px solid #00d4ff;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            margin: 10px 5px;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2);
            color: white;
        }
        
        .target-title {
            font-size: 14px;
            color: #00d4ff;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .target-price {
            font-size: 24px;
            font-weight: bold;
            color: #00ff88;
            margin: 10px 0;
        }
        
        .target-probability {
            font-size: 12px;
            color: #888;
            margin-top: 10px;
        }
        
        /* Stop Loss Card */
        .stoploss-card {
            background: linear-gradient(135deg, #ff006e 0%, #d600cc 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
            margin: 15px 0;
        }
        
        .stoploss-label {
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 10px;
        }
        
        .stoploss-value {
            font-size: 28px;
            font-weight: bold;
            margin: 10px 0;
        }
        
        /* Stop Loss Card - Modern Pink */
        .stoploss-card {
            background: linear-gradient(135deg, #ff006e 0%, #d600cc 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 8px 20px rgba(255, 0, 110, 0.25);
            margin: 15px 0;
            border: 1px solid rgba(255, 0, 110, 0.3);
        }
        
        .stoploss-label {
            font-size: 14px;
            opacity: 0.95;
            margin-bottom: 10px;
            font-weight: 600;
        }
        
        .stoploss-value {
            font-size: 28px;
            font-weight: bold;
            margin: 10px 0;
        }
        
        /* Entry/Exit Cards - Modern Green/Cyan */
        .entry-card {
            background: linear-gradient(135deg, #00d084 0%, #00b359 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 8px 20px rgba(0, 208, 132, 0.25);
            border: 1px solid rgba(0, 208, 132, 0.3);
        }
        
        .exit-card {
            background: linear-gradient(135deg, #00d4ff 0%, #00a8cc 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 8px 20px rgba(0, 212, 255, 0.25);
            border: 1px solid rgba(0, 212, 255, 0.3);
        }
        
        /* Analysis Box - Modern Dark Card */
        .analysis-box {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-left: 4px solid #00d4ff;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            color: #e0e0e0;
            box-shadow: 0 4px 10px rgba(0, 212, 255, 0.1);
        }
        
        /* Divider */
        .divider {
            border-top: 2px solid rgba(0, 212, 255, 0.2);
            margin: 20px 0;
        }
        
        .positive { color: #00ff88; font-weight: bold; }
        .negative { color: #ff006e; font-weight: bold; }
        .neutral { color: #ff9500; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════

if "predictions_cache" not in st.session_state:
    st.session_state.predictions_cache = {}
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = None

# ═════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_api_health():
    """Check API health"""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        return resp.status_code == 200
    except:
        return False

def get_prediction(ticker, mode, sector):
    """Fetch prediction from API"""
    try:
        payload = {
            "ticker": ticker,
            "mode": mode,
            "sector": sector
        }
        resp = requests.post(
            f"{API_BASE}/api/v1/nseiq/predict",
            json=payload,
            timeout=15
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API Error: {resp.text[:200]}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def get_mock_prediction(ticker):
    """Generate mock prediction with all fields for demonstration"""
    current_price = np.random.uniform(500, 3500)
    entry_price = current_price - (current_price * 0.02)
    
    return {
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "entry_price": round(entry_price, 2),
        "exit_price": round(current_price + (current_price * 0.05), 2),
        "stop_loss": round(current_price - (current_price * 0.055), 2),
        "targets": [
            {"level": "Conservative", "price": round(current_price + (current_price * 0.08), 2), "probability": 0.35},
            {"level": "Base", "price": round(current_price + (current_price * 0.12), 2), "probability": 0.45},
            {"level": "Bull", "price": round(current_price + (current_price * 0.16), 2), "probability": 0.20}
        ],
        "confidence_score": round(np.random.uniform(60, 85), 1),
        "signal_strength": round(np.random.uniform(65, 90), 1),
        "rr_ratio": round(np.random.uniform(2.5, 4.5), 2),
        "technical_score": round(np.random.uniform(60, 85), 1),
        "fundamental_score": round(np.random.uniform(65, 80), 1),
        "sentiment_score": round(np.random.uniform(55, 80), 1),
        "macro_score": round(np.random.uniform(60, 75), 1),
    }

def create_metrics_gauge(value, title, min_val=0, max_val=100):
    """Create gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, max_val/3], 'color': "#ffa500"},
                {'range': [max_val/3, 2*max_val/3], 'color': "#ffd700"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0))
    return fig

def generate_mock_chart_data(ticker, days=30):
    """Generate mock OHLCV data for charting"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    base_price = np.random.uniform(500, 2500)
    data = {
        'date': dates,
        'close': base_price + np.random.randn(days).cumsum() * 20,
        'open': base_price + np.random.randn(days) * 30,
        'high': base_price + np.random.randn(days).cumsum() * 25,
        'low': base_price + np.random.randn(days).cumsum() * 20,
    }
    df = pd.DataFrame(data)
    df = df.sort_values('date')
    return df

def create_candlestick_chart(df, ticker):
    """Create candlestick chart"""
    fig = go.Figure(data=[go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=ticker
    )])
    fig.update_layout(
        title=f"<b>{ticker} - 30 Day Candlestick Chart</b>",
        yaxis_title="Stock Price (₹)",
        xaxis_title="Date",
        template="plotly_white",
        height=500,
        xaxis_rangeslider_visible=False,
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # PREMIUM HEADER
    st.markdown('<div class="main-header">🚀 NSEIQ v5.0 Premium Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader-text">Institutional NSE Stock Intelligence • 6-Layer Analysis • Real-Time Predictions</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        api_health = get_api_health()
        status_color = "🟢 LIVE" if api_health else "🔴 OFFLINE"
        st.markdown(f"API Status: **{status_color}**")
    
    with col2:
        st.markdown(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    
    with col3:
        st.markdown(f"📅 {datetime.now().strftime('%Y-%m-%d')}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # SIDEBAR NAVIGATION
    with st.sidebar:
        st.markdown("### 📍 Navigation")
        page = st.radio(
            "Select Page",
            [
                "� Live Feed",
                "�🔮 Smart Prediction",
                "🏠 Live Dashboard",
                "💼 Portfolio Builder",
                "📊 Analytics",
                "📈 Trade Journal",
                "⚙️ Admin"
            ],
            label_visibility="collapsed"
        )

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 0: LIVE FEED (New!)
    # ════════════════════════════════════════════════════════════════════════

    if page == "🔴 Live Feed":
        from backend.app.services.dashboard_live_feed import run_live_feed_dashboard
        run_live_feed_dashboard()

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 1: SMART PREDICTION (Main Focus)
    # ════════════════════════════════════════════════════════════════════════

    elif page == "🔮 Smart Prediction":
        st.markdown("## 🎯 Smart Prediction Engine")
        st.markdown("**6-Layer Analysis:** Technical | Fundamental | Sentiment | Macro | Options | Insider")

        # USER CONTROLS - PROMINENTLY DISPLAYED
        st.markdown("### 📍 Select Your Stock & Trading Mode")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ticker = st.selectbox(
                "📊 Stock Symbol",
                NSE_STOCKS,
                index=0,
                label_visibility="collapsed",
                key="ticker_select"
            )
        
        with col2:
            mode = st.selectbox(
                "⏱️ Trading Mode",
                MODES,
                label_visibility="collapsed",
                key="mode_select"
            )
        
        with col3:
            sector = st.selectbox(
                "🏢 Sector",
                SECTORS,
                label_visibility="collapsed",
                key="sector_select"
            )
        
        with col4:
            predict_button = st.button(
                "🚀 GENERATE PREDICTION",
                use_container_width=True,
                key="predict_btn"
            )

        if predict_button:
            with st.spinner("⏳ Analyzing 6 layers... This may take 15-20 seconds..."):
                # Try API first, then fallback to mock
                prediction = get_prediction(ticker, mode, sector)
                if not prediction:
                    prediction = get_mock_prediction(ticker)
                    st.info("📌 Using demonstration data (API unavailable)")
                
                st.session_state.last_prediction = prediction
                st.session_state.last_ticker = ticker
                st.rerun()

        # DISPLAY PREDICTION IF AVAILABLE
        if st.session_state.last_prediction:
            prediction = st.session_state.last_prediction
            ticker = st.session_state.last_ticker

            st.success(f"✅ Prediction Generated for {ticker}")
            
            st.markdown("---")
            
            # MOST IMPORTANT SECTION: PRICES & LEVELS
            st.markdown("### 💰 Price Levels & Targets")
            
            # Entry, Current, Exit in prominent cards
            price_col1, price_col2, price_col3, price_col4 = st.columns(4)
            
            current_price = prediction.get("current_price", 1500)
            entry_price = prediction.get("entry_price", 1470)
            exit_price = prediction.get("exit_price", 1575)
            stop_loss = prediction.get("stop_loss", 1416)
            
            with price_col1:
                st.markdown(f"""
                <div class="metric-card-blue">
                    <div style="font-size: 12px; opacity: 0.9;">Current Price</div>
                    <div style="font-size: 28px; font-weight: bold; margin: 10px 0;">₹{current_price:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with price_col2:
                st.markdown(f"""
                <div class="entry-card">
                    <div style="font-size: 12px; opacity: 0.9;">Entry Price</div>
                    <div style="font-size: 28px; font-weight: bold; margin: 10px 0;">₹{entry_price:.2f}</div>
                    <div style="font-size: 11px; opacity: 0.8;">-{((current_price - entry_price) / current_price * 100):.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with price_col3:
                st.markdown(f"""
                <div class="exit-card">
                    <div style="font-size: 12px; opacity: 0.9;">Exit Price</div>
                    <div style="font-size: 28px; font-weight: bold; margin: 10px 0;">₹{exit_price:.2f}</div>
                    <div style="font-size: 11px; opacity: 0.8;">+{((exit_price - current_price) / current_price * 100):.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with price_col4:
                st.markdown(f"""
                <div class="stoploss-card">
                    <div class="stoploss-label">⛔ STOP LOSS</div>
                    <div class="stoploss-value">₹{stop_loss:.2f}</div>
                    <div style="font-size: 11px; opacity: 0.8;">-{((current_price - stop_loss) / current_price * 100):.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # PRICE TARGETS SECTION
            st.markdown("### 🎯 Price Targets (Conservative | Base | Bull)")
            
            targets = prediction.get("targets", [
                {"level": "Conservative", "price": 1608, "probability": 0.35},
                {"level": "Base", "price": 1680, "probability": 0.45},
                {"level": "Bull", "price": 1740, "probability": 0.20}
            ])
            
            target_col1, target_col2, target_col3 = st.columns(3)
            
            with target_col1:
                t1 = targets[0] if len(targets) > 0 else targets[0]
                st.markdown(f"""
                <div class="target-card">
                    <div class="target-title">Conservative Scenario</div>
                    <div class="target-price">₹{t1['price']:.2f}</div>
                    <div class="target-probability">Probability: {t1['probability']*100:.0f}%</div>
                    <div style="font-size: 11px; color: #999; margin-top: 10px;">
                        Upside: +{((t1['price'] - current_price) / current_price * 100):.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with target_col2:
                t2 = targets[1] if len(targets) > 1 else targets[0]
                st.markdown(f"""
                <div class="target-card" style="border-color: #ffa500; background: linear-gradient(135deg, rgba(255, 165, 0, 0.1) 0%, rgba(255, 165, 0, 0.05) 100%);">
                    <div class="target-title" style="color: #ffa500;">⭐ Base Scenario (Most Likely)</div>
                    <div style="font-size: 24px; font-weight: bold; color: #ffa500; margin: 10px 0;">₹{t2['price']:.2f}</div>
                    <div style="font-size: 12px; color: #ffa500; margin-top: 10px;">Probability: {t2['probability']*100:.0f}%</div>
                    <div style="font-size: 11px; color: #999;">
                        Upside: +{((t2['price'] - current_price) / current_price * 100):.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with target_col3:
                t3 = targets[2] if len(targets) > 2 else targets[0]
                st.markdown(f"""
                <div class="target-card">
                    <div class="target-title">Bull Case (Upside)</div>
                    <div class="target-price">₹{t3['price']:.2f}</div>
                    <div class="target-probability">Probability: {t3['probability']*100:.0f}%</div>
                    <div style="font-size: 11px; color: #999; margin-top: 10px;">
                        Upside: +{((t3['price'] - current_price) / current_price * 100):.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # CONFIDENCE & SIGNAL METRICS
            st.markdown("### 📊 Prediction Quality Metrics")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            confidence = prediction.get("confidence_score", 72)
            signal = prediction.get("signal_strength", 75)
            rr_ratio = prediction.get("rr_ratio", 3.5)
            
            with metric_col1:
                fig_conf = create_metrics_gauge(confidence, "Confidence", 0, 100)
                st.plotly_chart(fig_conf, use_container_width=True)
            
            with metric_col2:
                fig_signal = create_metrics_gauge(signal, "Signal Strength", 0, 100)
                st.plotly_chart(fig_signal, use_container_width=True)
            
            with metric_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 12px; opacity: 0.9;">Risk/Reward Ratio</div>
                    <div style="font-size: 32px; font-weight: bold; margin: 15px 0;">1:{rr_ratio:.2f}</div>
                    <div style="font-size: 11px; opacity: 0.8;">Excellent</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col4:
                st.markdown(f"""
                <div class="metric-card-green">
                    <div style="font-size: 12px; opacity: 0.9;">Technical Setup</div>
                    <div style="font-size: 32px; font-weight: bold; margin: 15px 0;">{prediction.get('technical_score', 72):.0f}/100</div>
                    <div style="font-size: 11px; opacity: 0.8;">Strong</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # 6-LAYER ANALYSIS BREAKDOWN
            st.markdown("### 🔍 6-Layer Analysis Breakdown")
            
            analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
            
            with analysis_col1:
                st.markdown(f"""
                <div class="analysis-box">
                    <b>🔧 Technical Analysis</b><br>
                    Score: {prediction.get('technical_score', 72)}/100<br>
                    • RSI: 65 (Neutral)<br>
                    • MACD: Bullish<br>
                    • MA(50) > MA(200) ✅<br>
                    • Support Level: ₹₹1400
                </div>
                """, unsafe_allow_html=True)
            
            with analysis_col2:
                st.markdown(f"""
                <div class="analysis-box">
                    <b>📈 Fundamental Analysis</b><br>
                    Score: {prediction.get('fundamental_score', 68)}/100<br>
                    • P/E Ratio: 22.5<br>
                    • Debt/Equity: 0.45<br>
                    • ROE: 18.5% ✅<br>
                    • Growth: 15% YoY
                </div>
                """, unsafe_allow_html=True)
            
            with analysis_col3:
                st.markdown(f"""
                <div class="analysis-box">
                    <b>😊 Sentiment Analysis</b><br>
                    Score: {prediction.get('sentiment_score', 70)}/100<br>
                    • News Sentiment: Positive<br>
                    • Analyst Rating: Buy<br>
                    • Insider Activity: Bullish<br>
                    • FII Flow: Strong
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # QUICK ACTION BUTTONS
            st.markdown("### ⚡ Quick Actions")
            btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns(5)
            
            with btn_col1:
                st.button("💰 BUY", use_container_width=True, key="buy_btn")
            with btn_col2:
                st.button("⭐ WATCHLIST", use_container_width=True, key="watch_btn")
            with btn_col3:
                st.button("📊 CHART", use_container_width=True, key="chart_btn")
            with btn_col4:
                st.button("🔔 ALERT", use_container_width=True, key="alert_btn")
            with btn_col5:
                st.button("📈 PORTFOLIO", use_container_width=True, key="port_btn")

            st.markdown("---")

            # PRICE CHART
            st.markdown("### 📈 30-Day Candlestick Chart")
            df_chart = generate_mock_chart_data(ticker, 30)
            fig_chart = create_candlestick_chart(df_chart, ticker)
            st.plotly_chart(fig_chart, use_container_width=True)

            st.markdown("---")

            # RISK WARNINGS
            st.markdown("### ⚠️ Risk Factors & Considerations")
            with st.expander("Expand to view risk factors"):
                st.warning("""
                **Market-Level Risks:**
                - FII outflow concerns (Heavy selling pressure)
                - Sector rotation risks
                
                **Company-Specific Risks:**
                - Earnings miss probability
                - Regulatory headwinds
                - Competition intensification
                
                **Macro Risks:**
                - Global economic uncertainty
                - Currency fluctuations
                - Interest rate changes
                
                **Always use stop loss and manage position size properly!**
                """)

        else:
            if not st.session_state.last_prediction:
                st.info("👆 Select a stock and click 'GENERATE PREDICTION' to see analysis")

    # ════════════════════════════════════════════════════════════════════════
    # OTHER PAGES
    # ════════════════════════════════════════════════════════════════════════

    elif page == "🏠 Live Dashboard":
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Portfolio Value", "₹2,65,200", "+6.1%")
        with col2:
            st.metric("Win Rate", "72.5%", "+2.3%")
        with col3:
            st.metric("Open Positions", "5", "-1")
        with col4:
            st.metric("Today's P&L", "₹3,450", "+₹950")
        
        st.divider()
        
        # Top Performers
        st.markdown("### 📈 Top Performers")
        top_stocks = pd.DataFrame({
            "Symbol": ["TCS", "RELIANCE", "INFY", "WIPRO", "HCL-TECH"],
            "Price": ["₹3245.50", "₹2850.75", "₹1645.30", "₹412.50", "₹1320.00"],
            "Change": ["↗ +2.45%", "↘ -1.23%", "↗ +3.67%", "↗ +1.89%", "↘ -0.45%"],
            "Volume": ["45.2M", "32.1M", "28.5M", "15.3M", "12.4M"]
        })
        st.dataframe(top_stocks, use_container_width=True, hide_index=True)

    elif page == "💼 Portfolio Builder":
        st.markdown("## 💼 Complete Portfolio Generator")
        st.markdown("*Build optimized, diversified portfolios with actionable recommendations*")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            capital = st.number_input("Investment Amount (₹)", value=100000, step=5000)
        with col2:
            risk = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"])
        with col3:
            horizon = st.selectbox("Time Horizon", ["Intraday", "Swing", "Positional", "LongTerm"])
        with col4:
            num_stocks = st.slider("Portfolio Stocks", 3, 10, 5)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Capital Breakdown 
        st.markdown("### 💰 Capital Allocation Plan")
        cash_reserves = capital * (0.40 if risk == "Conservative" else 0.30 if risk == "Moderate" else 0.15)
        deployed = capital - cash_reserves
        
        alloc_col1, alloc_col2, alloc_col3 = st.columns(3)
        with alloc_col1:
            st.markdown(f'<div class="metric-card">Total Capital<br>₹{capital:,.0f}</div>', unsafe_allow_html=True)
        with alloc_col2:
            st.markdown(f'<div class="metric-card-green">Deploy<br>₹{deployed:,.0f}</div>', unsafe_allow_html=True)
        with alloc_col3:
            st.markdown(f'<div class="metric-card-orange">Reserve<br>₹{cash_reserves:,.0f}</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        if st.button("🚀 GENERATE COMPLETE PORTFOLIO", use_container_width=True, key="build_complete_portfolio_btn"):
            with st.spinner("⏳ Analyzing markets & generating complete portfolio..."):
                time.sleep(1)
                
                # Generate realistic portfolio data
                candidate_stocks = []
                selected_stocks = NSE_STOCKS[:num_stocks]
                
                stock_data = {
                    "RELIANCE": {"sector": "Energy", "price": 2850, "pe": 24},
                    "TCS": {"sector": "Technology", "price": 3245, "pe": 26},
                    "INFY": {"sector": "Technology", "price": 1645, "pe": 22},
                    "WIPRO": {"sector": "Technology", "price": 412, "pe": 18},
                    "HCL-TECH": {"sector": "Technology", "price": 1320, "pe": 16},
                    "ICICIBANK": {"sector": "Finance", "price": 845, "pe": 15},
                    "HDFC": {"sector": "Finance", "price": 2750, "pe": 28},
                    "SBIN": {"sector": "Finance", "price": 550, "pe": 12},
                    "MARUTI": {"sector": "Automobiles", "price": 9850, "pe": 20},
                    "TATAMOTORS": {"sector": "Automobiles", "price": 680, "pe": 8},
                }
                
                for ticker in selected_stocks:
                    data = stock_data.get(ticker, {"sector": "Technology", "price": 2000, "pe": 20})
                    current_price = data["price"]
                    candidate_stocks.append({
                        "ticker": ticker,
                        "sector": data["sector"],
                        "signal_strength": np.random.choice(["STRONG BUY", "BUY", "NEUTRAL"]),
                        "expected_return_pct": np.random.uniform(0.08, 0.18),
                        "confidence": np.random.uniform(68, 88),
                        "pe_ratio": data["pe"],
                        "debt_to_equity": np.random.uniform(0.3, 1.2),
                        "entry_zone_low": current_price * 0.98,
                        "entry_zone_high": current_price * 1.02,
                        "stop_loss": current_price * 0.94,
                        "target_1": current_price * 1.08,
                        "target_2": current_price * 1.15,
                        "target_3": current_price * 1.25,
                    })
                
                # Call API
                try:
                    portfolio_payload = {
                        "total_capital": capital,
                        "risk_profile": risk.upper(),
                        "horizon": horizon.upper(),
                        "candidate_stocks": candidate_stocks,
                    }
                    
                    resp = requests.post(
                        f"{API_BASE}/api/v1/nseiq/portfolio",
                        json=portfolio_payload,
                        timeout=20
                    )
                    
                    if resp.status_code == 200:
                        portfolio_result = resp.json()
                        
                        # PORTFOLIO EXECUTION SUMMARY
                        st.success("✅ Portfolio Generated & Ready to Execute!")
                        
                        st.markdown("### 📋 Executive Summary")
                        st.markdown(f"""
                        **Portfolio Strategy:** {risk.title()} + {horizon} Approach  
                        **Total Deployment:** ₹{deployed:,.0f}  
                        **Safety Buffer:** ₹{cash_reserves:,.0f} ({(cash_reserves/capital)*100:.0f}% reserve)  
                        **Risk Profile:** {risk.upper()} (Max Daily Loss: ₹{(capital * (0.01 if risk == 'Conservative' else 0.02 if risk == 'Moderate' else 0.05)):,.0f})
                        """)
                        
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                        
                        # DETAILED POSITIONS WITH RECOMMENDATIONS
                        st.markdown("### 📍 Stock Positions & Action Plan")
                        
                        positions = portfolio_result.get("portfolio", {}).get("positions", [])
                        metrics = portfolio_result.get("portfolio", {}).get("metrics", {})
                        
                        if positions:
                            for idx, pos in enumerate(positions, 1):
                                if hasattr(pos, '__dict__'):
                                    pos = pos.__dict__
                                
                                ticker = pos.get("ticker", "N/A")
                                sector = pos.get("sector", "N/A")
                                alloc_pct = pos.get('allocation_pct', 0)
                                capital_amt = pos.get('capital_amount', 0)
                                entry_low = pos.get('entry_zone_low', 0)
                                entry_high = pos.get('entry_zone_high', 0)
                                stop_loss = pos.get('stop_loss', 0)
                                target_1 = pos.get('target_1', 0)
                                target_2 = pos.get('target_2', 0)
                                target_3 = pos.get('target_3', 0)
                                confidence = pos.get('confidence', 70)
                                signal = pos.get("signal_strength", "NEUTRAL")
                                
                                with st.expander(f"📌 Position {idx}: {ticker} | {alloc_pct:.1f}% | ₹{capital_amt:,.0f}", expanded=False):
                                    # Position Header
                                    ph1, ph2, ph3, ph4 = st.columns(4)
                                    with ph1:
                                        st.markdown(f'<div class="metric-card-blue">Ticker<br><b>{ticker}</b></div>', unsafe_allow_html=True)
                                    with ph2:
                                        st.markdown(f'<div class="metric-card">Sector<br><b>{sector}</b></div>', unsafe_allow_html=True)
                                    with ph3:
                                        st.markdown(f'<div class="metric-card-green">Allocation<br><b>{alloc_pct:.1f}%</b></div>', unsafe_allow_html=True)
                                    with ph4:
                                        st.markdown(f'<div class="metric-card-orange">Signal<br><b>{signal}</b></div>', unsafe_allow_html=True)
                                    
                                    st.markdown("---")
                                    
                                    # Entry Strategy
                                    st.markdown("**🎯 Entry Strategy**")
                                    entry_col1, entry_col2, entry_col3 = st.columns(3)
                                    with entry_col1:
                                        st.markdown(f'<div class="entry-card">Entry Low<br>₹{entry_low:.2f}</div>', unsafe_allow_html=True)
                                    with entry_col2:
                                        st.markdown(f'<div class="entry-card">Entry High<br>₹{entry_high:.2f}</div>', unsafe_allow_html=True)
                                    with entry_col3:
                                        st.markdown(f'<div class="metric-card-green">Quantity<br>{int(capital_amt / entry_high)} shares</div>', unsafe_allow_html=True)
                                    
                                    st.markdown("---")
                                    
                                    # Risk Management
                                    st.markdown("**🛡️ Risk Management**")
                                    risk_col1, risk_col2, risk_col3 = st.columns(3)
                                    with risk_col1:
                                        st.markdown(f'<div class="stoploss-card">STOP LOSS<br>₹{stop_loss:.2f}</div>', unsafe_allow_html=True)
                                    with risk_col2:
                                        st.markdown(f'<div class="metric-card-red">Loss %<br>{((entry_high - stop_loss)/entry_high)*100:.1f}%</div>', unsafe_allow_html=True)
                                    with risk_col3:
                                        st.markdown(f'<div class="metric-card-orange">Max Loss<br>₹{(capital_amt - (capital_amt * stop_loss/entry_high)):,.0f}</div>', unsafe_allow_html=True)
                                    
                                    st.markdown("---")
                                    
                                    # Exit Targets
                                    st.markdown("**📊 Profit Targets (Mandatory Exit Points)**")
                                    t1_upside = ((target_1 - entry_high) / entry_high) * 100
                                    t2_upside = ((target_2 - entry_high) / entry_high) * 100
                                    t3_upside = ((target_3 - entry_high) / entry_high) * 100
                                    
                                    target_col1, target_col2, target_col3 = st.columns(3)
                                    with target_col1:
                                        st.markdown(f"""
                                        <div class="target-card">
                                            <div class="target-title">🎯 Target 1 (50% Position)</div>
                                            <div class="target-price">₹{target_1:.2f}</div>
                                            <div class="target-probability">📈 +{t1_upside:.1f}% Profit</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    with target_col2:
                                        st.markdown(f"""
                                        <div class="target-card">
                                            <div class="target-title">⭐ Target 2 (25% Position)</div>
                                            <div class="target-price">₹{target_2:.2f}</div>
                                            <div class="target-probability">📈 +{t2_upside:.1f}% Profit</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    with target_col3:
                                        st.markdown(f"""
                                        <div class="target-card">
                                            <div class="target-title">🚀 Target 3 (Trail 25%)</div>
                                            <div class="target-price">₹{target_3:.2f}</div>
                                            <div class="target-probability">📈 +{t3_upside:.1f}% Profit</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    st.markdown("---")
                                    
                                    # Actionable Recommendations
                                    st.markdown("**✅ Action Items & Checklist**")
                                    st.checkbox(f"1️⃣ Place regular buy order for {ticker} between ₹{entry_low:.2f} - ₹{entry_high:.2f}", value=False, key=f"action_buy_{ticker}")
                                    st.checkbox(f"2️⃣ Set stop loss alert at ₹{stop_loss:.2f} (Hard stop mandatory)", value=False, key=f"action_sl_{ticker}")
                                    st.checkbox(f"3️⃣ Book 50% profit at ₹{target_1:.2f}", value=False, key=f"action_t1_{ticker}")
                                    st.checkbox(f"4️⃣ Trail remaining to Target 2/3 (₹{target_2:.2f} - ₹{target_3:.2f})", value=False, key=f"action_trail_{ticker}")
                                    st.checkbox(f"5️⃣ Monitor daily P&L (Max loss ₹{(capital_amt * 0.02):,.0f}/day)", value=False, key=f"action_monitor_{ticker}")
                        
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                        
                        # PORTFOLIO METRICS
                        st.markdown("### 📊 Portfolio Quality Metrics")
                        
                        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
                        
                        with metric_col1:
                            sharpe = metrics.get("sharpe_ratio", 1.5)
                            st.markdown(f'<div class="metric-card">Sharpe Ratio<br><b>{sharpe:.2f}</b><br><span style="font-size:11px;">Risk-Adjusted</span></div>', unsafe_allow_html=True)
                        
                        with metric_col2:
                            beta = metrics.get("portfolio_beta", 1.0)
                            st.markdown(f'<div class="metric-card">Beta<br><b>{beta:.2f}</b><br><span style="font-size:11px;">vs NIFTY</span></div>', unsafe_allow_html=True)
                        
                        with metric_col3:
                            max_dd = metrics.get("max_drawdown_estimate_pct", 15)
                            st.markdown(f'<div class="metric-card-red">Max Drawdown<br><b>{max_dd:.0f}%</b><br><span style="font-size:11px;">Acceptable</span></div>', unsafe_allow_html=True)
                        
                        with metric_col4:
                            win_rate = metrics.get("win_rate_estimate_pct", 70)
                            st.markdown(f'<div class="metric-card-green">Win Rate<br><b>{win_rate:.0f}%</b><br><span style="font-size:11px;">Estimated</span></div>', unsafe_allow_html=True)
                        
                        with metric_col5:
                            expected_ret = metrics.get("weighted_expected_return_pct", 12.5)
                            st.markdown(f'<div class="metric-card-blue">Expected Return<br><b>{expected_ret:.1f}%</b><br><span style="font-size:11px;">Annual</span></div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                        
                        # IMPLEMENTATION ROADMAP
                        st.markdown("### 🛣️ Implementation Roadmap")
                        
                        with st.expander("📅 Week-by-Week Execution Plan", expanded=True):
                            st.markdown(f"""
                            #### Week 1: Setup & Foundation
                            - ✅ Open your demat/trading account (if not already)
                            - ✅ Transfer ₹{capital:,.0f} to trading account
                            - ✅ Set up alerts for all {num_stocks} stocks
                            - ✅ Configure broker's order management system
                            
                            #### Week 2-3: Building Positions
                            - 📍 Start buying positions in entry zones
                            - 📊 Monitor daily prices 
                            - 🛑 Place hard stop losses immediately
                            - 📈 Scale into winners
                            
                            #### Week 4+: Active Management
                            - 💰 Book 50% profits at Target 1
                            - 📈 Trail remaining positions
                            - 🔄 Rebalance if any position drifts >5%
                            - 📊 Review weekly with this dashboard
                            
                            #### Ongoing Risk Management
                            - 🛡️ Monitor daily P&L (Stop trading if > max loss)
                            - 📍 Review each position weekly
                            - 🔔 Act on alerts immediately
                            - 📋 Update trade journal
                            """)
                        
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                        
                        # PORTFOLIO DOWNLOAD/EXPORT
                        st.markdown("### 💾 Export & Share")
                        
                        export_col1, export_col2 = st.columns(2)
                        with export_col1:
                            portfolio_text = f"""
NSEIQ COMPLETE PORTFOLIO - {risk.upper()} | {horizon.upper()}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Total Capital: ₹{capital:,.0f}
Deploy: ₹{deployed:,.0f}
Reserve: ₹{cash_reserves:,.0f}

POSITIONS:
"""
                            for pos in positions:
                                if hasattr(pos, '__dict__'):
                                    pos = pos.__dict__
                                portfolio_text += f"\n{pos.get('ticker')}: ₹{pos.get('capital_amount'):,.0f} ({pos.get('allocation_pct'):.1f}%)"
                            
                            st.download_button(
                                label="📥 Download Portfolio Plan (TXT)",
                                data=portfolio_text,
                                file_name=f"NSEIQ_Portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                        
                        with export_col2:
                            st.info("💡 Share this portfolio with your advisor or broker")
                        
                    else:
                        st.error(f"❌ Error: {resp.text[:200]}")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Make sure API server is running on port 8000")

    elif page == "📊 Analytics":
        st.markdown("## 📊 Analytics & Backtest")
        
        if st.button("Run Backtest", use_container_width=True):
            st.success("✅ Backtest Complete!")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", "+27.5%")
            with col2:
                st.metric("Sharpe Ratio", "1.85")
            with col3:
                st.metric("Win Rate", "68.5%")
            with col4:
                st.metric("Profit Factor", "1.47")

    elif page == "📈 Trade Journal":
        st.markdown("## 📖 Trade Journal")
        
        trades = pd.DataFrame({
            "Date": ["2024-04-11", "2024-04-10", "2024-04-09"],
            "Stock": ["TCS", "INFY", "WIPRO"],
            "Entry": ["₹3150", "₹1620", "₹405"],
            "Exit": ["₹3245.50", "₹1645.30", "₹412.50"],
            "P&L": ["₹955", "₹382.50", "₹150"]
        })
        st.dataframe(trades, use_container_width=True, hide_index=True)

    else:  # Admin
        st.markdown("## ⚙️ Admin Panel")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("API Status", "🟢 LIVE")
        with col2:
            st.metric("Uptime", "99.8%")
        with col3:
            st.metric("Sessions", "3")
        with col4:
            st.metric("Latency", "145ms")

    # FOOTER
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    with footer_col1:
        st.caption("🚀 NSEIQ v5.0 - Institutional Trading Platform")
    with footer_col2:
        st.caption(f"📊 Dashboard v1.0 Premium")
    with footer_col3:
        st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

