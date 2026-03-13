import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from modules.utils import (
    get_stock_predictions,
    get_portfolio_allocation,
    get_investment_advice,
    get_nse_stock_list,
    fetch_price_data,
)

# =========================
# 🧩 Streamlit Setup
# =========================
st.set_page_config(page_title="Digitrader — Smart Trading Assistant", page_icon="🚀", layout="wide")

# =========================
# 🎨 Custom CSS for eye-catching UI
# =========================
st.markdown("""
<style>
/* ---- Glassmorphism cards ---- */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(30,30,60,0.85), rgba(50,50,100,0.7));
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 18px 22px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    backdrop-filter: blur(8px);
}
div[data-testid="stMetric"] label {
    color: #a0aec0 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* ---- Sidebar glow ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #e0e0e0 !important;
}

/* ---- Glowing buttons ---- */
div.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
}

/* ---- Section dividers ---- */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(102,126,234,0.5), transparent);
    margin: 1.5rem 0;
}

/* ---- Trend badge ---- */
.trend-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.5px;
}
.trend-bullish { background: linear-gradient(135deg, #00b09b, #96c93d); color: #fff; }
.trend-bearish { background: linear-gradient(135deg, #fc4a1a, #f7b733); color: #fff; }
.trend-neutral { background: linear-gradient(135deg, #606c88, #3f4c6b); color: #fff; }

/* ---- Info cards ---- */
.info-card {
    background: linear-gradient(135deg, rgba(40,40,80,0.8), rgba(60,60,120,0.6));
    border-radius: 14px;
    padding: 20px;
    border-left: 4px solid #667eea;
    margin: 10px 0;
}

/* ---- Animated header ---- */
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.main-title {
    background: linear-gradient(270deg, #667eea, #764ba2, #f093fb, #667eea);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 4s ease infinite;
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0;
}
.sub-title {
    color: #a0aec0;
    font-size: 1rem;
    margin-top: 0;
}

/* ---- Portfolio table styling ---- */
.dataframe th {
    background: rgba(102, 126, 234, 0.3) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 🔄 Cached Helpers
# =========================
@st.cache_data(ttl=300, show_spinner="Fetching predictions...")
def cached_predictions(ticker, invest_amount, horizon):
    result = get_stock_predictions(ticker, invest_amount, horizon)
    # Do not cache empty/failed results — force a fresh attempt next run.
    if result.get("current_price", 0) == 0 or result.get("price_data") is None or (hasattr(result.get("price_data"), "empty") and result["price_data"].empty):
        st.cache_data.clear()
    return result

@st.cache_data(ttl=300, show_spinner="Generating portfolio...")
def cached_portfolio(total_amount, horizon, allocation_mode, top_n, max_weight_pct):
    return get_portfolio_allocation(total_amount, horizon, allocation_mode=allocation_mode, top_n=top_n, max_weight_pct=max_weight_pct)

@st.cache_data
def load_stock_list():
    try:
        return get_nse_stock_list()
    except Exception:
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS"]

stock_list = load_stock_list()

# =========================
# 🧭 Sidebar
# =========================
with st.sidebar:
    st.markdown("## 🚀 **Digitrader**")
    st.caption("Smart Trading Assistant")
    st.markdown("---")
    page = st.radio("Navigate", ["📊 Trading Dashboard", "💼 Portfolio Suggestions", "🔍 Stock Comparison", "📄 Research Results", "� Tracking Dashboard", "�📋 Browse All Stocks"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("##### ⏰ Market Hours")
    st.caption("NSE: Mon–Fri, 9:15 AM – 3:30 PM IST")
    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True, help="Clear cached data and reload fresh prices"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; opacity:0.5; font-size:0.75rem;'>v2.0 · Built with Streamlit</div>",
        unsafe_allow_html=True
    )


# =====================================================================
# 📊 PAGE 1: TRADING DASHBOARD
# =====================================================================
if page == "📊 Trading Dashboard":
    st.markdown('<p class="main-title">Trading Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Real-time predictions, sentiment analysis & actionable insights</p>', unsafe_allow_html=True)

    if not stock_list:
        st.error("⚠️ Unable to fetch NSE stocks. Please check your internet or the NSE API.")
        st.stop()
    
    # --- Stock count badge
    st.markdown(f"""
        <div style="text-align:center; margin-bottom:15px;">
            <span style="background: linear-gradient(135deg, rgba(102,126,234,0.3), rgba(118,75,162,0.3)); 
                         padding: 6px 16px; border-radius: 20px; font-size: 0.85rem; color: #a0aec0;">
                📊 {len(stock_list)} NSE Stocks Available for Analysis
            </span>
        </div>
    """, unsafe_allow_html=True)

    # --- Controls row with search
    col_stock, col_amount, col_horizon = st.columns([2, 1.5, 1.5])
    with col_stock:
        # Add search functionality
        search_term = st.text_input("🔍 Search Stock (Type to filter)", "", key="stock_search")
        if search_term:
            filtered_stocks = [s for s in stock_list if search_term.upper() in s.upper()]
            if filtered_stocks:
                stock_symbol = st.selectbox("🏢 Select Stock", filtered_stocks, index=0, key="stock_select")
            else:
                st.warning(f"No stocks found matching '{search_term}'")
                stock_symbol = st.selectbox("🏢 Select Stock", stock_list, index=0, key="stock_select_all")
        else:
            stock_symbol = st.selectbox("🏢 Select Stock", stock_list, index=0, key="stock_select_default")
    with col_amount:
        investment_amount = st.number_input("💰 Investment (₹)", min_value=100, value=1000, step=1)
    with col_horizon:
        horizon = st.selectbox("⏳ Horizon", ["Intraday", "Swing", "Long-Term"])

    st.markdown("---")

    try:
        prediction_data = cached_predictions(stock_symbol, investment_amount, horizon)
        trend = prediction_data["trend"]
        confidence = prediction_data["confidence"]
        sentiment = prediction_data["sentiment"]
        current_price = prediction_data["current_price"]
        predicted_price = prediction_data.get("predicted_price")
        predicted_return_pct = prediction_data.get("predicted_return_pct", 0.0)
        stop_loss = prediction_data.get("stop_loss")
        price_data = prediction_data.get("price_data")

        # --- Confidence as percent
        try:
            confidence_pct = float(confidence) * 100
        except Exception:
            confidence_pct = 0.0

        # ===== Metric Cards Row =====
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Current Price", f"₹{current_price:,.2f}")
        with m2:
            if predicted_price is not None:
                delta_val = predicted_price - current_price
                st.metric("Predicted Price", f"₹{predicted_price:,.2f}", delta=f"₹{delta_val:+,.2f}")
            else:
                st.metric("Predicted Price", "N/A")
        with m3:
            st.metric("Expected Return", f"{predicted_return_pct:+.2f}%")
        with m4:
            if stop_loss is not None:
                st.metric("Stop Loss", f"₹{stop_loss:,.2f}")
            else:
                st.metric("Stop Loss", "N/A")

        st.markdown("")

        # ===== Trend Badge + Confidence Gauge =====
        col_trend, col_gauge, col_sentiment = st.columns([1, 1.5, 1.5])

        with col_trend:
            trend_lower = trend.lower() if trend else "neutral"
            if "bull" in trend_lower:
                badge_class = "trend-bullish"
                trend_icon = "📈"
            elif "bear" in trend_lower:
                badge_class = "trend-bearish"
                trend_icon = "📉"
            else:
                badge_class = "trend-neutral"
                trend_icon = "➡️"
            st.markdown(f"""
                <div style="text-align:center; margin-top:10px;">
                    <p style="color:#a0aec0; font-size:0.85rem; text-transform:uppercase; letter-spacing:1px;">Predicted Trend</p>
                    <span class="trend-badge {badge_class}">{trend_icon} {trend}</span>
                </div>
            """, unsafe_allow_html=True)

        with col_gauge:
            # Confidence Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence_pct,
                number={"suffix": "%", "font": {"size": 28, "color": "white"}},
                title={"text": "Confidence", "font": {"size": 14, "color": "#a0aec0"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#555"},
                    "bar": {"color": "#667eea"},
                    "bgcolor": "rgba(30,30,60,0.5)",
                    "steps": [
                        {"range": [0, 33], "color": "rgba(252,74,26,0.3)"},
                        {"range": [33, 66], "color": "rgba(247,183,51,0.3)"},
                        {"range": [66, 100], "color": "rgba(0,176,155,0.3)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.8,
                        "value": confidence_pct,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=200, margin=dict(t=40, b=10, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_sentiment:
            # Sentiment Donut Chart
            sent_labels = ["Positive", "Neutral", "Negative"]
            sent_values = [
                sentiment["positive"] * 100,
                sentiment["neutral"] * 100,
                sentiment["negative"] * 100,
            ]
            sent_colors = ["#00b09b", "#667eea", "#fc4a1a"]
            fig_donut = go.Figure(go.Pie(
                labels=sent_labels, values=sent_values,
                hole=0.55, marker=dict(colors=sent_colors),
                textinfo="label+percent", textfont=dict(size=11, color="white"),
                hoverinfo="label+percent",
            ))
            fig_donut.update_layout(
                title=dict(text="Market Sentiment", font=dict(size=14, color="#a0aec0"), x=0.5),
                height=220, margin=dict(t=45, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                font={"color": "white"},
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("---")

        # ===== Candlestick Chart =====
        st.markdown("#### 📊 Price Chart")
        try:
            if price_data is None or price_data.empty:
                raise ValueError("No price data available")
            df_chart = price_data.copy()

            # Candlestick
            fig_candle = go.Figure()

            has_ohlc = all(c in df_chart.columns for c in ["Open", "High", "Low", "Close"])
            if has_ohlc:
                fig_candle.add_trace(go.Candlestick(
                    x=df_chart.index,
                    open=df_chart["Open"], high=df_chart["High"],
                    low=df_chart["Low"], close=df_chart["Close"],
                    name="OHLC",
                    increasing_line_color="#00b09b", decreasing_line_color="#fc4a1a",
                    increasing_fillcolor="#00b09b", decreasing_fillcolor="#fc4a1a",
                ))
            else:
                fig_candle.add_trace(go.Scatter(
                    x=df_chart.index, y=df_chart["Close"], mode="lines",
                    name="Close", line=dict(color="#667eea", width=2),
                ))

            # Volume as bars on secondary axis
            if "Volume" in df_chart.columns:
                fig_candle.add_trace(go.Bar(
                    x=df_chart.index, y=df_chart["Volume"],
                    name="Volume", marker_color="rgba(102,126,234,0.2)",
                    yaxis="y2",
                ))

            # Predicted price line
            if predicted_price is not None:
                fig_candle.add_hline(
                    y=predicted_price,
                    line=dict(dash="dash", color="#96c93d", width=1.5),
                    annotation_text=f"Target ₹{predicted_price:,.2f}",
                    annotation_font_color="#96c93d",
                )

            # Stop loss line
            if stop_loss is not None:
                fig_candle.add_hline(
                    y=stop_loss,
                    line=dict(dash="dot", color="#fc4a1a", width=1.5),
                    annotation_text=f"Stop Loss ₹{stop_loss:,.2f}",
                    annotation_font_color="#fc4a1a",
                    annotation_position="bottom left",
                )

            # Future predictions with dates
            future_predictions = prediction_data.get("future_predictions")
            if future_predictions is not None and not future_predictions.empty:
                fig_candle.add_trace(go.Scatter(
                    x=future_predictions["Date"],
                    y=future_predictions["Predicted_Price"],
                    mode="lines+markers",
                    name="Prediction Path",
                    line=dict(color="#96c93d", width=3, dash="dash"),
                    marker=dict(size=6, color="#96c93d"),
                    hovertemplate="<b>%{x|%d %b %Y, %H:%M}</b><br>Predicted: ₹%{y:,.2f}<extra></extra>",
                ))

            fig_candle.update_layout(
                height=480,
                title={
                    "text": (
                        f"📊 {stock_symbol} - {horizon} Prediction "
                        f"({df_chart.index[0].strftime('%d %b %Y')} to {df_chart.index[-1].strftime('%d %b %Y')}) "
                        f"→ Predicting till {future_predictions['Date'].max().strftime('%d %b %Y') if future_predictions is not None and not future_predictions.empty else 'N/A'}"
                    ) if len(df_chart) > 0 else f"📊 {stock_symbol} - {horizon} Prediction",
                    "font": {"size": 14, "color": "#a0aec0"},
                    "x": 0.5,
                    "xanchor": "center"
                },
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,35,0.8)",
                xaxis=dict(
                    rangeslider=dict(visible=False),
                    title="Date",
                    title_font=dict(size=12, color="#a0aec0"),
                    tickformat="%d %b",
                    tickangle=-45,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="rgba(102,126,234,0.1)",
                ),
                yaxis=dict(title="Price (₹)", side="left"),
                yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False, range=[0, df_chart.get("Volume", pd.Series([1])).max() * 4] if "Volume" in df_chart.columns else [0, 1]),
                margin=dict(t=60, b=60, l=60, r=60),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                font=dict(color="#ccc"),
            )
            st.plotly_chart(fig_candle, use_container_width=True)

        except Exception as e:
            st.warning(f"Chart not available: {e}")

        st.markdown("---")

        # ===== Investment Advice =====
        st.markdown("#### 💡 Investment Advice")
        try:
            advice_text = get_investment_advice(stock_symbol, horizon)
            st.markdown(f"""
                <div class="info-card">
                    {advice_text}
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

        st.markdown("---")
        
        # ===== Log to Sheets Button =====
        col_log1, col_log2, col_log3 = st.columns([1.5, 1, 1])
        with col_log1:
            st.markdown("#### 📊 Save to Tracking")
        with col_log2:
            if st.button("💾 Log to Sheets", use_container_width=True, key="log_to_sheets"):
                try:
                    from modules.sheets_tracker import get_tracker
                    sheets_url = st.session_state.get('sheets_url') or os.getenv("SHEETS_URL", "")
                    
                    if sheets_url:
                        tracker = get_tracker(sheets_url)
                        if tracker and tracker.authenticate() and tracker.open_sheet(sheets_url):
                            tracker.log_search(
                                symbol=stock_symbol,
                                trend=trend,
                                confidence=confidence,
                                current_price=current_price,
                                predicted_price=predicted_price,
                                expected_return=predicted_return_pct,
                                sentiment=sentiment
                            )
                            st.success(f"✅ Logged {stock_symbol} to Sheets!")
                        else:
                            st.warning("⚠️ Need to set up Google Sheets first. Go to 📊 Tracking Dashboard")
                    else:
                        st.warning("⚠️ No Sheets URL configured. Go to 📊 Tracking Dashboard to set up!")
                except ImportError:
                    st.info("📊 Sheets integration available with: `pip install gspread`")
                except Exception as e:
                    st.error(f"Error: {e}")
        with col_log3:
            if st.button("📈 To Tracking", use_container_width=True, key="nav_tracking"):
                # This would navigate to tracking dashboard
                st.info("Go to 📊 Tracking Dashboard in the sidebar to set up sheets!")

    except Exception as e:
        st.error(f"Error fetching predictions: {e}")

    st.markdown("---")
    horizon_tips = {
        "Intraday": "⚡ **Intraday** — Quick trades within market hours. Use tight stop-losses.",
        "Swing": "🌊 **Swing** — Hold 2–10 days. Look for breakout patterns.",
        "Long-Term": "🏛️ **Long-Term** — Buy and hold. Focus on fundamentals.",
    }
    st.caption(horizon_tips.get(horizon, ""))


# =====================================================================
# 💼 PAGE 2: PORTFOLIO SUGGESTIONS
# =====================================================================
elif page == "💼 Portfolio Suggestions":
    st.markdown('<p class="main-title">Portfolio Builder</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">AI-powered diversified allocation across NSE stocks</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        total_amount = st.number_input("💰 Total Investment (₹)", min_value=100, value=10000, step=1)
        horizon = st.selectbox("⏳ Investment Horizon", ["Intraday", "Swing", "Long-Term"])
    with col_b:
        allocation_mode = st.selectbox("🔧 Allocation Strategy", ["Proportional", "Equal", "Risk-adjusted"], index=0)
        col_cap, col_topn = st.columns(2)
        with col_cap:
            max_weight_pct = st.number_input("Max weight %", min_value=0.0, value=0.0, step=0.5, help="0 = no cap")
        with col_topn:
            show_top_n = st.number_input("Top N stocks", min_value=0, value=10, step=1, help="0 = all")

    allocation_mode_key = allocation_mode.lower().replace('-', '_')

    # Duration display
    duration_map = {
        "Intraday": ("⚡ Up to 4 Hours", "rgba(252,74,26,0.15)"),
        "Swing": ("🌊 2–10 Days", "rgba(247,183,51,0.15)"),
        "Long-Term": ("🏛️ 1–6 Months", "rgba(0,176,155,0.15)"),
    }
    dur_text, dur_bg = duration_map.get(horizon, ("—", "transparent"))
    st.markdown(f"""
        <div style="background:{dur_bg}; border-radius:10px; padding:10px 20px; display:inline-block; margin:10px 0;">
            <b>Investment Duration:</b> {dur_text}
        </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    if st.button("🚀 Generate Portfolio", use_container_width=True):
        try:
            portfolio = cached_portfolio(
                total_amount, horizon, allocation_mode_key,
                (None if show_top_n == 0 else int(show_top_n)),
                (None if max_weight_pct == 0 else float(max_weight_pct)),
            )
            if portfolio:
                df = pd.DataFrame(portfolio)
                df = df.sort_values(by="Weight (%)", ascending=False).reset_index(drop=True)

                total_profit = df["Expected Profit (₹)"].sum()
                total_return_pct = (total_profit / total_amount) * 100

                # ===== Summary Metrics =====
                sm1, sm2, sm3, sm4 = st.columns(4)
                with sm1:
                    st.metric("Stocks Selected", len(df))
                with sm2:
                    st.metric("Total Investment", f"₹{total_amount:,.0f}")
                with sm3:
                    st.metric("Expected Profit", f"₹{total_profit:,.2f}", delta=f"{total_return_pct:+.2f}%")
                with sm4:
                    st.metric("Strategy", allocation_mode)

                st.markdown("")

                # ===== Charts Row =====
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    # Pie chart — allocation by weight
                    top_display = df.head(15)
                    fig_pie = px.pie(
                        top_display, values="Weight (%)", names="Stock",
                        title="Portfolio Allocation",
                        color_discrete_sequence=px.colors.sequential.Plasma_r,
                        hole=0.4,
                    )
                    fig_pie.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=380,
                        margin=dict(t=50, b=20, l=20, r=20),
                        title_font=dict(size=14, color="#a0aec0"),
                        font=dict(color="#ccc"),
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with chart_col2:
                    # Bar chart — expected return per stock
                    top_bar = df.head(15).copy()
                    colors = ["#00b09b" if r >= 0 else "#fc4a1a" for r in top_bar["Expected Return (%)"]]
                    fig_bar = go.Figure(go.Bar(
                        x=top_bar["Stock"], y=top_bar["Expected Return (%)"],
                        marker_color=colors,
                        text=[f"{r:.1f}%" for r in top_bar["Expected Return (%)"]],
                        textposition="outside",
                    ))
                    fig_bar.update_layout(
                        title=dict(text="Expected Return by Stock", font=dict(size=14, color="#a0aec0")),
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15,15,35,0.8)",
                        height=380,
                        margin=dict(t=50, b=40, l=40, r=20),
                        yaxis_title="Return (%)",
                        font=dict(color="#ccc"),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.markdown("---")

                # ===== Data Table =====
                st.markdown("#### 📋 Full Allocation Table")

                # Color code the trend column
                def style_trend(val):
                    if val and "bull" in str(val).lower():
                        return "color: #00b09b; font-weight: bold"
                    elif val and "bear" in str(val).lower():
                        return "color: #fc4a1a; font-weight: bold"
                    return "color: #a0aec0"

                styled_df = df.style.applymap(style_trend, subset=["Trend"])
                st.dataframe(styled_df, use_container_width=True, height=400)

                # Warning for cap adjustment
                n_total = len(get_nse_stock_list())
                if max_weight_pct and max_weight_pct > 0 and max_weight_pct < float(100 / max(1, n_total)):
                    st.warning(f"Max cap ({max_weight_pct}%) was auto-adjusted to {round(100/n_total, 2)}%.")
            else:
                st.warning("No portfolio recommendations available.")
        except Exception as e:
            st.error(f"Error generating portfolio: {e}")


# =====================================================================
# 🔍 PAGE 3: STOCK COMPARISON
# =====================================================================
elif page == "🔍 Stock Comparison":
    st.markdown('<p class="main-title">Stock Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Compare multiple stocks side-by-side</p>', unsafe_allow_html=True)

    selected_stocks = st.multiselect("Select stocks to compare (2–5)", stock_list, default=stock_list[:3] if len(stock_list) >= 3 else stock_list, max_selections=5)
    horizon = st.selectbox("⏳ Horizon", ["Intraday", "Swing", "Long-Term"], key="compare_horizon")

    if selected_stocks and len(selected_stocks) >= 2:
        if st.button("⚡ Compare Stocks", use_container_width=True):
            comparison_data = []
            price_series = {}

            progress = st.progress(0)
            for i, stk in enumerate(selected_stocks):
                try:
                    pred = cached_predictions(stk, 10000, horizon)
                    comparison_data.append({
                        "Stock": stk,
                        "Price (₹)": f"₹{pred['current_price']:,.2f}",
                        "Predicted (₹)": f"₹{pred.get('predicted_price', 0):,.2f}" if pred.get("predicted_price") else "N/A",
                        "Return (%)": round(pred.get("predicted_return_pct", 0), 2),
                        "Trend": pred["trend"],
                        "Confidence (%)": round(float(pred["confidence"]) * 100, 2),
                        "Sentiment +": round(pred["sentiment"]["positive"] * 100, 1),
                        "Sentiment -": round(pred["sentiment"]["negative"] * 100, 1),
                    })
                    pd_data = pred.get("price_data")
                    if pd_data is not None and not pd_data.empty and "Close" in pd_data.columns:
                        # Normalize to percentage change from first day
                        close = pd_data["Close"].copy()
                        close_norm = (close / close.iloc[0] - 1) * 100
                        price_series[stk] = close_norm
                except Exception:
                    comparison_data.append({"Stock": stk, "Price (₹)": "Error", "Predicted (₹)": "—", "Return (%)": 0, "Trend": "N/A", "Confidence (%)": 0, "Sentiment +": 0, "Sentiment -": 0})
                progress.progress((i + 1) / len(selected_stocks))
            progress.empty()

            comp_df = pd.DataFrame(comparison_data)

            # ===== Comparison Metrics =====
            cols = st.columns(len(selected_stocks))
            for i, row in comp_df.iterrows():
                with cols[i]:
                    trend_lower = str(row["Trend"]).lower()
                    if "bull" in trend_lower:
                        delta_color = "normal"
                    elif "bear" in trend_lower:
                        delta_color = "inverse"
                    else:
                        delta_color = "off"
                    st.metric(row["Stock"], row["Price (₹)"], delta=f"{row['Return (%)']}%", delta_color=delta_color)

            st.markdown("---")

            # ===== Normalized price comparison chart =====
            if price_series:
                fig_compare = go.Figure()
                colors = ["#667eea", "#00b09b", "#fc4a1a", "#f7b733", "#764ba2"]
                for idx, (stk, series) in enumerate(price_series.items()):
                    fig_compare.add_trace(go.Scatter(
                        x=series.index, y=series.values,
                        mode="lines", name=stk,
                        line=dict(color=colors[idx % len(colors)], width=2.5),
                    ))
                fig_compare.update_layout(
                    title="Normalized Price Performance (%)",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,15,35,0.8)",
                    height=420,
                    margin=dict(t=50, b=40, l=60, r=20),
                    yaxis_title="Change (%)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    font=dict(color="#ccc"),
                )
                st.plotly_chart(fig_compare, use_container_width=True)

            # ===== Comparison Table =====
            st.markdown("#### 📋 Comparison Table")

            def style_comparison_trend(val):
                if val and "bull" in str(val).lower():
                    return "color: #00b09b; font-weight: bold"
                elif val and "bear" in str(val).lower():
                    return "color: #fc4a1a; font-weight: bold"
                return ""

            st.dataframe(
                comp_df.style.applymap(style_comparison_trend, subset=["Trend"]),
                use_container_width=True,
            )
    else:
        st.info("Select at least 2 stocks to compare.")


# =====================================================================
# 📄 PAGE 4: RESEARCH RESULTS
# =====================================================================
elif page == "📄 Research Results":
    st.markdown('<p class="main-title">Research Results</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">ML model evaluation, ablation study, and backtesting for paper</p>', unsafe_allow_html=True)

    # Check if results exist
    import os, glob
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 Model Comparison", "🧪 Ablation Study", "📊 Feature Importance",
        "💰 Trading Simulation", "⚡ Run Experiment"
    ])

    # --- Tab 5: Run experiment live
    with tab5:
        st.markdown("#### ⚡ Run Live Experiment")
        st.caption("Train all models on a single stock and see results instantly.")

        exp_stock = st.selectbox("Select Stock", stock_list, index=0, key="exp_stock")
        exp_horizon = st.selectbox("Horizon", ["Intraday", "Long-Term"], key="exp_horizon")

        if st.button("🚀 Run Experiment", use_container_width=True):
            with st.spinner("Training Random Forest, XGBoost, LSTM... (this may take 30-60 seconds)"):
                try:
                    from modules.predictive_ml import train_all_models
                    from modules.backtester import (
                        compute_classification_metrics, compute_regression_metrics,
                        get_confusion_matrix, get_feature_importance, simulate_trading,
                        run_ablation_study
                    )
                    from modules.feature_engineering import get_feature_columns

                    # Fetch data
                    if exp_horizon == "Intraday":
                        exp_data = fetch_price_data(exp_stock, period="5d", interval="1h")
                    else:
                        exp_data = fetch_price_data(exp_stock, period="6mo", interval="1d")

                    if exp_data is None or exp_data.empty:
                        st.error("No data available for this stock.")
                    else:
                        # Get sentiment
                        try:
                            from modules.sentiment_engine import analyze_hybrid_sentiment, analyze_finbert, analyze_general_sentiment, get_news_for_stock
                            headlines = get_news_for_stock(exp_stock)
                            if headlines:
                                fin_scores = [analyze_finbert(h["title"]) for h in headlines[:10]]
                                gen_scores = [analyze_general_sentiment(h["title"]) for h in headlines[:10]]
                                hyb_scores = [analyze_hybrid_sentiment(h["title"]) for h in headlines[:10]]
                                sent_scores = {
                                    "finbert": float(np.mean([s["positive"] - s["negative"] for s in fin_scores])),
                                    "textblob": float(np.mean([s["positive"] - s["negative"] for s in gen_scores])),
                                    "hybrid": float(np.mean([s["positive"] - s["negative"] for s in hyb_scores])),
                                }
                            else:
                                sent_scores = {"finbert": 0.0, "textblob": 0.0, "hybrid": 0.0}
                        except Exception:
                            sent_scores = {"finbert": 0.0, "textblob": 0.0, "hybrid": 0.0}

                        st.session_state["exp_sent_scores"] = sent_scores

                        # Train all models
                        results = train_all_models(exp_data, sentiment_score=sent_scores["hybrid"])

                        if results is None:
                            st.error("Not enough data to train models (need 30+ data points).")
                        else:
                            st.session_state["exp_results"] = results
                            st.session_state["exp_stock_name"] = exp_stock
                            st.session_state["exp_data"] = exp_data

                            # Run ablation study
                            ablation_df = run_ablation_study(exp_data, sent_scores)
                            st.session_state["exp_ablation"] = ablation_df

                            st.success(f"Experiment complete for {exp_stock}! Switch tabs to see results.")

                except Exception as e:
                    st.error(f"Experiment failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # Helper: check session state 
    exp_results = st.session_state.get("exp_results")
    exp_stock_name = st.session_state.get("exp_stock_name", "")

    # --- Tab 1: Model Comparison
    with tab1:
        st.markdown("#### 🤖 Model Comparison")

        # Try loading from saved CSVs first
        cls_file = None
        reg_file = None
        if os.path.exists(results_dir):
            cls_files = sorted(glob.glob(os.path.join(results_dir, "classification_metrics_*.csv")))
            reg_files = sorted(glob.glob(os.path.join(results_dir, "regression_metrics_*.csv")))
            if cls_files:
                cls_file = cls_files[-1]
            if reg_files:
                reg_file = reg_files[-1]

        # Source selection
        data_source = "live" if exp_results else ("saved" if cls_file else None)

        if data_source == "live" and exp_results:
            st.info(f"Showing results for **{exp_stock_name}** (live experiment)")

            from modules.backtester import compute_classification_metrics, compute_regression_metrics

            cls_rows = []
            reg_rows = []
            for mname, mdata in exp_results["models"].items():
                y_true_cls = exp_results["y_cls_test"][:len(mdata["cls_pred"])]
                y_true_reg = exp_results["y_reg_test"][:len(mdata["reg_pred"])]
                cls_rows.append(compute_classification_metrics(y_true_cls, mdata["cls_pred"], mname))
                reg_rows.append(compute_regression_metrics(y_true_reg, mdata["reg_pred"], mname))

            cls_df = pd.DataFrame(cls_rows)
            reg_df = pd.DataFrame(reg_rows)

        elif data_source == "saved" and cls_file:
            st.info("Showing saved experiment results")
            cls_df = pd.read_csv(cls_file)
            reg_df = pd.read_csv(reg_file) if reg_file else pd.DataFrame()
        else:
            st.warning("No results yet. Go to the **⚡ Run Experiment** tab first, or run `python run_experiments.py` from terminal.")
            cls_df = pd.DataFrame()
            reg_df = pd.DataFrame()

        if not cls_df.empty:
            # Classification metrics chart
            st.markdown("##### Classification Performance")
            metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
            available_metrics = [m for m in metrics_to_plot if m in cls_df.columns]

            if "Stock" in cls_df.columns:
                agg_cls = cls_df.groupby("Model")[available_metrics].mean().reset_index()
            else:
                agg_cls = cls_df

            fig_cls = go.Figure()
            colors = ["#667eea", "#00b09b", "#fc4a1a", "#f7b733"]
            for i, metric in enumerate(available_metrics):
                fig_cls.add_trace(go.Bar(
                    name=metric, x=agg_cls["Model"], y=agg_cls[metric],
                    marker_color=colors[i % len(colors)],
                    text=[f"{v:.3f}" for v in agg_cls[metric]],
                    textposition="outside",
                ))
            fig_cls.update_layout(
                barmode="group", template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,35,0.8)",
                height=400, margin=dict(t=30, b=40),
                yaxis_title="Score", yaxis_range=[0, 1.1],
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                font=dict(color="#ccc"),
            )
            st.plotly_chart(fig_cls, use_container_width=True)

            st.dataframe(agg_cls, use_container_width=True)

        if not reg_df.empty:
            st.markdown("##### Regression Performance")
            if "Stock" in reg_df.columns:
                agg_reg = reg_df.groupby("Model")[["MAE", "RMSE", "Directional Acc"]].mean().reset_index()
            else:
                agg_reg = reg_df

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                fig_rmse = px.bar(agg_reg, x="Model", y="RMSE", color="Model",
                    title="RMSE by Model", color_discrete_sequence=px.colors.sequential.Plasma_r)
                fig_rmse.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,15,35,0.8)", height=350, showlegend=False, font=dict(color="#ccc"))
                st.plotly_chart(fig_rmse, use_container_width=True)
            with col_r2:
                fig_dir = px.bar(agg_reg, x="Model", y="Directional Acc", color="Model",
                    title="Directional Accuracy", color_discrete_sequence=px.colors.sequential.Viridis)
                fig_dir.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,15,35,0.8)", height=350, showlegend=False,
                    yaxis_range=[0, 1.1], font=dict(color="#ccc"))
                st.plotly_chart(fig_dir, use_container_width=True)

            st.dataframe(agg_reg, use_container_width=True)

    # --- Tab 2: Ablation Study
    with tab2:
        st.markdown("#### 🧪 Ablation Study")
        st.caption("How each component (sentiment variant) affects prediction quality.")

        ablation_df = st.session_state.get("exp_ablation", pd.DataFrame())

        # Try loading from saved files
        if ablation_df.empty and os.path.exists(results_dir):
            ab_files = sorted(glob.glob(os.path.join(results_dir, "ablation_study_*.csv")))
            if ab_files:
                ablation_df = pd.read_csv(ab_files[-1])

        if ablation_df.empty:
            st.warning("No ablation results yet. Run an experiment first.")
        else:
            if "Stock" in ablation_df.columns:
                agg_ab = ablation_df.groupby("Variant")[["Accuracy", "F1-Score", "RMSE", "Dir. Accuracy"]].mean().reset_index()
            else:
                agg_ab = ablation_df

            # Bar chart
            fig_ab = go.Figure()
            ab_colors = ["#667eea", "#00b09b", "#fc4a1a"]
            for i, metric in enumerate(["Accuracy", "F1-Score", "Dir. Accuracy"]):
                if metric in agg_ab.columns:
                    fig_ab.add_trace(go.Bar(
                        name=metric, x=agg_ab["Variant"], y=agg_ab[metric],
                        marker_color=ab_colors[i % len(ab_colors)],
                        text=[f"{v:.3f}" for v in agg_ab[metric]],
                        textposition="outside",
                    ))
            fig_ab.update_layout(
                barmode="group", template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,35,0.8)",
                height=420, margin=dict(t=30, b=80),
                yaxis_title="Score", yaxis_range=[0, 1.1],
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                font=dict(color="#ccc"),
            )
            st.plotly_chart(fig_ab, use_container_width=True)

            st.dataframe(agg_ab, use_container_width=True)

            # Key insight
            if len(agg_ab) > 1:
                best = agg_ab.loc[agg_ab["Accuracy"].idxmax()]
                worst = agg_ab.loc[agg_ab["Accuracy"].idxmin()]
                st.markdown(f"""
                    <div class="info-card">
                        <b>Key Finding:</b> <i>{best['Variant']}</i> achieves the highest accuracy ({best['Accuracy']:.4f}),
                        outperforming <i>{worst['Variant']}</i> ({worst['Accuracy']:.4f}) by
                        <b>{(best['Accuracy'] - worst['Accuracy'])*100:.2f} percentage points</b>.
                    </div>
                """, unsafe_allow_html=True)

    # --- Tab 3: Feature Importance
    with tab3:
        st.markdown("#### 📊 Feature Importance")

        if exp_results:
            from modules.backtester import get_feature_importance
            from modules.feature_engineering import get_feature_columns
            feature_cols = get_feature_columns()

            for mname in ["RandomForest", "XGBoost"]:
                if mname in exp_results["models"] and "clf" in exp_results["models"][mname]:
                    fi = get_feature_importance(exp_results["models"][mname]["clf"], feature_cols)
                    if not fi.empty:
                        top_n = fi.head(15)
                        fig_fi = px.bar(
                            top_n, x="Importance", y="Feature", orientation="h",
                            title=f"{mname} — Top 15 Features",
                            color="Importance", color_continuous_scale="Viridis",
                        )
                        fig_fi.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,35,0.8)",
                            height=450, margin=dict(t=50, b=20, l=150),
                            yaxis=dict(autorange="reversed"),
                            font=dict(color="#ccc"),
                        )
                        st.plotly_chart(fig_fi, use_container_width=True)
        else:
            # Try loading from file
            fi_loaded = False
            if os.path.exists(results_dir):
                fi_files = sorted(glob.glob(os.path.join(results_dir, "feature_importance_*.csv")))
                if fi_files:
                    fi_df = pd.read_csv(fi_files[-1])
                    if not fi_df.empty:
                        fi_loaded = True
                        top_fi = fi_df.groupby("Feature")["Importance"].mean().sort_values(ascending=False).head(15).reset_index()
                        fig_fi = px.bar(
                            top_fi, x="Importance", y="Feature", orientation="h",
                            title="Top 15 Features (Averaged)",
                            color="Importance", color_continuous_scale="Viridis",
                        )
                        fig_fi.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,35,0.8)",
                            height=450, margin=dict(t=50, b=20, l=150),
                            yaxis=dict(autorange="reversed"),
                            font=dict(color="#ccc"),
                        )
                        st.plotly_chart(fig_fi, use_container_width=True)
            if not fi_loaded:
                st.warning("Run an experiment first to see feature importance.")

    # --- Tab 4: Trading Simulation
    with tab4:
        st.markdown("#### 💰 Trading Simulation (Backtest)")
        st.caption("Simulated P&L: invest when model predicts Bullish, hold cash when Bearish.")

        if exp_results:
            from modules.backtester import simulate_trading

            sim_results = []
            for mname, mdata in exp_results["models"].items():
                if "Baseline" in mname:
                    continue
                try:
                    y_true = exp_results["y_reg_test"][:len(mdata["cls_pred"])]
                    sim = simulate_trading(y_true, mdata["cls_pred"])
                    sim["Model"] = mname
                    sim_results.append(sim)
                except Exception:
                    pass

            # Also add buy & hold reference
            if sim_results:
                # Equity curve chart
                fig_eq = go.Figure()
                eq_colors = ["#667eea", "#00b09b", "#fc4a1a", "#f7b733", "#764ba2"]
                for i, sim in enumerate(sim_results):
                    fig_eq.add_trace(go.Scatter(
                        y=sim["strategy_equity"], mode="lines",
                        name=f"{sim['Model']} Strategy",
                        line=dict(color=eq_colors[i % len(eq_colors)], width=2.5),
                    ))
                # Add buy & hold from first sim
                fig_eq.add_trace(go.Scatter(
                    y=sim_results[0]["buyhold_equity"], mode="lines",
                    name="Buy & Hold",
                    line=dict(color="#888", width=2, dash="dash"),
                ))
                fig_eq.update_layout(
                    title="Equity Curve — Strategy vs Buy & Hold",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,35,0.8)",
                    height=420, margin=dict(t=50, b=40),
                    yaxis_title="Portfolio Value (₹)",
                    xaxis_title="Trading Period",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    font=dict(color="#ccc"),
                )
                st.plotly_chart(fig_eq, use_container_width=True)

                # Summary metrics
                sim_summary = []
                for sim in sim_results:
                    sim_summary.append({
                        "Model": sim["Model"],
                        "Strategy Return (%)": sim["total_return_pct"],
                        "Buy&Hold Return (%)": sim["buyhold_return_pct"],
                        "Sharpe Ratio": sim["sharpe_ratio"],
                        "Max Drawdown (%)": sim["max_drawdown_pct"],
                        "# Trades": sim["n_trades"],
                    })
                sim_df = pd.DataFrame(sim_summary)
                st.dataframe(sim_df, use_container_width=True)

                # Best model callout
                best_sim = max(sim_results, key=lambda s: s["total_return_pct"])
                bh_ret = sim_results[0]["buyhold_return_pct"]
                outperform = best_sim["total_return_pct"] - bh_ret
                st.markdown(f"""
                    <div class="info-card">
                        <b>Result:</b> {best_sim['Model']} achieves <b>{best_sim['total_return_pct']:.2f}%</b> return
                        vs Buy & Hold <b>{bh_ret:.2f}%</b>
                        ({'+' if outperform >= 0 else ''}{outperform:.2f}pp {'outperformance' if outperform >= 0 else 'underperformance'}).
                        Sharpe Ratio: <b>{best_sim['sharpe_ratio']:.4f}</b>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No simulation results available.")
        else:
            # Try loading from saved files
            trading_loaded = False
            if os.path.exists(results_dir):
                t_files = sorted(glob.glob(os.path.join(results_dir, "trading_simulation_*.csv")))
                if t_files:
                    t_df = pd.read_csv(t_files[-1])
                    if not t_df.empty:
                        trading_loaded = True
                        st.dataframe(t_df, use_container_width=True)

                        agg_t = t_df.groupby("Model")[["Strategy Return (%)", "Buy&Hold Return (%)", "Sharpe Ratio"]].mean().reset_index()
                        fig_t = px.bar(agg_t, x="Model", y=["Strategy Return (%)", "Buy&Hold Return (%)"],
                            barmode="group", title="Avg Returns: Strategy vs Buy & Hold",
                            color_discrete_sequence=["#667eea", "#888"])
                        fig_t.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(15,15,35,0.8)", height=380, font=dict(color="#ccc"))
                        st.plotly_chart(fig_t, use_container_width=True)
            if not trading_loaded:
                st.warning("Run an experiment first to see trading simulation.")

# =====================================================================
# � PAGE 5: TRACKING DASHBOARD (Google Sheets Integration)
# =====================================================================
elif page == "📊 Tracking Dashboard":
    st.markdown('<p class="main-title">📊 Investment Tracking Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Real-time portfolio tracking with Google Sheets integration</p>', unsafe_allow_html=True)
    
    try:
        from modules.sheets_tracker import get_tracker
    except ImportError:
        st.error("⚠️ Google Sheets integration not installed. Run: `pip install gspread google-auth-oauthlib`")
        st.stop()
    
    # Setup tracker from session or config
    tracker = None
    sheets_url = None
    
    # Check for saved sheet URL in .env or session
    if 'sheets_url' in st.session_state:
        sheets_url = st.session_state['sheets_url']
    else:
        # Try to load from .env
        import os
        from dotenv import load_dotenv
        load_dotenv()
        sheets_url = os.getenv("SHEETS_URL", "")
    
    # If we have a URL, try to connect
    if sheets_url:
        tracker = get_tracker(sheets_url)
        if tracker and tracker.authenticate():
            if tracker.open_sheet(sheets_url):
                st.session_state['sheets_url'] = sheets_url
    
    # Setup section
    with st.expander("⚙️ Setup Sheets & Start Tracking", expanded=not tracker):
        st.markdown("""
        ### Getting Started with Google Sheets
        
        #### Option 1: Create a New Sheet (Recommended)
        """)
        
        if st.button("🆕 Create New Investment Tracker Sheet", use_container_width=True):
            tracker = get_tracker()
            if tracker and tracker.authenticate():
                sheet_url = tracker.create_sheet("Investment Tracker - Digitrader")
                if sheet_url:
                    st.session_state['sheets_url'] = sheet_url
                    os.environ['SHEETS_URL'] = sheet_url
                    st.success(f"✅ Sheet created! 🎉\n\n**Your Sheet URL:**\n{sheet_url}")
                    st.info("📌 Bookmark this URL or save it somewhere safe!")
                    st.rerun()
            else:
                st.error("❌ Failed to authenticate. Check credentials setup in SHEETS_SETUP.md")
        
        st.markdown("#### Option 2: Connect Existing Sheet")
        sheet_url_input = st.text_input("📋 Paste your Google Sheet URL here:")
        if sheet_url_input and st.button("🔗 Connect", use_container_width=True):
            tracker = get_tracker(sheet_url_input)
            if tracker and tracker.authenticate():
                if tracker.open_sheet(sheet_url_input):
                    st.session_state['sheets_url'] = sheet_url_input
                    os.environ['SHEETS_URL'] = sheet_url_input
                    st.success("✅ Connected to sheet!")
                    st.rerun()
            else:
                st.error("❌ Failed to connect. Make sure credentials are set up.")
        
        st.markdown("---")
        st.markdown("""
        #### Setup Instructions:
        1. **Download credentials**: [Google Cloud Console](https://console.cloud.google.com/)
        2. **Save credentials**: Place `google_credentials.json` in project root
        3. **Read guide**: See SHEETS_SETUP.md for detailed instructions
        """)
        
        if st.button("📖 Open Setup Guide"):
            st.info("""
            **Google Sheets Integration Setup:**
            
            See SHEETS_SETUP.md in the project root for:
            - Step-by-step Google Cloud setup
            - Service account creation
            - Credentials configuration
            - Troubleshooting guide
            """)
    
    # Main tracking interface (only if connected)
    if tracker and sheets_url:
        st.success(f"✅ Connected! Sheet: {sheets_url}")
        st.markdown("---")
        
        # Tabs for different tracking functions
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Log Search", "💰 Log Investment", "📈 Daily Analysis", 
            "📋 Portfolio", "📜 History"
        ])
        
        # --- Tab 1: Log Stock Search
        with tab1:
            st.markdown("#### 📊 Log Stock Search")
            st.caption("Save your stock analysis to the sheets")
            
            col1, col2 = st.columns(2)
            with col1:
                search_symbol = st.selectbox("🏢 Select Stock", stock_list, key="track_symbol")
                search_amount = st.number_input("💰 Potential Investment (₹)", min_value=0, value=1000, step=100)
            with col2:
                search_horizon = st.selectbox("⏳ Horizon", ["Intraday", "Swing", "Long-Term"], key="track_horizon")
            
            if st.button("📌 Get & Log Analysis", use_container_width=True):
                with st.spinner("Fetching analysis..."):
                    pred = get_stock_predictions(search_symbol, search_amount, search_horizon)
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Trend", pred['trend'])
                with col_b:
                    st.metric("Confidence", f"{float(pred['confidence'])*100:.1f}%")
                with col_c:
                    st.metric("Current Price", f"₹{pred['current_price']:.2f}")
                with col_d:
                    st.metric("Predicted Price", f"₹{pred.get('predicted_price', 0):.2f}" if pred.get('predicted_price') else "N/A")
                
                # Log to sheets button
                if st.button("💾 Save to Sheets", use_container_width=True, key="log_search"):
                    if tracker.log_search(
                        symbol=search_symbol,
                        trend=pred['trend'],
                        confidence=pred['confidence'],
                        current_price=pred['current_price'],
                        predicted_price=pred.get('predicted_price', 0),
                        expected_return=pred.get('predicted_return_pct', 0),
                        sentiment=pred['sentiment']
                    ):
                        st.success(f"✅ Logged: {search_symbol} ({pred['trend']})")
        
        # --- Tab 2: Log Investment
        with tab2:
            st.markdown("#### 💰 Log Investment")
            st.caption("Record your stock purchases")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                inv_symbol = st.selectbox("🏢 Stock", stock_list, key="inv_symbol")
                inv_amount = st.number_input("💵 Investment Amount (₹)", min_value=100, value=5000, step=100)
            with col2:
                inv_price = st.number_input("💹 Entry Price (₹)", min_value=1.0, value=1000.0, step=0.5)
                inv_horizon = st.selectbox("⏳ Horizon", ["Intraday", "Swing", "Long-Term"], key="inv_horizon")
            with col3:
                inv_qty = inv_amount / inv_price
                st.metric("Quantity", f"{inv_qty:.2f} shares")
                st.metric("With Entry Price", f"₹{inv_price:.2f}")
            
            if st.button("💾 Log Investment", use_container_width=True, key="log_inv"):
                if tracker.log_investment(inv_symbol, inv_amount, inv_price, inv_horizon):
                    st.success(f"✅ Logged: {inv_symbol} @ ₹{inv_price:.2f}")
                    if st.button("Update Portfolio Now"):
                        tracker.update_portfolio(inv_symbol, inv_qty, inv_price, inv_amount)
                        st.success("📊 Portfolio updated!")
        
        # --- Tab 3: Daily Analysis
        with tab3:
            st.markdown("#### 📈 Daily Portfolio Analysis")
            st.caption("Log today's portfolio snapshot for daily tracking")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                daily_invested = st.number_input("💰 Total Invested Today (₹)", min_value=0, value=50000, step=1000)
            with col2:
                daily_current = st.number_input("📊 Current Portfolio Value (₹)", min_value=0, value=52000, step=1000)
            with col3:
                daily_best = st.text_input("📈 Best Performer Today", "RELIANCE.NS")
            with col4:
                daily_worst = st.text_input("📉 Worst Performer Today", "N/A")
            
            daily_notes = st.text_area("📝 Notes", "Market analysis, key events, etc.", height=80)
            
            pnl = daily_current - daily_invested
            ret_pct = (pnl / daily_invested * 100) if daily_invested > 0 else 0
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Daily P&L", f"₹{pnl:+,.0f}", f"{ret_pct:+.2f}%", delta_color="normal" if pnl >= 0 else "inverse")
            with col_stat2:
                pnl_color = "🟢" if pnl >= 0 else "🔴"
                st.markdown(f"{pnl_color} **{ret_pct:+.2f}%**")
            with col_stat3:
                st.markdown(f"Value: **₹{daily_current:,.0f}**")
            
            if st.button("📈 Log Daily Analysis", use_container_width=True, key="log_daily"):
                if tracker.log_daily_analysis(daily_invested, daily_current, daily_best, daily_worst, daily_notes):
                    st.success("✅ Daily analysis logged!")
        
        # --- Tab 4: Portfolio
        with tab4:
            st.markdown("#### 📋 Current Portfolio")
            st.caption("Your active holdings")
            
            portfolio_df = tracker.get_portfolio()
            
            if not portfolio_df.empty:
                # Display as styled table
                st.dataframe(portfolio_df, use_container_width=True, height=400)
                
                # Summary
                col_s1, col_s2, col_s3 = st.columns(3)
                try:
                    total_investment = float(portfolio_df['Investment (₹)'].astype(str).str.replace(',', '').sum())
                    total_value = float(portfolio_df['Current Value (₹)'].astype(str).str.replace(',', '').sum())
                    total_pnl = total_value - total_investment
                    total_return = (total_pnl / total_investment * 100) if total_investment > 0 else 0
                    
                    with col_s1:
                        st.metric("Total Invested", f"₹{total_investment:,.0f}")
                    with col_s2:
                        st.metric("Current Value", f"₹{total_value:,.0f}")
                    with col_s3:
                        st.metric("Total P&L", f"₹{total_pnl:+,.0f}", f"{total_return:+.2f}%", 
                                 delta_color="normal" if total_pnl >= 0 else "inverse")
                except:
                    st.info("Waiting for portfolio data...")
                
                # Update individual holding
                st.markdown("---")
                st.markdown("#### ✏️ Update Holdings")
                col1, col2, col3 = st.columns(3)
                with col1:
                    upd_symbol = st.selectbox("Select Stock", portfolio_df['Symbol'].tolist() if not portfolio_df.empty else ["N/A"])
                with col2:
                    upd_price = st.number_input("Current Price (₹)", min_value=1.0, value=1000.0, step=0.5)
                with col3:
                    if st.button("🔄 Update Price"):
                        # Find shares and investment
                        sym_data = portfolio_df[portfolio_df['Symbol'] == upd_symbol]
                        if not sym_data.empty:
                            shares = float(sym_data['Shares'].iloc[0])
                            investment = float(sym_data['Investment (₹)'].iloc[0])
                            tracker.update_portfolio(upd_symbol, shares, upd_price, investment)
                            st.success(f"✅ Updated {upd_symbol}")
                            st.rerun()
            else:
                st.info("📋 No portfolio data yet. Log an investment to start!")
        
        # --- Tab 5: History & Analysis
        with tab5:
            st.markdown("#### 📜 Search & Analysis History")
            
            col1, col2 = st.columns(2)
            with col1:
                days_filter = st.slider("📅 Last N days", 1, 90, 7)
            with col2:
                if st.button("🔄 Refresh Data"):
                    st.rerun()
            
            # Recent searches
            st.markdown("##### 📊 Recent Searches")
            searches_df = tracker.get_searches(days=days_filter)
            if not searches_df.empty:
                st.dataframe(searches_df, use_container_width=True, height=250)
                
                # Summary stats
                col_x1, col_x2, col_x3 = st.columns(3)
                with col_x1:
                    bullish = len(searches_df[searches_df['Trend'].str.contains('Bullish', na=False)])
                    st.metric("Bullish Signals", bullish)
                with col_x2:
                    bearish = len(searches_df[searches_df['Trend'].str.contains('Bearish', na=False)])
                    st.metric("Bearish Signals", bearish)
                with col_x3:
                    avg_conf = searches_df['Confidence (%)'].apply(lambda x: float(str(x).replace('%', '')) if x else 0).mean()
                    st.metric("Avg Confidence", f"{avg_conf:.1f}%")
            else:
                st.info("No searches logged yet.")
            
            # Daily analysis history
            st.markdown("##### 📈 Daily Analysis History")
            analysis_df = tracker.get_daily_analysis(days=days_filter)
            if not analysis_df.empty:
                st.dataframe(analysis_df, use_container_width=True, height=250)
                
                # Plot trend
                try:
                    analysis_df['Date'] = pd.to_datetime(analysis_df['Date'])
                    analysis_df['Overall Return (%)'] = analysis_df['Overall Return (%)'].astype(float)
                    
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=analysis_df['Date'], 
                        y=analysis_df['Overall Return (%)'],
                        mode='lines+markers',
                        name='Portfolio Return %',
                        line=dict(color='#667eea', width=3),
                        fill='tozeroy',
                    ))
                    fig_trend.update_layout(
                        title="Portfolio Return Trend",
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15,15,35,0.8)",
                        height=350,
                        xaxis_title="Date",
                        yaxis_title="Return (%)",
                        font=dict(color="#ccc"),
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                except:
                    pass
            else:
                st.info("No daily analysis logged yet.")
    else:
        st.warning("⚠️ Please connect to a Google Sheet first to start tracking!")
        st.info("""
        **Next Steps:**
        1. Expand **"⚙️ Setup Sheets & Start Tracking"** above
        2. Click **"🆕 Create New Investment Tracker Sheet"**
        3. Come back here to start logging!
        """)

# =====================================================================
# 📋 PAGE 6: BROWSE ALL NSE STOCKS
# =====================================================================
elif page == "📋 Browse All Stocks":
    st.markdown('<p class="main-title">Browse All NSE Stocks</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Complete list of available stocks for analysis</p>', unsafe_allow_html=True)
    
    # Stats overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Stocks", len(stock_list))
    with col2:
        nifty_50 = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", "ITC.NS", "SBIN.NS"]
        nifty_count = sum(1 for s in stock_list if any(n in s for n in nifty_50))
        st.metric("NIFTY 50 Stocks", "50+")
    with col3:
        st.metric("Categories", "14+")
    
    st.markdown("---")
    
    # Category definitions
    categories = {
        "🏆 NIFTY 50": [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", "ITC.NS",
            "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "LT.NS", "ASIANPAINT.NS",
            "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ADANIENT.NS", "ULTRACEMCO.NS",
            "WIPRO.NS", "NESTLEIND.NS", "HCLTECH.NS", "M&M.NS", "TATAMOTORS.NS", "NTPC.NS",
            "BAJAJFINSV.NS", "TATASTEEL.NS", "ONGC.NS", "COALINDIA.NS", "POWERGRID.NS", "JSWSTEEL.NS",
            "TECHM.NS", "INDUSINDBK.NS", "DIVISLAB.NS", "HINDALCO.NS", "ADANIPORTS.NS", "CIPLA.NS",
            "DRREDDY.NS", "EICHERMOT.NS", "BRITANNIA.NS", "BPCL.NS", "APOLLOHOSP.NS", "BAJAJ-AUTO.NS",
            "HEROMOTOCO.NS", "TRENT.NS", "GRASIM.NS", "HDFCLIFE.NS", "SBILIFE.NS", "SHRIRAMFIN.NS",
            "LTIM.NS", "BEL.NS"
        ],
        "💻 IT & Technology": [
            "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS", "COFORGE.NS",
            "MPHASIS.NS", "PERSISTENT.NS", "LTTS.NS", "KPITTECH.NS", "CYIENT.NS", "SONATSOFTW.NS",
            "ZENTEC.NS", "MASTEK.NS", "HAPPSTMNDS.NS", "ROUTE.NS", "RATEGAIN.NS"
        ],
        "🏦 Banking & Finance": [
            "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS",
            "BANDHANBNK.NS", "AUBANK.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS", "RBLBANK.NS", "PNB.NS",
            "CANBK.NS", "BANKBARODA.NS", "UNIONBANK.NS", "IOB.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
            "CHOLAFIN.NS", "LICHSGFIN.NS", "SRTRANSFIN.NS", "HDFCAMC.NS", "HDFCLIFE.NS", "SBILIFE.NS",
            "ICICIGI.NS", "ICICIPRULI.NS", "LICI.NS", "SBICARD.NS", "PFC.NS", "RECLTD.NS"
        ],
        "🚗 Auto & Components": [
            "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",
            "TVSMOTOR.NS", "ASHOKLEY.NS", "ESCORTS.NS", "MOTHERSON.NS", "BALKRISIND.NS", "MRF.NS",
            "APOLLOTYRE.NS", "BHARATFORG.NS", "BOSCHLTD.NS", "EXIDEIND.NS", "SONACOMS.NS"
        ],
        "💊 Pharma & Healthcare": [
            "SUNPHARMA.NS", "DIVISLAB.NS", "DRREDDY.NS", "CIPLA.NS", "LUPIN.NS", "AUROPHARMA.NS",
            "BIOCON.NS", "CADILAHC.NS", "GLENMARK.NS", "LAURUSLABS.NS", "TORNTPHARM.NS", "ALKEM.NS",
            "PFIZER.NS", "ABBOTINDIA.NS", "SYNGENE.NS", "LALPATHLAB.NS", "APOLLOHOSP.NS", "ZYDUSLIFE.NS"
        ],
        "🛍️ FMCG & Consumer": [
            "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "MARICO.NS",
            "GODREJCP.NS", "TATACONSUM.NS", "MCDOWELL-N.NS", "UBL.NS", "COLPAL.NS", "PIDILITIND.NS",
            "PGHH.NS", "VBL.NS", "JUBLFOOD.NS"
        ],
        "⚡ Energy & Oil/Gas": [
            "RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "HINDPETRO.NS", "COALINDIA.NS", "GAIL.NS",
            "NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", "IGL.NS",
            "MGL.NS", "PETRONET.NS"
        ],
        "🏗️ Infrastructure": [
            "LT.NS", "ULTRACEMCO.NS", "GRASIM.NS", "ADANIPORTS.NS", "AMBUJACEM.NS", "ACC.NS",
            "SHREECEM.NS", "JKCEMENT.NS", "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "CONCOR.NS"
        ],
        "⚙️ Metals & Mining": [
            "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "JINDALSTEL.NS", "VEDL.NS", "NATIONALUM.NS",
            "SAIL.NS", "NMDC.NS", "HINDZINC.NS"
        ],
        "📱 Telecom & Media": [
            "BHARTIARTL.NS", "IDEA.NS", "INDUSTOWER.NS", "TATACOMM.NS", "ZEEL.NS", "PVRINOX.NS"
        ],
        "🛒 E-commerce & New Age": [
            "ZOMATO.NS", "NYKAA.NS", "PAYTM.NS", "POLICYBZR.NS", "DELHIVERY.NS", "EASEMYTRIP.NS"
        ],
        "🏪 Retail & Hospitality": [
            "DMART.NS", "TRENT.NS", "TITAN.NS", "INDIGO.NS", "IRCTC.NS", "INDHOTEL.NS", "WESTLIFE.NS"
        ]
    }
    
    # Search and filter
    search = st.text_input("🔍 Search stocks by name or symbol", "")
    
    # Category tabs
    tab_names = ["📊 All Stocks"] + list(categories.keys())
    tabs = st.tabs(tab_names)
    
    # All Stocks tab
    with tabs[0]:
        filtered = stock_list
        if search:
            filtered = [s for s in stock_list if search.upper() in s.upper()]
        
        if filtered:
            # Display in columns for better viewing
            cols_per_row = 5
            num_stocks = len(filtered)
            for i in range(0, num_stocks, cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < num_stocks:
                        stock = filtered[idx]
                        # Remove .NS for display
                        display_name = stock.replace(".NS", "")
                        with col:
                            st.markdown(f"""
                                <div style="background: linear-gradient(135deg, rgba(40,40,80,0.5), rgba(60,60,120,0.3));
                                           border-radius: 8px; padding: 8px; margin: 4px 0; text-align: center;
                                           border: 1px solid rgba(102,126,234,0.2);">
                                    <span style="font-size: 0.85rem; color: #a0aec0;">{display_name}</span>
                                </div>
                            """, unsafe_allow_html=True)
            st.caption(f"Showing {len(filtered)} stocks")
        else:
            st.warning(f"No stocks found matching '{search}'")
    
    # Category tabs
    for idx, (cat_name, cat_stocks) in enumerate(categories.items(), 1):
        with tabs[idx]:
            filtered_cat = cat_stocks
            if search:
                filtered_cat = [s for s in cat_stocks if search.upper() in s.upper()]
            
            if filtered_cat:
                cols_per_row = 5
                num_stocks = len(filtered_cat)
                for i in range(0, num_stocks, cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        idx_s = i + j
                        if idx_s < num_stocks:
                            stock = filtered_cat[idx_s]
                            display_name = stock.replace(".NS", "")
                            with col:
                                st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, rgba(40,40,80,0.5), rgba(60,60,120,0.3));
                                               border-radius: 8px; padding: 8px; margin: 4px 0; text-align: center;
                                               border: 1px solid rgba(102,126,234,0.2);">
                                        <span style="font-size: 0.85rem; color: #a0aec0;">{display_name}</span>
                                    </div>
                                """, unsafe_allow_html=True)
                st.caption(f"Showing {len(filtered_cat)}/{len(cat_stocks)} stocks in {cat_name}")
            else:
                st.warning(f"No stocks found in {cat_name} matching '{search}'")

st.markdown("---")
st.markdown("""
    <div style="text-align:center; padding:20px 0;">
        <span style="font-size:1.1rem; font-weight:600; background: linear-gradient(90deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🚀 Digitrader
        </span>
        <br/>
        <span style="color:#666; font-size:0.8rem;">Smart Trading Assistant · For educational purposes only · Not financial advice</span>
    </div>
""", unsafe_allow_html=True)
