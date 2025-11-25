import streamlit as st
import pandas as pd
from modules.utils import (
    get_stock_predictions,
    get_portfolio_allocation,
    get_investment_advice,
    get_nse_stock_list,
)

# =========================
# 🧩 Streamlit Setup
# =========================
st.set_page_config(page_title="Smart Trading Assistant", page_icon="📊", layout="wide")

st.sidebar.title("📈 Navigation")
page = st.sidebar.radio("Go to", ["Trading Dashboard", "Portfolio Suggestions"])

# =========================
# ⚙️ Load NSE Stock List
# =========================
@st.cache_data
def load_stock_list():
    try:
        return get_nse_stock_list()
    except Exception:
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS"]

stock_list = load_stock_list()

# =========================
# 📊 PAGE 1: TRADING DASHBOARD
# =========================
if page == "Trading Dashboard":
    st.title("📊 Trading Dashboard")

    if not stock_list:
        st.error("⚠️ Unable to fetch NSE stocks. Please check your internet or the NSE API.")
        st.stop()

    # --- Stock Selection
    stock_symbol = st.selectbox("Select Stock", stock_list, index=0)
    investment_amount = st.number_input("💰 Investment Amount (₹)", min_value=1000, step=500)
    horizon = st.radio("⏳ Investment Horizon", ["Intraday", "Swing", "Long-Term"])

    st.markdown("---")

    # --- Stock Predictions & Sentiment
    st.subheader(f"📈 Predictions & Market Sentiment for {stock_symbol}")

    try:
        prediction_data = get_stock_predictions(stock_symbol, investment_amount, horizon)
        trend = prediction_data["trend"]
        confidence = prediction_data["confidence"]
        sentiment = prediction_data["sentiment"]
        current_price = prediction_data["current_price"]
        predicted_price = prediction_data.get("predicted_price")
        predicted_return_pct = prediction_data.get("predicted_return_pct", 0.0)
        stop_loss = prediction_data.get("stop_loss")

        st.markdown(f"**🔹 Current Price:** ₹{current_price:.2f}")
        st.markdown(f"**🔹 Predicted Trend:** {trend}")
        if predicted_price is not None:
            st.markdown(f"**🔹 Predicted Price:** ₹{predicted_price:.2f}")
            st.markdown(f"**🔹 Predicted Return (%):** {predicted_return_pct:.2f}%")
        if stop_loss is not None:
            st.markdown(f"**🔹 Suggested Stop Loss:** ₹{stop_loss:.2f}")
        # `confidence` from the prediction code is a float in [0, 1]; convert to percent for display
        try:
            confidence_pct = float(confidence) * 100
        except Exception:
            confidence_pct = 0.0
        # display small non-zero confidences as <0.01% instead of 0.00%
        if 0 < confidence_pct < 0.01:
            conf_display = "<0.01%"
        else:
            conf_display = f"{confidence_pct:.2f}%"
        st.markdown(f"**🔹 Confidence:** {conf_display}")

        # show a progress bar [0..100]
        prog_val = max(0, min(100, int(confidence_pct)))
        if prog_val == 0 and confidence_pct > 0:
            prog_val = 1
        st.progress(prog_val)

        st.markdown(
            f"🧠 **Sentiment:** "
            f"Positive: {sentiment['positive']*100:.1f}%, "
            f"Neutral: {sentiment['neutral']*100:.1f}%, "
            f"Negative: {sentiment['negative']*100:.1f}%"
        )

        # =========================
        # Chart: actual close series; annotate beginning, current, and predicted
        try:
            import plotly.graph_objects as go
            df_chart = data.copy()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Close'], mode='lines', name='Actual Close'))
            # Beginning marker
            fig.add_trace(go.Scatter(x=[df_chart.index[0]], y=[df_chart['Close'].iloc[0]], mode='markers', name='Beginning', marker=dict(color='blue', size=8)))
            # Current marker
            fig.add_trace(go.Scatter(x=[df_chart.index[-1]], y=[df_chart['Close'].iloc[-1]], mode='markers', name='Current', marker=dict(color='orange', size=10)))
            # Predicted price marker at last timestamp (indicates expected target)
            if predicted_price is not None:
                fig.add_trace(go.Scatter(x=[df_chart.index[-1]], y=[predicted_price], mode='markers', name='Predicted', marker=dict(color='green', size=10)))
                # add a dashed horizontal line for predicted price
                fig.add_hline(y=predicted_price, line=dict(dash='dash', color='green'), annotation_text='Predicted Price')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Chart not available: {e}")
    except Exception as e:
        st.error(f"Error fetching predictions: {e}")

    st.markdown("---")

    # --- Investment Advice
    st.subheader("💡 Investment Advice")
    try:
        advice_text = get_investment_advice(stock_symbol, horizon)
        st.info(advice_text)
    except Exception as e:
        st.error(f"Error generating investment advice: {e}")

    st.markdown("---")
    st.caption("📊 Tip: Intraday = quick trades (<4h), Swing = few days, Long-term = steady growth.")


# =========================
# 💼 PAGE 2: PORTFOLIO SUGGESTIONS
# =========================
elif page == "Portfolio Suggestions":
    st.title("💼 Portfolio Suggestions & Allocation")

    total_amount = st.number_input("💰 Total Investment Amount (₹)", min_value=1000, step=500)
    horizon = st.radio("Select Investment Horizon", ["Intraday", "Swing", "Long-Term"])

    # Allocation controls
    allocation_mode = st.selectbox("Allocation Mode", ["Proportional", "Equal", "Risk-adjusted"], index=0)
    allocation_mode_key = allocation_mode.lower().replace('-', '_')
    max_weight_pct = st.number_input("Max allocation per stock (%) (0 for no cap)", min_value=0.0, value=0.0, step=0.5)
    show_top_n = st.number_input("Show top N stocks (0 for all)", min_value=0, value=0, step=1)

    # Duration display
    if horizon == "Intraday":
        st.markdown("⏳ **Investment Duration:** Up to 4 Hours")
    elif horizon == "Swing":
        st.markdown("⏳ **Investment Duration:** 2–10 Days")
    else:
        st.markdown("⏳ **Investment Duration:** 1–6 Months")

    if st.button("📊 Generate Portfolio"):
        try:
            portfolio = get_portfolio_allocation(total_amount, horizon, allocation_mode=allocation_mode_key, top_n=(None if show_top_n==0 else int(show_top_n)), max_weight_pct=(None if max_weight_pct==0 else float(max_weight_pct)))
            if portfolio:
                df = pd.DataFrame(portfolio)
                # sort by weight desc
                df = df.sort_values(by="Weight (%)", ascending=False).reset_index(drop=True)
                # show the full allocation table; `width='stretch'` replaces deprecated `use_container_width`
                try:
                    st.dataframe(df, width='stretch')
                except Exception:
                    # fallback
                    st.dataframe(df)

                total_profit = df["Expected Profit (₹)"].sum()
                total_return_pct = (total_profit / total_amount) * 100

                st.markdown(f"**💹 Total Expected Profit:** ₹{total_profit:,.2f}")
                st.markdown(f"**📈 Expected Portfolio Return:** {total_return_pct:.2f}%")
                st.caption("Allocation strategy: weights are proportional to each stock's expected return for the selected horizon.")

                # Inform user if requested cap is smaller than allowable minimum (auto-adjusted)
                n_total = len(get_nse_stock_list())
                if max_weight_pct and max_weight_pct > 0 and max_weight_pct < float(100 / max(1, n_total)):
                    st.warning(f"The requested max cap ({max_weight_pct}%) is too small for {n_total} stocks; it was automatically adjusted to {round(100/n_total, 2)}%.")
            else:
                st.warning("No portfolio recommendations available.")
        except Exception as e:
            st.error(f"Error generating portfolio: {e}")

    st.info("📈 Tip: Intraday = high liquidity stocks, Long-term = blue-chip stable stocks.")

# =========================
# 🧾 FOOTER
# =========================
st.markdown("---")
st.caption("🚀 Smart Trading Assistant | Digitrader")
