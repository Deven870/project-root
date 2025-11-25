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

        st.markdown(f"**🔹 Current Price:** ₹{current_price:.2f}")
        st.markdown(f"**🔹 Predicted Trend:** {trend}")
        st.markdown(f"**🔹 Confidence:** {confidence:.2f}%")

        st.progress(int(confidence))

        st.markdown(
            f"🧠 **Sentiment:** "
            f"Positive: {sentiment['positive']*100:.1f}%, "
            f"Neutral: {sentiment['neutral']*100:.1f}%, "
            f"Negative: {sentiment['negative']*100:.1f}%"
        )
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

    # Duration display
    if horizon == "Intraday":
        st.markdown("⏳ **Investment Duration:** Up to 4 Hours")
    elif horizon == "Swing":
        st.markdown("⏳ **Investment Duration:** 2–10 Days")
    else:
        st.markdown("⏳ **Investment Duration:** 1–6 Months")

    if st.button("📊 Generate Portfolio"):
        try:
            portfolio = get_portfolio_allocation(total_amount, horizon)
            if portfolio:
                df = pd.DataFrame(portfolio)
                st.dataframe(df, use_container_width=True)

                total_profit = df["Expected Profit (₹)"].sum()
                total_return_pct = (total_profit / total_amount) * 100

                st.markdown(f"**💹 Total Expected Profit:** ₹{total_profit:,.2f}")
                st.markdown(f"**📈 Expected Portfolio Return:** {total_return_pct:.2f}%")
            else:
                st.warning("No portfolio recommendations available.")
        except Exception as e:
            st.error(f"Error generating portfolio: {e}")

    st.info("📈 Tip: Intraday = high liquidity stocks, Long-term = blue-chip stable stocks.")

# =========================
# 🧾 FOOTER
# =========================
st.markdown("---")
st.caption("🚀 Smart Trading Assistant | Made with ❤️ using Streamlit")
