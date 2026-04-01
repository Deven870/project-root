#!/usr/bin/env python3
"""
Streamlit Dashboard - Public Trading Signals Display
======================================================

Real-time dashboard showing:
- Today's trading signals (confidence, expected return)
- Historical performance (win rate, Sharpe ratio, drawdown)
- Subscription management (free tier vs premium)
- Live market prices

Run:
    streamlit run dashboard.py

Deploy:
    streamlit run dashboard.py -- --logger.level=warning --client.showErrorDetails=false
"""

import streamlit as st
import pandas as pd
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pytz
import requests

IST = pytz.timezone("Asia/Kolkata")

# Page config
st.set_page_config(
    page_title="VoiceBot Trading - Daily Signals",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar theme
st.sidebar.title("⚙️ Dashboard Settings")

# Load data functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_daily_signals():
    """Load today's signals from JSON."""
    try:
        signals_file = Path("logs/daily_signals.json")
        if signals_file.exists() and signals_file.is_file():
            with open(signals_file, 'r') as f:
                return json.load(f)
    except (PermissionError, json.JSONDecodeError, IOError) as e:
        st.error(f"Could not load signals: {e}")
    return None


@st.cache_data(ttl=300)
def load_validation_tracker():
    """Load validation metrics."""
    try:
        tracker_file = Path("logs/validation_tracker.json")
        if tracker_file.exists() and tracker_file.is_file():
            with open(tracker_file, 'r') as f:
                return json.load(f)
    except (PermissionError, json.JSONDecodeError, IOError) as e:
        st.warning(f"Could not load validation metrics: {e}")
    return None


@st.cache_data(ttl=300)
def load_paper_trading_history():
    """Load paper trading history."""
    try:
        history_file = Path("logs/paper_trading.json")
        if history_file.exists() and history_file.is_file():
            with open(history_file, 'r') as f:
                return json.load(f)
    except (PermissionError, json.JSONDecodeError, IOError) as e:
        st.warning(f"Could not load trading history: {e}")
    return None


# Main dashboard
def main():
    # Header
    st.markdown("# 📊 VoiceBot Trading - Live Signals Dashboard")
    st.markdown("**Real-time AI-powered trading signals for NSE stocks**")
    st.markdown("---")
    
    # Load data
    signals = load_daily_signals()
    metrics = load_validation_tracker()
    history = load_paper_trading_history()
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if metrics:
        with col1:
            st.metric(
                "Win Rate",
                f"{metrics.get('win_rate', 0):.1%}",
                "Target: 55%+",
                delta_color="normal"
            )
        
        with col2:
            total_return = metrics.get('total_return', 0)
            st.metric(
                "Total Return",
                f"{total_return:+.2%}",
                f"{metrics.get('total_trades', 0)} trades"
            )
        
        with col3:
            avg_return = metrics.get('avg_return_per_trade', 0)
            st.metric(
                "Avg Return/Trade",
                f"{avg_return:+.2%}",
                "Target: +0.5%+"
            )
        
        with col4:
            st.metric(
                "Active Trades",
                f"{metrics.get('total_trades', 0)}",
                "2-week validation"
            )
    else:
        st.info("📊 Metrics will appear after first week of trading")
    
    st.markdown("---")
    
    # Today's Signals Section
    st.markdown("## 🚀 Today's Trading Signals")
    
    if signals and signals.get('signals'):
        signal_list = signals.get('signals', [])
        
        # Summary
        summary = signals.get('summary', {})
        sig_col1, sig_col2, sig_col3, sig_col4 = st.columns(4)
        
        with sig_col1:
            st.info(f"🟢 BUY: {summary.get('buys', 0)}")
        with sig_col2:
            st.warning(f"🔴 SELL: {summary.get('sells', 0)}")
        with sig_col3:
            st.info(f"⚪ HOLD: {summary.get('holds', 0)}")
        with sig_col4:
            st.info(f"📊 Total: {summary.get('total', 0)}")
        
        st.markdown("")
        
        # Buy Signals
        buy_signals = [s for s in signal_list if s.get('prediction') == 1]
        if buy_signals:
            st.markdown("### 🟢 BUY SIGNALS")
            buy_df = pd.DataFrame([
                {
                    'Ticker': s.get('ticker', '').replace('.NS', ''),
                    'Confidence': f"{s.get('confidence', 0):.1%}",
                    'Expected Return': f"{s.get('expected_return', 0):+.2%}",
                    'Price': f"₹{s.get('last_close', 0):.2f}",
                    'Action': '📈 BUY'
                }
                for s in buy_signals
            ])
            st.dataframe(buy_df, use_container_width=True, hide_index=True)
        
        # Sell Signals
        sell_signals = [s for s in signal_list if s.get('prediction') == -1]
        if sell_signals:
            st.markdown("### 🔴 SELL SIGNALS")
            sell_df = pd.DataFrame([
                {
                    'Ticker': s.get('ticker', '').replace('.NS', ''),
                    'Confidence': f"{s.get('confidence', 0):.1%}",
                    'Expected Return': f"{s.get('expected_return', 0):+.2%}",
                    'Price': f"₹{s.get('last_close', 0):.2f}",
                    'Action': '📉 SELL'
                }
                for s in sell_signals
            ])
            st.dataframe(sell_df, use_container_width=True, hide_index=True)
    
    else:
        st.warning("📊 No signals generated today. Check back at 08:30 AM IST.")
    
    st.markdown("---")
    
    # Performance Section
    st.markdown("## 📈 Performance Metrics")
    
    if history and isinstance(history, dict) and 'trades' in history:
        trades = history.get('trades', [])
        
        if trades:
            trades_df = pd.DataFrame([
                {
                    'Date': t.get('date', ''),
                    'Ticker': t.get('ticker', ''),
                    'Signal': 'BUY' if t.get('prediction') == 1 else 'SELL' if t.get('prediction') == -1 else 'HOLD',
                    'Entry': f"₹{t.get('entry_price', 0):.2f}",
                    'Exit': f"₹{t.get('exit_price', 0):.2f}" if t.get('exit_price') else '-',
                    'Return': f"{t.get('return', 0):+.2%}" if t.get('return') else '-',
                    'Status': '✅ WIN' if t.get('outcome') == 'win' else '❌ LOSS' if t.get('outcome') == 'loss' else '⚪ OPEN'
                }
                for t in trades[-20:]  # Last 20 trades
            ])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Recent Trades")
                st.dataframe(trades_df, use_container_width=True, hide_index=True)
            
            with col2:
                # Win rate over time
                if len(trades) >= 5:
                    wins = sum(1 for t in trades if t.get('outcome') == 'win')
                    losses = sum(1 for t in trades if t.get('outcome') == 'loss')
                    total = wins + losses
                    
                    st.markdown("### Trade Statistics")
                    st.metric("Win Rate", f"{wins/total:.1%}" if total > 0 else "N/A")
                    st.metric("Winning Trades", wins)
                    st.metric("Losing Trades", losses)
                    st.metric("Total Closed", total)
    
    else:
        st.info("📊 Trade history will appear after first trades are logged.")
    
    st.markdown("---")
    
    # Subscription Section (Coming Soon)
    st.markdown("## 💳 Subscription Plan")
    st.info("🔮 **Payment system coming soon!** Premium tier with real-time signals and Telegram alerts will be available later.")
    
    st.markdown("---")
    
    # Info Section
    st.markdown("## ℹ️ About VoiceBot Trading")
    
    with st.expander("How It Works"):
        st.markdown("""
        1. **Daily Signal Generation** (08:30 AM IST)
           - 52-feature ML ensemble model
           - Technical indicators + sentiment + institutional flows
           - Validates with 55% win rate threshold
        
        2. **Signal Delivery** (09:00 AM IST)
           - Email to all users
           - Telegram alerts for premium subscribers
        
        3. **Performance Tracking**
           - Real-time win rate calculation
           - Historical performance dashboard
           - Monthly profit/loss reporting
        
        4. **Payment Processing**
           - Razorpay integration
           - 1st month ₹99 trial
           - Auto-renew at ₹299/month
        """)
    
    with st.expander("Risk Disclosure"):
        st.markdown("""
        ⚠️ **IMPORTANT DISCLAIMER**
        
        - These are AI-generated signals for educational purposes only
        - Past performance does not guarantee future results
        - Trading involves substantial risk of loss
        - Always do your own research before trading
        - Never invest money you cannot afford to lose
        - Consult a financial advisor before starting
        - This is NOT financial advice
        """)
    
    with st.expander("FAQ"):
        st.markdown("""
        **Q: How accurate are the signals?**
        A: Historical backtesting shows 55-60% win rate. Real validation in progress.
        
        **Q: Can I use this for live trading?**
        A: Currently paper trading only. Real trading coming after 2-week validation.
        
        **Q: When do I get signals?**
        A: Daily at 08:30 AM IST (weekdays only). Delivered via email/Telegram.
        
        **Q: What's the refund policy?**
        A: 7-day money-back guarantee. Contact support@voicebot.io
        
        **Q: Do you offer alerts?**
        A: Yes! Email (free) and Telegram (premium). SMS in coming soon.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 12px;">
        <p>VoiceBot Trading © 2026 | Signals generated at 08:30 AM IST daily</p>
        <p>Last updated: """ + datetime.now(IST).strftime("%Y-%m-%d %H:%M IST") + """</p>
        <p><a href="mailto:support@voicebot.io">Contact Support</a> | 
           <a href="#">Privacy Policy</a> | 
           <a href="#">Terms of Service</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
