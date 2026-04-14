"""
Trading Bot Dashboard Component for Streamlit

Displays real-time trading bot statistics, positions, and performance metrics.
Integrates with the FastAPI backend to fetch live data.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Configuration
API_BASE_URL = "http://localhost:8000"


def get_bot_status():
    """Fetch bot current status"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/bot/status", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"❌ Could not fetch bot status: {e}")
    return None


def get_open_positions():
    """Fetch open positions"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/bot/positions", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"❌ Could not fetch positions: {e}")
    return None


def get_account_stats():
    """Fetch account statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/bot/account/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"❌ Could not fetch stats: {e}")
    return None


def get_trade_history(limit=20):
    """Fetch trade history"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/bot/trades",
            params={"limit": limit, "status": "ALL"},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"❌ Could not fetch trade history: {e}")
    return None


def display_bot_header():
    """Display bot header with status indicator"""
    status = get_bot_status()
    
    if not status:
        st.error("⚠️ Bot API not available")
        return False
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🤖 Bot Status", status.get("status", "UNKNOWN"))
    
    with col2:
        uptime = status.get("uptime", 0)
        minutes = uptime // 60
        seconds = uptime % 60
        st.metric("⏱️ Uptime", f"{minutes}m {seconds}s")
    
    with col3:
        st.metric("📊 Signals", status.get("signals_received", 0))
    
    with col4:
        st.metric("💹 Trades Placed", status.get("trades_placed", 0))
    
    return True


def display_account_metrics():
    """Display key account metrics"""
    stats = get_account_stats()
    
    if not stats:
        return
    
    st.markdown("### 💰 Account Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        initial_cap = stats.get("initial_capital", 0)
        st.metric(
            "Initial Capital",
            f"₹{initial_cap:,.0f}",
            help="Starting capital"
        )
    
    with col2:
        current_cap = stats.get("current_capital", 0)
        pnl = current_cap - stats.get("initial_capital", 0)
        pnl_pct = (pnl / stats.get("initial_capital", 1)) * 100 if stats.get("initial_capital") else 0
        
        st.metric(
            "Current Capital",
            f"₹{current_cap:,.0f}",
            delta=f"₹{pnl:,.0f} ({pnl_pct:+.2f}%)",
            delta_color="normal" if pnl >= 0 else "inverse"
        )
    
    with col3:
        daily_pnl = stats.get("daily_pnl", 0)
        st.metric(
            "Daily P&L",
            f"₹{daily_pnl:,.0f}",
            delta_color="normal" if daily_pnl >= 0 else "inverse"
        )
    
    with col4:
        deployed = stats.get("total_deployed", 0)
        available = current_cap - deployed
        st.metric(
            "Capital Available",
            f"₹{available:,.0f}",
            help=f"Deployed: ₹{deployed:,.0f}"
        )


def display_positions_table():
    """Display open positions in interactive table"""
    positions = get_open_positions()
    
    if not positions or not positions.get("positions"):
        st.info("📭 No open positions")
        return
    
    st.markdown("### 📍 Open Positions")
    
    pos_list = positions.get("positions", [])
    
    # Create DataFrame
    df_data = []
    for pos in pos_list:
        df_data.append({
            "Stock": pos.get("stock", "N/A"),
            "Entry Price": f"₹{pos.get('entry_price', 0):.2f}",
            "Current Price": f"₹{pos.get('current_price', 0):.2f}",
            "Target": f"₹{pos.get('target_price', 0):.2f}",
            "SL": f"₹{pos.get('stop_loss', 0):.2f}",
            "Qty": pos.get("quantity", 0),
            "P&L": f"₹{pos.get('unrealized_pnl', 0):,.0f}",
            "P&L %": f"{pos.get('unrealized_pnl_pct', 0):+.2f}%",
            "Time": pos.get("elapsed_time", "0m")
        })
    
    df = pd.DataFrame(df_data)
    
    # Color code P&L column
    def highlight_pnl(val):
        if "+" in val:
            return 'color: green'
        elif "-" in val:
            return 'color: red'
        return 'color: black'
    
    st.dataframe(
        df.style.applymap(highlight_pnl, subset=["P&L %"]),
        use_container_width=True
    )
    
    # Position summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Open", len(pos_list))
    
    with col2:
        capital_deployed = sum(p.get("entry_value", 0) for p in pos_list)
        st.metric("Capital Deployed", f"₹{capital_deployed:,.0f}")
    
    with col3:
        total_pnl = sum(p.get("unrealized_pnl", 0) for p in pos_list)
        st.metric(
            "Total Unrealized P&L",
            f"₹{total_pnl:,.0f}",
            delta_color="normal" if total_pnl >= 0 else "inverse"
        )


def display_performance_metrics():
    """Display trading performance metrics"""
    stats = get_account_stats()
    
    if not stats:
        return
    
    st.markdown("### 📈 Performance Metrics")
    
    trades = stats.get("trades", {})
    perf = stats.get("performance", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_trades = trades.get("total", 0)
        st.metric("Total Trades", total_trades)
    
    with col2:
        win_rate = perf.get("win_rate", 0)
        st.metric(
            "Win Rate",
            f"{win_rate:.1f}%",
            delta_color="normal" if win_rate >= 50 else "inverse"
        )
    
    with col3:
        avg_win = perf.get("avg_win", 0)
        avg_loss = abs(perf.get("avg_loss", 0))
        ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        st.metric(
            "Win/Loss Ratio",
            f"1:{ratio:.2f}" if ratio > 0 else "N/A"
        )
    
    with col4:
        largest_win = perf.get("largest_win", 0)
        st.metric("Largest Win", f"₹{largest_win:,.0f}")


def display_trade_history():
    """Display recent trade history"""
    history = get_trade_history(20)
    
    if not history or not history.get("trades"):
        st.info("📭 No trades yet")
        return
    
    st.markdown("### 📋 Recent Trades")
    
    trades = history.get("trades", [])
    
    # Create DataFrame
    df_data = []
    for trade in trades:
        status = trade.get("status", "UNKNOWN")
        pnl = trade.get("pnl", "-")
        
        df_data.append({
            "Stock": trade.get("stock", "N/A"),
            "Entry": f"₹{trade.get('entry_price', 0):.2f}",
            "Exit": f"₹{trade.get('exit_price', '-'):.2f}" if trade.get("exit_price") else "-",
            "Signal": trade.get("signal", "N/A"),
            "Confidence": f"{trade.get('confidence', 0)*100:.0f}%",
            "Status": "✅ Closed" if status == "CLOSED" else "⏳ Open",
            "P&L": f"₹{pnl:,.0f}" if pnl != "-" else "-",
            "Reason": trade.get("exit_reason", "-")
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)


def display_performance_charts():
    """Display performance visualization charts"""
    stats = get_account_stats()
    
    if not stats or not stats.get("trades", {}).get("total", 0):
        st.info("📊 Insufficient data for charts")
        return
    
    st.markdown("### 📊 Performance Charts")
    
    trades = stats.get("trades", {})
    perf = stats.get("performance", {})
    
    col1, col2 = st.columns(2)
    
    # Chart 1: Win vs Loss
    with col1:
        winning = trades.get("winning", 0)
        losing = trades.get("losing", 0)
        
        if winning > 0 or losing > 0:
            fig = go.Figure(data=[
                go.Pie(
                    labels=["Wins", "Losses"],
                    values=[winning, losing],
                    marker=dict(colors=["green", "red"]),
                    textposition="inside",
                    textinfo="label+percent"
                )
            ])
            fig.update_layout(title="Win vs Loss Trades", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Chart 2: Risk Profile
    with col2:
        avg_win = abs(perf.get("avg_win", 0))
        avg_loss = abs(perf.get("avg_loss", 0))
        
        if avg_win > 0 or avg_loss > 0:
            fig = go.Figure(data=[
                go.Bar(
                    x=["Average Win", "Average Loss"],
                    y=[avg_win, avg_loss],
                    marker=dict(color=["green", "red"]),
                    text=[f"₹{avg_win:.0f}", f"₹{avg_loss:.0f}"],
                    textposition="outside"
                )
            ])
            fig.update_layout(title="Average Win vs Loss", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


def display_risk_limits():
    """Display risk management limits"""
    stats = get_account_stats()
    
    if not stats:
        return
    
    st.markdown("### 🛡️ Risk Management")
    
    limits = stats.get("limits", {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        daily_remaining = limits.get("daily_loss_remaining", 0)
        daily_limit = limits.get("daily_loss_limit", 0)
        used_pct = 100 - (daily_remaining / daily_limit * 100) if daily_limit else 0
        
        st.metric(
            "Daily Loss Limit",
            f"₹{daily_remaining:,.0f}",
            delta=f"{used_pct:.1f}% used",
            delta_color="inverse"
        )
    
    with col2:
        max_pos = limits.get("max_positions", 0)
        pos_remaining = limits.get("positions_remaining", 0)
        
        st.metric(
            "Position Slots",
            f"{pos_remaining}/{max_pos}",
            help="Open position limit"
        )
    
    with col3:
        max_risk = limits.get("max_risk_per_trade", 0)
        st.metric(
            "Max Risk/Trade",
            f"₹{max_risk:,.0f}",
            help="8% of capital"
        )


def run_trading_bot_dashboard():
    """Main dashboard function"""
    st.set_page_config(
        page_title="Trading Bot Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("# 🤖 Trading Bot Dashboard")
    st.markdown("Real-time monitoring and analytics for automated trading")
    
    # Refresh button
    col1, col2 = st.columns([0.8, 0.2])
    with col2:
        if st.button("🔄 Refresh", key="refresh_bot_dashboard"):
            st.rerun()
    
    # Auto-refresh toggle
    with col1:
        auto_refresh = st.checkbox("Auto-refresh every 30s", value=False)
    
    # Main content
    try:
        if not display_bot_header():
            st.stop()
        
        st.divider()
        
        # Display sections
        display_account_metrics()
        st.divider()
        
        display_positions_table()
        st.divider()
        
        display_performance_metrics()
        st.divider()
        
        display_performance_charts()
        st.divider()
        
        display_risk_limits()
        st.divider()
        
        display_trade_history()
        
        # Auto-refresh logic
        if auto_refresh:
            st.info("⏳ Auto-refreshing in 30 seconds...")
            time.sleep(30)
            st.rerun()
        
    except Exception as e:
        st.error(f"❌ Dashboard error: {e}")
        st.info("Make sure the API server is running on port 8000")


if __name__ == "__main__":
    run_trading_bot_dashboard()
