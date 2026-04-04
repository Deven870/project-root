"""
FAST ANALYTICS: Paper Trading Validation (Apr 4-6)
Target: 68% accuracy to go LIVE | Optimized for speed
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import json


@st.cache_data(ttl=60)
def load_real_trades(start_date="2026-04-04", end_date="2026-04-06"):
    """Load real trades from paper trading logs (CACHED)"""
    trades_list = []
    logs_dir = Path("paper_trading_logs")
    
    try:
        if logs_dir.exists():
            metrics_file = logs_dir / "metrics_tracker.json"
            if metrics_file.exists():
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                    if "daily" in metrics:
                        for date_str, day_data in metrics["daily"].items():
                            if start_date <= date_str <= end_date:
                                if "trades" in day_data:
                                    trades_list.extend(day_data["trades"])
    except Exception:
        pass
    
    return trades_list


@st.cache_data
def convert_trades_to_dataframe(real_trades):
    """Convert real trade data to DataFrame (CACHED)"""
    if not real_trades:
        return pd.DataFrame()
    
    trades_data = []
    for i, trade in enumerate(real_trades, 1):
        try:
            entry = float(trade.get('entry', 0))
            exit_price = float(trade.get('exit', 0))
            capital = float(trade.get('capital', 25000))
            
            if entry > 0:
                profit_loss = (exit_price - entry) / entry * 100 * capital / 100
                status = '✅ Win' if profit_loss > 0 else '❌ Loss'
            else:
                profit_loss = 0
                status = '⏳ Open'
            
            trades_data.append({
                'Trade #': i,
                'Stock': trade.get('stock', 'N/A'),
                'Entry': entry,
                'Exit': exit_price,
                'P&L (₹)': profit_loss,
                'Return (%)': (exit_price - entry) / entry * 100 if entry > 0 else 0,
                'Status': status,
                'Time': trade.get('time', datetime.now().isoformat())[:10]
            })
        except:
            pass
    
    return pd.DataFrame(trades_data) if trades_data else pd.DataFrame()


def render_advanced_analytics(stock_list):
    """FAST RENDER: Only essential metrics for April 4-6 validation"""
    
    st.markdown('<p class="main-title">📈 Paper Trading Analytics (Apr 4-6)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">🎯 Target: 68% accuracy | Live trading decision on Apr 17</p>', unsafe_allow_html=True)
    
    # Load real trades
    real_trades = load_real_trades("2026-04-04", "2026-04-06")
    trades_df = convert_trades_to_dataframe(real_trades)
    
    # If no trades, show empty state
    if trades_df.empty:
        st.warning("⏳ No trades executed yet (Apr 4-6)")
        st.info("Execute: `python execute_daily_trades.py`")
        return
    
    # CRITICAL METRICS AT TOP
    st.markdown("### 🎯 CUMULATIVE ACCURACY TRACKING")
    
    col1, col2, col3, col4 = st.columns(4)
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['P&L (₹)'] > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_pnl = trades_df['P&L (₹)'].sum()
    
    # Color coding
    if win_rate >= 68:
        accuracy_color = "✅ "
    elif win_rate >= 65:
        accuracy_color = "🟡 "
    else:
        accuracy_color = "❌ "
    
    with col1:
        st.metric(f"{accuracy_color}Accuracy", f"{win_rate:.1f}%", delta=f"{winning_trades}/{total_trades} wins")
    with col2:
        st.metric("Trades Count", total_trades, delta="Apr 4-6")
    with col3:
        st.metric("Total P&L", f"₹{total_pnl:,.0f}", delta=f"Avg: ₹{total_pnl/total_trades:.0f}" if total_trades > 0 else "")
    with col4:
        if win_rate >= 68:
            status = "✅ LIVE READY"
        elif win_rate >= 65:
            status = "🟡 ALMOST READY"
        else:
            status = "❌ NEEDS WORK"
        st.metric("Status", status, delta=f"Target: 68%")
    
    st.divider()
    
    # TWO KEY TABS ONLY
    tab1, tab2 = st.tabs(["📊 Trade Analysis", "📈 Win/Loss Analysis"])
    
    # === TAB 1: Trade Analysis ===
    with tab1:
        st.markdown("#### Real Trades Executed (Apr 4-6)")
        
        # Fast bar chart
        fig_trades = go.Figure()
        winners = trades_df[trades_df['P&L (₹)'] > 0]
        losers = trades_df[trades_df['P&L (₹)'] <= 0]
        
        if len(winners) > 0:
            fig_trades.add_trace(go.Bar(
                y=winners['P&L (₹)'], name='Wins',
                marker_color='#00CC88', text=winners['P&L (₹)'].round(0),
                textposition='auto'
            ))
        if len(losers) > 0:
            fig_trades.add_trace(go.Bar(
                y=losers['P&L (₹)'], name='Losses',
                marker_color='#FF4444', text=losers['P&L (₹)'].round(0),
                textposition='auto'
            ))
        
        fig_trades.update_layout(
            title='Trade P&L Analysis', template='plotly_dark',
            height=350, hovermode='x unified', showlegend=True
        )
        st.plotly_chart(fig_trades, use_container_width=True)
        
        # Trades table
        st.dataframe(trades_df[['Trade #', 'Stock', 'Entry', 'Exit', 'P&L (₹)', 'Status']], use_container_width=True)
    
    # === TAB 2: Win/Loss Analysis ===
    with tab2:
        st.markdown("#### Win/Loss Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Win Rate", f"{win_rate:.1f}%", f"{winning_trades}/{total_trades}")
        with col2:
            avg_win = trades_df[trades_df['P&L (₹)'] > 0]['P&L (₹)'].mean() if winning_trades > 0 else 0
            st.metric("Avg Win", f"₹{avg_win:.0f}")
        with col3:
            avg_loss = abs(trades_df[trades_df['P&L (₹)'] <= 0]['P&L (₹)'].mean()) if len(losers) > 0 else 0
            st.metric("Avg Loss", f"-₹{avg_loss:.0f}")
        
        # Pie chart
        fig_pie = go.Figure(data=[
            go.Pie(labels=['Wins', 'Losses'], values=[winning_trades, len(losers)],
                   marker=dict(colors=['#00CC88', '#FF4444']))
        ])
        fig_pie.update_layout(title='Win Distribution', template='plotly_dark', height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Performance summary
        st.markdown("#### Performance Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            profit_factor = (avg_win / max(avg_loss, 1)) if avg_loss > 0 else avg_win
            st.metric("Profit Factor", f"{profit_factor:.2f}x")
        
        with col2:
            best_trade = trades_df['P&L (₹)'].max()
            worst_trade = trades_df['P&L (₹)'].min()
            st.metric("Best/Worst Trade", f"₹{best_trade:,.0f} / ₹{worst_trade:,.0f}")
    
    # Footer with decision
    st.divider()
    if win_rate >= 68:
        st.success("🎉 **READY FOR LIVE DEPLOYMENT** - Accuracy threshold met! Decision: GO LIVE")
    elif win_rate >= 65:
        st.info(f"🟡 Close to target ({win_rate:.1f}%). Accumulate 2-3 more wins to reach 68%")
    else:
        st.warning(f"❌ Below target ({win_rate:.1f}%). Continue trading, improve strategy")
