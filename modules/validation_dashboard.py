"""
STREAMLIT DASHBOARD: Paper Trading Validation Monitor
Real-time 30-day validation tracking with go-live readiness indicator
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

from modules.paper_trading_validator import get_validator, TARGETS


def render_validation_dashboard():
    """Main validation dashboard for Streamlit."""
    
    st.markdown("## 📊 30-Day Paper Trading Validation")
    
    validator = get_validator()
    report = validator.generate_validation_report()
    
    if not report["metrics"]:
        st.info("⏳ No trading data yet. Come back after first trades are executed!")
        return
    
    metrics = report["metrics"]
    validation = report["validation"]
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        win_pct = metrics["win_rate_pct"]
        target = TARGETS["win_rate_pct"]
        color = "🟢" if win_pct >= target else "🟡" if win_pct >= target*0.8 else "🔴"
        st.metric(
            f"{color} Win Rate",
            f"{win_pct:.1f}%",
            f"Target: {target}%"
        )
    
    with col2:
        sharpe = metrics["sharpe_ratio"]
        target = TARGETS["sharpe_ratio"]
        color = "🟢" if sharpe >= target else "🟡" if sharpe >= target*0.8 else "🔴"
        st.metric(
            f"{color} Sharpe Ratio",
            f"{sharpe:.2f}",
            f"Target: {target}"
        )
    
    with col3:
        dd = abs(metrics["max_drawdown_pct"])
        target = TARGETS["max_drawdown_pct"]
        color = "🟢" if dd <= target else "🟡" if dd <= target*1.2 else "🔴"
        st.metric(
            f"{color} Max Drawdown",
            f"{dd:.1f}%",
            f"Target: ≤{target}%"
        )
    
    with col4:
        days = metrics["days_active"]
        target = TARGETS["days_active"]
        pct = (days / target) * 100
        st.metric(
            "📅 Duration",
            f"{days}/{target} days",
            f"{pct:.0f}% complete"
        )
    
    # Status indicator
    st.markdown("---")
    status = validation["status"]
    passed = validation["passed"]
    total = validation["total"]
    
    if status == "READY_FOR_LIVE":
        st.success(f"✅ GO-LIVE APPROVED! ({passed}/{total} targets met)")
        st.balloons()
    else:
        st.warning(f"⏳ Continue testing ({passed}/{total} targets met)")
    
    # Progress by metric
    st.markdown("### Validation Progress")
    
    checks = validation["checks"]
    progress_data = []
    
    for check_name, passed_check in checks.items():
        if check_name == "win_rate":
            current = metrics["win_rate_pct"]
            target = TARGETS["win_rate_pct"]
            unit = "%"
        elif check_name == "sharpe":
            current = metrics["sharpe_ratio"]
            target = TARGETS["sharpe_ratio"]
            unit = ""
        elif check_name == "drawdown":
            current = abs(metrics["max_drawdown_pct"])
            target = TARGETS["max_drawdown_pct"]
            unit = "%"
        elif check_name == "profit_factor":
            current = metrics["profit_factor"]
            target = TARGETS["profit_factor"]
            unit = ""
        else:
            current = metrics["days_active"]
            target = TARGETS["days_active"]
            unit = "d"
        
        status_icon = "✓" if passed_check else "✗"
        progress_pct = min((current / target) * 100, 100) if target > 0 else 0
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.write(f"{status_icon} {check_name.replace('_', ' ').title()}")
        with col2:
            st.progress(progress_pct / 100)
        with col3:
            st.write(f"{current:.1f}{unit} / {target}{unit}")
    
    # Performance details
    st.markdown("### Performance Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Trades", metrics["total_trades"])
        st.metric("Winning Trades", metrics["winning_trades"])
        st.metric("Losing Trades", metrics["losing_trades"])
    
    with col2:
        st.metric("Average Win", f"₹{metrics['avg_win']:,.2f}")
        st.metric("Average Loss", f"₹{metrics['avg_loss']:,.2f}")
        st.metric("Trades/Day", f"{metrics['trades_per_day']:.1f}")
    
    with col3:
        st.metric("Total P&L", f"₹{metrics['total_pnl']:,.2f}", delta=f"{metrics['total_pnl_pct']}%")
        st.metric("Final Equity", f"₹{metrics['final_equity']:,.2f}")
        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
    
    # Equity curve
    st.markdown("### Equity Curve")
    
    try:
        trades_df = pd.DataFrame(validator.trades_log)
        if len(trades_df) > 0:
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            trades_df = trades_df.sort_values('date')
            
            capital = float(validator.capital)
            trades_df['equity'] = capital + trades_df['pnl'].cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trades_df['date'],
                y=trades_df['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title="Account Equity Over Time",
                xaxis_title="Date",
                yaxis_title="Equity (₹)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Equity curve will appear once trades are logged")
    
    # Win/Loss distribution
    st.markdown("### Trade Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        win_data = {
            "Status": ["Wins", "Losses"],
            "Count": [metrics["winning_trades"], metrics["losing_trades"]]
        }
        fig = px.pie(
            win_data,
            values="Count",
            names="Status",
            title="Win vs Loss Distribution",
            color_discrete_map={"Wins": "#1f77b4", "Losses": "#ff7f0e"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # P&L by day
        try:
            daily_df = pd.DataFrame(validator.daily_metrics)
            if len(daily_df) > 0:
                daily_df['date'] = pd.to_datetime(daily_df['date'])
                daily_df['color'] = daily_df['pnl'].apply(lambda x: '#1f77b4' if x > 0 else '#ff7f0e')
                
                fig = px.bar(
                    daily_df,
                    x='date',
                    y='pnl',
                    title='Daily P&L',
                    labels={'date': 'Date', 'pnl': 'P&L (₹)'},
                    color='color',
                    color_discrete_map={'#1f77b4': '#1f77b4', '#ff7f0e': '#ff7f0e'}
                )
                fig.update_xaxes(title_text="")
                st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Daily P&L chart will appear once trading data is available")
    
    # Trade details table
    st.markdown("### Recent Trades")
    
    try:
        trades_display = pd.DataFrame(validator.trades_log).tail(20).copy()
        trades_display = trades_display[[
            'date', 'symbol', 'entry_price', 'exit_price', 'pnl', 'pnl_pct', 'exit_reason'
        ]].rename(columns={
            'date': 'Date',
            'symbol': 'Symbol',
            'entry_price': 'Entry',
            'exit_price': 'Exit',
            'pnl': 'P&L (₹)',
            'pnl_pct': 'Return %',
            'exit_reason': 'Exit Reason'
        })
        
        # Style P&L column
        def color_pnl(val):
            return 'background-color: #90EE90' if val > 0 else 'background-color: #FFB6C6'
        
        styled_df = trades_display.style.applymap(
            color_pnl,
            subset=['P&L (₹)']
        )
        
        st.dataframe(styled_df, use_container_width=True)
    except:
        st.info("Trade history will appear once trades are logged")
    
    # Download report
    if st.button("📥 Download Full Report (Excel)"):
        try:
            validator.export_summary()
            st.success("✓ Report exported to paper_trading_logs/")
        except Exception as e:
            st.error(f"Error exporting: {e}")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    render_validation_dashboard()
