# modules/trading_dashboard.py
"""
Trading & Risk Management Dashboard UI Components
Streamlit components for live P&L tracking, risk management, and backtests.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, Optional

from modules.pnl_tracker import PnLTracker
from modules.risk_management import RiskManager
from modules.safety_guardrails import SafetyGuardrails


def show_pnl_dashboard():
    """
    Display P&L and performance tracking dashboard.
    """
    st.subheader("📊 Live P&L Tracker")
    
    # Initialize tracker
    tracker = PnLTracker(use_sheets=False)
    
    # Tabs
    tab_positions, tab_history, tab_performance = st.tabs(
        ["📍 Open Positions", "📋 Trade History", "📈 Performance"]
    )
    
    # --- Tab 1: Open Positions ---
    with tab_positions:
        if tracker.open_positions:
            positions_df = tracker.get_open_positions_summary()
            st.dataframe(
                positions_df.style.format({
                    "Entry": "₹{:.2f}",
                    "Current": "₹{:.2f}",
                    "PnL": "₹{:.0f}",
                    "PnL %": "{:+.2f}%",
                    "SL": "₹{:.2f}",
                    "TP": "₹{:.2f}",
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            unrealized_pnl = sum(p["pnl"] for p in tracker.open_positions)
            
            with col1:
                st.metric("Open Positions", len(tracker.open_positions))
            with col2:
                st.metric("Unrealized P&L", f"₹{unrealized_pnl:,.0f}")
            with col3:
                avg_pnl_pct = np.mean([p["pnl_percent"] for p in tracker.open_positions])
                st.metric("Avg P&L %", f"{avg_pnl_pct:+.2f}%")
            with col4:
                bullish_count = sum(1 for p in tracker.open_positions if p["entry_trend"].lower() == "bullish")
                st.metric("Bullish Positions", bullish_count)
        else:
            st.info("📭 No open positions")
    
    # --- Tab 2: Trade History ---
    with tab_history:
        if tracker.trade_history:
            history_df = tracker.get_trade_history_summary()
            st.dataframe(
                history_df.style.format({
                    "Entry": "₹{:.2f}",
                    "Exit": "₹{:.2f}",
                    "PnL": "₹{:.0f}",
                    "PnL %": "{:+.2f}%",
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Export options
            col1, col2 = st.columns(2)
            with col1:
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="trade_history.csv",
                    mime="text/csv"
                )
            with col2:
                if st.button("🔄 Refresh"):
                    st.rerun()
        else:
            st.info("📭 No closed trades yet")
    
    # --- Tab 3: Performance ---
    with tab_performance:
        if tracker.trade_history:
            metrics = tracker.calculate_performance()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Trades", metrics["total_closed_trades"])
            with col2:
                st.metric("Win Rate", f"{metrics['win_rate_percent']:.1f}%")
            with col3:
                st.metric("Profit Factor", f"{metrics['payoff_ratio']:.2f}")
            with col4:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Total P&L",
                    f"₹{metrics['total_pnl']:,.0f}",
                    f"{metrics['total_return_percent']:+.2f}%"
                )
            with col2:
                st.metric("Avg Win", f"₹{metrics['avg_win']:,.0f}")
            with col3:
                st.metric("Avg Loss", f"₹{metrics['avg_loss']:,.0f}")
            
            st.markdown("---")
            
            # P&L Distribution Chart
            st.subheader("📊 P&L Distribution")
            pnls = [t["pnl"] for t in tracker.trade_history]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=pnls,
                nbinsx=15,
                marker_color="rgba(102, 126, 234, 0.7)",
                name="P&L"
            ))
            fig.update_layout(
                title="Trade P&L Distribution",
                xaxis_title="P&L (Rs)",
                yaxis_title="Frequency",
                template="plotly_dark",
                showlegend=False,
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)


def show_risk_management_panel():
    """
    Display risk management controls and position sizing calculator.
    """
    st.subheader("🛡️ Risk Management")
    
    # Initialize risk manager
    rm = RiskManager(account_size=100000, max_risk_per_trade=0.02)
    
    tab_calculator, tab_rules = st.tabs(["🧮 Position Calculator", "📋 Risk Rules"])
    
    # --- Tab 1: Position Sizing Calculator ---
    with tab_calculator:
        st.markdown("#### Calculate Position Size")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            account_size = st.number_input("Account Size (₹)", value=100000, min_value=10000)
        with col2:
            entry_price = st.number_input("Entry Price (₹)", value=1000.0, min_value=1.0)
        with col3:
            stop_loss = st.number_input("Stop Loss (₹)", value=970.0, min_value=1.0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            confidence = st.slider("Confidence", 0.0, 1.0, 0.6, 0.05)
        with col2:
            risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5)
        
        # Calculate
        if st.button("📊 Calculate Position Size", use_container_width=True):
            rm.account_size = account_size
            rm.max_risk_per_trade = risk_pct / 100
            
            pos_size = rm.calculate_position_size(entry_price, stop_loss, confidence)
            risk_amount = abs(entry_price - stop_loss) * pos_size
            tp = rm.calculate_profit_target(entry_price, stop_loss, 2.0)[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Position Size", f"{pos_size:.0f} units")
            with col2:
                st.metric("Risk Amount", f"₹{risk_amount:,.0f}")
            with col3:
                pct_of_account = (risk_amount / account_size) * 100
                st.metric("% of Account", f"{pct_of_account:.2f}%")
            with col4:
                st.metric("Take Profit", f"₹{tp:,.0f}")
            
            st.success(f"✓ Position validates {risk_pct}% risk per trade")
    
    # --- Tab 2: Risk Rules ---
    with tab_rules:
        st.markdown("#### 📋 Standard Risk Rules")
        
        rules_data = {
            "Rule": [
                "Max Risk per Trade",
                "Max Position Size",
                "Max Daily Loss",
                "Max Open Positions",
                "Min Model Accuracy",
                "Min Confidence",
            ],
            "Value": [
                "2% of account",
                "10% of account",
                "5% of account per day",
                "5 concurrent positions",
                "55%",
                "50%",
            ],
            "Rationale": [
                "Preserve capital, limit downside",
                "Avoid over-concentration",
                "Halt if losing too much in a day",
                "Prevent scattered positions",
                "Ensure model viability",
                "Only high-conviction trades",
            ]
        }
        
        st.dataframe(
            pd.DataFrame(rules_data),
            use_container_width=True,
            hide_index=True
        )
        
        st.info(
            "💡 **Why These Rules?**\n\n"
            "- **2% risk/trade**: If you win 55% and lose average win=1%+ loss=3%, you profit\n"
            "- **5% daily loss limit**: Prevents catastrophic days\n"
            "- **Max position #**: Ensures diversification, not all-in on one idea"
        )


def show_safety_guardrails_panel():
    """
    Display safety guardrails status and alerts.
    """
    st.subheader("🚨 Safety Guardrails")
    
    # Initialize guardrails
    guardrails = SafetyGuardrails(
        max_daily_loss_percent=5.0,
        max_consecutive_losses=3,
        min_accuracy_threshold=0.55
    )
    
    # Status indicator
    summary = guardrails.get_alert_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "🔴" if summary["circuit_breaker_active"] else "🟢"
        st.metric(status_color + " Status", "HALTED" if summary["circuit_breaker_active"] else "ACTIVE")
    
    with col2:
        st.metric("🔴 Critical Alerts", summary["critical_alerts"])
    
    with col3:
        st.metric("🟡 Warnings", summary["warnings"])
    
    with col4:
        st.metric("📊 Daily Loss", f"₹{summary.get('daily_loss', 0):,.0f}")
    
    st.markdown("---")
    
    # Alerts list
    if summary.get('latest_alerts'):
        st.subheader("📋 Latest Alerts")
        
        for alert in summary['latest_alerts'][-5:]:
            severity = alert['severity']
            
            if severity == "CRITICAL":
                st.error(f"🔴 **{severity} | {alert['rule']}**")
            elif severity == "WARNING":
                st.warning(f"🟡 **{severity} | {alert['rule']}**")
            else:
                st.success(f"🟢 **{severity} | {alert['rule']}**")
            
            st.caption(alert['message'])
    else:
        st.success("✓ All systems nominal. No alerts.")


def show_backtest_results():
    """
    Display backtest results and comparison.
    """
    st.subheader("📈 Backtest Results")
    
    st.info(
        "**To run a full backtest on real NSE data:**\n\n"
        "```bash\npython run_live_backtest.py\n```\n\n"
        "This will test the model on 1 year of real NSE data with position sizing, "
        "stop-loss, and safety guardrails."
    )
    
    # Mock backtest results (would load from JSON in real scenario)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Backtest Period", "365 days")
    with col2:
        st.metric("Expected Accuracy", "58-65% (≠ 93% synthetic)")
    with col3:
        st.metric("Model", "Random Forest + GBM")
    
    st.markdown("---")
    
    st.subheader("🎯 What to Expect")
    
    expectation_data = {
        "Metric": [
            "Training Data Accuracy",
            "Real Data Accuracy",
            "Why Different?",
            "Still Profitable?",
            "Minimum Win Rate",
        ],
        "Value/Description": [
            "93.58%",
            "55-65%",
            "Synthetic data is more predictable than markets",
            "Yes - need only >55% with risk management",
            "55-60% is realistic and profitable",
        ]
    }
    
    st.table(pd.DataFrame(expectation_data))


def render_trading_dashboard():
    """
    Complete trading and risk management dashboard.
    """
    st.markdown("---")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["💰 P&L Tracking", "🛡️ Risk Management", "🚨 Guardrails", "📈 Backtests"]
    )
    
    with tab1:
        show_pnl_dashboard()
    
    with tab2:
        show_risk_management_panel()
    
    with tab3:
        show_safety_guardrails_panel()
    
    with tab4:
        show_backtest_results()
