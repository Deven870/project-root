"""
╔════════════════════════════════════════════════════════════════════════════╗
║          LIVE PREDICTIONS DASHBOARD - Real-Time Feed Component            ║
║                Integration module for Streamlit dashboard                  ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

API_BASE = "http://localhost:8000"


def display_live_feed_page():
    """Display the live predictions feed page"""
    
    st.markdown("## 🔴 LIVE PREDICTIONS FEED")
    st.markdown("*Real-time 6-layer analysis updated every 60 seconds during market hours*")
    
    # Service Status Section
    with st.expander("📊 Service Status & Details", expanded=True):
        status_col1, status_col2, status_col3, status_col4 = st.columns(4)
        
        try:
            response = requests.get(f"{API_BASE}/api/v1/live/status", timeout=5)
            status = response.json()
            
            with status_col1:
                service_status = "🟢 RUNNING" if status.get('status') == 'running' else "🔴 STOPPED"
                st.metric("Service", service_status)
            
            with status_col2:
                st.metric("Stocks Monitored", status.get('stocks_monitored', 0))
            
            with status_col3:
                market_status = "📈 OPEN" if status.get('is_market_open') else "📉 CLOSED"
                st.metric("Market Status", market_status)
            
            with status_col4:
                st.metric("Active Updates", status.get('total_updates', 0))
            
            # Additional details
            det_col1, det_col2, det_col3 = st.columns(3)
            with det_col1:
                st.text(f"Update Interval: {status.get('update_interval')}s")
            with det_col2:
                st.text(f"Subscribers: {status.get('active_subscribers')}")
            with det_col3:
                if status.get('last_update'):
                    last = status['last_update']
                    st.text(f"Last Update: {last['timestamp']}")
        
        except Exception as e:
            st.error(f"❌ Could not fetch service status: {e}")


def display_live_predictions():
    """Display live predictions cards"""
    
    st.markdown("### 📈 Real-Time Predictions")
    
    # Refresh button
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col2:
        if st.button("🔄 Refresh Now", use_container_width=True):
            with st.spinner("⏳ Fetching latest predictions..."):
                try:
                    response = requests.post(f"{API_BASE}/api/v1/live/refresh", timeout=30)
                    result = response.json()
                    if result.get('status') == 'success':
                        st.success(f"✅ Updated {result.get('predictions_updated', 0)} predictions")
                    else:
                        st.warning("⚠️  No new predictions available")
                except Exception as e:
                    st.error(f"❌ Refresh failed: {e}")
            st.rerun()
    
    with col3:
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.show_settings = not st.session_state.get('show_settings', False)
    
    st.markdown("---")
    
    # Fetch current predictions
    try:
        response = requests.get(f"{API_BASE}/api/v1/live/predictions", timeout=10)
        data = response.json()
        predictions = data.get('data', {})
        count = data.get('count', 0)
        
        if count == 0:
            st.info("⏳ No predictions available yet. Waiting for first batch during market hours...")
            return
        
        st.success(f"✅ Displaying {count} Real-Time Predictions")
        
        # Filter options
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            signal_filter = st.multiselect(
                "Filter by Signal",
                ["STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL"],
                default=["STRONG BUY", "BUY"],
                key="signal_filter"
            )
        
        with filter_col2:
            min_confidence = st.slider(
                "Min Confidence (%)",
                0, 100, 50,
                key="confidence_filter"
            )
        
        with filter_col3:
            sort_by = st.selectbox(
                "Sort By",
                ["Confidence", "Target Upside", "Stock Name"],
                key="sort_filter"
            )
        
        st.markdown("---")
        
        # Filter predictions
        filtered = []
        for stock, pred in predictions.items():
            signal = pred.get('signal', 'NEUTRAL')
            confidence = pred.get('confidence', 0)
            
            if signal in signal_filter and confidence >= min_confidence:
                filtered.append((stock, pred))
        
        # Sort
        if sort_by == "Confidence":
            filtered.sort(key=lambda x: x[1].get('confidence', 0), reverse=True)
        elif sort_by == "Target Upside":
            filtered.sort(
                key=lambda x: ((x[1].get('target_price', 0) - x[1].get('current_price', 1)) / 
                              max(x[1].get('current_price', 1), 1) * 100),
                reverse=True
            )
        else:
            filtered.sort(key=lambda x: x[0])
        
        if not filtered:
            st.warning(f"⚠️  No predictions match your filters")
            return
        
        st.markdown(f"**Showing {len(filtered)} of {count} predictions**")
        
        # Display as cards (3 columns)
        cols = st.columns(3)
        
        for idx, (stock, pred) in enumerate(filtered):
            col = cols[idx % 3]
            
            with col:
                # Card styling
                signal = pred.get('signal', 'NEUTRAL')
                confidence = pred.get('confidence', 0)
                current_price = pred.get('current_price', 0)
                target_price = pred.get('target_price', 0)
                stop_loss = pred.get('stop_loss', 0)
                
                # Calculate upside/downside
                upside = ((target_price - current_price) / max(current_price, 1)) * 100
                downside = -((current_price - stop_loss) / max(current_price, 1)) * 100
                
                # Signal color
                if signal in ["STRONG BUY", "BUY"]:
                    signal_color = "green"
                    signal_emoji = "🟢"
                elif signal in ["SELL", "STRONG SELL"]:
                    signal_color = "red"
                    signal_emoji = "🔴"
                else:
                    signal_color = "gray"
                    signal_emoji = "⚪"
                
                with st.container(border=True):
                    # Header
                    st.markdown(f"## {signal_emoji} {stock}")
                    
                    # Signal badge
                    st.markdown(
                        f'<span style="background-color: {signal_color}; color: white; '
                        f'padding: 4px 12px; border-radius: 20px; font-weight: bold;">'
                        f'{signal}</span>',
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("---")
                    
                    # Price levels
                    st.metric("Current Price", f"₹{current_price:.2f}")
                    st.metric("Target Price", f"₹{target_price:.2f}", delta=f"+{upside:.2f}%")
                    st.metric("Stop Loss", f"₹{stop_loss:.2f}", delta=f"{downside:.2f}%")
                    
                    st.markdown("---")
                    
                    # Scores
                    st.markdown("**Scores:**")
                    score_col1, score_col2, score_col3 = st.columns(3)
                    with score_col1:
                        st.metric("Technical", f"{pred.get('technical_score', 0):.1f}")
                    with score_col2:
                        st.metric("Fundamental", f"{pred.get('fundamental_score', 0):.1f}")
                    with score_col3:
                        st.metric("Sentiment", f"{pred.get('sentiment_score', 0):.1f}")
                    
                    # Confidence
                    st.markdown("---")
                    st.markdown(f"**Confidence:** {confidence:.0f}%")
                    
                    # Progress bar
                    st.progress(confidence / 100)
                    
                    # Timestamp
                    st.caption(f"Last Updated: {pred.get('timestamp', '').split('T')[-1][:5]}")
                    
                    # Action buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📊 Chart", key=f"chart_{stock}", use_container_width=True):
                            st.session_state.selected_stock = stock
                            st.info(f"📈 Chart for {stock} (would integrate with TradingView)")
                    
                    with col2:
                        if st.button("⭐ Alert", key=f"alert_{stock}", use_container_width=True):
                            st.success(f"🔔 Alert set for {stock}")
    
    except Exception as e:
        st.error(f"❌ Failed to fetch live predictions: {e}")
        st.info("💡 Tip: Make sure the NSEIQ API server is running on http://localhost:8000")


def display_prediction_history():
    """Display historical predictions"""
    
    st.markdown("### 📊 Prediction History & Accuracy")
    
    try:
        response = requests.get(f"{API_BASE}/api/v1/live/status", timeout=5)
        status = response.json()
        
        history = status.get('last_update', {})
        total_updates = status.get('total_updates', 0)
        
        if total_updates == 0:
            st.info("📋 No historical data yet - updates start when service begins sending predictions")
            return
        
        # Display as metrics
        hist_col1, hist_col2, hist_col3 = st.columns(3)
        
        with hist_col1:
            st.metric("Total Updates", total_updates)
        
        with hist_col2:
            st.metric("Stocks per Update", history.get('count', 0))
        
        with hist_col3:
            st.metric("Last Update", history.get('timestamp', 'N/A')[-8:])
        
        st.markdown("---")
        
        # Sample update details
        if history:
            st.markdown("**Latest Update Details:**")
            st.json(history)
    
    except Exception as e:
        st.warning(f"⚠️  Could not fetch history: {e}")


def display_websocket_info():
    """Display WebSocket connection information"""
    
    st.markdown("### 🔌 WebSocket Connection Details")
    
    with st.expander("WebSocket Integration Guide", expanded=False):
        st.code("""
# JavaScript Client Example
const ws = new WebSocket('ws://localhost:8000/ws/predictions');

ws.onopen = () => {
    console.log('Connected to live predictions');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if(data.type === 'predictions_update') {
        console.log('Updated predictions:', data.data);
        // Update UI with new predictions
    }
    else if(data.type === 'stock_update') {
        console.log(`Updated ${data.ticker}:`, data.data);
    }
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};
        """, language="javascript")
        
        st.markdown("**Endpoints:**")
        st.code("""
# Get all current predictions (HTTP)
GET /api/v1/live/predictions

# Get specific stock prediction
GET /api/v1/live/predictions?stock=RELIANCE

# Get service status
GET /api/v1/live/status

# Manual refresh
POST /api/v1/live/refresh

# WebSocket - All predictions
WS /ws/predictions

# WebSocket - Single stock
WS /ws/stock/{symbol}
        """, language="text")


def run_live_feed_dashboard():
    """Main live feed dashboard page"""
    
    # Initialize session state
    if 'show_settings' not in st.session_state:
        st.session_state.show_settings = False
    
    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = None
    
    # Auto-refresh
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### ⏱️ Auto-Refresh")
        refresh_interval = st.selectbox(
            "Refresh Interval",
            [30, 60, 120, 300],
            format_func=lambda x: f"{x} seconds",
            key="refresh_interval"
        )
    
    with col2:
        st.markdown("### 🎯 Mode")
        mode = st.selectbox(
            "Display Mode",
            ["Cards (3-col)", "Table", "Detailed"],
            key="display_mode"
        )
    
    st.markdown("---")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Feed",
        "📈 History",
        "🔌 WebSocket",
        "ℹ️ About"
    ])
    
    with tab1:
        display_live_feed_page()
        display_live_predictions()
    
    with tab2:
        display_prediction_history()
    
    with tab3:
        display_websocket_info()
    
    with tab4:
        st.markdown("""
        ## 🎯 Live Predictions Feed
        
        This real-time predictions dashboard connects to the **NSEIQ v5.0 API Server**
        and displays continuously updated stock predictions.
        
        ### How It Works
        
        1. **Live Service** runs on the backend, analyzing 15+ NSE stocks every 60 seconds
        2. **6-Layer Analysis** combines:
           - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
           - Fundamental metrics (P/E, Debt/Equity, ROE)
           - Sentiment scores (News, Social, Analyst ratings)
           - Macro factors (NIFTY trend, VIX, FII flows)
           - Options data (PCR, Max Pain)
           - Insider activity
        
        3. **Real-Time Updates** broadcast via WebSocket to all connected clients
        4. **Confidence Scoring** ensures only high-quality predictions displayed
        
        ### Key Features
        
        ✅ Auto-updating every 60 seconds during market hours  
        ✅ Filter by signal strength and confidence  
        ✅ View detailed scores for each prediction  
        ✅ WebSocket integration for live updates  
        ✅ Manual refresh anytime  
        ✅ Google Sheets logging (automatic)
        
        ### Getting Started
        
        1. Ensure API server is running: `python -m uvicorn backend.app.main:app --port 8000`
        2. Select refresh interval above
        3. Filter predictions by signal and confidence
        4. Click on any prediction to view details
        
        ⚠️ **Note:** Predictions only update during market hours (9:15 AM - 3:30 PM IST)
        """)
