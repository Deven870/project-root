"""
╔════════════════════════════════════════════════════════════════════════════╗
║         LIVE PREDICTIONS CLIENT - Streamlit Dashboard Integration          ║
║                    Real-Time WebSocket Consumer                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import websocket
import json
import threading
import logging
from datetime import datetime
from typing import Dict, List, Callable, Optional
import requests

logger = logging.getLogger(__name__)


class LivePredictionsClient:
    """Client for receiving real-time predictions via WebSocket"""
    
    def __init__(self, api_url: str = "http://localhost:8000", ws_url: str = "ws://localhost:8000"):
        self.api_url = api_url
        self.ws_url = ws_url
        self.ws = None
        self.is_connected = False
        self.predictions: Dict = {}
        self.callbacks: List[Callable] = []
        self.thread = None
        
    def add_callback(self, callback: Callable):
        """Add callback to be called on new predictions"""
        self.callbacks.append(callback)
    
    def connect_live_feed(self):
        """Establish WebSocket connection to live predictions"""
        try:
            ws_url = f"{self.ws_url.replace('http', 'ws')}/ws/predictions"
            logger.info(f"🔌 Connecting to live predictions at {ws_url}")
            
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            # Run in separate thread
            self.thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            self.thread.start()
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to live feed: {e}")
            self.is_connected = False
    
    def _on_open(self, ws):
        """Called when WebSocket opens"""
        self.is_connected = True
        logger.info("✅ Connected to live predictions feed")
    
    def _on_message(self, ws, message: str):
        """Called when message received"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "predictions_update":
                # Update predictions
                self.predictions = data.get("data", {})
                logger.debug(f"📊 Received {len(self.predictions)} predictions")
                
                # Call callbacks
                for callback in self.callbacks:
                    try:
                        callback(self.predictions)
                    except Exception as e:
                        logger.error(f"❌ Callback error: {e}")
            
            elif message_type == "stock_update":
                # Update single stock
                stock = data.get("ticker")
                pred = data.get("data", {})
                if stock:
                    self.predictions[stock] = pred
                    logger.debug(f"📌 Updated {stock}")
            
            elif message_type == "heartbeat":
                logger.debug("💓 Heartbeat received")
            
            elif message_type == "initial_predictions":
                self.predictions = data.get("data", {})
                stats = data.get("service_stats", {})
                logger.info(f"🚀 Initial load: {len(self.predictions)} stocks ready")
                
        except Exception as e:
            logger.error(f"❌ Message error: {e}")
    
    def _on_error(self, ws, error):
        """Called on WebSocket error"""
        logger.error(f"❌ WebSocket error: {error}")
        self.is_connected = False
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Called when WebSocket closes"""
        logger.info("📌 Disconnected from live feed")
        self.is_connected = False
    
    def disconnect(self):
        """Disconnect from WebSocket"""
        if self.ws:
            self.ws.close()
        self.is_connected = False
        logger.info("✅ Disconnected from live predictions")
    
    def get_predictions(self) -> Dict:
        """Get current predictions"""
        return self.predictions
    
    def get_stock_prediction(self, stock: str) -> Dict:
        """Get prediction for specific stock"""
        return self.predictions.get(stock.upper(), {})
    
    def get_status(self) -> Dict:
        """Get live service status via HTTP"""
        try:
            response = requests.get(f"{self.api_url}/api/v1/live/status")
            return response.json()
        except Exception as e:
            logger.error(f"❌ Status fetch error: {e}")
            return {}
    
    def refresh_now(self) -> Dict:
        """Manually trigger prediction refresh"""
        try:
            response = requests.post(f"{self.api_url}/api/v1/live/refresh")
            return response.json()
        except Exception as e:
            logger.error(f"❌ Refresh error: {e}")
            return {"status": "error", "message": str(e)}


def create_live_predictions_display(client: LivePredictionsClient):
    """Create Streamlit display for live predictions"""
    
    st.markdown("### 🔴 LIVE PREDICTIONS FEED")
    
    # Connection status
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        status = "🟢 CONNECTED" if client.is_connected else "🔴 DISCONNECTED"
        st.metric("Connection", status)
    
    with status_col2:
        st.metric("Predictions", len(client.get_predictions()))
    
    with status_col3:
        if st.button("🔄 Refresh Now"):
            result = client.refresh_now()
            st.success(f"Updated {result.get('predictions_updated', 0)} predictions")
    
    # Display live predictions
    predictions = client.get_predictions()
    
    if not predictions:
        st.warning("⏳ Waiting for predictions...")
        return
    
    # Create columns for stock cards
    cols = st.columns(3)
    
    for idx, (stock, pred) in enumerate(predictions.items()):
        col = cols[idx % 3]
        
        with col:
            with st.container(border=True):
                st.subheader(f"📈 {stock}")
                
                signal = pred.get("signal", "NEUTRAL")
                signal_color = "green" if signal in ["STRONG BUY", "BUY"] else "red" if signal in ["STRONG SELL", "SELL"] else "gray"
                
                st.markdown(f"**Signal:** `{signal}`")
                st.metric("Current Price", f"₹{pred.get('current_price', 0):.2f}")
                st.metric("Target", f"₹{pred.get('target_price', 0):.2f}")
                st.metric("Stop Loss", f"₹{pred.get('stop_loss', 0):.2f}")
                st.metric("Confidence", f"{pred.get('confidence', 0):.1%}")
                
                # Scores
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Technical", f"{pred.get('technical_score', 0):.1f}")
                with col2:
                    st.metric("Fundamental", f"{pred.get('fundamental_score', 0):.1f}")
                with col3:
                    st.metric("Sentiment", f"{pred.get('sentiment_score', 0):.1f}")
                
                st.caption(pred.get("timestamp", ""))


def display_live_status(client: LivePredictionsClient):
    """Display live service status"""
    
    with st.expander("📊 Live Service Status", expanded=False):
        status = client.get_status()
        
        if status:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Status", status.get("status", "unknown"))
            with col2:
                st.metric("Stocks Monitored", status.get("stocks_monitored", 0))
            with col3:
                st.metric("Update Interval", f"{status.get('update_interval', 0)}s")
            with col4:
                st.metric("Active Subscribers", status.get("active_subscribers", 0))
            
            st.metric("Market Open", status.get("is_market_open", False))
            st.metric("Total Updates", status.get("total_updates", 0))
            
            if status.get("last_update"):
                st.json(status.get("last_update"))


# Helper function for dashboard integration
@st.cache_resource
def init_live_client() -> LivePredictionsClient:
    """Initialize and cache live predictions client"""
    client = LivePredictionsClient()
    
    # Auto-refresh callback
    def on_predictions_update(predictions):
        st.rerun()
    
    client.add_callback(on_predictions_update)
    client.connect_live_feed()
    
    return client
