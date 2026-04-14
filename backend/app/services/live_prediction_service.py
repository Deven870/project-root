"""
╔════════════════════════════════════════════════════════════════════════════╗
║              LIVE PREDICTION SERVICE - Real-Time Updates                   ║
║          Continuously fetches predictions and broadcasts to clients        ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
from datetime import datetime, time
import json
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class LivePredictionService:
    """Manages continuous live prediction updates"""

    def __init__(self, connection_manager=None):
        self.connection_manager = connection_manager
        self.sheets_logger = None
        self.is_running = False
        self.update_interval = 60  # seconds (1 minute)
        self.market_open = time(9, 15)  # 9:15 AM IST
        self.market_close = time(15, 30)  # 3:30 PM IST
        
        # Primary NSE stocks for continuous monitoring
        self.stocks = [
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
            "SBIN", "WIPRO", "AXISBANK", "LT", "MARUTI",
            "BAJAJ-AUTO", "SUNPHARMA", "ASIANPAINT", "KOTAKBANK", "M&M"
        ]
        
        self.current_predictions: Dict[str, Dict] = {}
        self.update_history: List[Dict] = []
        
        # Initialize sheets logger
        try:
            from .live_predictions_sheets_logger import get_live_predictions_sheets_logger
            self.sheets_logger = get_live_predictions_sheets_logger()
            logger.info("✅ Sheets logger registered for live predictions")
        except Exception as e:
            logger.warning(f"⚠️  Sheets logger not available: {e}")
        
    def set_connection_manager(self, manager):
        """Set WebSocket connection manager"""
        self.connection_manager = manager
        logger.info("✅ Connection manager registered for live updates")
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now().time()
        return self.market_open <= now <= self.market_close
    
    async def start(self):
        """Start the live prediction loop"""
        if self.is_running:
            logger.warning("⚠️  Live prediction service already running")
            return
        
        self.is_running = True
        logger.info("🟢 Live Prediction Service STARTED")
        logger.info(f"📊 Monitoring {len(self.stocks)} stocks at {self.update_interval}s intervals")
        
        try:
            await self.prediction_loop()
        except Exception as e:
            logger.error(f"❌ Live prediction service error: {e}")
            self.is_running = False
    
    async def stop(self):
        """Stop the live prediction loop"""
        self.is_running = False
        logger.info("🔴 Live Prediction Service STOPPED")
    
    async def prediction_loop(self):
        """Main loop for fetching and broadcasting predictions"""
        first_run = True
        
        while self.is_running:
            try:
                # Only update during market hours (with first run exception)
                if not self.is_market_open() and not first_run:
                    logger.info("⏸️  Market closed - Live updates paused")
                    await asyncio.sleep(300)  # Check every 5 mins
                    continue
                
                if first_run:
                    logger.info("🚀 First run - fetching initial predictions...")
                    first_run = False
                
                # Fetch predictions for all stocks
                predictions = await self.fetch_batch_predictions()
                
                if predictions:
                    # Store current predictions
                    self.current_predictions = predictions
                    
                    # Add to history
                    self.update_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "count": len(predictions),
                        "stocks": list(predictions.keys())
                    })
                    
                    # Keep only last 100 updates
                    if len(self.update_history) > 100:
                        self.update_history = self.update_history[-100:]
                    
                    # Log to Google Sheets
                    if self.sheets_logger:
                        try:
                            await self.sheets_logger.log_predictions_batch(
                                predictions,
                                datetime.now()
                            )
                        except Exception as e:
                            logger.debug(f"⚠️  Sheets logging error: {e}")
                    
                    # Broadcast to all connected clients
                    await self.broadcast_predictions(predictions)
                    
                    logger.info(f"✅ Updated {len(predictions)} predictions | Subscribers: {self.get_subscriber_count()}")
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"❌ Prediction loop error: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def fetch_batch_predictions(self) -> Dict[str, Dict]:
        """Fetch predictions for all monitored stocks"""
        from .nseiq_prediction_engine import NSEIQPredictionEngine
        
        predictions = {}
        engine = NSEIQPredictionEngine()
        
        for stock in self.stocks:
            try:
                # Get prediction
                pred = await asyncio.to_thread(
                    engine.predict,
                    stock,
                    "INTRADAY"
                )
                
                if pred and "error" not in pred:
                    predictions[stock] = {
                        "ticker": stock,
                        "signal": pred.get("signal", "NEUTRAL"),
                        "current_price": pred.get("current_price", 0),
                        "target_price": pred.get("target_price", 0),
                        "stop_loss": pred.get("stop_loss", 0),
                        "confidence": pred.get("confidence", 0),
                        "timestamp": datetime.now().isoformat(),
                        "technical_score": pred.get("technical_score", 0),
                        "fundamental_score": pred.get("fundamental_score", 0),
                        "sentiment_score": pred.get("sentiment_score", 0),
                    }
                    
            except Exception as e:
                logger.debug(f"⚠️  Error fetching {stock}: {e}")
                continue
        
        return predictions
    
    async def broadcast_predictions(self, predictions: Dict[str, Dict]):
        """Broadcast predictions to all connected clients via WebSocket"""
        if not self.connection_manager or not self.connection_manager.active_connections:
            return
        
        # Format message
        message = {
            "type": "predictions_update",
            "timestamp": datetime.now().isoformat(),
            "count": len(predictions),
            "data": predictions
        }
        
        try:
            await self.connection_manager.broadcast(message)
        except Exception as e:
            logger.error(f"❌ Broadcast error: {e}")
    
    async def broadcast_stock_update(self, stock: str, prediction: Dict):
        """Broadcast individual stock prediction update"""
        if not self.connection_manager:
            return
        
        message = {
            "type": "stock_update",
            "ticker": stock,
            "timestamp": datetime.now().isoformat(),
            "data": prediction
        }
        
        try:
            await self.connection_manager.broadcast_to_subscription(stock, message)
        except Exception as e:
            logger.error(f"❌ Stock broadcast error: {e}")
    
    def get_current_predictions(self, stock: Optional[str] = None) -> Dict:
        """Get current predictions"""
        if stock:
            return self.current_predictions.get(stock, {})
        return self.current_predictions
    
    def get_subscriber_count(self) -> int:
        """Get number of active subscribers"""
        if self.connection_manager:
            return len(self.connection_manager.active_connections)
        return 0
    
    def get_stats(self) -> Dict:
        """Get service statistics"""
        return {
            "status": "running" if self.is_running else "stopped",
            "stocks_monitored": len(self.stocks),
            "update_interval": self.update_interval,
            "market_open": self.market_open.isoformat(),
            "market_close": self.market_close.isoformat(),
            "is_market_open": self.is_market_open(),
            "current_predictions": len(self.current_predictions),
            "active_subscribers": self.get_subscriber_count(),
            "total_updates": len(self.update_history),
            "last_update": self.update_history[-1] if self.update_history else None
        }


# Global singleton instance
_live_service_instance: Optional[LivePredictionService] = None


def get_live_prediction_service() -> LivePredictionService:
    """Get or create singleton instance"""
    global _live_service_instance
    if _live_service_instance is None:
        _live_service_instance = LivePredictionService()
    return _live_service_instance


def create_live_prediction_service(connection_manager=None) -> LivePredictionService:
    """Create and register live prediction service"""
    global _live_service_instance
    _live_service_instance = LivePredictionService(connection_manager)
    return _live_service_instance
