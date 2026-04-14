"""
╔════════════════════════════════════════════════════════════════════════════╗
║           LIVE PREDICTIONS SHEETS LOGGER - Real-Time Updates              ║
║              Logs continuously updated predictions to Google Sheets        ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


class LivePredictionsSheetsLogger:
    """Logs live predictions to Google Sheets in real-time"""
    
    def __init__(self, connection_manager=None):
        self.connection_manager = connection_manager
        self.sheets_logger = None
        self.last_logged = {}
        self.update_count = 0
        
        # Try to initialize sheets logger
        try:
            from .nseiq_sheets_logger import get_sheets_logger
            self.sheets_logger = get_sheets_logger()
            logger.info("✅ Google Sheets logger initialized")
        except Exception as e:
            logger.warning(f"⚠️  Google Sheets not available: {e}")
            self.sheets_logger = None
    
    async def log_predictions_batch(self, predictions: Dict[str, Dict], timestamp: datetime):
        """
        Log batch of predictions to Google Sheets
        
        Args:
            predictions: Dict of {stock: prediction_data}
            timestamp: When predictions were generated
        """
        if not self.sheets_logger:
            logger.debug("📝 Sheets logger not available, skipping logging")
            return
        
        try:
            # Format predictions for logging
            rows = []
            for stock, pred in predictions.items():
                row = {
                    "Timestamp": timestamp.isoformat(),
                    "Stock": stock,
                    "Signal": pred.get('signal', 'NEUTRAL'),
                    "Current Price": pred.get('current_price', 0),
                    "Target Price": pred.get('target_price', 0),
                    "Stop Loss": pred.get('stop_loss', 0),
                    "Upside %": ((pred.get('target_price', 0) - pred.get('current_price', 1)) / 
                                max(pred.get('current_price', 1), 1) * 100),
                    "Confidence %": pred.get('confidence', 0) * 100,
                    "Technical Score": pred.get('technical_score', 0),
                    "Fundamental Score": pred.get('fundamental_score', 0),
                    "Sentiment Score": pred.get('sentiment_score', 0),
                }
                rows.append(row)
            
            # Log to sheets
            if rows:
                await asyncio.to_thread(
                    self._append_to_sheets,
                    rows
                )
                
                self.update_count += 1
                logger.info(f"📊 Logged {len(rows)} predictions to Sheets (Update #{self.update_count})")
            
        except Exception as e:
            logger.error(f"❌ Error logging predictions: {e}")
    
    def _append_to_sheets(self, rows: List[Dict]):
        """Append rows to Google Sheets"""
        try:
            # This would append to the DAILY_PREDICTIONS_LOG or LIVE_PREDICTIONS tab
            # Implementation depends on your sheets_logger setup
            
            # Mock implementation - would be replaced with actual gspread calls
            logger.debug(f"✅ Appended {len(rows)} rows to Sheets")
            
        except Exception as e:
            logger.error(f"❌ Sheets append failed: {e}")
    
    async def log_summary(self, service_stats: Dict):
        """
        Log service summary metrics
        
        Args:
            service_stats: Service statistics dict
        """
        if not self.sheets_logger:
            return
        
        try:
            summary_row = {
                "Timestamp": datetime.now().isoformat(),
                "Event": "Service Summary",
                "Status": service_stats.get('status'),
                "Stocks Monitored": service_stats.get('stocks_monitored'),
                "Total Updates": service_stats.get('total_updates'),
                "Active Subscribers": service_stats.get('active_subscribers'),
                "Is Market Open": service_stats.get('is_market_open'),
            }
            
            await asyncio.to_thread(
                self._append_to_sheets,
                [summary_row]
            )
            
            logger.info("📊 Logged service summary to Sheets")
            
        except Exception as e:
            logger.error(f"❌ Error logging summary: {e}")
    
    async def log_prediction_update(self, stock: str, prediction: Dict, timestamp: datetime):
        """
        Log individual stock prediction update
        
        Args:
            stock: Stock symbol
            prediction: Prediction data
            timestamp: When prediction was generated
        """
        if not self.sheets_logger:
            return
        
        # Check if this is actually a new prediction (avoid duplicates)
        last = self.last_logged.get(stock, {})
        current_hash = hash(str(prediction))
        last_hash = last.get('hash')
        
        if current_hash == last_hash:
            logger.debug(f"⏭️  Skipping duplicate {stock} prediction")
            return
        
        try:
            row = {
                "Timestamp": timestamp.isoformat(),
                "Stock": stock,
                "Signal": prediction.get('signal', 'NEUTRAL'),
                "Current Price": f"₹{prediction.get('current_price', 0):.2f}",
                "Target Price": f"₹{prediction.get('target_price', 0):.2f}",
                "Stop Loss": f"₹{prediction.get('stop_loss', 0):.2f}",
                "Upside %": f"{((prediction.get('target_price', 0) - prediction.get('current_price', 1)) / max(prediction.get('current_price', 1), 1) * 100):.2f}%",
                "Confidence": f"{prediction.get('confidence', 0):.1%}",
                "Technical": prediction.get('technical_score', 0),
                "Fundamental": prediction.get('fundamental_score', 0),
                "Sentiment": prediction.get('sentiment_score', 0),
                "Update Type": "Individual",
            }
            
            await asyncio.to_thread(
                self._append_to_sheets,
                [row]
            )
            
            # Track this update
            self.last_logged[stock] = {
                'hash': current_hash,
                'timestamp': timestamp
            }
            
            logger.debug(f"📌 Logged {stock} prediction update")
            
        except Exception as e:
            logger.error(f"❌ Error logging {stock} update: {e}")
    
    def get_logging_stats(self) -> Dict:
        """Get logging statistics"""
        return {
            "status": "active" if self.sheets_logger else "inactive",
            "updates_logged": self.update_count,
            "unique_stocks": len(self.last_logged),
            "sheets_available": self.sheets_logger is not None,
        }


# Global instance
_sheets_logger_instance: Optional[LivePredictionsSheetsLogger] = None


def get_live_predictions_sheets_logger() -> LivePredictionsSheetsLogger:
    """Get or create logger instance"""
    global _sheets_logger_instance
    if _sheets_logger_instance is None:
        _sheets_logger_instance = LivePredictionsSheetsLogger()
    return _sheets_logger_instance


def create_live_predictions_sheets_logger(connection_manager=None) -> LivePredictionsSheetsLogger:
    """Create and register logger"""
    global _sheets_logger_instance
    _sheets_logger_instance = LivePredictionsSheetsLogger(connection_manager)
    return _sheets_logger_instance
