"""
╔════════════════════════════════════════════════════════════════════════════╗
║                 NSEIQ GOOGLE SHEETS LOGGING ENGINE v5.0                    ║
║              Real-Time Analysis Logging & Performance Tracking              ║
╚════════════════════════════════════════════════════════════════════════════╝

Tabs:
  1. DAILY_PREDICTIONS_LOG - Every prediction with results
  2. PORTFOLIO_SNAPSHOT - Current holdings & P&L
  3. TRADE_JOURNAL - Historical trades + lessons
  4. PORTFOLIO_METRICS_DAILY - EOD metrics
  5. NEWS_SENTIMENT_LOG - News items with sentiment
  6. ALERTS_LOG - All alerts (SL, Target, Macro)
"""

import gspread
from gspread_dataframe import set_with_dataframe, get_as_dataframe
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import os
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Alert classifications"""
    SL_HIT = "SL Hit"
    TARGET_HIT = "Target Hit"
    MACRO_ALERT = "Macro Alert"
    REBALANCE = "Rebalance"
    VIX_SPIKE = "VIX Spike"
    CIRCUIT_BREAKER = "Circuit Breaker"
    NEWS_EVENT = "News Event"


class NSEIQSheetsLogger:
    """Google Sheets logging system for real-time analysis tracking"""

    def __init__(self, sheets_id: str, credentials_json: Optional[str] = None):
        """
        Initialize Sheets logger

        Args:
            sheets_id: Google Sheets ID (from URL)
            credentials_json: Path to service account JSON (optional)
        """
        self.sheets_id = sheets_id

        try:
            # Authenticate with Google Sheets
            if credentials_json and os.path.exists(credentials_json):
                self.gc = gspread.service_account(filename=credentials_json)
            else:
                # Use default authentication (if credentials are set in environment)
                self.gc = gspread.oauth()

            self.worksheet = self.gc.open_by_key(sheets_id)
            logger.info(f"✅ Google Sheets authenticated: {sheets_id[:20]}...")

            # Ensure all required tabs exist
            self._ensure_tabs()

        except Exception as e:
            logger.error(f"❌ Sheets authentication failed: {e}")
            self.worksheet = None

    def _ensure_tabs(self):
        """Create tabs if they don't exist"""
        required_tabs = [
            "DAILY_PREDICTIONS_LOG",
            "PORTFOLIO_SNAPSHOT",
            "TRADE_JOURNAL",
            "PORTFOLIO_METRICS_DAILY",
            "NEWS_SENTIMENT_LOG",
            "ALERTS_LOG",
        ]

        existing_tabs = [sheet.title for sheet in self.worksheet.worksheets()]

        for tab in required_tabs:
            if tab not in existing_tabs:
                self.worksheet.add_worksheet(title=tab, rows=1000, cols=15)
                logger.info(f"✅ Created tab: {tab}")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1: DAILY PREDICTIONS LOG
    # ═════════════════════════════════════════════════════════════════════════

    def log_prediction(self, prediction: Dict) -> bool:
        """
        Log prediction to DAILY_PREDICTIONS_LOG

        Columns: Date | Time | Ticker | Mode | Entry | SL | T1 | T2 | T3 | 
                 Signal | Confidence | CMP | Exit Price | Hit T1? | Hit SL? | P&L ₹ | Notes
        """
        try:
            if not self.worksheet:
                return False

            sheet = self.worksheet.worksheet("DAILY_PREDICTIONS_LOG")

            # Build row
            now = datetime.now()
            row = [
                now.strftime("%d-%b-%Y"),  # Date
                now.strftime("%H:%M:%S"),  # Time
                prediction.get("ticker", ""),  # Ticker
                prediction.get("mode", ""),  # Mode
                prediction.get("entry_zone_low", ""),  # Entry
                prediction.get("stop_loss", ""),  # SL
                prediction.get("target_1", ""),  # T1
                prediction.get("target_2", ""),  # T2
                prediction.get("target_3", ""),  # T3
                prediction.get("signal", ""),  # Signal
                prediction.get("confidence", ""),  # Confidence
                prediction.get("current_price", ""),  # CMP
                "",  # Exit Price (filled later)
                "",  # Hit T1?
                "",  # Hit SL?
                "",  # P&L
                prediction.get("thesis_summary", "")[:50],  # Notes
            ]

            # Append to sheet
            sheet.append_row(row)
            logger.info(f"✅ Logged prediction: {prediction.get('ticker')}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to log prediction: {e}")
            return False

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2: PORTFOLIO SNAPSHOT
    # ═════════════════════════════════════════════════════════════════════════

    def log_portfolio_snapshot(self, portfolio_state: Dict) -> bool:
        """
        Log portfolio holdings to PORTFOLIO_SNAPSHOT

        Columns: Date | Stock | Qty | Avg Buy Price | CMP | Current Value | 
                 P&L ₹ | P&L % | Days Held | Status | Exit Price | Exit Date | Reason
        """
        try:
            if not self.worksheet:
                return False

            sheet = self.worksheet.worksheet("PORTFOLIO_SNAPSHOT")

            # Clear existing data and write new
            sheet.clear()

            # Header
            headers = [
                "Date",
                "Stock",
                "Qty",
                "Avg Buy Price",
                "CMP",
                "Current Value",
                "P&L ₹",
                "P&L %",
                "Days Held",
                "Status",
                "Entry Price",
                "Exit Price",
                "Exit Date",
                "Reason",
            ]
            sheet.append_row(headers)

            # Build rows from portfolio
            positions = portfolio_state.get("positions", [])
            now = datetime.now()

            for pos in positions:
                row = [
                    now.strftime("%d-%b-%Y"),
                    pos.get("ticker", ""),
                    pos.get("quantity", 0),
                    pos.get("avg_buy_price", 0),
                    pos.get("current_price", 0),
                    pos.get("current_value", 0),
                    pos.get("pnl_rupees", 0),
                    pos.get("pnl_pct", 0),
                    pos.get("days_held", 0),
                    "OPEN" if pos.get("status") == "open" else "CLOSED",
                    pos.get("entry_price", ""),
                    pos.get("exit_price", ""),
                    pos.get("exit_date", ""),
                    pos.get("exit_reason", ""),
                ]
                sheet.append_row(row)

            logger.info(f"✅ Logged portfolio snapshot: {len(positions)} positions")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to log portfolio snapshot: {e}")
            return False

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3: TRADE JOURNAL
    # ═════════════════════════════════════════════════════════════════════════

    def log_trade(self, trade: Dict) -> bool:
        """
        Log completed trade to TRADE_JOURNAL

        Columns: Trade ID | Entry Date | Stock | Setup Type | Entry Price | SL | 
                 Target | Exit Date | Exit Price | P&L ₹ | P&L % | What Worked | 
                 What Didn't | Lesson
        """
        try:
            if not self.worksheet:
                return False

            sheet = self.worksheet.worksheet("TRADE_JOURNAL")

            # Build row
            row = [
                trade.get("trade_id", ""),  # Trade ID
                trade.get("entry_date", ""),  # Entry Date
                trade.get("ticker", ""),  # Stock
                trade.get("setup_type", ""),  # Setup Type
                trade.get("entry_price", 0),  # Entry Price
                trade.get("stop_loss", 0),  # SL
                trade.get("target", 0),  # Target
                trade.get("exit_date", ""),  # Exit Date
                trade.get("exit_price", 0),  # Exit Price
                trade.get("pnl_rupees", 0),  # P&L ₹
                trade.get("pnl_pct", 0),  # P&L %
                trade.get("what_worked", "")[:50],  # What Worked
                trade.get("what_didnt", "")[:50],  # What Didn't
                trade.get("lesson", "")[:80],  # Lesson
            ]

            sheet.append_row(row)
            logger.info(f"✅ Logged trade: {trade.get('trade_id')}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to log trade: {e}")
            return False

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 4: PORTFOLIO METRICS DAILY
    # ═════════════════════════════════════════════════════════════════════════

    def log_daily_metrics(self, metrics: Dict) -> bool:
        """
        Log daily metrics to PORTFOLIO_METRICS_DAILY (called at 3:30 PM IST)

        Columns: Date | Total Invested | Current Value | Total P&L ₹ | Total P&L % |
                 Day's Gain/Loss | Portfolio Beta | Win Rate % | Avg Win ₹ | 
                 Avg Loss ₹ | Expectancy | VIX | NIFTY Close
        """
        try:
            if not self.worksheet:
                return False

            sheet = self.worksheet.worksheet("PORTFOLIO_METRICS_DAILY")

            # Build row
            now = datetime.now()
            row = [
                now.strftime("%d-%b-%Y"),  # Date
                metrics.get("total_invested", 0),  # Total Invested
                metrics.get("current_value", 0),  # Current Value
                metrics.get("total_pnl_rupees", 0),  # Total P&L ₹
                metrics.get("total_pnl_pct", 0),  # Total P&L %
                metrics.get("days_gain_loss", 0),  # Day's Gain/Loss
                metrics.get("portfolio_beta", 0),  # Portfolio Beta
                metrics.get("win_rate_pct", 0),  # Win Rate %
                metrics.get("avg_win", 0),  # Avg Win ₹
                metrics.get("avg_loss", 0),  # Avg Loss ₹
                metrics.get("expectancy", 0),  # Expectancy
                metrics.get("vix", 0),  # VIX
                metrics.get("nifty_close", 0),  # NIFTY Close
            ]

            sheet.append_row(row)
            logger.info(f"✅ Logged daily metrics")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to log daily metrics: {e}")
            return False

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 5: NEWS & SENTIMENT LOG
    # ═════════════════════════════════════════════════════════════════════════

    def log_news_sentiment(self, news_item: Dict) -> bool:
        """
        Log news item to NEWS_SENTIMENT_LOG

        Columns: Date | Time | Ticker | Headline | Source | Sentiment | 
                 Score | Impact Level | Action Triggered
        """
        try:
            if not self.worksheet:
                return False

            sheet = self.worksheet.worksheet("NEWS_SENTIMENT_LOG")

            # Build row
            now = datetime.now()
            row = [
                now.strftime("%d-%b-%Y"),  # Date
                now.strftime("%H:%M:%S"),  # Time
                news_item.get("ticker", ""),  # Ticker
                news_item.get("headline", "")[:80],  # Headline
                news_item.get("source", ""),  # Source
                news_item.get("sentiment", ""),  # Sentiment
                news_item.get("score", 0),  # Score
                news_item.get("impact_level", "MEDIUM"),  # Impact
                news_item.get("action_triggered", "")[:50],  # Action
            ]

            sheet.append_row(row)
            logger.info(f"✅ Logged news sentiment: {news_item.get('ticker')}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to log news sentiment: {e}")
            return False

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 6: ALERTS LOG
    # ═════════════════════════════════════════════════════════════════════════

    def log_alert(self, alert: Dict) -> bool:
        """
        Log alert to ALERTS_LOG

        Columns: Date | Time | Type | Stock | Details | Recommended Action | 
                 Actioned by User?
        """
        try:
            if not self.worksheet:
                return False

            sheet = self.worksheet.worksheet("ALERTS_LOG")

            # Build row
            now = datetime.now()
            row = [
                now.strftime("%d-%b-%Y"),  # Date
                now.strftime("%H:%M:%S"),  # Time
                alert.get("alert_type", ""),  # Type
                alert.get("ticker", ""),  # Stock
                alert.get("details", "")[:80],  # Details
                alert.get("recommended_action", "")[:80],  # Action
                alert.get("actioned", "N"),  # Actioned?
            ]

            sheet.append_row(row)
            logger.info(f"✅ Logged alert: {alert.get('alert_type')}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to log alert: {e}")
            return False

    # ═════════════════════════════════════════════════════════════════════════
    # BATCH OPERATIONS
    # ═════════════════════════════════════════════════════════════════════════

    def log_batch_predictions(self, predictions: List[Dict]) -> int:
        """Log multiple predictions (batch operation)"""
        count = 0
        for pred in predictions:
            if self.log_prediction(pred):
                count += 1
        logger.info(f"✅ Logged {count}/{len(predictions)} predictions")
        return count

    def get_daily_summary(self) -> Dict:
        """Get today's trading summary from sheets"""
        try:
            if not self.worksheet:
                return {}

            # Get predictions logged today
            pred_sheet = self.worksheet.worksheet("DAILY_PREDICTIONS_LOG")
            predictions = pred_sheet.get_all_values()

            # Get metrics logged today
            metrics_sheet = self.worksheet.worksheet("PORTFOLIO_METRICS_DAILY")
            metrics = metrics_sheet.get_all_values()

            # Get alerts logged today
            alerts_sheet = self.worksheet.worksheet("ALERTS_LOG")
            alerts = alerts_sheet.get_all_values()

            today = datetime.now().strftime("%d-%b-%Y")

            today_predictions = [
                p for p in predictions[1:] if p[0] == today
            ]  # Skip header
            today_alerts = [a for a in alerts[1:] if a[0] == today]

            return {
                "date": today,
                "predictions_count": len(today_predictions),
                "predictions": today_predictions[:5],  # First 5
                "alerts_count": len(today_alerts),
                "alerts": today_alerts[:5],
            }

        except Exception as e:
            logger.error(f"❌ Failed to get daily summary: {e}")
            return {}

    # ═════════════════════════════════════════════════════════════════════════
    # HEALTH CHECK
    # ═════════════════════════════════════════════════════════════════════════

    def health_check(self) -> bool:
        """Verify Sheets connection is active"""
        try:
            if not self.worksheet:
                return False

            # Try to read first cell
            sheet = self.worksheet.worksheet("DAILY_PREDICTIONS_LOG")
            sheet.cell(1, 1).value
            logger.info("✅ Google Sheets connected & healthy")
            return True

        except Exception as e:
            logger.error(f"❌ Sheets health check failed: {e}")
            return False


# ═════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═════════════════════════════════════════════════════════════════════════════

def get_sheets_logger() -> Optional[NSEIQSheetsLogger]:
    """Get or create Sheets logger instance"""
    try:
        from app.config import GOOGLE_SHEETS_ID, SERVICE_ACCOUNT_FILE

        return NSEIQSheetsLogger(GOOGLE_SHEETS_ID, SERVICE_ACCOUNT_FILE)
    except Exception as e:
        logger.error(f"❌ Could not initialize Sheets logger: {e}")
        return None


sheets_logger = None
