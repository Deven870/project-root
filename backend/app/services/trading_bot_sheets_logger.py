"""
╔════════════════════════════════════════════════════════════════════════════╗
║          TRADING BOT GOOGLE SHEETS LOGGER  - Real-Time Trade Tracking     ║
║                  Auto-logs all trades, P&L, and statistics                ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import logging
from typing import Dict, List, Optional
import asyncio

logger = logging.getLogger(__name__)

# Sheet ID from: https://docs.google.com/spreadsheets/d/1RuJHwfu2xfAYbSNMc05yzbz1M15kex6pL4MkyyiOxVw
SHEET_ID = "1RuJHwfu2xfAYbSNMc05yzbz1M15kex6pL4MkyyiOxVw"

# Sheet tabs
TRADES_SHEET = "Trades"
DAILY_STATS_SHEET = "Daily Stats"
POSITIONS_SHEET = "Open Positions"


class TradingBotSheetsLogger:
    """Logs trading bot trades to Google Sheets in real-time"""

    def __init__(self):
        self.sheet = None
        self.trades_ws = None
        self.stats_ws = None
        self.positions_ws = None
        self._initialize()

    def _initialize(self):
        """Initialize Google Sheets connection"""
        try:
            # Check if credentials file exists
            import os
            creds_path = os.path.expanduser("~/.config/gspread/service_account.json")
            
            if not os.path.exists(creds_path):
                logger.warning("⚠️  Google Sheets credentials not configured")
                logger.info("   To enable: Set up service account JSON at ~/.config/gspread/service_account.json")
                return

            # Authorize
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
            gc = gspread.authorize(creds)

            # Open spreadsheet
            self.sheet = gc.open_by_key(SHEET_ID)
            
            # Create/get worksheets
            try:
                self.trades_ws = self.sheet.worksheet(TRADES_SHEET)
            except gspread.exceptions.WorksheetNotFound:
                self.trades_ws = self.sheet.add_worksheet(title=TRADES_SHEET, rows=1000, cols=15)
                self._initialize_trades_headers()

            try:
                self.stats_ws = self.sheet.worksheet(DAILY_STATS_SHEET)
            except gspread.exceptions.WorksheetNotFound:
                self.stats_ws = self.sheet.add_worksheet(title=DAILY_STATS_SHEET, rows=500, cols=10)
                self._initialize_stats_headers()

            try:
                self.positions_ws = self.sheet.worksheet(POSITIONS_SHEET)
            except gspread.exceptions.WorksheetNotFound:
                self.positions_ws = self.sheet.add_worksheet(title=POSITIONS_SHEET, rows=500, cols=12)
                self._initialize_positions_headers()

            logger.info("✅ Google Sheets Logger Connected")
            logger.info(f"   Sheet: {SHEET_ID}")
            logger.info(f"   Tabs: {TRADES_SHEET}, {DAILY_STATS_SHEET}, {POSITIONS_SHEET}")

        except Exception as e:
            logger.warning(f"⚠️  Google Sheets not available: {e}")
            logger.info("   Trades will be logged locally only")

    def _initialize_trades_headers(self):
        """Add headers to Trades sheet"""
        if not self.trades_ws:
            return

        headers = [
            "Timestamp",
            "Symbol",
            "Signal",
            "Confidence",
            "Entry Price",
            "Target Price",
            "Stop Loss",
            "Quantity",
            "Capital Used",
            "Risk Amount",
            "Risk/Reward",
            "Status",
            "Exit Price",
            "P&L",
            "Return %"
        ]

        self.trades_ws.insert_row(headers, 1)
        logger.info("✅ Trades sheet initialized with headers")

    def _initialize_stats_headers(self):
        """Add headers to Daily Stats sheet"""
        if not self.stats_ws:
            return

        headers = [
            "Date",
            "Total Trades",
            "Winning Trades",
            "Losing Trades",
            "Win Rate %",
            "Total P&L",
            "Daily Return %",
            "Max Drawdown %",
            "Capital Used",
            "Notes"
        ]

        self.stats_ws.insert_row(headers, 1)
        logger.info("✅ Daily Stats sheet initialized with headers")

    def _initialize_positions_headers(self):
        """Add headers to Open Positions sheet"""
        if not self.positions_ws:
            return

        headers = [
            "Symbol",
            "Entry Date",
            "Entry Price",
            "Current Price",
            "Quantity",
            "Capital Used",
            "Target",
            "Stop Loss",
            "Current P&L",
            "P&L %",
            "Days Open",
            "Status"
        ]

        self.positions_ws.insert_row(headers, 1)
        logger.info("✅ Open Positions sheet initialized with headers")

    async def log_trade(self, trade_data: Dict):
        """Log a trade to Google Sheets"""
        if not self.trades_ws:
            logger.debug("📝 Google Sheets not available - skipping trade log")
            return

        try:
            row = [
                trade_data.get("timestamp", datetime.now().isoformat()),
                trade_data.get("symbol", "N/A"),
                trade_data.get("signal", "STRONG_BUY"),
                f"{trade_data.get('confidence', 0):.1f}%",
                f"₹{trade_data.get('entry_price', 0):.2f}",
                f"₹{trade_data.get('target_price', 0):.2f}",
                f"₹{trade_data.get('stop_loss', 0):.2f}",
                int(trade_data.get("quantity", 0)),
                f"₹{trade_data.get('capital_used', 0):.2f}",
                f"₹{trade_data.get('risk_amount', 0):.2f}",
                f"1:{trade_data.get('risk_reward_ratio', 1):.2f}",
                trade_data.get("status", "OPEN"),
                f"₹{trade_data.get('exit_price', 0):.2f}",
                f"₹{trade_data.get('pnl', 0):.2f}",
                f"{trade_data.get('return_pct', 0):.2f}%"
            ]

            self.trades_ws.append_row(row)
            logger.info(f"📊 Trade logged: {trade_data.get('symbol')} @ ₹{trade_data.get('entry_price', 0):.2f}")

        except Exception as e:
            logger.error(f"❌ Error logging trade: {e}")

    async def log_daily_stats(self, stats: Dict):
        """Log daily statistics"""
        if not self.stats_ws:
            return

        try:
            row = [
                stats.get("date", datetime.now().strftime("%Y-%m-%d")),
                stats.get("total_trades", 0),
                stats.get("winning_trades", 0),
                stats.get("losing_trades", 0),
                f"{stats.get('win_rate', 0):.1f}%",
                f"₹{stats.get('total_pnl', 0):.2f}",
                f"{stats.get('daily_return_pct', 0):.2f}%",
                f"{stats.get('max_drawdown_pct', 0):.2f}%",
                f"₹{stats.get('capital_used', 0):.2f}",
                stats.get("notes", "")
            ]

            self.stats_ws.append_row(row)
            logger.info(f"📈 Daily stats logged: {stats.get('total_trades')} trades, P&L: ₹{stats.get('total_pnl', 0):.2f}")

        except Exception as e:
            logger.error(f"❌ Error logging daily stats: {e}")

    async def update_open_positions(self, positions: List[Dict]):
        """Update open positions sheet"""
        if not self.positions_ws:
            return

        try:
            # Clear existing data (keep headers)
            if self.positions_ws.row_count > 1:
                self.positions_ws.delete_rows(2, self.positions_ws.row_count)

            # Add current positions
            for pos in positions:
                row = [
                    pos.get("symbol", "N/A"),
                    pos.get("entry_date", ""),
                    f"₹{pos.get('entry_price', 0):.2f}",
                    f"₹{pos.get('current_price', 0):.2f}",
                    int(pos.get("quantity", 0)),
                    f"₹{pos.get('capital_used', 0):.2f}",
                    f"₹{pos.get('target', 0):.2f}",
                    f"₹{pos.get('stop_loss', 0):.2f}",
                    f"₹{pos.get('current_pnl', 0):.2f}",
                    f"{pos.get('pnl_pct', 0):.2f}%",
                    pos.get("days_open", 0),
                    "OPEN"
                ]
                self.positions_ws.append_row(row)

            logger.info(f"📍 Open positions updated: {len(positions)} positions")

        except Exception as e:
            logger.error(f"❌ Error updating positions: {e}")


# Global singleton instance
_sheets_logger_instance: Optional[TradingBotSheetsLogger] = None


def get_trading_bot_sheets_logger() -> TradingBotSheetsLogger:
    """Get or create singleton instance"""
    global _sheets_logger_instance
    if _sheets_logger_instance is None:
        _sheets_logger_instance = TradingBotSheetsLogger()
    return _sheets_logger_instance
