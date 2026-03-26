"""
Google Sheets integration for real-time trade tracking in Digitrader.
Uses service account authentication (not OAuth).
Supports: live signals, trade logging, price updates, PnL tracking, news feed.
"""
import os
import gspread
import time
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

GOOGLE_SHEETS_ID = os.getenv("GOOGLE_SHEETS_ID", "")
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")

# Sheet names (tabs)
SHEET_LIVE_SIGNALS = "Live Signals"
SHEET_MY_TRADES = "My Trades"
SHEET_PNL_DASHBOARD = "PnL Dashboard"
SHEET_NEWS_FEED = "News Feed"
SHEET_CONFIG = "Config"

# Max retries for API calls
MAX_RETRIES = 3
RETRY_DELAY = 2

logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL STATE
# ============================================================================

_spreadsheet = None  # Cached spreadsheet connection
_worksheets_cache = {}  # Cached worksheet objects
_trade_counter = {}  # For generating TradeIDs


# ============================================================================
# AUTHENTICATION & CONNECTION
# ============================================================================

def _get_credentials():
    """
    Load service account credentials from JSON file.
    Raises error if credentials not found.
    """
    if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
        raise FileNotFoundError(
            f"Credentials file not found: {GOOGLE_CREDENTIALS_PATH}\n"
            f"Download from Google Cloud Console and save as {GOOGLE_CREDENTIALS_PATH}"
        )
    
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    
    try:
        credentials = Credentials.from_service_account_file(
            GOOGLE_CREDENTIALS_PATH,
            scopes=scopes
        )
        return credentials
    except Exception as e:
        raise ValueError(f"Failed to load credentials: {e}")


def _get_spreadsheet():
    """
    Get cached spreadsheet connection or create new one.
    """
    global _spreadsheet
    
    if _spreadsheet is not None:
        return _spreadsheet
    
    if not GOOGLE_SHEETS_ID:
        raise ValueError("GOOGLE_SHEETS_ID not configured in .env")
    
    try:
        credentials = _get_credentials()
        client = gspread.authorize(credentials)
        _spreadsheet = client.open_by_key(GOOGLE_SHEETS_ID)
        logger.info("✓ Spreadsheet connected")
        return _spreadsheet
    except Exception as e:
        logger.error(f"Failed to connect to spreadsheet: {e}")
        raise


def _retry_gspread_call(func, *args, max_retries=MAX_RETRIES, **kwargs):
    """
    Retry gspread API calls with exponential backoff.
    
    Parameters
    ----------
    func : callable
        gspread function to call
    max_retries : int
        Maximum number of retry attempts
    *args, **kwargs
        Arguments to pass to func
    
    Returns
    -------
    Result of func call, or None if all retries fail
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except gspread.exceptions.APIError as e:
            if attempt < max_retries - 1:
                wait_time = RETRY_DELAY ** attempt
                logger.warning(f"API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"API failed after {max_retries} attempts: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error in gspread call: {e}")
            raise


# ============================================================================
# WORKSHEET MANAGEMENT
# ============================================================================

def get_worksheet(tab_name: str):
    """
    Get worksheet by name with caching.
    Creates tab if it doesn't exist.
    
    Parameters
    ----------
    tab_name : str
        Name of the worksheet/tab
    
    Returns
    -------
    gspread.Worksheet
        The worksheet object
    """
    if tab_name in _worksheets_cache:
        return _worksheets_cache[tab_name]
    
    try:
        spreadsheet = _get_spreadsheet()
        
        # Try to get existing worksheet
        try:
            worksheet = spreadsheet.worksheet(tab_name)
            logger.info(f"✓ Opened worksheet: {tab_name}")
        except gspread.exceptions.WorksheetNotFound:
            # Create new worksheet
            worksheet = spreadsheet.add_worksheet(title=tab_name, rows=1000, cols=20)
            logger.info(f"✓ Created worksheet: {tab_name}")
        
        _worksheets_cache[tab_name] = worksheet
        return worksheet
    
    except Exception as e:
        logger.error(f"Error getting worksheet '{tab_name}': {e}")
        raise


# ============================================================================
# SETUP HEADERS
# ============================================================================

def setup_sheet_headers():
    """
    Create all 5 tabs with proper headers if they don't exist.
    """
    headers = {
        SHEET_LIVE_SIGNALS: [
            "Date", "Time", "Symbol", "Trend", "Confidence%", "CurrentPrice",
            "PredictedPrice", "ExpectedReturn%", "StopLoss", "Horizon",
            "SentimentScore", "AlertFired"
        ],
        SHEET_MY_TRADES: [
            "TradeID", "Symbol", "BuyPrice", "Qty", "BuyTime", "CurrentPrice",
            "PnL_Rs", "PnL_Pct", "StopLoss", "Target", "Status",
            "SellPrice", "SellTime", "Horizon"
        ],
        SHEET_PNL_DASHBOARD: [
            "Date", "TotalRealised_Rs", "TotalUnrealised_Rs", "WinCount",
            "LossCount", "WinRate%", "BestTrade_Rs", "WorstTrade_Rs", "TotalTrades"
        ],
        SHEET_NEWS_FEED: [
            "Timestamp", "Symbol", "Headline", "SentimentScore",
            "SentimentLabel", "Source", "AlertTriggered"
        ],
        SHEET_CONFIG: [
            "Key", "Value"
        ]
    }
    
    try:
        for sheet_name, header_row in headers.items():
            worksheet = get_worksheet(sheet_name)
            
            # Check if headers already exist
            try:
                existing_headers = worksheet.row_values(1)
                if existing_headers and len(existing_headers) > 0:
                    logger.info(f"✓ {sheet_name} already has headers")
                    continue
            except:
                pass
            
            # Insert headers
            def insert_headers():
                worksheet.insert_row(header_row, 1)
            
            _retry_gspread_call(insert_headers)
            logger.info(f"✓ Added headers to {sheet_name}")
    
    except Exception as e:
        logger.error(f"Error setting up headers: {e}")
        raise


# ============================================================================
# SIGNAL LOGGING
# ============================================================================

def log_signal(
    symbol: str,
    trend: str,
    confidence: float,
    current_price: float,
    predicted_price: float,
    expected_return: float,
    stop_loss: float,
    horizon: str,
    sentiment_score: float
) -> bool:
    """
    Log a trading signal to "Live Signals" sheet with current timestamp.
    
    Parameters
    ----------
    symbol : str
        Stock symbol (e.g., "RELIANCE.NS")
    trend : str
        "BULLISH", "BEARISH", or "NEUTRAL"
    confidence : float
        Confidence as percentage (0-100)
    current_price : float
        Current stock price
    predicted_price : float
        Predicted target price
    expected_return : float
        Expected return as percentage
    stop_loss : float
        Stop loss level
    horizon : str
        "intraday", "short_term", "medium_term"
    sentiment_score : float
        Sentiment score (-1 to 1)
    
    Returns
    -------
    bool
        True if logged successfully
    """
    try:
        now = datetime.now()
        row = [
            now.strftime("%Y-%m-%d"),  # Date
            now.strftime("%H:%M:%S"),  # Time
            symbol,
            trend,
            f"{confidence:.2f}",
            f"{current_price:.2f}",
            f"{predicted_price:.2f}",
            f"{expected_return:.2f}",
            f"{stop_loss:.2f}",
            horizon,
            f"{sentiment_score:.4f}",
            "YES"
        ]
        
        worksheet = get_worksheet(SHEET_LIVE_SIGNALS)
        
        def append_row():
            worksheet.append_row(row)
        
        _retry_gspread_call(append_row)
        logger.info(f"✓ Logged signal for {symbol}")
        return True
    
    except Exception as e:
        logger.error(f"Error logging signal: {e}")
        return False


# ============================================================================
# TRADE LOGGING
# ============================================================================

def _generate_trade_id():
    """
    Generate unique TradeID in format: TRD-YYYYMMDD-###
    """
    today = datetime.now().strftime("%Y%m%d")
    
    if today not in _trade_counter:
        _trade_counter[today] = 0
    
    _trade_counter[today] += 1
    trade_num = _trade_counter[today]
    
    return f"TRD-{today}-{trade_num:03d}"


def log_trade(
    symbol: str,
    buy_price: float,
    qty: int,
    stop_loss: float,
    target: float,
    horizon: str
) -> str:
    """
    Log a new trade entry to "My Trades" sheet.
    
    Parameters
    ----------
    symbol : str
        Stock symbol
    buy_price : float
        Entry price
    qty : int
        Quantity traded
    stop_loss : float
        Stop loss level
    target : float
        Target/profit level
    horizon : str
        Trade horizon
    
    Returns
    -------
    str
        TradeID (or empty string if failed)
    """
    try:
        trade_id = _generate_trade_id()
        now = datetime.now()
        
        row = [
            trade_id,
            symbol,
            f"{buy_price:.2f}",
            qty,
            now.strftime("%Y-%m-%d %H:%M:%S"),
            f"{buy_price:.2f}",  # CurrentPrice (initially same as buy_price)
            "0.00",  # PnL_Rs
            "0.00",  # PnL_Pct
            f"{stop_loss:.2f}",
            f"{target:.2f}",
            "OPEN",  # Status
            "",  # SellPrice
            "",  # SellTime
            horizon
        ]
        
        worksheet = get_worksheet(SHEET_MY_TRADES)
        
        def append_trade():
            worksheet.append_row(row)
        
        _retry_gspread_call(append_trade)
        logger.info(f"✓ Logged trade {trade_id} for {symbol}")
        return trade_id
    
    except Exception as e:
        logger.error(f"Error logging trade: {e}")
        return ""


# ============================================================================
# TRADE CLOSING
# ============================================================================

def close_trade(trade_id: str, sell_price: float) -> bool:
    """
    Close a trade and update its status, sell price, and P&L.
    
    Parameters
    ----------
    trade_id : str
        TradeID to close
    sell_price : float
        Exit price
    
    Returns
    -------
    bool
        True if closed successfully
    """
    try:
        worksheet = get_worksheet(SHEET_MY_TRADES)
        
        # Get all rows
        def get_all_rows():
            return worksheet.get_all_records()
        
        all_trades = _retry_gspread_call(get_all_rows) or []
        
        # Find the trade
        trade_row_idx = None
        trade_data = None
        
        for idx, trade in enumerate(all_trades, start=2):  # Start from 2 (row 1 is header)
            if trade.get("TradeID") == trade_id:
                trade_row_idx = idx
                trade_data = trade
                break
        
        if not trade_data:
            logger.warning(f"Trade {trade_id} not found")
            return False

        if trade_row_idx is None:
            logger.warning(f"Trade row index missing for {trade_id}")
            return False
        
        # Calculate P&L
        buy_price = float(trade_data.get("BuyPrice", 0))
        qty = int(trade_data.get("Qty", 0))
        pnl_rs = (sell_price - buy_price) * qty
        pnl_pct = ((sell_price - buy_price) / buy_price * 100) if buy_price > 0 else 0
        
        # Prepare updates
        now = datetime.now()
        updates = {
            "Status": "CLOSED",
            "SellPrice": f"{sell_price:.2f}",
            "SellTime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "PnL_Rs": f"{pnl_rs:.2f}",
            "PnL_Pct": f"{pnl_pct:.2f}"
        }
        
        # Update using batch_update
        def batch_update():
            cells_to_update = []
            headers = worksheet.row_values(1)
            
            for col_idx, header in enumerate(headers, start=1):
                if header in updates:
                    cells_to_update.append(gspread.Cell(trade_row_idx, col_idx, updates[header]))
            
            if cells_to_update:
                worksheet.update_cells(cells_to_update)
        
        _retry_gspread_call(batch_update)
        logger.info(f"✓ Closed trade {trade_id}: P&L = ₹{pnl_rs:.2f} ({pnl_pct:.2f}%)")
        return True
    
    except Exception as e:
        logger.error(f"Error closing trade {trade_id}: {e}")
        return False


# ============================================================================
# LIVE PRICE UPDATES (BATCH)
# ============================================================================

def update_live_prices(price_dict: Dict[str, float]) -> bool:
    """
    Update current prices and PnL for all OPEN trades (batch update).
    Called periodically (e.g., every 60 seconds during market hours).
    
    Parameters
    ----------
    price_dict : dict
        Dictionary of {symbol: current_price}
    
    Returns
    -------
    bool
        True if updated successfully
    """
    try:
        worksheet = get_worksheet(SHEET_MY_TRADES)
        
        # Get all open trades
        def get_all_trades():
            return worksheet.get_all_records()
        
        all_trades = _retry_gspread_call(get_all_trades) or []
        
        # Collect cells to update
        cells_to_update = []
        headers = worksheet.row_values(1)
        
        for row_idx, trade in enumerate(all_trades, start=2):
            if trade.get("Status") != "OPEN":
                continue
            
            symbol = trade.get("Symbol")
            if symbol not in price_dict:
                continue
            
            current_price = price_dict[symbol]
            buy_price = float(trade.get("BuyPrice", 0))
            qty = int(trade.get("Qty", 0))
            
            # Calculate P&L
            pnl_rs = (current_price - buy_price) * qty
            pnl_pct = ((current_price - buy_price) / buy_price * 100) if buy_price > 0 else 0
            
            # Find column indices
            current_price_col = None
            pnl_rs_col = None
            pnl_pct_col = None
            
            for col_idx, header in enumerate(headers, start=1):
                if header == "CurrentPrice":
                    current_price_col = col_idx
                elif header == "PnL_Rs":
                    pnl_rs_col = col_idx
                elif header == "PnL_Pct":
                    pnl_pct_col = col_idx
            
            # Add cells to update batch
            if current_price_col:
                cells_to_update.append(
                    gspread.Cell(row_idx, current_price_col, f"{current_price:.2f}")
                )
            if pnl_rs_col:
                cells_to_update.append(
                    gspread.Cell(row_idx, pnl_rs_col, f"{pnl_rs:.2f}")
                )
            if pnl_pct_col:
                cells_to_update.append(
                    gspread.Cell(row_idx, pnl_pct_col, f"{pnl_pct:.2f}")
                )
        
        # Batch update all cells at once
        if cells_to_update:
            def batch_update():
                worksheet.update_cells(cells_to_update)
            
            _retry_gspread_call(batch_update)
            logger.info(f"✓ Updated prices for {len(set(t.get('Symbol') for t in all_trades if t.get('Status') == 'OPEN' and t.get('Symbol') in price_dict))} symbols")
        
        return True
    
    except Exception as e:
        logger.error(f"Error updating live prices: {e}")
        return False


# ============================================================================
# OPEN TRADES
# ============================================================================

def get_open_trades() -> List[Dict]:
    """
    Get all OPEN trades from "My Trades" sheet.
    
    Returns
    -------
    list of dict
        List of open trade records
    """
    try:
        worksheet = get_worksheet(SHEET_MY_TRADES)
        
        def get_all_trades():
            return worksheet.get_all_records()
        
        all_trades = _retry_gspread_call(get_all_trades) or []
        open_trades = [t for t in all_trades if t.get("Status") == "OPEN"]
        
        logger.info(f"✓ Retrieved {len(open_trades)} open trades")
        return open_trades
    
    except Exception as e:
        logger.error(f"Error getting open trades: {e}")
        return []


# ============================================================================
# P&L DASHBOARD
# ============================================================================

def update_pnl_dashboard() -> bool:
    """
    Calculate daily P&L aggregates and update/append row in "PnL Dashboard".
    """
    try:
        trades_worksheet = get_worksheet(SHEET_MY_TRADES)
        dashboard_worksheet = get_worksheet(SHEET_PNL_DASHBOARD)
        
        # Get all trades
        def get_trades():
            return trades_worksheet.get_all_records()
        
        all_trades = _retry_gspread_call(get_trades) or []
        
        # Filter today's closed trades
        today = date.today().strftime("%Y-%m-%d")
        today_trades = [
            t for t in all_trades
            if str(t.get("SellTime", "")).startswith(today) and t.get("Status") == "CLOSED"
        ]
        
        if not today_trades:
            logger.info("No closed trades today")
            return True
        
        # Calculate aggregates
        total_realised = sum(float(t.get("PnL_Rs", 0)) for t in today_trades)
        
        wins = [t for t in today_trades if float(t.get("PnL_Rs", 0)) > 0]
        losses = [t for t in today_trades if float(t.get("PnL_Rs", 0)) < 0]
        
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / len(today_trades) * 100) if today_trades else 0
        
        best_trade = max(float(t.get("PnL_Rs", 0)) for t in today_trades) if today_trades else 0
        worst_trade = min(float(t.get("PnL_Rs", 0)) for t in today_trades) if today_trades else 0
        
        # For this implementation, assume no unrealised (all trades are closed today)
        total_unrealised = 0
        
        row = [
            today,
            f"{total_realised:.2f}",
            f"{total_unrealised:.2f}",
            win_count,
            loss_count,
            f"{win_rate:.2f}",
            f"{best_trade:.2f}",
            f"{worst_trade:.2f}",
            len(today_trades)
        ]
        
        # Check if today's row already exists
        def get_dashboard_rows():
            return dashboard_worksheet.get_all_records()
        
        dashboard_rows = _retry_gspread_call(get_dashboard_rows) or []
        
        today_row_exists = any(r.get("Date") == today for r in dashboard_rows)
        
        if today_row_exists:
            # Update existing row
            headers = dashboard_worksheet.row_values(1)
            for row_idx, dashboard_row in enumerate(dashboard_rows, start=2):
                if dashboard_row.get("Date") == today:
                    cells = []
                    for col_idx, header in enumerate(headers, start=1):
                        value = row[col_idx - 1] if col_idx <= len(row) else ""
                        cells.append(gspread.Cell(row_idx, col_idx, value))
                    
                    def batch_update():
                        dashboard_worksheet.update_cells(cells)
                    
                    _retry_gspread_call(batch_update)
                    logger.info(f"✓ Updated P&L Dashboard for {today}")
                    break
        else:
            # Append new row
            def append_row():
                dashboard_worksheet.append_row(row)
            
            _retry_gspread_call(append_row)
            logger.info(f"✓ Appended P&L to Dashboard for {today}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error updating P&L Dashboard: {e}")
        return False


# ============================================================================
# NEWS LOGGING
# ============================================================================

def log_news(
    symbol: str,
    headline: str,
    sentiment_score: float,
    sentiment_label: str,
    source: str
) -> bool:
    """
    Log news to "News Feed" sheet.
    
    Parameters
    ----------
    symbol : str
        Stock symbol
    headline : str
        News headline
    sentiment_score : float
        Sentiment score (-1 to 1)
    sentiment_label : str
        "POSITIVE", "NEUTRAL", or "NEGATIVE"
    source : str
        News source
    
    Returns
    -------
    bool
        True if logged successfully
    """
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        row = [
            now,
            symbol,
            headline,
            f"{sentiment_score:.4f}",
            sentiment_label,
            source,
            "NO"  # AlertTriggered
        ]
        
        worksheet = get_worksheet(SHEET_NEWS_FEED)
        
        def append_row():
            worksheet.append_row(row)
        
        _retry_gspread_call(append_row)
        logger.info(f"✓ Logged news for {symbol}")
        return True
    
    except Exception as e:
        logger.error(f"Error logging news: {e}")
        return False


# ============================================================================
# TEST/MAIN
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Sheets Live module...")
    print(f"  Config Path: {GOOGLE_CREDENTIALS_PATH}")
    print(f"  Sheets ID: {GOOGLE_SHEETS_ID}")
    
    if not GOOGLE_SHEETS_ID:
        print("✗ GOOGLE_SHEETS_ID not configured")
    elif not os.path.exists(GOOGLE_CREDENTIALS_PATH):
        print(f"✗ Credentials file not found: {GOOGLE_CREDENTIALS_PATH}")
    else:
        try:
            setup_sheet_headers()
            print("✓ Headers setup complete")
        except Exception as e:
            print(f"✗ Error: {e}")
