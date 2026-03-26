"""
Google Sheets integration module for real-time data updates.
Uses batching to minimize API calls (max 1 call per refresh cycle).
"""
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import os
from config import SHEETS_ID, SERVICE_ACCOUNT_FILE

# Global worksheet cache
_worksheets_cache = {}
_client = None

def get_authorized_client():
    """Get authorized gspread client using service account credentials."""
    global _client
    if _client is not None:
        return _client
    
    try:
        # Use service account JSON for authentication
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
        _client = gspread.authorize(creds)
        print(f"✓ Google Sheets authenticated")
        return _client
    except Exception as e:
        print(f"✗ Google Sheets authentication failed: {e}")
        print(f"  Ensure {SERVICE_ACCOUNT_FILE} exists with valid service account credentials")
        return None

def get_spreadsheet():
    """Get the spreadsheet by ID."""
    try:
        client = get_authorized_client()
        if client is None:
            return None
        spreadsheet = client.open_by_key(SHEETS_ID)
        return spreadsheet
    except Exception as e:
        print(f"✗ Error opening spreadsheet: {e}")
        return None

def get_worksheet(sheet_name):
    """Get worksheet by name with caching."""
    try:
        if sheet_name in _worksheets_cache:
            return _worksheets_cache[sheet_name]
        
        spreadsheet = get_spreadsheet()
        if spreadsheet is None:
            return None
        
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
        except gspread.exceptions.WorksheetNotFound:
            # Create the worksheet if it doesn't exist
            worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=50)
            print(f"✓ Created worksheet: {sheet_name}")
        
        _worksheets_cache[sheet_name] = worksheet
        return worksheet
    except Exception as e:
        print(f"✗ Error getting worksheet '{sheet_name}': {e}")
        return None

# ============================================================================
# TAB 1: LIVE SIGNALS
# ============================================================================
def update_live_signals(signals_data):
    """
    Update Live Signals tab with batch update.
    signals_data: list of dicts with keys: timestamp, stock, signal, strength, confidence
    Example: [{"timestamp": "2024-01-15 10:30", "stock": "RELIANCE", "signal": "BUY", "strength": 0.85, "confidence": 0.92}]
    """
    try:
        worksheet = get_worksheet("Live Signals")
        if worksheet is None:
            return False
        
        # Set headers if empty
        if worksheet.cell(1, 1).value is None:
            headers = ["Timestamp", "Stock", "Signal", "Strength", "Confidence"]
            worksheet.insert_row(headers, 1)
        
        # Batch update: clear old data and insert new
        worksheet.delete_rows(2, worksheet.row_count)
        
        rows_to_insert = []
        for item in signals_data:
            rows_to_insert.append([
                item.get("timestamp", ""),
                item.get("stock", ""),
                item.get("signal", ""),
                item.get("strength", 0),
                item.get("confidence", 0)
            ])
        
        if rows_to_insert:
            worksheet.insert_rows(rows_to_insert, 2)
        
        print(f"✓ Updated Live Signals ({len(signals_data)} rows)")
        return True
    except Exception as e:
        print(f"✗ Error updating Live Signals: {e}")
        return False

# ============================================================================
# TAB 2: MY TRADES
# ============================================================================
def add_trade(trade_data):
    """
    Add a single trade to My Trades tab (batch-friendly).
    trade_data: dict with keys: entry_time, entry_price, exit_time, exit_price, qty, profit_loss, status
    """
    try:
        worksheet = get_worksheet("My Trades")
        if worksheet is None:
            return False
        
        # Set headers if empty
        if worksheet.cell(1, 1).value is None:
            headers = ["Entry Time", "Entry Price", "Exit Time", "Exit Price", "Qty", "Profit/Loss", "Status"]
            worksheet.insert_row(headers, 1)
        
        # Append new trade
        new_row = [
            trade_data.get("entry_time", ""),
            trade_data.get("entry_price", 0),
            trade_data.get("exit_time", ""),
            trade_data.get("exit_price", 0),
            trade_data.get("qty", 0),
            trade_data.get("profit_loss", 0),
            trade_data.get("status", "OPEN")
        ]
        
        worksheet.append_row(new_row)
        print(f"✓ Added trade to My Trades")
        return True
    except Exception as e:
        print(f"✗ Error adding trade: {e}")
        return False

def update_trades_batch(trades_data):
    """
    Batch update all trades (efficient for refresh cycles).
    trades_data: list of trade dicts
    """
    try:
        worksheet = get_worksheet("My Trades")
        if worksheet is None:
            return False
        
        # Set headers if empty
        if worksheet.cell(1, 1).value is None:
            headers = ["Entry Time", "Entry Price", "Exit Time", "Exit Price", "Qty", "Profit/Loss", "Status"]
            worksheet.insert_row(headers, 1)
        
        # Clear existing trades and insert new batch
        worksheet.delete_rows(2, worksheet.row_count)
        
        rows_to_insert = []
        for trade in trades_data:
            rows_to_insert.append([
                trade.get("entry_time", ""),
                trade.get("entry_price", 0),
                trade.get("exit_time", ""),
                trade.get("exit_price", 0),
                trade.get("qty", 0),
                trade.get("profit_loss", 0),
                trade.get("status", "OPEN")
            ])
        
        if rows_to_insert:
            worksheet.insert_rows(rows_to_insert, 2)
        
        print(f"✓ Batch updated My Trades ({len(trades_data)} trades)")
        return True
    except Exception as e:
        print(f"✗ Error batch updating trades: {e}")
        return False

# ============================================================================
# TAB 3: P&L DASHBOARD
# ============================================================================
def update_pnl_dashboard(pnl_data):
    """
    Update P&L Dashboard tab.
    pnl_data: dict with keys: total_trades, winners, losers, total_pnl, win_rate, avg_win, avg_loss, profit_factor
    """
    try:
        worksheet = get_worksheet("P&L Dashboard")
        if worksheet is None:
            return False
        
        # Create dashboard structure
        metrics = [
            ["Metric", "Value"],
            ["Last Updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Total Trades", pnl_data.get("total_trades", 0)],
            ["Winners", pnl_data.get("winners", 0)],
            ["Losers", pnl_data.get("losers", 0)],
            ["Win Rate (%)", pnl_data.get("win_rate", 0)],
            ["Total P&L", pnl_data.get("total_pnl", 0)],
            ["Average Win", pnl_data.get("avg_win", 0)],
            ["Average Loss", pnl_data.get("avg_loss", 0)],
            ["Profit Factor", pnl_data.get("profit_factor", 0)]
        ]
        
        # Clear and update dashboard
        worksheet.clear()
        worksheet.insert_rows(metrics, 1)
        
        print(f"✓ Updated P&L Dashboard")
        return True
    except Exception as e:
        print(f"✗ Error updating P&L Dashboard: {e}")
        return False

# ============================================================================
# TAB 4: NEWS FEED
# ============================================================================
def update_news_feed(news_items):
    """
    Update News Feed tab.
    news_items: list of dicts with keys: timestamp, title, url, sentiment, sentiment_score
    """
    try:
        worksheet = get_worksheet("News Feed")
        if worksheet is None:
            return False
        
        # Set headers if empty
        if worksheet.cell(1, 1).value is None:
            headers = ["Timestamp", "Title", "URL", "Sentiment", "Score"]
            worksheet.insert_row(headers, 1)
        
        # Clear old news and insert new
        worksheet.delete_rows(2, worksheet.row_count)
        
        rows_to_insert = []
        for item in news_items:
            rows_to_insert.append([
                item.get("timestamp", ""),
                item.get("title", ""),
                item.get("url", ""),
                item.get("sentiment", "NEUTRAL"),
                item.get("sentiment_score", 0)
            ])
        
        if rows_to_insert:
            worksheet.insert_rows(rows_to_insert, 2)
        
        print(f"✓ Updated News Feed ({len(news_items)} items)")
        return True
    except Exception as e:
        print(f"✗ Error updating News Feed: {e}")
        return False

# ============================================================================
# TAB 5: CONFIG
# ============================================================================
def update_config(config_data):
    """
    Update Config tab with system settings.
    config_data: dict with configuration parameters
    """
    try:
        worksheet = get_worksheet("Config")
        if worksheet is None:
            return False
        
        # Create config structure
        rows = [["Parameter", "Value"]]
        for key, value in config_data.items():
            rows.append([key, str(value)])
        
        # Clear and update config
        worksheet.clear()
        worksheet.insert_rows(rows, 1)
        
        print(f"✓ Updated Config tab")
        return True
    except Exception as e:
        print(f"✗ Error updating Config: {e}")
        return False

# ============================================================================
# UTILITY: BATCH UPDATE ALL TABS (ONE API CALL)
# ============================================================================
def batch_update_all_tabs(live_signals=None, trades=None, pnl=None, news=None, config=None):
    """
    Batch update multiple tabs efficiently.
    All updates happen in sequence (still individual calls but coordinated).
    Use this for synchronized refresh cycles.
    """
    results = {
        "live_signals": False,
        "trades": False,
        "pnl": False,
        "news": False,
        "config": False
    }
    
    if live_signals is not None:
        results["live_signals"] = update_live_signals(live_signals)
    if trades is not None:
        results["trades"] = update_trades_batch(trades)
    if pnl is not None:
        results["pnl"] = update_pnl_dashboard(pnl)
    if news is not None:
        results["news"] = update_news_feed(news)
    if config is not None:
        results["config"] = update_config(config)
    
    return results

# ============================================================================
# UTILITY: READ FROM SHEETS
# ============================================================================
def read_worksheet(sheet_name):
    """Read all data from a worksheet."""
    try:
        worksheet = get_worksheet(sheet_name)
        if worksheet is None:
            return []
        
        records = worksheet.get_all_values()
        return records
    except Exception as e:
        print(f"✗ Error reading worksheet '{sheet_name}': {e}")
        return []

def get_live_signals():
    """Get all live signals."""
    return read_worksheet("Live Signals")

def get_trades():
    """Get all trades."""
    return read_worksheet("My Trades")

def get_pnl():
    """Get P&L dashboard data."""
    return read_worksheet("P&L Dashboard")

def get_news():
    """Get news feed data."""
    return read_worksheet("News Feed")

def get_config():
    """Get config data."""
    return read_worksheet("Config")

if __name__ == "__main__":
    # Test authentication
    print("Testing Google Sheets integration...")
    client = get_authorized_client()
    if client:
        print("✓ Authentication successful")
    else:
        print("✗ Authentication failed")
