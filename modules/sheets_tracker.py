# modules/sheets_tracker.py
"""
Google Sheets Integration for Real-Time Portfolio & Search Tracking
Allows logging of stock searches, investments, and daily analytics to Google Sheets.
"""

import gspread
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from google.auth import default
import pandas as pd
from datetime import datetime, timedelta
import json
import os

# Google Sheets API scope
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

class SheetsTracker:
    def __init__(self, sheet_url=None, credentials_path=None):
        """
        Initialize Sheets tracker.
        
        Args:
            sheet_url: URL of the Google Sheet (optional, will create if not provided)
            credentials_path: Path to Google service account JSON (optional, will prompt if not provided)
        """
        self.sheet = None
        self.sheet_url = sheet_url
        self.credentials_path = credentials_path or os.path.join(
            os.path.dirname(__file__), "..", "google_credentials.json"
        )
        self.client = None
        self.sheets_dict = {}
        
    def authenticate(self):
        """Authenticate with Google Sheets API."""
        try:
            if os.path.exists(self.credentials_path):
                self.client = gspread.service_account(filename=self.credentials_path)
                return True
            else:
                print(f"⚠️ Credentials not found at {self.credentials_path}")
                print("To set up Google Sheets integration:")
                print("1. Go to https://console.cloud.google.com/")
                print("2. Create a service account and download JSON key")
                print(f"3. Save it as: {self.credentials_path}")
                return False
        except Exception as e:
            print(f"Authentication error: {e}")
            return False
    
    def create_sheet(self, sheet_name="Investment Tracker"):
        """Create a new Google Sheet and return the URL."""
        try:
            if not self.client:
                if not self.authenticate():
                    return None
            
            # Create a new spreadsheet
            sheet = self.client.create(sheet_name)
            self.sheet = sheet
            self.sheet_url = sheet.url
            
            # Initialize worksheets
            self._init_worksheets()
            
            print(f"✅ Sheet created: {self.sheet_url}")
            return self.sheet_url
        except Exception as e:
            print(f"Error creating sheet: {e}")
            return None
    
    def open_sheet(self, sheet_url):
        """Open an existing Google Sheet by URL."""
        try:
            if not self.client:
                if not self.authenticate():
                    return False
            
            # Extract sheet ID from URL
            if "docs.google.com/spreadsheets" in sheet_url:
                # Extract ID from URL: https://docs.google.com/spreadsheets/d/{ID}/edit#gid=0
                parts = sheet_url.split('/d/')
                if len(parts) > 1:
                    sheet_id = parts[1].split('/')[0]
                    self.sheet = self.client.open_by_key(sheet_id)
                    self.sheet_url = sheet_url
                    self._ensure_worksheets()
                    return True
            else:
                # Try as sheet name
                self.sheet = self.client.open(sheet_url)
                self.sheet_url = self.sheet.url
                self._ensure_worksheets()
                return True
        except Exception as e:
            print(f"Error opening sheet: {e}")
            return False
    
    def _init_worksheets(self):
        """Initialize worksheet structure."""
        try:
            # Delete default "Sheet1" if exists
            try:
                default_sheet = self.sheet.worksheet("Sheet1")
                self.sheet.del_worksheet(default_sheet)
            except:
                pass
            
            # Create worksheets
            worksheets_config = {
                "Searches": ["Date", "Symbol", "Trend", "Confidence (%)", "Current Price (₹)", "Predicted Price (₹)", "Expected Return (%)", "Sentiment"],
                "Investments": ["Date", "Symbol", "Investment Amount (₹)", "Entry Price (₹)", "Current Price (₹)", "P&L (₹)", "Return (%)", "Horizon", "Status"],
                "Daily Analysis": ["Date", "Total Invested (₹)", "Current Value (₹)", "Total P&L (₹)", "Overall Return (%)", "Best Performer", "Worst Performer", "Notes"],
                "Portfolio": ["Symbol", "Shares", "Entry Price", "Current Price", "Investment (₹)", "Current Value (₹)", "P&L (₹)", "Return (%)", "Last Updated"],
            }
            
            for ws_name, headers in worksheets_config.items():
                try:
                    ws = self.sheet.worksheet(ws_name)
                except:
                    ws = self.sheet.add_worksheet(title=ws_name, rows=1000, cols=len(headers))
                
                # Add headers
                ws.clear()
                ws.append_row(headers)
                self.sheets_dict[ws_name] = ws
        except Exception as e:
            print(f"Error initializing worksheets: {e}")
    
    def _ensure_worksheets(self):
        """Ensure required worksheets exist."""
        try:
            existing = [ws.title for ws in self.sheet.worksheets()]
            required = ["Searches", "Investments", "Daily Analysis", "Portfolio"]
            
            for ws_name in required:
                if ws_name not in existing:
                    # Create if missing
                    if ws_name == "Searches":
                        headers = ["Date", "Symbol", "Trend", "Confidence (%)", "Current Price (₹)", "Predicted Price (₹)", "Expected Return (%)", "Sentiment"]
                    elif ws_name == "Investments":
                        headers = ["Date", "Symbol", "Investment Amount (₹)", "Entry Price (₹)", "Current Price (₹)", "P&L (₹)", "Return (%)", "Horizon", "Status"]
                    elif ws_name == "Daily Analysis":
                        headers = ["Date", "Total Invested (₹)", "Current Value (₹)", "Total P&L (₹)", "Overall Return (%)", "Best Performer", "Worst Performer", "Notes"]
                    else:  # Portfolio
                        headers = ["Symbol", "Shares", "Entry Price", "Current Price", "Investment (₹)", "Current Value (₹)", "P&L (₹)", "Return (%)", "Last Updated"]
                    
                    ws = self.sheet.add_worksheet(title=ws_name, rows=1000, cols=len(headers))
                    ws.append_row(headers)
            
            # Load worksheets
            for ws in self.sheet.worksheets():
                self.sheets_dict[ws.title] = ws
        except Exception as e:
            print(f"Error ensuring worksheets: {e}")
    
    def log_search(self, symbol, trend, confidence, current_price, predicted_price, expected_return, sentiment):
        """Log a stock search to 'Searches' sheet."""
        try:
            if "Searches" not in self.sheets_dict:
                return False
            
            ws = self.sheets_dict["Searches"]
            sentiment_str = f"Pos: {sentiment.get('positive', 0):.1%} | Neu: {sentiment.get('neutral', 0):.1%} | Neg: {sentiment.get('negative', 0):.1%}"
            
            row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                symbol,
                trend,
                f"{float(confidence)*100:.2f}",
                f"{float(current_price):.2f}",
                f"{float(predicted_price):.2f}" if predicted_price else "N/A",
                f"{float(expected_return):.2f}" if expected_return else "N/A",
                sentiment_str,
            ]
            
            ws.append_row(row)
            print(f"✅ Logged search: {symbol}")
            return True
        except Exception as e:
            print(f"Error logging search: {e}")
            return False
    
    def log_investment(self, symbol, investment_amount, entry_price, horizon="Long-Term"):
        """Log an investment to 'Investments' sheet."""
        try:
            if "Investments" not in self.sheets_dict:
                return False
            
            ws = self.sheets_dict["Investments"]
            
            row = [
                datetime.now().strftime("%Y-%m-%d"),
                symbol,
                f"{float(investment_amount):.2f}",
                f"{float(entry_price):.2f}",
                f"{float(entry_price):.2f}",  # Current price = entry price initially
                "0.00",  # P&L
                "0.00",  # Return %
                horizon,
                "Open",
            ]
            
            ws.append_row(row)
            print(f"✅ Logged investment: {symbol} ₹{investment_amount}")
            return True
        except Exception as e:
            print(f"Error logging investment: {e}")
            return False
    
    def log_daily_analysis(self, total_invested, current_value, best, worst, notes=""):
        """Log daily portfolio analysis."""
        try:
            if "Daily Analysis" not in self.sheets_dict:
                return False
            
            ws = self.sheets_dict["Daily Analysis"]
            pnl = current_value - total_invested
            return_pct = (pnl / total_invested * 100) if total_invested > 0 else 0
            
            row = [
                datetime.now().strftime("%Y-%m-%d"),
                f"{float(total_invested):.2f}",
                f"{float(current_value):.2f}",
                f"{float(pnl):.2f}",
                f"{float(return_pct):.2f}",
                best or "N/A",
                worst or "N/A",
                notes or "",
            ]
            
            ws.append_row(row)
            print(f"✅ Logged daily analysis: P&L ₹{pnl:.2f} ({return_pct:+.2f}%)")
            return True
        except Exception as e:
            print(f"Error logging daily analysis: {e}")
            return False
    
    def update_portfolio(self, symbol, shares, current_price, investment_amt):
        """Update or add a stock to portfolio sheet."""
        try:
            if "Portfolio" not in self.sheets_dict:
                return False
            
            ws = self.sheets_dict["Portfolio"]
            all_rows = ws.get_all_values()
            
            # Check if symbol exists
            symbol_row = None
            for i, row in enumerate(all_rows[1:], start=2):  # Skip header
                if row and row[0] == symbol:
                    symbol_row = i
                    break
            
            current_value = shares * current_price
            pnl = current_value - investment_amt
            return_pct = (pnl / investment_amt * 100) if investment_amt > 0 else 0
            entry_price = investment_amt / shares if shares > 0 else current_price
            
            new_row = [
                symbol,
                f"{float(shares):.2f}",
                f"{float(entry_price):.2f}",
                f"{float(current_price):.2f}",
                f"{float(investment_amt):.2f}",
                f"{float(current_value):.2f}",
                f"{float(pnl):.2f}",
                f"{float(return_pct):.2f}",
                datetime.now().strftime("%Y-%m-%d %H:%M"),
            ]
            
            if symbol_row:
                # Update existing row
                for col_idx, val in enumerate(new_row, start=1):
                    ws.update_cell(symbol_row, col_idx, val)
            else:
                # Add new row
                ws.append_row(new_row)
            
            print(f"✅ Updated portfolio: {symbol}")
            return True
        except Exception as e:
            print(f"Error updating portfolio: {e}")
            return False
    
    def get_searches(self, days=7):
        """Get search history from last N days."""
        try:
            if "Searches" not in self.sheets_dict:
                return pd.DataFrame()
            
            ws = self.sheets_dict["Searches"]
            all_rows = ws.get_all_values()
            
            if len(all_rows) <= 1:
                return pd.DataFrame()
            
            df = pd.DataFrame(all_rows[1:], columns=all_rows[0])
            if df.empty:
                return df
            
            # Parse dates and filter
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            cutoff = datetime.now() - timedelta(days=days)
            df = df[df['Date'] >= cutoff].sort_values('Date', ascending=False)
            
            return df
        except Exception as e:
            print(f"Error fetching searches: {e}")
            return pd.DataFrame()
    
    def get_investments(self):
        """Get all investments."""
        try:
            if "Investments" not in self.sheets_dict:
                return pd.DataFrame()
            
            ws = self.sheets_dict["Investments"]
            all_rows = ws.get_all_values()
            
            if len(all_rows) <= 1:
                return pd.DataFrame()
            
            df = pd.DataFrame(all_rows[1:], columns=all_rows[0])
            return df
        except Exception as e:
            print(f"Error fetching investments: {e}")
            return pd.DataFrame()
    
    def get_portfolio(self):
        """Get current portfolio."""
        try:
            if "Portfolio" not in self.sheets_dict:
                return pd.DataFrame()
            
            ws = self.sheets_dict["Portfolio"]
            all_rows = ws.get_all_values()
            
            if len(all_rows) <= 1:
                return pd.DataFrame()
            
            df = pd.DataFrame(all_rows[1:], columns=all_rows[0])
            return df
        except Exception as e:
            print(f"Error fetching portfolio: {e}")
            return pd.DataFrame()
    
    def get_daily_analysis(self, days=30):
        """Get daily analysis history."""
        try:
            if "Daily Analysis" not in self.sheets_dict:
                return pd.DataFrame()
            
            ws = self.sheets_dict["Daily Analysis"]
            all_rows = ws.get_all_values()
            
            if len(all_rows) <= 1:
                return pd.DataFrame()
            
            df = pd.DataFrame(all_rows[1:], columns=all_rows[0])
            if df.empty:
                return df
            
            # Parse dates and filter
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
            cutoff = datetime.now() - timedelta(days=days)
            df = df[df['Date'] >= cutoff].sort_values('Date', ascending=False)
            
            return df
        except Exception as e:
            print(f"Error fetching daily analysis: {e}")
            return pd.DataFrame()


def get_tracker(sheet_url=None):
    """Convenience function to get or create a tracker instance."""
    tracker = SheetsTracker(sheet_url=sheet_url)
    if tracker.authenticate():
        if sheet_url:
            tracker.open_sheet(sheet_url)
        return tracker
    return None
