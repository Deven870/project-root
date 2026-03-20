# modules/pnl_tracker.py
"""
Real-time P&L Tracking System
Tracks open positions, closed trades, portfolio performance, and syncs with storage.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional
from pathlib import Path


class PnLTracker:
    """
    Tracks and manages P&L for trading journal and performance monitoring.
    """
    
    def __init__(self, storage_dir: str = "results/pnl_data", use_sheets: bool = False):
        """
        Initialize P&L Tracker.
        
        Args:
            storage_dir: Directory to save tracking data
            use_sheets: If True, sync with Google Sheets via sheets_tracker
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_sheets = use_sheets
        self.sheets_tracker = None
        if use_sheets:
            try:
                from modules.sheets_tracker import SheetsTracker
                self.sheets_tracker = SheetsTracker()
            except:
                self.use_sheets = False
        
        self.positions_file = self.storage_dir / "open_positions.json"
        self.trades_file = self.storage_dir / "trade_history.json"
        self.performance_file = self.storage_dir / "performance.json"
        
        # Load existing data
        self.open_positions = self._load_json(self.positions_file, [])
        self.trade_history = self._load_json(self.trades_file, [])
        self.performance_data = self._load_json(self.performance_file, {})
    
    
    def _load_json(self, filepath: Path, default):
        """Safely load JSON file."""
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except:
                pass
        return default
    
    
    def _save_json(self, filepath: Path, data):
        """Save data to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving {filepath}: {e}")
    
    
    def add_trade(
        self,
        ticker: str,
        entry_price: float,
        position_size: float,
        entry_trend: str = "Bullish",
        entry_confidence: float = 0.6,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict:
        """
        Log a new open position.
        """
        trade = {
            "id": f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "ticker": ticker,
            "entry_price": float(entry_price),
            "position_size": float(position_size),
            "entry_trend": entry_trend,
            "entry_confidence": float(entry_confidence),
            "stop_loss": float(stop_loss) if stop_loss else None,
            "take_profit": float(take_profit) if take_profit else None,
            "entry_time": datetime.now().isoformat(),
            "exit_price": None,
            "exit_time": None,
            "exit_reason": None,
            "pnl": 0.0,
            "pnl_percent": 0.0,
            "status": "open",
        }
        
        self.open_positions.append(trade)
        self._save_json(self.positions_file, self.open_positions)
        
        # Sync to Sheets if available
        if self.use_sheets and self.sheets_tracker:
            try:
                self.sheets_tracker.log_investment(
                    ticker=ticker,
                    entry_price=entry_price,
                    quantity=position_size,
                    investment_type=entry_trend.lower(),
                    confidence_score=entry_confidence,
                )
            except:
                pass
        
        return trade
    
    
    def update_position_price(self, ticker: str, current_price: float) -> Optional[Dict]:
        """
        Update market price for an open position.
        Returns updated position.
        """
        for pos in self.open_positions:
            if pos["ticker"] == ticker and pos["status"] == "open":
                pos["current_price"] = float(current_price)
                
                # Calculate P&L
                if pos["entry_trend"].lower() == "bullish":
                    pnl = (current_price - pos["entry_price"]) * pos["position_size"]
                else:
                    pnl = (pos["entry_price"] - current_price) * pos["position_size"]
                
                pos["pnl"] = float(pnl)
                entry_capital = pos["entry_price"] * pos["position_size"]
                pos["pnl_percent"] = (pnl / entry_capital * 100) if entry_capital > 0 else 0
                
                self._save_json(self.positions_file, self.open_positions)
                return pos
        
        return None
    
    
    def close_trade(
        self,
        ticker: str,
        exit_price: float,
        exit_reason: str = "manual"
    ) -> Optional[Dict]:
        """
        Close an open position and move to trade history.
        """
        for i, pos in enumerate(self.open_positions):
            if pos["ticker"] == ticker and pos["status"] == "open":
                pos["exit_price"] = float(exit_price)
                pos["exit_time"] = datetime.now().isoformat()
                pos["exit_reason"] = exit_reason
                pos["status"] = "closed"
                
                # Final P&L
                if pos["entry_trend"].lower() == "bullish":
                    pnl = (exit_price - pos["entry_price"]) * pos["position_size"]
                else:
                    pnl = (pos["entry_price"] - exit_price) * pos["position_size"]
                
                pos["pnl"] = float(pnl)
                entry_capital = pos["entry_price"] * pos["position_size"]
                pos["pnl_percent"] = (pnl / entry_capital * 100) if entry_capital > 0 else 0
                
                # Move to history
                self.trade_history.append(pos)
                self.open_positions.pop(i)
                
                self._save_json(self.positions_file, self.open_positions)
                self._save_json(self.trades_file, self.trade_history)
                
                return pos
        
        return None
    
    
    def get_open_positions_summary(self) -> pd.DataFrame:
        """Get DataFrame of all open positions with current P&L."""
        if not self.open_positions:
            return pd.DataFrame()
        
        data = []
        for pos in self.open_positions:
            data.append({
                "Ticker": pos["ticker"],
                "Entry": pos["entry_price"],
                "Current": pos.get("current_price", pos["entry_price"]),
                "Size": pos["position_size"],
                "Trend": pos["entry_trend"],
                "Confidence": pos["entry_confidence"],
                "PnL": pos["pnl"],
                "PnL %": pos["pnl_percent"],
                "SL": pos["stop_loss"],
                "TP": pos["take_profit"],
                "Entry Time": pos["entry_time"],
            })
        
        return pd.DataFrame(data)
    
    
    def get_trade_history_summary(self) -> pd.DataFrame:
        """Get DataFrame of closed trades."""
        if not self.trade_history:
            return pd.DataFrame()
        
        data = []
        for trade in self.trade_history:
            data.append({
                "Ticker": trade["ticker"],
                "Entry": trade["entry_price"],
                "Exit": trade["exit_price"],
                "Size": trade["position_size"],
                "Trend": trade["entry_trend"],
                "Confidence": trade["entry_confidence"],
                "Exit Reason": trade["exit_reason"],
                "PnL": trade["pnl"],
                "PnL %": trade["pnl_percent"],
                "Entry Time": trade["entry_time"],
                "Exit Time": trade["exit_time"],
            })
        
        return pd.DataFrame(data)
    
    
    def calculate_performance(self, initial_capital: float = 100000) -> Dict:
        """
        Calculate comprehensive performance metrics.
        """
        # Closed trades metrics
        closed_pnls = [t["pnl"] for t in self.trade_history]
        
        total_closed_pnl = sum(closed_pnls)
        win_count = sum(1 for p in closed_pnls if p > 0)
        loss_count = sum(1 for p in closed_pnls if p < 0)
        total_closed_trades = len(closed_pnls)
        
        win_rate = (win_count / total_closed_trades * 100) if total_closed_trades > 0 else 0
        
        # Average win/loss
        wins = [p for p in closed_pnls if p > 0]
        losses = [p for p in closed_pnls if p < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        max_loss = min(losses) if losses else 0
        
        # Payoff ratio
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Open positions P&L
        open_pnl = sum(p["pnl"] for p in self.open_positions)
        
        # Total return
        total_pnl = total_closed_pnl + open_pnl
        total_return_pct = (total_pnl / initial_capital) * 100
        
        # Drawdown
        cumulative_pnl = 0
        peak_pnl = 0
        max_dd = 0
        
        for pnl in closed_pnls:
            cumulative_pnl += pnl
            peak_pnl = max(peak_pnl, cumulative_pnl)
            dd = peak_pnl - cumulative_pnl
            max_dd = max(max_dd, dd)
        
        max_dd_pct = (max_dd / initial_capital * 100) if initial_capital > 0 else 0
        
        # Sharpe ratio
        if len(closed_pnls) > 1:
            sharpe = (np.mean(closed_pnls) / np.std(closed_pnls)) * np.sqrt(252) if np.std(closed_pnls) > 0 else 0
        else:
            sharpe = 0
        
        metrics = {
            "total_closed_trades": total_closed_trades,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate_percent": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "max_loss": round(max_loss, 2),
            "payoff_ratio": round(payoff_ratio, 2),
            "closed_pnl": round(total_closed_pnl, 2),
            "open_pnl": round(open_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_percent": round(total_return_pct, 2),
            "max_drawdown_percent": round(max_dd_pct, 2),
            "sharpe_ratio": round(sharpe, 2),
            "open_positions": len(self.open_positions),
            "timestamp": datetime.now().isoformat(),
        }
        
        self.performance_data = metrics
        self._save_json(self.performance_file, metrics)
        
        return metrics
    
    
    def get_daily_summary(self) -> Dict:
        """
        Get today's trading summary.
        """
        today = datetime.now().date()
        
        today_trades = [
            t for t in self.trade_history
            if pd.to_datetime(t["exit_time"]).date() == today
        ]
        
        today_pnl = sum(t["pnl"] for t in today_trades)
        today_trades_count = len(today_trades)
        today_wins = sum(1 for t in today_trades if t["pnl"] > 0)
        
        return {
            "date": str(today),
            "trades": today_trades_count,
            "wins": today_wins,
            "losses": today_trades_count - today_wins,
            "pnl": round(today_pnl, 2),
            "trades": today_trades,
        }
    
    
    def export_to_csv(self) -> Tuple[str, str]:
        """
        Export trade history and positions to CSV.
        Returns (positions_file, history_file)
        """
        positions_df = self.get_open_positions_summary()
        history_df = self.get_trade_history_summary()
        
        positions_path = self.storage_dir / "open_positions.csv"
        history_path = self.storage_dir / "trade_history.csv"
        
        positions_df.to_csv(positions_path, index=False)
        history_df.to_csv(history_path, index=False)
        
        return str(positions_path), str(history_path)
