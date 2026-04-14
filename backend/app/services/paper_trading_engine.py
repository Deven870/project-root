"""
╔════════════════════════════════════════════════════════════════════════════╗
║              PAPER TRADING ENGINE - Simulated Account System               ║
║                   Tracks all trades, positions, and P&L                    ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import os
from dataclasses import dataclass, asdict
import pandas as pd

logger = logging.getLogger(__name__)


class TradeStatus(Enum):
    """Trade lifecycle statuses"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    SL_HIT = "SL_HIT"
    TARGET_HIT = "TARGET_HIT"


@dataclass
class Trade:
    """Single trade record"""
    trade_id: str
    timestamp_entry: str  # ISO format
    stock: str
    entry_price: float
    entry_quantity: int
    entry_capital: float
    target_price: float
    stop_loss: float
    signal_confidence: float
    status: str = "OPEN"
    exit_price: Optional[float] = None
    exit_quantity: Optional[int] = None
    timestamp_exit: Optional[str] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    exit_reason: Optional[str] = None  # TARGET_HIT, SL_HIT, MANUAL, etc.
    
    def to_dict(self):
        return asdict(self)


class PaperTradingAccount:
    """Simulated trading account"""
    
    def __init__(self, initial_capital: float, account_name: str = "Paper Trading"):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.account_name = account_name
        self.trades: List[Trade] = []
        self.open_positions: Dict[str, Trade] = {}  # {stock: Trade}
        self.trade_counter = 0
        self.account_created = datetime.now().isoformat()
        
        logger.info(f"✅ Paper Trading Account Created")
        logger.info(f"   Account: {account_name}")
        logger.info(f"   Initial Capital: ₹{initial_capital:,.0f}")
    
    def place_trade(
        self,
        stock: str,
        entry_price: float,
        target_price: float,
        stop_loss: float,
        quantity: int,
        signal_confidence: float,
        entry_capital: float
    ) -> Tuple[bool, str, Optional[Trade]]:
        """
        Place a new trade
        
        Returns: (success, message, trade_object)
        """
        # Check if already have position
        if stock in self.open_positions:
            return False, f"❌ {stock} already has open position", None
        
        # Check if enough capital
        if entry_capital > self.current_capital:
            return False, f"❌ Insufficient capital. Need ₹{entry_capital:,.0f}, Have ₹{self.current_capital:,.0f}", None
        
        # Create trade
        self.trade_counter += 1
        trade_id = f"TR{datetime.now().strftime('%Y%m%d%H%M%S')}{self.trade_counter:03d}"
        
        trade = Trade(
            trade_id=trade_id,
            timestamp_entry=datetime.now().isoformat(),
            stock=stock,
            entry_price=entry_price,
            entry_quantity=quantity,
            entry_capital=entry_capital,
            target_price=target_price,
            stop_loss=stop_loss,
            signal_confidence=signal_confidence,
            status="OPEN"
        )
        
        # Update account
        self.current_capital -= entry_capital
        self.open_positions[stock] = trade
        self.trades.append(trade)
        
        logger.info(f"✅ TRADE OPENED: {stock}")
        logger.info(f"   Entry: ₹{entry_price:.2f} (Qty: {quantity})")
        logger.info(f"   Target: ₹{target_price:.2f} | SL: ₹{stop_loss:.2f}")
        logger.info(f"   Capital Used: ₹{entry_capital:,.0f}")
        logger.info(f"   Remaining: ₹{self.current_capital:,.0f}")
        
        return True, f"✅ {stock} position opened", trade
    
    def close_trade(
        self,
        stock: str,
        exit_price: float,
        exit_reason: str = "MANUAL"
    ) -> Tuple[bool, str, Optional[float]]:
        """
        Close an open trade
        
        Returns: (success, message, pnl)
        """
        if stock not in self.open_positions:
            return False, f"❌ No open position for {stock}", None
        
        trade = self.open_positions[stock]
        
        # Calculate P&L
        pnl = (exit_price - trade.entry_price) * trade.entry_quantity
        pnl_percent = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        
        # Update trade
        trade.exit_price = exit_price
        trade.exit_quantity = trade.entry_quantity
        trade.timestamp_exit = datetime.now().isoformat()
        trade.pnl = pnl
        trade.pnl_percent = pnl_percent
        trade.exit_reason = exit_reason
        trade.status = "CLOSED"
        
        # Update account
        self.current_capital += trade.entry_capital + pnl
        del self.open_positions[stock]
        
        result_text = "✅ PROFIT" if pnl > 0 else "❌ LOSS"
        logger.info(f"{result_text} TRADE CLOSED: {stock}")
        logger.info(f"   Exit: ₹{exit_price:.2f} | Reason: {exit_reason}")
        logger.info(f"   P&L: ₹{pnl:,.0f} ({pnl_percent:+.2f}%)")
        logger.info(f"   Account Balance: ₹{self.current_capital:,.0f}")
        
        return True, f"{result_text} {stock} closed", pnl
    
    def check_exit_conditions(self, stock: str, current_price: float) -> Optional[Tuple[str, float]]:
        """
        Check if trade should auto-exit (target hit or SL hit)
        
        Returns: (exit_reason, exit_price) or None
        """
        if stock not in self.open_positions:
            return None
        
        trade = self.open_positions[stock]
        
        # Check target hit
        if current_price >= trade.target_price:
            return ("TARGET_HIT", trade.target_price)
        
        # Check stop loss hit
        if current_price <= trade.stop_loss:
            return ("SL_HIT", trade.stop_loss)
        
        return None
    
    def get_account_stats(self) -> Dict:
        """Get comprehensive account statistics"""
        total_trades = len(self.trades)
        closed_trades = [t for t in self.trades if t.status == "CLOSED"]
        open_trades = len(self.open_positions)
        
        winning_trades = [t for t in closed_trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl and t.pnl < 0]
        
        total_pnl = sum(t.pnl for t in closed_trades if t.pnl)
        
        return {
            "account_name": self.account_name,
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "account_balance": self.current_capital,
            "total_pnl": total_pnl,
            "pnl_percent": (total_pnl / self.initial_capital) * 100,
            "total_trades": total_trades,
            "closed_trades": len(closed_trades),
            "open_trades": open_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0,
            "avg_win": (sum(t.pnl for t in winning_trades) / len(winning_trades)) if winning_trades else 0,
            "avg_loss": (sum(t.pnl for t in losing_trades) / len(losing_trades)) if losing_trades else 0,
            "open_positions": list(self.open_positions.keys()),
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        positions = []
        for stock, trade in self.open_positions.items():
            positions.append({
                "stock": stock,
                "entry_price": trade.entry_price,
                "target_price": trade.target_price,
                "stop_loss": trade.stop_loss,
                "quantity": trade.entry_quantity,
                "entry_capital": trade.entry_capital,
                "signal_confidence": trade.signal_confidence,
                "status": trade.status,
            })
        return positions
    
    def get_trade_history(self) -> List[Dict]:
        """Get all trades"""
        return [trade.to_dict() for trade in self.trades]
    
    def export_trades_csv(self, filepath: str = None) -> str:
        """Export trades to CSV"""
        if not filepath:
            filepath = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = pd.DataFrame([trade.to_dict() for trade in self.trades])
        df.to_csv(filepath, index=False)
        
        logger.info(f"📊 Trades exported to: {filepath}")
        return filepath
    
    def export_stats_json(self, filepath: str = None) -> str:
        """Export account stats to JSON"""
        if not filepath:
            filepath = f"account_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        stats = self.get_account_stats()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"📊 Account stats exported to: {filepath}")
        return filepath


# Global paper trading account instance
_account_instance: Optional[PaperTradingAccount] = None


def create_paper_trading_account(initial_capital: float, name: str = "Paper Trading") -> PaperTradingAccount:
    """Create global paper trading account"""
    global _account_instance
    _account_instance = PaperTradingAccount(initial_capital, name)
    return _account_instance


def get_paper_trading_account() -> PaperTradingAccount:
    """Get global paper trading account"""
    global _account_instance
    if _account_instance is None:
        _account_instance = PaperTradingAccount(250000)
    return _account_instance
