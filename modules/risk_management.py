# modules/risk_management.py
"""
Risk Management Module
Handles position sizing, stop-loss calculations, portfolio constraints,
and trade validation for safe trading execution.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class RiskManager:
    """
    Manages position sizing, stop-loss, profit targets, and portfolio-level risk.
    """
    
    def __init__(
        self,
        account_size: float = 100000,  # Initial capital in Rs
        max_risk_per_trade: float = 0.02,  # 2% risk per trade
        max_position_size: float = 0.1,  # 10% of capital per position
        max_daily_loss: float = 0.05,  # Quit if 5% down in a day
        max_open_positions: int = 5,  # Max concurrent positions
        min_win_rate: float = 0.55,  # Require >55% win rate for profitability
    ):
        """
        Initialize Risk Manager.
        
        Args:
            account_size: Total capital available (in Rs)
            max_risk_per_trade: Risk per trade as % of capital
            max_position_size: Max position size as % of capital
            max_daily_loss: Max acceptable loss per day as % of capital
            max_open_positions: Max concurrent open positions
            min_win_rate: Minimum required accuracy for strategy profitability
        """
        self.account_size = account_size
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_open_positions = max_open_positions
        self.min_win_rate = min_win_rate
        
        self.current_equity = account_size
        self.daily_pnl = 0.0
        self.open_positions = []  # List of active trades
        self.trade_history = []  # Closed trades
    
    
    def calculate_position_size(
        self,
        current_price: float,
        stop_loss_price: float,
        confidence: float = 0.5
    ) -> float:
        """
        Calculate position size using risk-based approach.
        Risk = (current_price - stop_loss_price) * position_size
        
        Args:
            current_price: Entry price
            stop_loss_price: Stop-loss price
            confidence: Model confidence (0-1). Higher confidence → larger position
            
        Returns:
            Position size in units (shares/quantity)
        """
        risk_per_trade = self.current_equity * self.max_risk_per_trade
        price_risk = abs(current_price - stop_loss_price)
        
        if price_risk <= 0:
            return 0
        
        # Base position size
        position_size = risk_per_trade / price_risk
        
        # Adjust for confidence: higher confidence = more aggressive
        position_size *= (0.7 + 0.3 * confidence)  # Scale confidence to 0.7-1.0
        
        # Cap by max position size constraint
        max_capital_per_position = self.current_equity * self.max_position_size
        max_position_size_by_capital = max_capital_per_position / current_price
        
        position_size = min(position_size, max_position_size_by_capital)
        
        return max(0, position_size)
    
    
    def calculate_stop_loss(
        self,
        current_price: float,
        trend: str = "Bullish",
        method: str = "atr",
        atr: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate stop-loss price.
        
        Args:
            current_price: Entry price
            trend: "Bullish" or "Bearish"
            method: "atr" (ATR-based), "percent" (fixed %), or "volatility"
            atr: Average True Range (if method='atr')
            volatility: Daily volatility (if method='volatility')
            
        Returns:
            Stop-loss price
        """
        if trend.lower() == "bullish":
            if method == "atr" and atr:
                # For longs: SL = entry - 2*ATR
                return current_price - (2.0 * atr)
            elif method == "volatility" and volatility:
                # SL = entry - 2*volatility
                return current_price - (2.0 * volatility * current_price)
            else:
                # Default: 3% below entry for bullish
                return current_price * 0.97
        else:  # Bearish
            if method == "atr" and atr:
                # For shorts: SL = entry + 2*ATR
                return current_price + (2.0 * atr)
            elif method == "volatility" and volatility:
                return current_price + (2.0 * volatility * current_price)
            else:
                # Default: 3% above entry for bearish
                return current_price * 1.03
    
    
    def calculate_profit_target(
        self,
        entry_price: float,
        stop_loss_price: float,
        risk_reward_ratio: float = 2.0
    ) -> Tuple[float, float]:
        """
        Calculate profit targets using risk-reward ratio.
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop-loss price
            risk_reward_ratio: TP = Entry + (Entry - SL) * ratio
            
        Returns:
            (take_profit_price, risk_in_rupees)
        """
        risk = abs(entry_price - stop_loss_price)
        reward = risk * risk_reward_ratio
        
        if entry_price > stop_loss_price:  # Long
            tp = entry_price + reward
        else:  # Short
            tp = entry_price - reward
        
        return tp, risk
    
    
    def can_take_trade(
        self,
        position_size: float,
        current_price: float,
        confidence: float,
        model_accuracy: float = 0.58
    ) -> Tuple[bool, str]:
        """
        Validate if a new trade should be taken.
        
        Args:
            position_size: Proposed position size
            current_price: Entry price
            confidence: Model confidence (0-1)
            model_accuracy: Historical model win rate
            
        Returns:
            (can_trade, reason)
        """
        # Check 1: Max open positions
        if len(self.open_positions) >= self.max_open_positions:
            return False, "Max open positions reached"
        
        # Check 2: Daily loss limit
        if self.daily_pnl < -(self.current_equity * self.max_daily_loss):
            return False, "Daily loss limit exceeded - HALT TRADING"
        
        # Check 3: Model accuracy check
        if model_accuracy < self.min_win_rate:
            return False, f"Model accuracy {model_accuracy:.1%} < {self.min_win_rate:.1%} required"
        
        # Check 4: Confidence check
        if confidence < 0.5:
            return False, "Model confidence too low"
        
        # Check 5: Position size reasonable
        position_capital = position_size * current_price
        if position_capital > self.current_equity * self.max_position_size:
            return False, "Position size exceeds max limit"
        
        # Check 6: Sufficient capital
        if position_capital > self.current_equity * 0.95:
            return False, "Insufficient capital for position"
        
        return True, "Trade approved"
    
    
    def add_open_position(
        self,
        ticker: str,
        entry_price: float,
        position_size: float,
        stop_loss: float,
        take_profit: float,
        trend: str,
        confidence: float
    ) -> Dict:
        """
        Add a new open position.
        
        Returns:
            Position record dict
        """
        position = {
            "ticker": ticker,
            "entry_price": entry_price,
            "position_size": position_size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "trend": trend,
            "confidence": confidence,
            "entry_time": pd.Timestamp.now(),
            "current_price": entry_price,
            "pnl": 0.0,
            "pnl_percent": 0.0,
        }
        self.open_positions.append(position)
        return position
    
    
    def close_position(
        self,
        position_idx: int,
        exit_price: float,
        exit_reason: str = "manual"
    ) -> Dict:
        """
        Close an open position.
        
        Args:
            position_idx: Index in open_positions list
            exit_price: Closing price
            exit_reason: "sl_hit", "tp_hit", "manual", "timeout"
            
        Returns:
            Closed position record
        """
        if position_idx >= len(self.open_positions):
            return None
        
        position = self.open_positions[position_idx]
        position["exit_price"] = exit_price
        position["exit_time"] = pd.Timestamp.now()
        position["exit_reason"] = exit_reason
        
        # Calculate P&L
        if position["trend"].lower() == "bullish":
            pnl = (exit_price - position["entry_price"]) * position["position_size"]
        else:
            pnl = (position["entry_price"] - exit_price) * position["position_size"]
        
        position["pnl_final"] = pnl
        position["pnl_percent"] = (pnl / (position["entry_price"] * position["position_size"])) * 100
        
        # Update equity
        self.current_equity += pnl
        self.daily_pnl += pnl
        
        # Move to history
        self.trade_history.append(position)
        self.open_positions.pop(position_idx)
        
        return position
    
    
    def update_position_price(self, position_idx: int, current_price: float) -> Dict:
        """
        Update position market price and unrealized P&L.
        """
        if position_idx >= len(self.open_positions):
            return None
        
        position = self.open_positions[position_idx]
        position["current_price"] = current_price
        
        if position["trend"].lower() == "bullish":
            pnl = (current_price - position["entry_price"]) * position["position_size"]
        else:
            pnl = (position["entry_price"] - current_price) * position["position_size"]
        
        position["pnl"] = pnl
        position["pnl_percent"] = (pnl / (position["entry_price"] * position["position_size"])) * 100
        
        return position
    
    
    def get_portfolio_metrics(self) -> Dict:
        """
        Calculate overall portfolio metrics.
        """
        open_pnl = sum(p["pnl"] for p in self.open_positions)
        closed_pnl = sum(p.get("pnl_final", 0) for p in self.trade_history)
        
        win_count = sum(1 for p in self.trade_history if p.get("pnl_final", 0) > 0)
        loss_count = sum(1 for p in self.trade_history if p.get("pnl_final", 0) < 0)
        total_trades = len(self.trade_history)
        
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        return {
            "current_equity": self.current_equity,
            "total_return": self.current_equity - self.account_size,
            "return_percent": ((self.current_equity - self.account_size) / self.account_size) * 100,
            "open_pnl": open_pnl,
            "closed_pnl": closed_pnl,
            "daily_pnl": self.daily_pnl,
            "open_positions_count": len(self.open_positions),
            "total_trades": total_trades,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": win_rate,
            "max_drawdown": self._calculate_max_drawdown(),
            "sharpe_ratio": self._calculate_sharpe_ratio(),
        }
    
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from closed trades."""
        if not self.trade_history:
            return 0.0
        
        peak_equity = self.account_size
        max_dd = 0.0
        
        for trade in self.trade_history:
            peak_equity = max(peak_equity, peak_equity + trade.get("pnl_final", 0))
            dd = (peak_equity - (peak_equity + trade.get("pnl_final", 0))) / peak_equity
            max_dd = max(max_dd, dd)
        
        return max_dd * 100
    
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe Ratio (simplified: assumes daily returns)."""
        if len(self.trade_history) < 2:
            return 0.0
        
        pnls = np.array([p.get("pnl_final", 0) for p in self.trade_history])
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        
        if std_pnl == 0:
            return 0.0
        
        # Annualized Sharpe (assuming 252 trading days)
        return (mean_pnl / std_pnl) * np.sqrt(252)
    
    
    def reset_daily_pnl(self):
        """Reset daily P&L for next day."""
        self.daily_pnl = 0.0
    
    
    def to_dict(self) -> Dict:
        """Export risk manager state for storage/API."""
        return {
            "account_size": self.account_size,
            "current_equity": self.current_equity,
            "daily_pnl": self.daily_pnl,
            "max_risk_per_trade": self.max_risk_per_trade,
            "max_position_size": self.max_position_size,
            "max_daily_loss": self.max_daily_loss,
            "max_open_positions": self.max_open_positions,
            "open_positions": self.open_positions,
            "trade_history": self.trade_history,
            "portfolio_metrics": self.get_portfolio_metrics(),
        }
