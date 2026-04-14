"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    RISK MANAGEMENT SYSTEM                                  ║
║           Position sizing, stop loss, and daily loss limits                ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class RiskProfile(Enum):
    """Risk profiles"""
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"
    CUSTOM = "CUSTOM"


class RiskManager:
    """Manages position sizing and risk constraints"""
    
    def __init__(
        self,
        account_balance: float,
        risk_per_trade_pct: float = 0.08,  # 8%
        daily_loss_limit_pct: float = 0.07,  # 7%
        max_open_positions: int = 4
    ):
        self.account_balance = account_balance
        self.risk_per_trade_pct = risk_per_trade_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_open_positions = max_open_positions
        self.daily_pnl = 0
        self.daily_pnl_reset_time = datetime.now()
        
        logger.info("✅ Risk Manager Initialized")
        logger.info(f"   Account Balance: ₹{account_balance:,.0f}")
        logger.info(f"   Risk per Trade: {risk_per_trade_pct*100:.0f}%")
        logger.info(f"   Daily Loss Limit: {daily_loss_limit_pct*100:.0f}%")
        logger.info(f"   Max Open Positions: {max_open_positions}")
    
    def check_daily_loss_limit(self, daily_pnl: float) -> Tuple[bool, str]:
        """
        Check if daily loss limit exceeded
        
        Returns: (allowed, message)
        """
        daily_loss_limit = self.account_balance * self.daily_loss_limit_pct
        
        if daily_pnl < -daily_loss_limit:
            return False, f"❌ Daily loss limit exceeded. Limit: ₹{daily_loss_limit:,.0f}, Current: ₹{daily_pnl:,.0f}"
        
        return True, f"✅ Daily loss limit OK. Limit: ₹{daily_loss_limit:,.0f}, Current: ₹{daily_pnl:,.0f}"
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float
    ) -> Tuple[float, int]:
        """
        Calculate position size based on risk per trade
        
        Returns: (capital_to_deploy, quantity)
        """
        # Risk amount = account balance * risk%
        risk_amount = self.account_balance * self.risk_per_trade_pct
        
        # Points at risk per share
        risk_per_share = entry_price - stop_loss_price
        
        if risk_per_share <= 0:
            logger.error(f"❌ Invalid SL: Entry {entry_price} should be > SL {stop_loss_price}")
            return 0, 0
        
        # Quantity = risk amount / risk per share
        quantity = int(risk_amount / risk_per_share)
        
        # Capital to deploy
        capital_to_deploy = quantity * entry_price
        
        logger.debug(f"Position Sizing:")
        logger.debug(f"  Risk per Trade: ₹{risk_amount:,.0f}")
        logger.debug(f"  Risk per Share: ₹{risk_per_share:.2f}")
        logger.debug(f"  Quantity: {quantity}")
        logger.debug(f"  Capital: ₹{capital_to_deploy:,.0f}")
        
        return capital_to_deploy, quantity
    
    def validate_trade(
        self,
        entry_price: float,
        stop_loss_price: float,
        target_price: float,
        current_capital: float,
        open_positions_count: int,
        daily_pnl: float
    ) -> Tuple[bool, str]:
        """
        Validate if trade satisfies all risk constraints
        
        Returns: (allowed, message)
        """
        checks = []
        
        # Check 1: Stop loss below entry
        if stop_loss_price >= entry_price:
            checks.append((False, f"❌ SL (₹{stop_loss_price:.2f}) must be below entry (₹{entry_price:.2f})"))
        else:
            checks.append((True, f"✅ SL check passed"))
        
        # Check 2: Target above entry
        if target_price <= entry_price:
            checks.append((False, f"❌ Target (₹{target_price:.2f}) must be above entry (₹{entry_price:.2f})"))
        else:
            checks.append((True, f"✅ Target check passed"))
        
        # Check 3: Risk/Reward ratio
        risk = entry_price - stop_loss_price
        reward = target_price - entry_price
        rr_ratio = reward / risk if risk > 0 else 0
        
        if rr_ratio < 1:
            checks.append((False, f"❌ RR Ratio (1:{rr_ratio:.2f}) too low, need at least 1:1"))
        else:
            checks.append((True, f"✅ RR Ratio check passed (1:{rr_ratio:.2f})"))
        
        # Check 4: Capital available
        capital_needed, qty = self.calculate_position_size(entry_price, stop_loss_price)
        if capital_needed > current_capital:
            checks.append((False, f"❌ Capital needed (₹{capital_needed:,.0f}) > available (₹{current_capital:,.0f})"))
        else:
            checks.append((True, f"✅ Capital check passed"))
        
        # Check 5: Max open positions
        if open_positions_count >= self.max_open_positions:
            checks.append((False, f"❌ Max open positions ({self.max_open_positions}) reached"))
        else:
            checks.append((True, f"✅ Position limit OK ({open_positions_count}/{self.max_open_positions})"))
        
        # Check 6: Daily loss limit
        allowed, msg = self.check_daily_loss_limit(daily_pnl)
        checks.append((allowed, msg))
        
        # Summary
        all_passed = all(check[0] for check in checks)
        messages = "\n".join(check[1] for check in checks)
        
        if all_passed:
            logger.info(f"✅ ALL RISK CHECKS PASSED")
        else:
            logger.warning(f"❌ RISK CHECKS FAILED")
        
        logger.debug(messages)
        
        return all_passed, messages
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        return {
            "account_balance": self.account_balance,
            "risk_per_trade": self.risk_per_trade_pct * 100,
            "daily_loss_limit": self.daily_loss_limit_pct * 100,
            "max_open_positions": self.max_open_positions,
            "daily_loss_limit_amount": self.account_balance * self.daily_loss_limit_pct,
            "risk_per_trade_amount": self.account_balance * self.risk_per_trade_pct,
        }


def create_risk_manager(
    account_balance: float,
    risk_per_trade_pct: float = 0.08,
    daily_loss_limit_pct: float = 0.07,
    max_open_positions: int = 4
) -> RiskManager:
    """Factory function to create risk manager"""
    return RiskManager(
        account_balance,
        risk_per_trade_pct,
        daily_loss_limit_pct,
        max_open_positions
    )
