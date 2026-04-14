"""
╔════════════════════════════════════════════════════════════════════════════╗
║                      AUTOMATED TRADING BOT v1.0                            ║
║        Connects to live predictions and auto-trades based on signals      ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
import json
import requests
import yfinance as yf
from enum import Enum

from .paper_trading_engine import create_paper_trading_account, get_paper_trading_account
from .risk_manager import create_risk_manager

logger = logging.getLogger(__name__)


class BotStatus(Enum):
    """Bot status"""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"


class TradingBot:
    """Automated trading bot using live predictions"""
    
    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        initial_capital: float = 300000,
        min_confidence: float = 0.75,  # 75%
        signal_filter: str = "STRONG_BUY",  # STRONG_BUY, BUY, ALL
        risk_per_trade: float = 0.08,  # 8%
        daily_loss_limit: float = 0.07,  # 7%
        max_positions: int = 4
    ):
        self.api_base_url = api_base_url
        self.initial_capital = initial_capital
        self.min_confidence = min_confidence
        self.signal_filter = signal_filter
        self.status = BotStatus.IDLE
        self.is_running = False
        self.update_interval = 60  # seconds
        
        # Initialize account and risk manager
        self.account = create_paper_trading_account(initial_capital, "Trading Bot")
        self.risk_manager = create_risk_manager(
            initial_capital,
            risk_per_trade,
            daily_loss_limit,
            max_positions
        )
        
        # Statistics
        self.signals_received = 0
        self.trades_placed = 0
        self.trades_closed = 0
        self.daily_pnl = 0
        self.last_check_time = None
        
        logger.info("🤖 Trading Bot Initialized")
        logger.info(f"   Capital: ₹{initial_capital:,.0f}")
        logger.info(f"   Min Confidence Filter: {min_confidence*100:.0f}%")
        logger.info(f"   Signal Filter: {signal_filter}")
        logger.info(f"   Risk per Trade: {risk_per_trade*100:.0f}%")
        logger.info(f"   Daily Loss Limit: {daily_loss_limit*100:.0f}%")
    
    async def start(self):
        """Start the bot"""
        self.is_running = True
        self.status = BotStatus.RUNNING
        logger.info("🟢 TRADING BOT STARTED")
        
        try:
            await self.main_loop()
        except Exception as e:
            logger.error(f"❌ Bot error: {e}")
            self.status = BotStatus.STOPPED
    
    async def stop(self):
        """Stop the bot"""
        self.is_running = False
        self.status = BotStatus.STOPPED
        logger.info("🔴 TRADING BOT STOPPED")
    
    async def main_loop(self):
        """Main trading loop"""
        while self.is_running:
            try:
                # Step 1: Get live predictions
                predictions = await self.get_live_predictions()
                
                if predictions:
                    # Step 2: Filter predictions
                    filtered = self.filter_predictions(predictions)
                    logger.info(f"📊 Received {len(predictions)} predictions, {len(filtered)} passed filters")
                    
                    # Step 3: Check exit conditions for open trades
                    await self.check_exits(predictions)
                    
                    # Step 4: Process buy signals
                    for stock, pred in filtered.items():
                        await self.process_signal(stock, pred)
                
                # Sleep before next check
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"❌ Loop error: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def get_live_predictions(self) -> Dict:
        """Fetch live predictions from API"""
        try:
            response = requests.get(
                f"{self.api_base_url}/api/v1/live/predictions",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('data', {})
            
        except Exception as e:
            logger.warning(f"⚠️  Could not fetch predictions: {e}")
        
        return {}
    
    def filter_predictions(self, predictions: Dict) -> Dict:
        """Filter predictions based on criteria"""
        filtered = {}
        
        for stock, pred in predictions.items():
            # Check confidence
            confidence = pred.get('confidence', 0)
            if confidence < self.min_confidence:
                continue
            
            # Check signal
            signal = pred.get('signal', '')
            if self.signal_filter == "STRONG_BUY" and signal != "STRONG BUY":
                continue
            elif self.signal_filter == "BUY" and signal not in ["STRONG BUY", "BUY"]:
                continue
            
            # Check if already have position
            if stock in self.account.open_positions:
                continue
            
            filtered[stock] = pred
            self.signals_received += 1
        
        return filtered
    
    async def check_exits(self, current_predictions: Dict):
        """Check if any open positions should be exited"""
        open_positions = list(self.account.open_positions.keys())
        
        for stock in open_positions:
            pred = current_predictions.get(stock)
            if not pred:
                continue
            
            current_price = pred.get('current_price', 0)
            
            # Check exit conditions
            exit_condition = self.account.check_exit_conditions(stock, current_price)
            
            if exit_condition:
                exit_reason, exit_price = exit_condition
                success, msg, pnl = self.account.close_trade(stock, exit_price, exit_reason)
                
                if success:
                    self.trades_closed += 1
                    self.daily_pnl += pnl
                    logger.info(f"📊 {msg}")
    
    async def process_signal(self, stock: str, prediction: Dict):
        """Process a buy signal"""
        
        try:
            entry_price = prediction.get('current_price', 0)
            target_price = prediction.get('target_price', 0)
            stop_loss = prediction.get('stop_loss', 0)
            confidence = prediction.get('confidence', 0)
            
            logger.debug(f"🔔 Processing signal: {stock} @ ₹{entry_price:.2f}")
            
            # Validate trade
            allowed, msg = self.risk_manager.validate_trade(
                entry_price,
                stop_loss,
                target_price,
                self.account.current_capital,
                len(self.account.open_positions),
                self.daily_pnl
            )
            
            if not allowed:
                logger.warning(f"⚠️  {stock} trade rejected:")
                logger.warning(msg)
                return
            
            # Calculate position size
            capital, qty = self.risk_manager.calculate_position_size(entry_price, stop_loss)
            
            # Check capital available
            if capital > self.account.current_capital:
                logger.warning(f"⚠️  {stock} insufficient capital")
                return
            
            # Place trade
            success, msg, trade = self.account.place_trade(
                stock=stock,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                quantity=qty,
                signal_confidence=confidence,
                entry_capital=capital
            )
            
            if success:
                self.trades_placed += 1
                logger.info(f"✅ {msg}")
                logger.info(f"   Entry: ₹{entry_price:.2f} | Target: ₹{target_price:.2f} | SL: ₹{stop_loss:.2f}")
                logger.info(f"   Qty: {qty} | Capital: ₹{capital:,.0f}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {stock} signal: {e}")
    
    def get_bot_status(self) -> Dict:
        """Get bot status and statistics"""
        account_stats = self.account.get_account_stats()
        
        return {
            "bot_status": self.status.value,
            "is_running": self.is_running,
            "signals_received": self.signals_received,
            "trades_placed": self.trades_placed,
            "trades_closed": self.trades_closed,
            "daily_pnl": self.daily_pnl,
            "open_positions": len(self.account.open_positions),
            "current_capital": self.account.current_capital,
            "capital_deployed": self.initial_capital - self.account.current_capital,
            **account_stats
        }
    
    def get_positions(self) -> Dict:
        """Get all open positions"""
        return self.account.get_open_positions()
    
    def close_position(self, stock: str, reason: str = "MANUAL") -> Dict:
        """Manually close a position"""
        
        if stock not in self.account.open_positions:
            return {"success": False, "message": f"No open position for {stock}"}
        
        # Get current price
        try:
            data = yf.download(f"{stock}.NS", period="1d", progress=False)
            current_price = data['Close'].iloc[-1]
        except:
            return {"success": False, "message": "Could not fetch current price"}
        
        success, msg, pnl = self.account.close_trade(stock, current_price, reason)
        
        if success:
            self.trades_closed += 1
            self.daily_pnl += pnl
        
        return {
            "success": success,
            "message": msg,
            "pnl": pnl
        }


# Global bot instance
_bot_instance: Optional[TradingBot] = None


def create_trading_bot(**kwargs) -> TradingBot:
    """Create global trading bot instance"""
    global _bot_instance
    _bot_instance = TradingBot(**kwargs)
    return _bot_instance


def get_trading_bot() -> TradingBot:
    """Get global trading bot instance"""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = TradingBot()
    return _bot_instance
