"""
Paper Trading Framework for 70% Accuracy System
Simulates real trading with virtual capital to validate predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PaperTradingAccount:
    """Simulates a paper trading account"""

    def __init__(self, initial_capital=10000, account_name="paper_trading"):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.account_name = account_name
        self.trades = []
        self.positions = {}
        self.daily_pnl = []
        self.start_date = datetime.now()
        self.max_position_size = 0.05 * initial_capital  # Max 5% per position
        self.max_positions = 4

    def place_buy_order(self, ticker, quantity, entry_price, confidence, reasoning=""):
        """
        Place a buy order
        
        Args:
            ticker: Stock ticker
            quantity: Number of shares
            entry_price: Entry price per share
            confidence: Confidence 0-1
            reasoning: Trade reasoning note
        """
        cost = quantity * entry_price
        
        if cost > self.current_capital * 0.2:  # Max 20% of capital per trade
            logger.warning(f"Order too large for {ticker}: ${cost:.0f} > ${self.current_capital * 0.2:.0f}")
            return False

        if len(self.positions) >= self.max_positions:
            logger.warning(f"Max positions ({self.max_positions}) reached")
            return False

        # Execute trade
        self.current_capital -= cost
        
        trade = {
            'date': datetime.now(),
            'type': 'BUY',
            'ticker': ticker,
            'quantity': quantity,
            'entry_price': entry_price,
            'confidence': confidence,
            'reasoning': reasoning,
            'status': 'OPEN'
        }
        
        self.trades.append(trade)
        self.positions[ticker] = {
            'quantity': quantity,
            'entry_price': entry_price,
            'current_price': entry_price,
            'confidence': confidence,
            'trade_id': len(self.trades) - 1
        }
        
        logger.info(f"BUY {quantity} {ticker} @ ${entry_price:.2f} (conf: {confidence:.0%})")
        return True

    def place_sell_order(self, ticker, reason=""):
        """Close a position"""
        if ticker not in self.positions:
            logger.warning(f"No position for {ticker}")
            return False

        position = self.positions[ticker]
        quantity = position['quantity']
        exit_price = position['current_price']
        entry_price = position['entry_price']
        
        proceeds = quantity * exit_price
        pnl = proceeds - (quantity * entry_price)
        pnl_pct = (pnl / (quantity * entry_price)) * 100
        
        self.current_capital += proceeds
        
        # Update trade
        trade_id = position['trade_id']
        self.trades[trade_id]['status'] = 'CLOSED'
        self.trades[trade_id]['exit_price'] = exit_price
        self.trades[trade_id]['pnl'] = pnl
        self.trades[trade_id]['pnl_pct'] = pnl_pct
        self.trades[trade_id]['close_date'] = datetime.now
        self.trades[trade_id]['reason'] = reason
        
        del self.positions[ticker]
        
        logger.info(f"SELL {quantity} {ticker} @ ${exit_price:.2f} | "
                   f"P&L: ${pnl:+.0f} ({pnl_pct:+.1f}%) [{reason}]")
        
        return True

    def update_price(self, ticker, current_price):
        """Update position price (mark-to-market)"""
        if ticker in self.positions:
            self.positions[ticker]['current_price'] = current_price

    def get_portfolio_value(self):
        """Get total portfolio value (cash + positions)"""
        value = self.current_capital
        for ticker, position in self.positions.items():
            value += position['quantity'] * position['current_price']
        return value

    def get_unrealized_pnl(self):
        """Get total unrealized P&L"""
        pnl = 0
        for ticker, position in self.positions.items():
            current_value = position['quantity'] * position['current_price']
            entry_value = position['quantity'] * position['entry_price']
            pnl += current_value - entry_value
        return pnl

    def get_daily_return(self):
        """Get today's return %"""
        portfolio_value = self.get_portfolio_value()
        return ((portfolio_value - self.initial_capital) / self.initial_capital) * 100

    def get_stats(self):
        """Get comprehensive account statistics"""
        closed_trades = [t for t in self.trades if t['status'] == 'CLOSED']
        
        if not closed_trades:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        else:
            winners = [t for t in closed_trades if t['pnl'] > 0]
            losers = [t for t in closed_trades if t['pnl'] < 0]
            
            win_rate = len(winners) / len(closed_trades) * 100 if closed_trades else 0
            
            total_wins = sum(t['pnl'] for t in winners) if winners else 0
            total_losses = abs(sum(t['pnl'] for t in losers)) if losers else 0
            
            avg_win = total_wins / len(winners) if winners else 0
            avg_loss = total_losses / len(losers) if losers else 0
            
            profit_factor = total_wins / total_losses if total_losses > 0 else 0

        return {
            'account_value': self.get_portfolio_value(),
            'total_pnl': self.get_portfolio_value() - self.initial_capital,
            'total_pnl_pct': self.get_daily_return(),
            'trades_total': len(self.trades),
            'trades_closed': len(closed_trades),
            'trades_open': len(self.positions),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'unrealized_pnl': self.get_unrealized_pnl()
        }

    def save_to_json(self, filepath):
        """Save account state"""
        stats = self.get_stats()
        stats['account_name'] = self.account_name
        stats['start_date'] = self.start_date.isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Account saved to {filepath}")


class PaperTradingManager:
    """Manages multiple paper trading sessions"""

    def __init__(self, base_dir="paper_trading_logs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.accounts = {}

    def create_account(self, name, capital):
        """Create new paper trading account"""
        account = PaperTradingAccount(initial_capital=capital, account_name=name)
        self.accounts[name] = account
        logger.info(f"Created account '{name}' with ${capital:,.0f}")
        return account

    def get_account(self, name):
        """Get account by name"""
        return self.accounts.get(name)

    def save_all_accounts(self):
        """Save all accounts"""
        for name, account in self.accounts.items():
            filepath = self.base_dir / f"{name}_stats.json"
            account.save_to_json(filepath)


class PaperTradeExecutor:
    """Executes trades on paper trading account based on predictions"""

    def __init__(self, account, max_risk_pct=0.02):
        self.account = account
        self.max_risk_pct = max_risk_pct  # Max 2% risk per trade

    def execute_prediction(self, prediction, ticker, current_price, features):
        """
        Execute trade based on prediction
        
        Args:
            prediction: dict {trend, confidence, signal, regime}
            ticker: Stock ticker
            current_price: Current price
            features: Feature dict with technical indicators
        """
        trend = prediction.get('trend', 0)
        confidence = prediction.get('confidence', 0)
        signal = prediction.get('signal', 0)
        regime = prediction.get('regime', 'neutral')

        # Determine position size based on confidence
        base_size = 100
        position_size = int(base_size * confidence)

        # Execute if confidence > 60%
        if confidence < 0.6:
            logger.debug(f"Low confidence ({confidence:.0%}) for {ticker}, skipping")
            return

        if trend == 1:  # Bullish
            entry_price = current_price * 1.001  # Slightly above market
            self.account.place_buy_order(
                ticker=ticker,
                quantity=position_size,
                entry_price=entry_price,
                confidence=confidence,
                reasoning=f"Bullish signal ({regime}), conf={confidence:.0%}"
            )
        elif trend == 0:  # Bearish
            # Short sell or close existing long position
            if ticker in self.account.positions:
                self.account.place_sell_order(ticker, reason="Bearish signal")
            else:
                logger.debug(f"No position to close for {ticker}")

    def evaluate_positions(self, market_data):
        """
        Evaluate open positions and close if targets or stops hit
        
        Args:
            market_data: dict {ticker: current_price}
        """
        for ticker, position in list(self.account.positions.items()):
            if ticker not in market_data:
                continue

            current_price = market_data[ticker]
            self.account.update_price(ticker, current_price)
            
            entry_price = position['entry_price']
            gain_pct = ((current_price - entry_price) / entry_price) * 100

            # Close on profit target (5%)
            if gain_pct >= 5:
                self.account.place_sell_order(ticker, reason="Take profit (5%)")

            # Close on stop loss (2%)
            elif gain_pct <= -2:
                self.account.place_sell_order(ticker, reason="Stop loss (2%)")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create account
    account = PaperTradingAccount(initial_capital=10000)

    # Simulate trades
    print("\n" + "=" * 60)
    print("PAPER TRADING SIMULATION")
    print("=" * 60)

    # Trade 1: Buy low-confidence
    account.place_buy_order(
        ticker="RELIANCE",
        quantity=10,
        entry_price=1350,
        confidence=0.70,
        reasoning="Strong bullish setup"
    )

    # Update price and close with profit
    account.update_price("RELIANCE", 1380)
    account.place_sell_order("RELIANCE", reason="Take profit")

    # Trade 2: Buy another stock
    account.place_buy_order(
        ticker="TCS",
        quantity=5,
        entry_price=3500,
        confidence=0.65,
        reasoning="Swing trade setup"
    )

    # Close at loss
    account.update_price("TCS", 3430)
    account.place_sell_order("TCS", reason="Cut loss")

    # Print stats
    stats = account.get_stats()
    print("\n" + "=" * 60)
    print("ACCOUNT STATISTICS")
    print("=" * 60)
    print(f"Portfolio Value: ${stats['account_value']:,.0f}")
    print(f"Total P&L: ${stats['total_pnl']:+,.0f} ({stats['total_pnl_pct']:+.2f}%)")
    print(f"Trades: {stats['trades_closed']} closed, {stats['trades_open']} open")
    print(f"Win Rate: {stats['win_rate']:.1f}%")
    print(f"Profit Factor: {stats['profit_factor']:.2f}")
