"""
30-DAY PAPER TRADING AUTOMATION & VALIDATION
Automatically tracks performance and validates readiness for live trading.
Targets: 60%+ win rate, Sharpe >1.2, Max drawdown <15%
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configuration
PAPER_TRADING_DIR = "paper_trading_logs"
VALIDATION_REPORT = f"{PAPER_TRADING_DIR}/validation_report.json"
DAILY_TRADES_CSV = f"{PAPER_TRADING_DIR}/daily_trades.csv"
METRICS_LOG = f"{PAPER_TRADING_DIR}/metrics_log.csv"

# Validation targets
TARGETS = {
    "win_rate_pct": 60.0,          # Minimum 60% win rate
    "sharpe_ratio": 1.2,           # Minimum 1.2 Sharpe
    "max_drawdown_pct": 15.0,      # Maximum 15% drawdown
    "profit_factor": 1.5,          # Minimum 1.5 (wins/losses ratio)
    "days_active": 30               # Full 30-day period
}

# Create directory
os.makedirs(PAPER_TRADING_DIR, exist_ok=True)


class PaperTradingValidator:
    """Tracks 30-day paper trading performance and validates go-live readiness."""
    
    def __init__(self):
        self.trades_log = []
        self.daily_metrics = []
        self.start_date = None
        self.capital = float(os.getenv("STARTING_CAPITAL", "100000"))
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load previous trading data if exists."""
        if os.path.exists(DAILY_TRADES_CSV):
            self.trades_log = pd.read_csv(DAILY_TRADES_CSV).to_dict('records')
        
        if os.path.exists(METRICS_LOG):
            self.daily_metrics = pd.read_csv(METRICS_LOG).to_dict('records')
        
        if os.path.exists(VALIDATION_REPORT):
            with open(VALIDATION_REPORT) as f:
                report = json.load(f)
                self.start_date = report.get("start_date")
    
    def log_trade(self, symbol: str, entry_price: float, exit_price: float, 
                  entry_time: datetime, exit_time: datetime, 
                  stop_loss: float, target: float, exit_reason: str = ""):
        """
        Log a completed paper trade.
        
        Parameters
        ----------
        symbol : str
            Stock symbol
        entry_price : float
            Entry price
        exit_price : float
            Exit price
        entry_time : datetime
            Entry time
        exit_time : datetime
            Exit time
        stop_loss : float
            Stop loss level
        target : float
            Target price
        exit_reason : str
            How trade exited (stop_loss, target_hit, manual_close)
        """
        pnl = exit_price - entry_price
        pnl_pct = (pnl / entry_price) * 100
        drawdown = (stop_loss - entry_price) / entry_price * 100
        upside = (target - entry_price) / entry_price * 100
        
        trade = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "symbol": symbol,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "win": 1 if pnl > 0 else 0,
            "stop_loss": stop_loss,
            "target": target,
            "drawdown_risk_pct": round(drawdown, 2),
            "upside_pct": round(upside, 2),
            "exit_reason": exit_reason,
            "entry_time": entry_time.isoformat() if isinstance(entry_time, datetime) else entry_time,
            "exit_time": exit_time.isoformat() if isinstance(exit_time, datetime) else exit_time,
        }
        
        self.trades_log.append(trade)
        self.save_trades()
        
        return trade
    
    def save_trades(self):
        """Save trades to CSV."""
        if self.trades_log:
            df = pd.DataFrame(self.trades_log)
            df.to_csv(DAILY_TRADES_CSV, index=False)
    
    def calculate_daily_metrics(self):
        """Calculate daily performance metrics."""
        if not self.trades_log:
            print("No trades logged yet")
            return None
        
        df = pd.DataFrame(self.trades_log)
        df['date'] = pd.to_datetime(df['date'])
        
        # Aggregate by date
        daily_pnl = df.groupby('date').agg({
            'pnl': 'sum',
            'win': 'sum',
            'symbol': 'count'  # Number of trades
        }).rename(columns={'symbol': 'trades'})
        
        daily_pnl['date'] = daily_pnl.index.strftime("%Y-%m-%d")
        daily_pnl = daily_pnl.reset_index(drop=True)
        
        self.daily_metrics = daily_pnl.to_dict('records')
        daily_pnl.to_csv(METRICS_LOG, index=False)
        
        return daily_pnl
    
    def get_cumulative_metrics(self):
        """Calculate cumulative performance metrics."""
        if not self.trades_log:
            return None
        
        df = pd.DataFrame(self.trades_log)
        
        # Calculate metrics
        total_trades = len(df)
        winning_trades = df['win'].sum()
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = df['pnl'].sum()
        avg_win = df[df['win'] == 1]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(df[df['win'] == 0]['pnl'].mean()) if losing_trades > 0 else 0
        
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Sharpe ratio (daily returns)
        if len(df) > 1:
            pnl_pcts = df['pnl_pct'].values
            daily_returns = np.array(pnl_pcts) / 100
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
        else:
            sharpe = 0
        
        # Max drawdown
        equity_curve = self.capital + df['pnl'].cumsum()
        peak = equity_curve.max()
        trough = equity_curve.min()
        max_dd = ((trough - peak) / peak * 100) if peak > 0 else 0
        
        # Days active
        df['date'] = pd.to_datetime(df['date'])
        days_active = (df['date'].max() - df['date'].min()).days + 1
        
        metrics = {
            "total_trades": int(total_trades),
            "winning_trades": int(winning_trades),
            "losing_trades": int(losing_trades),
            "win_rate_pct": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round((total_pnl / self.capital) * 100, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 4),
            "days_active": int(days_active),
            "trades_per_day": round(total_trades / max(days_active, 1), 2),
            "final_equity": round(self.capital + total_pnl, 2),
        }
        
        return metrics
    
    def check_validation_status(self):
        """Check if trading system meets go-live targets."""
        metrics = self.get_cumulative_metrics()
        
        if not metrics:
            return {"status": "INSUFFICIENT_DATA", "message": "No trades yet"}
        
        checks = {
            "win_rate": metrics["win_rate_pct"] >= TARGETS["win_rate_pct"],
            "sharpe": metrics["sharpe_ratio"] >= TARGETS["sharpe_ratio"],
            "drawdown": abs(metrics["max_drawdown_pct"]) <= TARGETS["max_drawdown_pct"],
            "profit_factor": metrics["profit_factor"] >= TARGETS["profit_factor"],
            "duration": metrics["days_active"] >= TARGETS["days_active"],
        }
        
        status = "READY_FOR_LIVE" if all(checks.values()) else "CONTINUE_TESTING"
        passed = sum(checks.values())
        total = len(checks)
        
        return {
            "status": status,
            "passed": passed,
            "total": total,
            "checks": checks,
            "metrics": metrics,
            "targets": TARGETS
        }
    
    def generate_validation_report(self):
        """Generate comprehensive validation report."""
        metrics = self.get_cumulative_metrics()
        validation = self.check_validation_status()
        
        if not self.start_date:
            self.start_date = datetime.now().isoformat()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "start_date": self.start_date,
            "metrics": metrics,
            "validation": validation,
            "targets": TARGETS,
            "status": validation["status"],
            "progress": {
                "win_rate": f"{metrics['win_rate_pct']:.1f}% (target: {TARGETS['win_rate_pct']:.1f}%)",
                "sharpe": f"{metrics['sharpe_ratio']:.2f} (target: {TARGETS['sharpe_ratio']:.2f})",
                "drawdown": f"{abs(metrics['max_drawdown_pct']):.1f}% (max: {TARGETS['max_drawdown_pct']:.1f}%)",
                "duration": f"{metrics['days_active']} days (target: {TARGETS['days_active']} days)",
            }
        }
        
        with open(VALIDATION_REPORT, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def print_status(self):
        """Print current validation status."""
        report = self.generate_validation_report()
        
        print("\n" + "="*70)
        print("           30-DAY PAPER TRADING VALIDATION STATUS")
        print("="*70)
        
        if report["metrics"]:
            m = report["metrics"]
            print(f"\n📊 PERFORMANCE METRICS:")
            print(f"  Trades: {m['total_trades']} | Wins: {m['winning_trades']} | Losses: {m['losing_trades']}")
            print(f"  Win Rate: {m['win_rate_pct']}% (target: ≥60%)")
            print(f"  Sharpe Ratio: {m['sharpe_ratio']} (target: ≥1.2)")
            print(f"  Max Drawdown: {m['max_drawdown_pct']}% (target: ≤15%)")
            print(f"  Profit Factor: {m['profit_factor']} (target: ≥1.5)")
            print(f"  Total P&L: ₹{m['total_pnl']:,.2f} ({m['total_pnl_pct']}%)")
            print(f"  Days Active: {m['days_active']}/30")
        
        val = report["validation"]
        print(f"\n✅ VALIDATION CHECK:")
        print(f"  Status: {val['status']}")
        print(f"  Passed: {val['passed']}/{val['total']} checks")
        
        if val['checks']:
            for key, passed in val['checks'].items():
                symbol = "✓" if passed else "✗"
                print(f"    {symbol} {key.replace('_', ' ').title()}")
        
        if val['status'] == "READY_FOR_LIVE":
            print("\n🎉 SYSTEM IS READY FOR LIVE TRADING!")
        else:
            print("\n⏳ CONTINUE TESTING...")
        
        print("="*70 + "\n")
        
        return report
    
    def export_summary(self):
        """Export summary for review."""
        report = self.generate_validation_report()
        
        # Create Excel export
        with pd.ExcelWriter(f"{PAPER_TRADING_DIR}/validation_summary.xlsx") as writer:
            if self.trades_log:
                df_trades = pd.DataFrame(self.trades_log)
                df_trades.to_excel(writer, sheet_name='Trades', index=False)
            
            if self.daily_metrics:
                df_daily = pd.DataFrame(self.daily_metrics)
                df_daily.to_excel(writer, sheet_name='Daily', index=False)
            
            # Summary sheet
            if report["metrics"]:
                summary_data = {
                    "Metric": list(report["metrics"].keys()),
                    "Value": list(report["metrics"].values()),
                    "Target": [TARGETS.get(k, "N/A") for k in report["metrics"].keys()]
                }
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"✓ Summary exported to {PAPER_TRADING_DIR}/validation_summary.xlsx")
        
        return report


def get_validator():
    """Singleton instance."""
    global _validator
    if '_validator' not in globals():
        _validator = PaperTradingValidator()
    return _validator


def log_paper_trade(symbol, entry_price, exit_price, entry_time, exit_time,
                   stop_loss, target, exit_reason=""):
    """Convenience function to log a trade."""
    validator = get_validator()
    return validator.log_trade(symbol, entry_price, exit_price, entry_time, exit_time,
                               stop_loss, target, exit_reason)


def check_validation_status():
    """Check if ready for live trading."""
    validator = get_validator()
    return validator.check_validation_status()


def print_validation_status():
    """Print current status to console."""
    validator = get_validator()
    return validator.print_status()


if __name__ == "__main__":
    validator = get_validator()
    
    # Example: Log some test trades
    if not validator.trades_log:
        print("Logging sample trades for demo...")
        today = datetime.now()
        
        # Sample trades
        validator.log_trade("RELIANCE", 2500, 2625, today, today + timedelta(hours=2),
                           2450, 2550, "target_hit")
        validator.log_trade("TCS", 3500, 3450, today, today + timedelta(hours=1),
                           3520, 3600, "stop_loss")
        validator.log_trade("INFY", 2200, 2280, today, today + timedelta(hours=4),
                           2150, 2300, "target_hit")
    
    validator.print_status()
    validator.export_summary()
