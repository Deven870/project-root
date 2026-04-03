#!/usr/bin/env python
"""
Daily Trading Metrics Tracker
Track your 70% system performance throughout the 2-week validation
"""

import json
from datetime import datetime
from pathlib import Path

class DailyMetricsTracker:
    """Track daily trading metrics"""
    
    def __init__(self, metrics_file="paper_trading_logs/metrics_tracker.json"):
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(exist_ok=True)
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.data = self.load_metrics()
    
    def load_metrics(self):
        """Load existing metrics or create new"""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {"daily": {}, "weekly": {}}
    
    def save_metrics(self):
        """Save metrics to file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def add_trade(self, stock: str, entry: float, exit: float, 
                  target: float, stoploss: float, capital: float = 25000):
        """Log a single trade"""
        
        if self.today not in self.data["daily"]:
            self.data["daily"][self.today] = {
                "date": self.today,
                "trades": [],
                "summary": {}
            }
        
        # Calculate results
        pnl = (exit - entry) / entry * 100  # % return
        pnl_amount = capital * (pnl / 100)
        
        # Determine if win or loss
        if exit >= target:
            result = "WIN"
            accuracy_match = "✅" if exit > entry else "❌"
        elif exit <= stoploss:
            result = "LOSS"
            accuracy_match = "❌"
        else:
            result = "MANUAL"
            accuracy_match = "?" if exit > entry else "?"
        
        trade = {
            "stock": stock,
            "entry": entry,
            "exit": exit,
            "target": target,
            "stoploss": stoploss,
            "capital": capital,
            "pnl_pct": pnl,
            "pnl_amount": pnl_amount,
            "result": result,
            "accuracy": accuracy_match,
            "time": datetime.now().isoformat()
        }
        
        self.data["daily"][self.today]["trades"].append(trade)
        self.save_metrics()
        
        return trade
    
    def update_daily_summary(self):
        """Update daily summary statistics"""
        
        if self.today not in self.data["daily"]:
            return None
        
        trades = self.data["daily"][self.today]["trades"]
        
        if not trades:
            return None
        
        wins = len([t for t in trades if t["result"] == "WIN"])
        losses = len([t for t in trades if t["result"] == "LOSS"])
        win_rate = (wins / len(trades) * 100) if trades else 0
        
        total_pnl = sum(t["pnl_amount"] for t in trades)
        daily_return = sum(t["pnl_pct"] for t in trades) / len(trades)
        
        summary = {
            "trades_executed": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(win_rate, 1),
            "total_pnl_rupees": round(total_pnl, 0),
            "avg_return_pct": round(daily_return, 2),
            "capital_used": sum(t["capital"] for t in trades)
        }
        
        self.data["daily"][self.today]["summary"] = summary
        self.save_metrics()
        
        return summary
    
    def get_weekly_stats(self, start_date="2026-04-03", end_date="2026-04-07"):
        """Calculate weekly statistics"""
        
        total_trades = 0
        total_wins = 0
        total_losses = 0
        total_pnl = 0
        
        for day_data in self.data["daily"].values():
            if start_date <= day_data["date"] <= end_date:
                summary = day_data.get("summary", {})
                if summary:
                    total_trades += summary.get("trades_executed", 0)
                    total_wins += summary.get("wins", 0)
                    total_losses += summary.get("losses", 0)
                    total_pnl += summary.get("total_pnl_rupees", 0)
        
        weekly_accuracy = (total_wins / total_trades * 100) if total_trades > 0 else 0
        capital_start = 250000
        capital_end = capital_start + total_pnl
        return_pct = (total_pnl / capital_start * 100) if capital_start > 0 else 0
        
        return {
            "period": f"{start_date} to {end_date}",
            "total_trades": total_trades,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "accuracy_pct": round(weekly_accuracy, 1),
            "total_pnl": round(total_pnl, 0),
            "return_pct": round(return_pct, 1),
            "capital_start": capital_start,
            "capital_end": round(capital_end, 0),
            "meets_68_target": weekly_accuracy >= 68,
            "meets_65_winrate_target": weekly_accuracy >= 65
        }


def print_daily_report():
    """Print today's trading report"""
    
    tracker = DailyMetricsTracker()
    tracker.update_daily_summary()
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    if today not in tracker.data["daily"] or not tracker.data["daily"][today]["trades"]:
        print(f"No trades recorded for {today}")
        return
    
    today_data = tracker.data["daily"][today]
    summary = today_data["summary"]
    trades = today_data["trades"]
    
    print("\n" + "="*70)
    print(f"📊 DAILY TRADING REPORT - {today}")
    print("="*70)
    
    print(f"\n📈 Summary:")
    print(f"  Trades Executed: {summary.get('trades_executed', 0)}")
    print(f"  Wins: {summary.get('wins', 0)}")
    print(f"  Losses: {summary.get('losses', 0)}")
    print(f"  Win Rate: {summary.get('win_rate_pct', 0):.1f}%")
    print(f"  Daily P&L: ₹{summary.get('total_pnl_rupees', 0):+,.0f}")
    print(f"  Avg Return: {summary.get('avg_return_pct', 0):+.2f}%")
    
    print(f"\n📋 Trades:")
    for i, trade in enumerate(trades, 1):
        print(f"\n  Trade #{i}: {trade['stock']}")
        print(f"    Entry: ₹{trade['entry']:,.2f}")
        print(f"    Exit: ₹{trade['exit']:,.2f}")
        print(f"    Result: {trade['result']} {trade['accuracy']} (P&L: {trade['pnl_pct']:+.2f}%)")
        print(f"    Amount: ₹{trade['pnl_amount']:+,.0f}")
    
    print("\n" + "="*70)


def print_weekly_report(week_num=1):
    """Print weekly summary"""
    
    tracker = DailyMetricsTracker()
    
    if week_num == 1:
        stats = tracker.get_weekly_stats("2026-04-03", "2026-04-07")
    else:
        stats = tracker.get_weekly_stats("2026-04-10", "2026-04-17")
    
    print("\n" + "="*70)
    print(f"📊 WEEKLY REPORT - Week {week_num}")
    print("="*70)
    
    print(f"\nPeriod: {stats['period']}")
    print(f"\n✅ Trading Results:")
    print(f"  Total Trades: {stats['total_trades']}")
    print(f"  Wins: {stats['total_wins']}")
    print(f"  Losses: {stats['total_losses']}")
    print(f"  Accuracy: {stats['accuracy_pct']:.1f}%")
    
    print(f"\n💰 Financial Results:")
    print(f"  Starting Capital: ₹{stats['capital_start']:,.0f}")
    print(f"  Ending Capital: ₹{stats['capital_end']:,.0f}")
    print(f"  Total P&L: ₹{stats['total_pnl']:+,.0f}")
    print(f"  Return %: {stats['return_pct']:+.1f}%")
    
    print(f"\n🎯 Validation Targets:")
    accuracy_status = "✅ PASS" if stats['meets_68_target'] else "❌ FAIL"
    winrate_status = "✅ PASS" if stats['meets_65_winrate_target'] else "❌ FAIL"
    
    print(f"  Accuracy 68%+: {accuracy_status} ({stats['accuracy_pct']:.1f}%)")
    print(f"  Win Rate 65%+: {winrate_status} ({stats['accuracy_pct']:.1f}%)")
    
    decision = "✅ READY FOR LIVE DEPLOYMENT" if (
        stats['meets_68_target'] and stats['meets_65_winrate_target']
    ) else "⏳ CONTINUE PAPER TRADING"
    
    print(f"\n🚀 Decision: {decision}")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "daily":
            print_daily_report()
        elif sys.argv[1] == "weekly":
            week = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            print_weekly_report(week)
    else:
        print("Usage:")
        print("  python track_metrics.py daily          # Today's report")
        print("  python track_metrics.py weekly [1|2]   # Week 1 or 2 report")
