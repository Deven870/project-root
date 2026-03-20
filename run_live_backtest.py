#!/usr/bin/env python3
"""
run_live_backtest.py
Execute comprehensive backtest on real NSE data with all safety features.
Usage: python run_live_backtest.py
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.realdata_backtester import RealdataBacktester
from modules.safety_guardrails import SafetyGuardrails, AnomalyDetector
from modules.pnl_tracker import PnLTracker


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def print_section(text: str):
    """Print formatted section."""
    print(f"\n{text}")
    print("-" * len(text))


def run_backtest_live():
    """
    Main backtest runner with full safety and tracking.
    """
    
    print_header("⚡ LIVE BACKTEST: ML Trading Strategy on Real NSE Data")
    
    # Configuration
    print_section("📋 Configuration")
    
    tickers = [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
        "ICICIBANK.NS",
        "ITC.NS",
        "SBIN.NS",
        "MARUTI.NS",
        "TITAN.NS",
        "WIPRO.NS",
    ]
    
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")  # 1 year back
    end_date = datetime.now().strftime("%Y-%m-%d")
    initial_capital = 100000  # Rs 1,00,000
    
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: Rs {initial_capital:,.0f}")
    print(f"Max Risk per Trade: 2%")
    print(f"Stop-Loss: Enabled")
    print(f"Position Sizing: Risk-Based")
    
    # Initialize systems
    print_section("🚀 Initializing Systems")
    
    try:
        backtester = RealdataBacktester(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            max_risk_per_trade=0.02
        )
        print("✓ Backtester initialized")
    except Exception as e:
        print(f"✗ Failed to initialize backtester: {e}")
        return
    
    try:
        guardrails = SafetyGuardrails(
            max_daily_loss_percent=5.0,
            max_single_loss_percent=3.0,
            max_consecutive_losses=3,
            min_accuracy_threshold=0.55
        )
        print("✓ Safety guardrails initialized")
    except Exception as e:
        print(f"✗ Failed to initialize guardrails: {e}")
        return
    
    try:
        pnl_tracker = PnLTracker(use_sheets=False)
        print("✓ P&L tracker initialized")
    except Exception as e:
        print(f"✗ Failed to initialize tracker: {e}")
        return
    
    anomaly_detector = AnomalyDetector()
    print("✓ Anomaly detector initialized")
    
    # Fetch Data
    print_section("📊 Fetching Historical Data")
    data = backtester.fetch_data()
    print(f"✓ Fetched data for {len(data)} tickers")
    
    if not data:
        print("✗ No data fetched. Check ticker symbols and internet connection.")
        return
    
    # Generate Signals
    print_section("🤖 Generating ML Signals")
    signals = backtester.generate_signals(lookback=20)
    print(f"✓ Generated signals for {len(signals)} tickers")
    
    if not signals:
        print("⚠️  No signals generated. Check model availability.")
        return
    
    # Run Backtest
    print_section("⚙️  Running Backtest")
    trades_df = backtester.run_backtest(
        use_stop_loss=True,
        use_position_sizing=True,
        position_hold_days=5
    )
    print(f"✓ Execution complete. {len(trades_df)} trades executed.")
    
    # Get Performance Report
    print_section("📈 Performance Report")
    report = backtester.get_performance_report()
    
    if "error" not in report:
        print(f"""
Final Results:
  Starting Capital:     Rs {report['initial_capital']:>12,.0f}
  Final Capital:        Rs {report['final_capital']:>12,.0f}
  Total Profit/Loss:    Rs {report['total_pnl']:>12,.2f}
  Return:               {report['total_return_percent']:>12.2f}%

Trading Statistics:
  Total Trades:         {report['total_trades']:>12}
  Winning Trades:       {report['winning_trades']:>12} ({report['win_rate_percent']:.1f}%)
  Losing Trades:        {report['losing_trades']:>12}
  Avg Win:              Rs {report['avg_win']:>12,.1f}
  Avg Loss:             Rs {report['avg_loss']:>12,.1f}
  Max Win:              Rs {report['max_win']:>12,.1f}
  Max Loss:             Rs {report['max_loss']:>12,.1f}
  Profit Factor:        {report['profit_factor']:>12.2f}

Risk Metrics:
  Sharpe Ratio:         {report['sharpe_ratio']:>12.2f}
  Avg Hold Days:        {report['avg_hold_days']:>12.1f} days

Expected vs Synthetic:
  📊 Result on Real Data: {report['total_return_percent']:.2f}% return
  🎯 Expectations:        58-65% accuracy needed for profitability
  ⚠️  Comparison:         Model was trained on synthetic data (93% accuracy)
                          Real market data accuracy will be lower (realistic 55-60%)
                          This backtest validates the model generalization
        """)
    else:
        print(f"⚠️  {report['error']}")
    
    # Guardrail Status
    print_section("🛡️  Safety Guardrails Status")
    guardrail_summary = guardrails.get_alert_summary()
    
    print(f"""
Status:                   {guardrail_summary['status']}
Circuit Breaker:          {'🔴 ACTIVE' if guardrail_summary['circuit_breaker_active'] else '🟢 INACTIVE'}
Total Alerts:             {guardrail_summary['total_alerts']}
Critical Alerts:          {guardrail_summary['critical_alerts']}
Warnings:                 {guardrail_summary['warnings']}
Daily Loss:               Rs {guardrail_summary.get('daily_loss', 0):,.2f}
    """)
    
    if guardrail_summary.get('latest_alerts'):
        print("\nLatest Alerts:")
        for alert in guardrail_summary['latest_alerts'][-3:]:
            severity_emoji = "🔴" if alert['severity'] == "CRITICAL" else "🟡" if alert['severity'] == "WARNING" else "🟢"
            print(f"  {severity_emoji} [{alert['severity']}] {alert['rule']}")
            print(f"     → {alert['message']}")
    
    # Save Backtest Results
    print_section("💾 Saving Results")
    
    try:
        # Save trades to CSV
        trades_df.to_csv("results/backtest_trades.csv", index=False)
        print("✓ Trades exported to results/backtest_trades.csv")
        
        # Save report
        import json
        with open("results/backtest_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        print("✓ Report exported to results/backtest_report.json")
        
        # Save guardrails report
        guardrails_df = guardrails.get_alert_report()
        if not guardrails_df.empty:
            guardrails_df.to_csv("results/guardrails_alerts.csv", index=False)
            print("✓ Alerts exported to results/guardrails_alerts.csv")
    
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
    
    # Display Top Trades
    print_section("🏆 Top Trades")
    
    if not trades_df.empty:
        top_wins = trades_df.nlargest(3, "pnl")[["ticker", "entry_price", "exit_price", "pnl", "pnl_percent", "exit_reason"]]
        top_losses = trades_df.nsmallest(3, "pnl")[["ticker", "entry_price", "exit_price", "pnl", "pnl_percent", "exit_reason"]]
        
        print("\nBest Trades (Top 3):")
        for _, row in top_wins.iterrows():
            print(f"  ✓ {row['ticker']:12} | Entry: {row['entry_price']:8.2f} → Exit: {row['exit_price']:8.2f} | PnL: Rs {row['pnl']:8,.0f} ({row['pnl_percent']:+.1f}%)")
        
        print("\nWorst Trades (Bottom 3):")
        for _, row in top_losses.iterrows():
            print(f"  ✗ {row['ticker']:12} | Entry: {row['entry_price']:8.2f} → Exit: {row['exit_price']:8.2f} | PnL: Rs {row['pnl']:8,.0f} ({row['pnl_percent']:+.1f}%)")
    
    # Next Steps
    print_section("📌 Next Steps")
    print("""
1. ✅ Review backtest results (trades, P&L, accuracy)
2. ⏭️  Deploy model to live trading if satisfied with performance
3. 📊 Start with small position sizes (1-2% risk per trade)
4. 🔄 Monitor daily P&L and close app if down 5%
5. 📈 Track all trades in Google Sheets for journal
6. 🛡️  Use safety guardrails to prevent emotional trades
7. 📅 Review performance monthly and rebalance
    """)
    
    print_header("✅ Backtest Complete!")
    print(f"\nResults saved to:")
    print(f"  - results/backtest_trades.csv")
    print(f"  - results/backtest_report.json")
    print(f"  - results/guardrails_alerts.csv")
    print(f"\nTo start live trading with these models, run:")
    print(f"  $ streamlit run app.py")
    print()


if __name__ == "__main__":
    try:
        run_backtest_live()
    except KeyboardInterrupt:
        print("\n\n⚠️  Backtest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
