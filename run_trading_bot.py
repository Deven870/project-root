"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    TRADING BOT LAUNCHER & MONITOR                          ║
║              Start watching live trades in real-time                        ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import sys
import logging
from datetime import datetime
import os
import io

# Fix UTF-8 encoding on Windows
if sys.platform == 'win32':
    # Wrap stdout with UTF-8 encoding
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"trading_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

from backend.app.services.trading_bot import create_trading_bot


async def main():
    """Main entry point"""
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                   🤖 NSEIQ TRADING BOT v1.0                               ║
║              Automated Trading using Live Predictions                      ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Bot configuration (from user specifications)
    config = {
        "api_base_url": "http://localhost:8000",
        "initial_capital": 300000,      # ₹300,000
        "min_confidence": 0.75,          # 75%
        "signal_filter": "STRONG_BUY",   # STRONG_BUY only
        "risk_per_trade": 0.08,          # 8%
        "daily_loss_limit": 0.07,        # 7%
        "max_positions": 4
    }
    
    print("\n📋 BOT CONFIGURATION:")
    print(f"   Capital: ₹{config['initial_capital']:,.0f}")
    print(f"   Min Confidence: {config['min_confidence']*100:.0f}%")
    print(f"   Signal Filter: {config['signal_filter']}")
    print(f"   Risk per Trade: {config['risk_per_trade']*100:.0f}%")
    print(f"   Daily Loss Limit: {config['daily_loss_limit']*100:.0f}%")
    print(f"   Max Open Positions: {config['max_positions']}")
    print()
    
    # Create bot
    bot = create_trading_bot(**config)
    
    # Confirm before start
    response = input("\n🚀 Start trading bot? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Bot startup cancelled")
        return
    
    print("\n✅ Starting trading bot...")
    print("📌 Press Ctrl+C to stop\n")
    
    try:
        # Start bot
        bot_task = asyncio.create_task(bot.start())
        
        # Monitor loop
        while bot.is_running:
            await asyncio.sleep(30)  # Print status every 30 seconds
            
            status = bot.get_bot_status()
            
            print("\n" + "="*70)
            print(f"📊 BOT STATUS - {datetime.now().strftime('%H:%M:%S')}")
            print("="*70)
            print(f"Status: {status['bot_status']}")
            print(f"Signals Received: {status['signals_received']}")
            print(f"Trades Placed: {status['trades_placed']} | Trades Closed: {status['trades_closed']}")
            print(f"Daily P&L: ₹{status['daily_pnl']:,.0f}")
            print(f"Open Positions: {status['open_positions']}")
            print(f"Current Capital: ₹{status['current_capital']:,.0f}")
            print(f"Capital Deployed: ₹{status['capital_deployed']:,.0f}")
            print(f"Win Rate: {status['win_rate']:.1f}%")
            print(f"Account Balance: ₹{status['account_balance']:,.0f}")
            print(f"Total P&L: ₹{status['total_pnl']:,.0f} ({status['pnl_percent']:+.2f}%)")
            
            positions = bot.get_positions()
            if positions:
                print(f"\nOpen Positions ({len(positions)}):")
                for pos in positions:
                    print(f"  • {pos['stock']}: ₹{pos['entry_price']:.2f} → ₹{pos['target_price']:.2f} (SL: ₹{pos['stop_loss']:.2f})")
            
            print("="*70)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping trading bot...")
        await bot.stop()
        
        # Final stats
        print("\n" + "="*70)
        print("📊 FINAL TRADING STATISTICS")
        print("="*70)
        
        final_status = bot.get_bot_status()
        print(f"Signals Received: {final_status['signals_received']}")
        print(f"Trades Placed: {final_status['trades_placed']}")
        print(f"Trades Closed: {final_status['trades_closed']}")
        print(f"Winning Trades: {final_status['winning_trades']}")
        print(f"Losing Trades: {final_status['losing_trades']}")
        print(f"Win Rate: {final_status['win_rate']:.1f}%")
        print(f"Daily P&L: ₹{final_status['daily_pnl']:,.0f}")
        print(f"Total Account P&L: ₹{final_status['total_pnl']:,.0f}")
        print(f"Account P&L %: {final_status['pnl_percent']:+.2f}%")
        print(f"Final Capital: ₹{final_status['current_capital']:,.0f}")
        print(f"Average Win: ₹{final_status['avg_win']:,.0f}")
        print(f"Average Loss: ₹{final_status['avg_loss']:,.0f}")
        print("="*70)
        
        # Export data
        print("\n💾 Exporting trade data...")
        trades_file = bot.account.export_trades_csv()
        stats_file = bot.account.export_stats_json()
        print(f"✅ Trades exported to: {trades_file}")
        print(f"✅ Stats exported to: {stats_file}")
        
        print("\n👋 Trading bot stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
