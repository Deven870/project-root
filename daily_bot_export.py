"""
═══════════════════════════════════════════════════════════════════════════════
Daily Bot Data Export to CSV (Automated)
═══════════════════════════════════════════════════════════════════════════════
Runs daily, exports bot data to timestamped CSV files, ready for Google Sheets
"""

import csv
from datetime import datetime
from pathlib import Path
import json
import shutil
import logging

# Setup logging
log_file = Path.cwd() / "logs" / "bot_export.log"
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
EXPORT_DIR = Path.cwd() / "bot_exports"
ARCHIVE_DIR = Path.cwd() / "bot_exports" / "archive"
EXPORT_DIR.mkdir(exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

def get_bot_data():
    """Get bot data from API or local files"""
    # In production, this would fetch from the API
    # For now, returning sample data structure
    return {
        "status": {
            "state": "RUNNING",
            "uptime_seconds": 3600,
            "trades_executed": 1,
            "positions_open": 1,
            "last_signal_time": datetime.now().isoformat()
        },
        "account": {
            "initial_capital": 300000,
            "current_capital": 300000,
            "deployed_capital": 0,
            "available_capital": 300000,
            "total_pnl": 0,
            "pnl_percent": 0
        },
        "positions": [
            {
                "ticker": "M&M",
                "quantity": 480,
                "entry_price": 450,
                "entry_value": 216000,
                "current_price": 450,
                "current_value": 216000,
                "pnl": 0,
                "pnl_percent": 0,
                "target_price": 530,
                "stop_loss": 400,
                "entry_time": "2026-04-15 11:56:00",
                "status": "OPEN"
            }
        ],
        "trades": [
            {
                "trade_id": "BOT_20260415_001",
                "ticker": "M&M",
                "type": "BUY",
                "quantity": 480,
                "entry_price": 450,
                "entry_value": 216000,
                "exit_price": None,
                "exit_value": None,
                "pnl": None,
                "pnl_percent": None,
                "entry_time": "2026-04-15 11:56:00",
                "exit_time": None,
                "status": "OPEN",
                "signal": "STRONG_BUY",
                "confidence": 80.5
            }
        ]
    }

def export_positions_csv(bot_data, filename):
    """Export positions to CSV"""
    csv_file = EXPORT_DIR / filename
    
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'Timestamp', 'Ticker', 'Quantity', 'Entry Price', 'Entry Value',
                'Current Price', 'Current Value', 'P&L', 'P&L %', 'Target Price',
                'Stop Loss', 'Entry Time', 'Status'
            ])
            writer.writeheader()
            
            for pos in bot_data.get('positions', []):
                writer.writerow({
                    'Timestamp': datetime.now().isoformat(),
                    'Ticker': pos['ticker'],
                    'Quantity': pos['quantity'],
                    'Entry Price': pos['entry_price'],
                    'Entry Value': pos['entry_value'],
                    'Current Price': pos['current_price'],
                    'Current Value': pos['current_value'],
                    'P&L': pos['pnl'],
                    'P&L %': pos['pnl_percent'],
                    'Target Price': pos['target_price'],
                    'Stop Loss': pos['stop_loss'],
                    'Entry Time': pos['entry_time'],
                    'Status': pos['status']
                })
        
        logger.info(f"✅ Positions exported: {csv_file}")
        return csv_file
    except Exception as e:
        logger.error(f"❌ Error exporting positions: {e}")
        return None

def export_trades_csv(bot_data, filename):
    """Export trades to CSV"""
    csv_file = EXPORT_DIR / filename
    
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'Timestamp', 'Trade ID', 'Ticker', 'Type', 'Quantity', 'Entry Price',
                'Entry Value', 'Exit Price', 'Exit Value', 'P&L', 'P&L %',
                'Entry Time', 'Exit Time', 'Status', 'Signal', 'Confidence'
            ])
            writer.writeheader()
            
            for trade in bot_data.get('trades', []):
                writer.writerow({
                    'Timestamp': datetime.now().isoformat(),
                    'Trade ID': trade['trade_id'],
                    'Ticker': trade['ticker'],
                    'Type': trade['type'],
                    'Quantity': trade['quantity'],
                    'Entry Price': trade['entry_price'],
                    'Entry Value': trade['entry_value'],
                    'Exit Price': trade['exit_price'],
                    'Exit Value': trade['exit_value'],
                    'P&L': trade['pnl'],
                    'P&L %': trade['pnl_percent'],
                    'Entry Time': trade['entry_time'],
                    'Exit Time': trade['exit_time'],
                    'Status': trade['status'],
                    'Signal': trade['signal'],
                    'Confidence': trade['confidence']
                })
        
        logger.info(f"✅ Trades exported: {csv_file}")
        return csv_file
    except Exception as e:
        logger.error(f"❌ Error exporting trades: {e}")
        return None

def export_stats_csv(bot_data, filename):
    """Export daily stats to CSV"""
    csv_file = EXPORT_DIR / filename
    
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'Date', 'Time', 'Initial Capital', 'Current Capital', 'Deployed Capital',
                'Available Capital', 'Total P&L', 'P&L %', 'Positions Open',
                'Trades Executed', 'Status'
            ])
            writer.writeheader()
            
            account = bot_data.get('account', {})
            status = bot_data.get('status', {})
            
            writer.writerow({
                'Date': datetime.now().strftime("%Y-%m-%d"),
                'Time': datetime.now().strftime("%H:%M:%S"),
                'Initial Capital': account['initial_capital'],
                'Current Capital': account['current_capital'],
                'Deployed Capital': account['deployed_capital'],
                'Available Capital': account['available_capital'],
                'Total P&L': account['total_pnl'],
                'P&L %': account['pnl_percent'],
                'Positions Open': status['positions_open'],
                'Trades Executed': status['trades_executed'],
                'Status': status['state']
            })
        
        logger.info(f"✅ Stats exported: {csv_file}")
        return csv_file
    except Exception as e:
        logger.error(f"❌ Error exporting stats: {e}")
        return None

def export_json(bot_data, filename):
    """Export all data to JSON"""
    json_file = EXPORT_DIR / filename
    
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(bot_data, f, indent=2, default=str)
        
        logger.info(f"✅ JSON exported: {json_file}")
        return json_file
    except Exception as e:
        logger.error(f"❌ Error exporting JSON: {e}")
        return None

def archive_old_exports(days_to_keep=30):
    """Archive exports older than specified days"""
    try:
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 86400)
        
        for file in EXPORT_DIR.glob("bot_*.csv"):
            if file.stat().st_mtime < cutoff_date:
                archived_file = ARCHIVE_DIR / file.name
                shutil.move(str(file), str(archived_file))
                logger.info(f"📦 Archived: {file.name}")
    except Exception as e:
        logger.error(f"⚠️  Archiving error: {e}")

def main():
    logger.info("=" * 80)
    logger.info("🤖 NSEIQ Bot Daily Export Started")
    logger.info("=" * 80)
    
    try:
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get bot data
        logger.info("📦 Fetching bot data...")
        bot_data = get_bot_data()
        
        # Export all formats with timestamp
        logger.info("📤 Exporting to CSV files...\n")
        
        files_exported = []
        
        # Latest versions (for manual upload)
        files_exported.append(export_positions_csv(bot_data, "bot_positions.csv"))
        files_exported.append(export_trades_csv(bot_data, "bot_trades.csv"))
        files_exported.append(export_stats_csv(bot_data, "bot_stats.csv"))
        files_exported.append(export_json(bot_data, "bot_data.json"))
        
        # Timestamped versions (for archiving)
        export_positions_csv(bot_data, f"bot_positions_{timestamp}.csv")
        export_trades_csv(bot_data, f"bot_trades_{timestamp}.csv")
        export_stats_csv(bot_data, f"bot_stats_{timestamp}.csv")
        export_json(bot_data, f"bot_data_{timestamp}.json")
        
        # Archive old files
        logger.info("🗂️  Archiving old exports...")
        archive_old_exports(days_to_keep=30)
        
        logger.info("=" * 80)
        logger.info("✅ Daily export completed successfully!")
        logger.info(f"📁 Export directory: {EXPORT_DIR}")
        logger.info(f"📊 Files ready for Google Sheets upload")
        logger.info("=" * 80)
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Export failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
