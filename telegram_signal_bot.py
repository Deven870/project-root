#!/usr/bin/env python3
"""
Telegram Signal Bot - Send trading signals to Telegram
=======================================================

Sends daily trading signals from the ML model to a Telegram group/channel.

Usage:
    python telegram_signal_bot.py send      # Send signals now
    python telegram_signal_bot.py daemon    # Run daily scheduler
    python telegram_signal_bot.py test      # Test connection
"""

import requests
import json
import logging
from pathlib import Path
from datetime import datetime
import pytz
import os
from apscheduler.schedulers.background import BackgroundScheduler

IST = pytz.timezone("Asia/Kolkata")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/telegram_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TelegramBot')


class TelegramBot:
    """Send messages to Telegram using Bot API."""
    
    def __init__(self, bot_token=None, chat_id=None):
        """Initialize Telegram bot.
        
        Args:
            bot_token: Telegram bot token from @BotFather
            chat_id: Target chat ID (group or channel)
        """
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID', '')
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.history_file = Path('logs/telegram_history.json')
    
    def send_message(self, message):
        """Send text message to Telegram.
        
        Args:
            message: Message text to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not configured")
            return False
        
        try:
            url = f"{self.api_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"✓ Message sent to Telegram")
                return True
            else:
                logger.error(f"✗ Telegram error: {response.text}")
                return False
        
        except Exception as e:
            logger.error(f"✗ Failed to send message: {e}")
            return False
    
    def send_document(self, file_path):
        """Send file to Telegram.
        
        Args:
            file_path: Path to file (CSV, JSON, etc.)
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not configured")
            return False
        
        try:
            url = f"{self.api_url}/sendDocument"
            
            with open(file_path, 'rb') as f:
                files = {'document': f}
                data = {'chat_id': self.chat_id}
                response = requests.post(url, files=files, data=data, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"✓ File sent to Telegram: {file_path}")
                return True
            else:
                logger.error(f"✗ Telegram error: {response.text}")
                return False
        
        except Exception as e:
            logger.error(f"✗ Failed to send file: {e}")
            return False
    
    def test_connection(self):
        """Test Telegram bot connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.bot_token or not self.chat_id:
            logger.error("✗ Telegram credentials not configured")
            logger.error("  Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
            return False
        
        try:
            # Get bot info to test connection
            url = f"{self.api_url}/getMe"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                bot_info = response.json()
                bot_name = bot_info.get('result', {}).get('username', 'Unknown')
                logger.info(f"✓ Telegram bot connected: @{bot_name}")
                return True
            else:
                logger.error(f"✗ Telegram error: {response.text}")
                return False
        
        except Exception as e:
            logger.error(f"✗ Connection failed: {e}")
            return False


def load_daily_signals():
    """Load today's signals from JSON file.
    
    Returns:
        Dictionary of signals or None if file doesn't exist
    """
    signals_file = Path('logs/daily_signals.json')
    
    if not signals_file.exists():
        logger.warning("No signals file found")
        return None
    
    try:
        with open(signals_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load signals: {e}")
        return None


def format_signal_message(signals_data):
    """Format signals for Telegram message.
    
    Args:
        signals_data: Dictionary of signals
        
    Returns:
        Formatted message string with emojis and tables
    """
    if not signals_data:
        return "📊 No signals generated today"
    
    date = signals_data.get('date', datetime.now(IST).strftime('%Y-%m-%d'))
    signals = signals_data.get('signals', [])
    summary = signals_data.get('summary', {})
    
    # Format message
    message = f"""
📊 <b>Trading Signals - {date}</b>

<b>Signal Summary</b>
🟢 BUY: {summary.get('total_buy', 0)} signals
🔴 SELL: {summary.get('total_sell', 0)} signals  
⚪ HOLD: {summary.get('total_hold', 0)} signals
📊 Total: {len(signals)} stocks analyzed

<b>Key Signals</b>
"""
    
    # Add BUY signals
    buy_signals = [s for s in signals if s.get('prediction') == 1]
    if buy_signals:
        message += "\n🟢 <b>BUY Opportunities:</b>\n"
        for sig in buy_signals[:5]:  # Show top 5
            ticker = sig.get('ticker', '').replace('.NS', '')
            confidence = sig.get('confidence', 0) * 100
            ret = sig.get('expected_return', 0) * 100
            price = sig.get('last_close', 0)
            message += f"  • {ticker}: {confidence:.0f}% confidence, {ret:+.2f}% target (₹{price:.2f})\n"
    
    # Add SELL signals  
    sell_signals = [s for s in signals if s.get('prediction') == -1]
    if sell_signals:
        message += "\n🔴 <b>SELL Signals:</b>\n"
        for sig in sell_signals[:5]:  # Show top 5
            ticker = sig.get('ticker', '').replace('.NS', '')
            confidence = sig.get('confidence', 0) * 100
            ret = sig.get('expected_return', 0) * 100
            price = sig.get('last_close', 0)
            message += f"  • {ticker}: {confidence:.0f}% confidence, {ret:+.2f}% target (₹{price:.2f})\n"
    
    message += f"\n⏰ Next update: Tomorrow 08:30 IST\n"
    message += "📲 Dashboard: voicebot.trading\n"
    
    return message


def send_signals():
    """Send today's signals to Telegram immediately."""
    logger.info("Sending signals to Telegram...")
    
    signals = load_daily_signals()
    if not signals:
        logger.warning("No signals available")
        return False
    
    bot = TelegramBot()
    message = format_signal_message(signals)
    
    success = bot.send_message(message)
    
    if success:
        save_sent_message(message, signals)
    
    return success


def save_sent_message(message, signals_data):
    """Save sent message to history file.
    
    Args:
        message: Message that was sent
        signals_data: Original signals data
    """
    try:
        history_file = Path('logs/telegram_history.json')
        
        # Load existing history
        history = {}
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        
        # Add new entry
        date = signals_data.get('date', datetime.now(IST).strftime('%Y-%m-%d'))
        history[date] = {
            'timestamp': datetime.now(IST).isoformat(),
            'sent': True,
            'signal_count': len(signals_data.get('signals', [])),
            'message_preview': message[:200]
        }
        
        # Save updated history
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Message saved to history")
    
    except Exception as e:
        logger.error(f"Failed to save message history: {e}")


def run_daemon():
    """Run background scheduler to send signals daily at 9:00 AM IST."""
    logger.info("Starting Telegram bot daemon...")
    
    scheduler = BackgroundScheduler()
    
    # Schedule for 9:00 AM IST (after signals are generated at 8:30 AM)
    job = scheduler.add_job(
        send_signals,
        'cron',
        hour=9,
        minute=0,
        timezone='Asia/Kolkata',
        id='daily_telegram_signal'
    )
    
    logger.info("✓ Scheduled to send signals daily at 09:00 AM IST")
    
    try:
        scheduler.start()
        logger.info("✓ Daemon started. Press Ctrl+C to stop.")
        
        # Keep running
        import time
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Daemon stopped")
        scheduler.shutdown()
    
    except Exception as e:
        logger.error(f"Daemon error: {e}")
        scheduler.shutdown()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'send':
            send_signals()
        
        elif command == 'daemon':
            run_daemon()
        
        elif command == 'test':
            bot = TelegramBot()
            if bot.test_connection():
                logger.info("✓ Ready to send signals!")
            else:
                logger.error("✗ Connection failed. Check credentials in .env")
        
        else:
            print("Usage: python telegram_signal_bot.py [send|daemon|test]")
    
    else:
        print("Usage: python telegram_signal_bot.py [send|daemon|test]")
        print("  send   - Send signals now")
        print("  daemon - Run daily scheduler")
        print("  test   - Test connection")
