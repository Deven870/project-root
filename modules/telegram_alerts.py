"""
FIX 6: Telegram Alert System for Trade Notifications
Sends real-time trade signals and alerts to Telegram.
"""

import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


async def _send(text):
    """Send a message to Telegram via Bot API."""
    if not BOT_TOKEN or not CHAT_ID:
        print("Warning: Telegram credentials not configured in .env")
        return
    
    try:
        from telegram import Bot
        bot = Bot(token=BOT_TOKEN)
        await bot.send_message(
            chat_id=CHAT_ID,
            text=text,
            parse_mode="HTML"
        )
    except Exception as e:
        print(f"Telegram send error: {e}")


def send_telegram_alert(signals: list):
    """
    Send daily signal summary to Telegram.
    
    Parameters
    ----------
    signals : list
        List of prediction dicts from get_stock_predictions, sorted by confidence
    """
    try:
        lines = ["🚀 <b>Digitrader Daily Signals</b>"]
        
        for i, sig in enumerate(signals[:3], 1):
            emoji = "📈" if "bull" in sig.get("trend", "").lower() else "📉"
            symbol = sig.get("symbol", "N/A")
            cur_price = sig.get("current_price", 0)
            pred_price = sig.get("predicted_price", 0)
            ret_pct = sig.get("predicted_return_pct", 0)
            conf = sig.get("confidence", 0) * 100
            
            lines.append(
                f"\n{i}. {emoji} <b>{symbol}</b>\n"
                f"   Current: ₹{cur_price:,.0f} → Predicted: ₹{pred_price:,.0f}\n"
                f"   Return: {ret_pct:+.1f}%  |  Confidence: {conf:.0f}%"
            )
        
        lines.append(
            "\n\n⚠️ <i>Paper trade only — always verify before acting.</i>"
        )
        
        msg = "\n".join(lines)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_send(msg))
        
    except Exception as e:
        print(f"Error sending Telegram alert: {e}")


def send_stop_alert(symbol, price, stop_loss):
    """Send stop-loss hit alert."""
    try:
        msg = (
            f"🛑 <b>STOP-LOSS HIT</b>\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Price: ₹{price:,.2f}\n"
            f"Stop: ₹{stop_loss:,.2f}"
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_send(msg))
    except Exception as e:
        print(f"Error sending stop alert: {e}")


def send_target_alert(symbol, price, target):
    """Send target hit alert."""
    try:
        msg = (
            f"🎯 <b>TARGET HIT</b>\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Price: ₹{price:,.2f}\n"
            f"Target: ₹{target:,.2f}"
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_send(msg))
    except Exception as e:
        print(f"Error sending target alert: {e}")


def send_alert_message(text):
    """Send a custom alert message."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_send(text))
    except Exception as e:
        print(f"Error sending message: {e}")
