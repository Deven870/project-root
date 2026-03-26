"""
Comprehensive alert system for Digitrader.
Supports Telegram (async) and Gmail notifications.
All functions have silent fallbacks if credentials not configured.
"""
import os
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional, Dict, List
from dotenv import load_dotenv
import pandas as pd

# Try to use nest_asyncio to handle already-running loops
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

try:
    from telegram import Bot
except ImportError:
    Bot = None

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")

_TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
_EMAIL_ENABLED = bool(GMAIL_ADDRESS and GMAIL_APP_PASSWORD)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _run_async(coro):
    """
    Safely run async coroutine even if event loop is already running.
    Handles RuntimeError when loop is running in Jupyter/Streamlit.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop running, safe to use asyncio.run()
        return asyncio.run(coro)
    else:
        # Loop already running, use create_task
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()


# ============================================================================
# TELEGRAM FUNCTIONS
# ============================================================================

async def _send_telegram_async(message: str, parse_mode: str = "Markdown") -> bool:
    """
    Async implementation of Telegram send.
    """
    if not _TELEGRAM_ENABLED:
        return False
    
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=message,
            parse_mode=parse_mode
        )
        return True
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


def send_telegram(message: str, parse_mode: str = "Markdown") -> bool:
    """
    Send message to Telegram (synchronous wrapper).
    
    Parameters
    ----------
    message : str
        Message text (supports Markdown if parse_mode="Markdown")
    parse_mode : str
        "Markdown" or "HTML"
    
    Returns
    -------
    bool
        True if sent successfully, False otherwise
    """
    if not _TELEGRAM_ENABLED:
        return False
    
    try:
        return _run_async(_send_telegram_async(message, parse_mode))
    except Exception as e:
        print(f"Error sending Telegram: {e}")
        return False


# ============================================================================
# EMAIL FUNCTIONS
# ============================================================================

def send_email(subject: str, html_body: str) -> bool:
    """
    Send HTML email via Gmail SMTP.
    
    Parameters
    ----------
    subject : str
        Email subject
    html_body : str
        Email body in HTML format
    
    Returns
    -------
    bool
        True if sent successfully, False otherwise
    """
    if not _EMAIL_ENABLED:
        return False
    
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = GMAIL_ADDRESS
        msg["To"] = GMAIL_ADDRESS
        
        # Attach HTML
        msg.attach(MIMEText(html_body, "html"))
        
        # Send via Gmail SMTP
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, GMAIL_ADDRESS, msg.as_string())
        
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False


# ============================================================================
# BUY SIGNAL
# ============================================================================

def fire_buy_signal(
    symbol: str,
    price: float,
    predicted_price: float,
    stop_loss: float,
    confidence: float,
    sentiment_score: float,
    horizon: str = "intraday"
) -> bool:
    """
    Send buy signal alert (Telegram + Email).
    Only fires if confidence >= 0.70.
    
    Parameters
    ----------
    symbol : str
        Stock symbol (e.g., "RELIANCE.NS")
    price : float
        Current stock price
    predicted_price : float
        Predicted target price
    stop_loss : float
        Stop loss price
    confidence : float
        Model confidence (0-1)
    sentiment_score : float
        Sentiment score (-1 to 1)
    horizon : str
        "intraday", "short_term", "medium_term"
    
    Returns
    -------
    bool
        True if alert was sent
    """
    if confidence < 0.70:
        print(f"Confidence {confidence:.2%} below 0.70 threshold. Signal not fired.")
        return False
    
    expected_return = ((predicted_price - price) / price) * 100
    risk_reward = ((predicted_price - price) / (price - stop_loss)) if price != stop_loss else 0
    
    # Telegram message (Markdown)
    telegram_msg = f"""
🚀 **BUY SIGNAL** - {horizon.upper()}

**Stock:** {symbol}
**Current Price:** ₹{price:.2f}
**Target Price:** 🎯 ₹{predicted_price:.2f}
**Stop Loss:** 🛑 ₹{stop_loss:.2f}

**Expected Return:** +{expected_return:.2f}%
**Risk/Reward Ratio:** {risk_reward:.2f}x

**Confidence:** {confidence:.1%}
**Sentiment:** {'Positive' if sentiment_score > 0 else 'Negative' if sentiment_score < 0 else 'Neutral'} ({sentiment_score:.2f})

⏰ Alert Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Email message (HTML)
    email_html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f5f5f5; }}
            .container {{ max-width: 600px; margin: 20px auto; background: white; padding: 20px; border-radius: 8px; }}
            .header {{ background: #28a745; color: white; padding: 20px; border-radius: 8px; text-align: center; font-size: 24px; font-weight: bold; }}
            .details {{ margin: 20px 0; }}
            .row {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #eee; }}
            .label {{ font-weight: bold; color: #333; }}
            .value {{ color: #666; }}
            .positive {{ color: #28a745; font-weight: bold; }}
            .footer {{ margin-top: 20px; font-size: 12px; color: #999; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">🚀 BUY SIGNAL - {horizon.upper()}</div>
            <div class="details">
                <div class="row">
                    <span class="label">Stock:</span>
                    <span class="value">{symbol}</span>
                </div>
                <div class="row">
                    <span class="label">Current Price:</span>
                    <span class="value">₹{price:.2f}</span>
                </div>
                <div class="row">
                    <span class="label">Target Price:</span>
                    <span class="value positive">₹{predicted_price:.2f}</span>
                </div>
                <div class="row">
                    <span class="label">Stop Loss:</span>
                    <span class="value">₹{stop_loss:.2f}</span>
                </div>
                <div class="row">
                    <span class="label">Expected Return:</span>
                    <span class="value positive">+{expected_return:.2f}%</span>
                </div>
                <div class="row">
                    <span class="label">Risk/Reward Ratio:</span>
                    <span class="value">{risk_reward:.2f}x</span>
                </div>
                <div class="row">
                    <span class="label">Confidence:</span>
                    <span class="value positive">{confidence:.1%}</span>
                </div>
                <div class="row">
                    <span class="label">Sentiment:</span>
                    <span class="value">{'Positive' if sentiment_score > 0 else 'Negative' if sentiment_score < 0 else 'Neutral'} ({sentiment_score:.2f})</span>
                </div>
            </div>
            <div class="footer">
                Alert Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}
            </div>
        </div>
    </body>
    </html>
    """
    
    send_telegram(telegram_msg)
    send_email(f"🚀 BUY Signal: {symbol}", email_html)
    
    return True


# ============================================================================
# SELL SIGNAL
# ============================================================================

def fire_sell_signal(
    symbol: str,
    price: float,
    reason: str,
    pnl_pct: float
) -> bool:
    """
    Send urgent sell signal alert.
    
    Parameters
    ----------
    symbol : str
        Stock symbol
    price : float
        Current price
    reason : str
        "stop_loss_breach", "target_reached", or "manual"
    pnl_pct : float
        Profit/Loss percentage
    
    Returns
    -------
    bool
        True if alert was sent
    """
    reason_emoji = "🔴" if reason == "stop_loss_breach" else "✅" if reason == "target_reached" else "⚪"
    color = "#dc3545" if reason == "stop_loss_breach" else "#28a745" if reason == "target_reached" else "#6c757d"
    reason_text = "STOP LOSS" if reason == "stop_loss_breach" else "TARGET REACHED" if reason == "target_reached" else "MANUAL EXIT"
    
    # Telegram
    telegram_msg = f"""
{reason_emoji} **SELL SIGNAL** - {reason_text}

**Stock:** {symbol}
**Exit Price:** ₹{price:.2f}
**P&L:** {'+' if pnl_pct > 0 else ''}{pnl_pct:.2f}%

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Email HTML
    email_html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f5f5f5; }}
            .container {{ max-width: 600px; margin: 20px auto; background: white; padding: 20px; border-radius: 8px; }}
            .header {{ background: {color}; color: white; padding: 20px; border-radius: 8px; text-align: center; font-size: 24px; font-weight: bold; }}
            .details {{ margin: 20px 0; }}
            .row {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #eee; }}
            .label {{ font-weight: bold; color: #333; }}
            .value {{ color: #666; }}
            .pnl {{ font-weight: bold; color: {color}; }}
            .footer {{ margin-top: 20px; font-size: 12px; color: #999; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">{reason_emoji} SELL SIGNAL - {reason_text}</div>
            <div class="details">
                <div class="row">
                    <span class="label">Stock:</span>
                    <span class="value">{symbol}</span>
                </div>
                <div class="row">
                    <span class="label">Exit Price:</span>
                    <span class="value">₹{price:.2f}</span>
                </div>
                <div class="row">
                    <span class="label">P&L:</span>
                    <span class="pnl">{'+' if pnl_pct > 0 else ''}{pnl_pct:.2f}%</span>
                </div>
            </div>
            <div class="footer">
                {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}
            </div>
        </div>
    </body>
    </html>
    """
    
    send_telegram(telegram_msg)
    send_email(f"{reason_emoji} SELL Signal: {symbol}", email_html)
    
    return True


# ============================================================================
# TOP 10 ALERT
# ============================================================================

def send_top10_alert(top10_df: pd.DataFrame, horizon: str, scan_time: str = None) -> bool:
    """
    Send top 10 stocks alert with formatted Markdown (Telegram) and HTML (Email).
    
    Parameters
    ----------
    top10_df : pd.DataFrame
        Must have columns: Stock, Trend, Confidence, CurrentPrice, PredictedPrice, ExpectedReturn
    horizon : str
        "intraday", "short_term", etc.
    scan_time : str
        Timestamp of scan (optional)
    
    Returns
    -------
    bool
        True if alerts were sent
    """
    if top10_df.empty or len(top10_df) == 0:
        print("No stocks to alert")
        return False
    
    scan_time = scan_time or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Telegram message (Markdown table)
    telegram_msg = f"""
📊 **TOP 10 STOCKS** - {horizon.upper()}
Scan Time: {scan_time}

"""
    
    telegram_msg += "| Stock | Trend | Confidence | Current | Predicted | Return |\n"
    telegram_msg += "|-------|-------|-----------|---------|-----------|--------|\n"
    
    for _, row in top10_df.iterrows():
        trend_emoji = "📈" if row.get("Trend", "BULLISH") == "BULLISH" else "📉"
        telegram_msg += f"| {row.get('Stock', 'N/A')} | {trend_emoji} | {row.get('Confidence', 0):.1%} | ₹{row.get('CurrentPrice', 0):.2f} | ₹{row.get('PredictedPrice', 0):.2f} | +{row.get('ExpectedReturn', 0):.2f}% |\n"
    
    # Email HTML table
    email_html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f5f5f5; }}
            .container {{ max-width: 1000px; margin: 20px auto; background: white; padding: 20px; border-radius: 8px; }}
            .header {{ background: #007bff; color: white; padding: 20px; border-radius: 8px; text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th {{ background: #333; color: white; padding: 12px; text-align: left; font-weight: bold; }}
            td {{ padding: 12px; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .bullish {{ background-color: #d4edda; color: #155724; font-weight: bold; }}
            .bearish {{ background-color: #f8d7da; color: #721c24; font-weight: bold; }}
            .positive {{ color: #28a745; font-weight: bold; }}
            .footer {{ margin-top: 20px; font-size: 12px; color: #999; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">📊 TOP 10 STOCKS - {horizon.upper()}</div>
            <table>
                <tr>
                    <th>Stock</th>
                    <th>Trend</th>
                    <th>Confidence</th>
                    <th>Current Price</th>
                    <th>Predicted Price</th>
                    <th>Expected Return</th>
                </tr>
    """
    
    for _, row in top10_df.iterrows():
        trend = row.get("Trend", "BULLISH")
        trend_class = "bullish" if trend == "BULLISH" else "bearish"
        email_html += f"""
                <tr class="{trend_class}">
                    <td>{row.get('Stock', 'N/A')}</td>
                    <td>{trend}</td>
                    <td>{row.get('Confidence', 0):.1%}</td>
                    <td>₹{row.get('CurrentPrice', 0):.2f}</td>
                    <td>₹{row.get('PredictedPrice', 0):.2f}</td>
                    <td class="positive">+{row.get('ExpectedReturn', 0):.2f}%</td>
                </tr>
        """
    
    email_html += """
            </table>
            <div class="footer">"""
    email_html += f"Scan Time: {scan_time} IST\n"
    email_html += """
            </div>
        </div>
    </body>
    </html>
    """
    
    send_telegram(telegram_msg)
    send_email(f"📊 Top 10 Stocks - {horizon.upper()}", email_html)
    
    return True


# ============================================================================
# EOD REPORT
# ============================================================================

def send_eod_report(pnl_summary: Dict) -> bool:
    """
    Send end-of-day P&L summary report.
    
    Parameters
    ----------
    pnl_summary : dict
        Keys: total_pnl, win_count, loss_count, best_trade, worst_trade, total_trades
    
    Returns
    -------
    bool
        True if alerts were sent
    """
    total_pnl = pnl_summary.get("total_pnl", 0)
    win_count = pnl_summary.get("win_count", 0)
    loss_count = pnl_summary.get("loss_count", 0)
    best_trade = pnl_summary.get("best_trade", 0)
    worst_trade = pnl_summary.get("worst_trade", 0)
    total_trades = pnl_summary.get("total_trades", 0)
    
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
    
    # Telegram
    telegram_msg = f"""
📈 **END OF DAY REPORT**

**Total P&L:** {pnl_emoji} ₹{total_pnl:.2f}
**Total Trades:** {total_trades}
**Wins:** ✅ {win_count} | **Losses:** ❌ {loss_count}
**Win Rate:** {win_rate:.1f}%

**Best Trade:** ₹{best_trade:.2f}
**Worst Trade:** ₹{worst_trade:.2f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Email HTML
    color = "#28a745" if total_pnl >= 0 else "#dc3545"
    email_html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f5f5f5; }}
            .container {{ max-width: 600px; margin: 20px auto; background: white; padding: 20px; border-radius: 8px; }}
            .header {{ background: {color}; color: white; padding: 20px; border-radius: 8px; text-align: center; font-size: 24px; font-weight: bold; }}
            .summary {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
            .metric {{ background: #f9f9f9; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid {color}; }}
            .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; margin-bottom: 10px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: {color}; }}
            .details {{ margin: 20px 0; }}
            .row {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #eee; }}
            .label {{ font-weight: bold; color: #333; }}
            .value {{ color: #666; }}
            .footer {{ margin-top: 20px; font-size: 12px; color: #999; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">📈 END OF DAY REPORT</div>
            <div class="summary">
                <div class="metric">
                    <div class="metric-label">Total P&L</div>
                    <div class="metric-value">₹{total_pnl:.2f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Trades</div>
                    <div class="metric-value">{total_trades}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">{win_rate:.1f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Wins / Losses</div>
                    <div class="metric-value">{win_count} / {loss_count}</div>
                </div>
            </div>
            <div class="details">
                <div class="row">
                    <span class="label">Best Trade:</span>
                    <span class="value">₹{best_trade:.2f}</span>
                </div>
                <div class="row">
                    <span class="label">Worst Trade:</span>
                    <span class="value">₹{worst_trade:.2f}</span>
                </div>
            </div>
            <div class="footer">
                {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}
            </div>
        </div>
    </body>
    </html>
    """
    
    send_telegram(telegram_msg)
    send_email("📈 End of Day Report", email_html)
    
    return True


# ============================================================================
# TEST/MAIN
# ============================================================================

if __name__ == "__main__":
    print("Alert Engine Configuration:")
    print(f"  Telegram: {'✓' if _TELEGRAM_ENABLED else '✗'}")
    print(f"  Email: {'✓' if _EMAIL_ENABLED else '✗'}")
    
    if not _TELEGRAM_ENABLED:
        print("  → Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to enable")
    if not _EMAIL_ENABLED:
        print("  → Set GMAIL_ADDRESS and GMAIL_APP_PASSWORD to enable")
