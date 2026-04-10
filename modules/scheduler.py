import logging
from datetime import datetime

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from modules.alert_engine import fire_buy_signal, fire_sell_signal, send_eod_report, send_top10_alert
from modules.finnhub_feed import get_realtime_quote
from modules.sheets_live import (
    close_trade,
    get_open_trades,
    get_worksheet,
    setup_sheet_headers,
    update_live_prices,
    update_pnl_dashboard,
)
from modules.auto_trader import run_daily_paper_trading, run_daily_validation_check
from modules.top10_scanner import get_top10

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

_fired_alerts_today = set()
_fired_alerts_day = None

# Job execution tracking for status panel
_job_last_run = {
    "morning_scan": None,
    "check_signals_live": None,
    "check_stoploss_breaches": None,
    "refresh_open_trade_prices": None,
    "eod_report": None,
    "daily_paper_trading": None,
    "daily_validation_check": None,
}
_scheduler_instance = None


def _now_ist():
    return datetime.now(IST)


def _is_market_minutes(now=None):
    now = now or _now_ist()
    if now.weekday() >= 5:
        return False
    hhmm = now.hour * 100 + now.minute
    return 915 <= hhmm <= 1525


def _reset_day_state_if_needed():
    global _fired_alerts_day
    today = _now_ist().strftime("%Y-%m-%d")
    if _fired_alerts_day != today:
        _fired_alerts_today.clear()
        _fired_alerts_day = today


def _alert_already_fired_today(symbol):
    _reset_day_state_if_needed()
    day = _now_ist().strftime("%Y-%m-%d")
    key = (day, symbol)
    if key in _fired_alerts_today:
        return True

    try:
        ws = get_worksheet("Live Signals")
        rows = ws.get_all_records()
        for row in rows:
            if (
                str(row.get("Date", "")).strip() == day
                and str(row.get("Symbol", "")).strip() == symbol
                and str(row.get("AlertFired", "")).strip().upper() in ("YES", "TRUE", "1")
            ):
                _fired_alerts_today.add(key)
                return True
    except Exception as e:
        logger.warning("Could not verify prior alerts from sheet: %s", e)

    return False


def _mark_alert_fired(symbol):
    _reset_day_state_if_needed()
    day = _now_ist().strftime("%Y-%m-%d")
    _fired_alerts_today.add((day, symbol))


def _track_job_run(job_id):
    """Record last run time for a job."""
    global _job_last_run
    _job_last_run[job_id] = _now_ist()


def run_morning_scan():
    """
    Run morning scans for Intraday and Swing and send top-10 alerts.
    """
    _track_job_run("morning_scan")
    try:
        setup_sheet_headers()
    except Exception:
        pass

    intraday_df = get_top10(horizon="Intraday")
    swing_df = get_top10(horizon="Swing")

    if intraday_df is not None and not intraday_df.empty:
        send_top10_alert(intraday_df, "Intraday", _now_ist().strftime("%Y-%m-%d %H:%M:%S"))

    if swing_df is not None and not swing_df.empty:
        send_top10_alert(swing_df, "Swing", _now_ist().strftime("%Y-%m-%d %H:%M:%S"))

    total = int((0 if intraday_df is None else len(intraday_df)) + (0 if swing_df is None else len(swing_df)))
    logger.info("Morning scan complete: %s signals found", total)


def check_signals_live():
    """
    Check high-confidence live signals and fire deduplicated buy alerts.
    """
    _track_job_run("check_signals_live")
    if not _is_market_minutes():
        return

    live_df = get_top10(horizon="Intraday", min_confidence=0.75)
    if live_df is None or live_df.empty:
        return

    for _, row in live_df.iterrows():
        symbol = str(row.get("Stock", "")).strip()
        if not symbol or _alert_already_fired_today(symbol):
            continue

        try:
            fire_buy_signal(
                symbol=symbol,
                price=float(row.get("CurrentPrice", 0.0)),
                predicted_price=float(row.get("PredictedPrice", 0.0)),
                stop_loss=float(row.get("StopLoss", 0.0)),
                confidence=float(row.get("Confidence", 0.0)),
                sentiment_score=float(row.get("SentimentScore", 0.0)),
                horizon="Intraday",
            )
            _mark_alert_fired(symbol)
        except Exception as e:
            logger.warning("Buy alert failed for %s: %s", symbol, e)


def check_stoploss_breaches():
    """
    Monitor open trades for stop-loss breaches and target hits.
    """
    _track_job_run("check_stoploss_breaches")
    if not _is_market_minutes():
        return

    open_trades = get_open_trades()
    for trade in open_trades:
        try:
            trade_id = str(trade.get("TradeID", "")).strip()
            symbol = str(trade.get("Symbol", "")).strip()
            stop_loss = float(trade.get("StopLoss", 0.0) or 0.0)
            target = float(trade.get("Target", 0.0) or 0.0)
            buy_price = float(trade.get("BuyPrice", 0.0) or 0.0)

            quote = get_realtime_quote(symbol)
            current_price = float(quote.get("price", 0.0) or 0.0)
            if current_price <= 0:
                continue

            pnl_pct = ((current_price - buy_price) / buy_price * 100) if buy_price > 0 else 0.0

            if stop_loss > 0 and current_price <= stop_loss:
                fire_sell_signal(symbol, current_price, "stop_loss_breach", pnl_pct)
                if trade_id:
                    close_trade(trade_id, current_price)
                continue

            if target > 0 and current_price >= target:
                fire_sell_signal(symbol, current_price, "target_reached", pnl_pct)
                if trade_id:
                    close_trade(trade_id, current_price)
        except Exception as e:
            logger.warning("Stop-loss monitor error for trade %s: %s", trade.get("TradeID", ""), e)


def _refresh_open_trade_prices():
    """
    Build a symbol:price map from Finnhub and push batch updates to My Trades.
    """
    _track_job_run("refresh_open_trade_prices")
    if not _is_market_minutes():
        return

    open_trades = get_open_trades()
    if not open_trades:
        return

    symbols = sorted({str(t.get("Symbol", "")).strip() for t in open_trades if str(t.get("Symbol", "")).strip()})
    price_dict = {}
    for symbol in symbols:
        try:
            quote = get_realtime_quote(symbol)
            price = float(quote.get("price", 0.0) or 0.0)
            if price > 0:
                price_dict[symbol] = price
        except Exception:
            continue

    if price_dict:
        update_live_prices(price_dict)


def _build_today_pnl_summary():
    """
    Build EOD summary from today's closed trades.
    """
    today = _now_ist().strftime("%Y-%m-%d")
    ws = get_worksheet("My Trades")
    rows = ws.get_all_records()
    closed_today = [
        r for r in rows
        if str(r.get("Status", "")).strip().upper() == "CLOSED"
        and str(r.get("SellTime", "")).startswith(today)
    ]

    pnl_values = [float(r.get("PnL_Rs", 0.0) or 0.0) for r in closed_today]
    wins = [x for x in pnl_values if x > 0]
    losses = [x for x in pnl_values if x < 0]

    return {
        "total_pnl": float(sum(pnl_values)) if pnl_values else 0.0,
        "win_count": len(wins),
        "loss_count": len(losses),
        "best_trade": max(pnl_values) if pnl_values else 0.0,
        "worst_trade": min(pnl_values) if pnl_values else 0.0,
        "total_trades": len(closed_today),
    }


def run_eod_report():
    """
    Update dashboard and send end-of-day summary alert.
    """
    _track_job_run("eod_report")
    update_pnl_dashboard()
    try:
        summary = _build_today_pnl_summary()
        send_eod_report(summary)
    except Exception as e:
        logger.warning("EOD summary send failed: %s", e)


def start_scheduler():
    """
    Start APScheduler with IST jobs for market automation.
    """
    global _scheduler_instance
    scheduler = BackgroundScheduler(timezone=IST)

    scheduler.add_job(
        run_morning_scan,
        CronTrigger(day_of_week="mon-fri", hour=9, minute=0, timezone=IST),
        id="morning_scan",
        replace_existing=True,
    )

    scheduler.add_job(
        check_signals_live,
        CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/5", timezone=IST),
        id="check_signals_live",
        replace_existing=True,
    )

    scheduler.add_job(
        check_stoploss_breaches,
        CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/2", timezone=IST),
        id="check_stoploss_breaches",
        replace_existing=True,
    )

    scheduler.add_job(
        _refresh_open_trade_prices,
        IntervalTrigger(seconds=60, timezone=IST),
        id="refresh_open_trade_prices",
        replace_existing=True,
    )

    scheduler.add_job(
        run_eod_report,
        CronTrigger(day_of_week="mon-fri", hour=15, minute=45, timezone=IST),
        id="eod_report",
        replace_existing=True,
    )

    scheduler.add_job(
        run_daily_paper_trading,
        CronTrigger(day_of_week="mon-fri", hour=9, minute=15, timezone=IST),
        id="daily_paper_trading",
        replace_existing=True,
    )

    scheduler.add_job(
        run_daily_validation_check,
        CronTrigger(day_of_week="mon-fri", hour=15, minute=35, timezone=IST),
        id="daily_validation_check",
        replace_existing=True,
    )

    scheduler.start()
    _scheduler_instance = scheduler
    logger.info("Scheduler started (IST).")
    return scheduler


def get_scheduler_status():
    """
    Return scheduler status and last run times for each job.
    
    Returns:
        dict: {
            'is_running': bool,
            'job_last_run': dict with job_id: datetime or None,
            'next_run_times': dict with job_id: next run time
        }
    """
    is_running = _scheduler_instance is not None and _scheduler_instance.running
    
    next_runs = {}
    if is_running and _scheduler_instance:
        for job in _scheduler_instance.get_jobs():
            next_runs[job.id] = job.next_run_time
    
    return {
        'is_running': is_running,
        'job_last_run': _job_last_run.copy(),
        'next_run_times': next_runs,
    }


def stop_scheduler(scheduler):
    """
    Gracefully stop the running scheduler.
    """
    if scheduler is None:
        return
    try:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")
    except Exception as e:
        logger.warning("Scheduler stop warning: %s", e)
