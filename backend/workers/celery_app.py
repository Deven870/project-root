"""
Celery configuration and task definitions
"""
from celery import Celery
from app.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND
import logging

logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "digitrader",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,          # 30 min hard limit
    task_soft_time_limit=25 * 60,     # 25 min soft limit
)

# ===== ANALYSIS TASKS =====

@celery_app.task(name="analyze_stock", bind=True)
def analyze_stock_task(self, symbol: str):
    """
    Background task: Analyze single stock
    Can be called: analyze_stock_task.delay(symbol)
    """
    try:
        import asyncio
        from app.services.signal_service import signal_service
        from modules.utils import fetch_price_data
        
        logger.info(f"📊 Analyzing {symbol}...")
        
        price_data = fetch_price_data(symbol)
        if price_data is None or price_data.empty:
            return {"error": f"No data for {symbol}"}
        
        # Run async analysis
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            signal_service.analyze_stock(symbol, price_data)
        )
        
        logger.info(f"✅ {symbol}: {result.get('signal')}")
        return result
    
    except Exception as e:
        logger.error(f"❌ Analysis failed for {symbol}: {e}")
        return {"error": str(e)[:100]}

@celery_app.task(name="analyze_all_stocks")
def analyze_all_stocks_task():
    """
    Background task: Analyze all NSE stocks
    Queue all stocks for parallel analysis
    """
    from app.config import NIFTY_50_STOCKS
    
    logger.info(f"📊 Queueing {len(NIFTY_50_STOCKS)} stocks for analysis...")
    
    for symbol in NIFTY_50_STOCKS:
        analyze_stock_task.delay(symbol)
    
    return {
        "queued": len(NIFTY_50_STOCKS),
        "status": "Analysis jobs queued"
    }

@celery_app.task(name="update_prices")
def update_prices_task():
    """
    Background task: Update all prices
    Runs every minute
    """
    from app.config import NIFTY_50_STOCKS
    import asyncio
    from app.services.price_service import price_service
    from app.services.cache_service import cache
    
    logger.info("💰 Updating prices...")
    
    try:
        loop = asyncio.get_event_loop()
        prices = loop.run_until_complete(
            price_service.get_batch_prices(NIFTY_50_STOCKS[:10])  # Top 10
        )
        
        logger.info(f"✅ Updated {len(prices)} prices")
        return {"updated": len(prices)}
    
    except Exception as e:
        logger.error(f"Price update failed: {e}")
        return {"error": str(e)}

# ===== SCHEDULED TASKS =====

from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    # Every minute during market hours (9:15 AM - 3:30 PM)
    "update-prices": {
        "task": "update_prices",
        "schedule": crontab(minute="*/1", hour="9-15", day_of_week="0-4"),
    },
    # Every 5 minutes
    "analyze-all-stocks": {
        "task": "analyze_all_stocks",
        "schedule": crontab(minute="*/5", hour="9-15", day_of_week="0-4"),
    },
}

logger.info("✅ Celery configured with beat schedule")
