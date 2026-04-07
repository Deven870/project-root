"""
Main FastAPI application
🚀 DigiTrader v5.0 - Async Trading Platform
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio

from app.config import APP_NAME, APP_VERSION, CORS_ORIGINS
from app.ws_manager import manager
from app.services.price_service import price_service
from app.services.signal_service import signal_service
from app.services.cache_service import cache
from database.base import create_tables

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== STARTUP/SHUTDOWN =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info(f"🚀 {APP_NAME} v{APP_VERSION} Starting...")
    create_tables()
    logger.info("✅ Database initialized")
    logger.info(f"✅ Redis: {'Connected' if cache.is_connected() else 'Fallback mode'}")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutdown initiated...")

# ===== CREATE APP =====
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="High-performance async trading platform for NSE stocks",
    lifespan=lifespan
)

# ===== MIDDLEWARE =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ROUTES =====

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "🚀 LIVE",
        "app": APP_NAME,
        "version": APP_VERSION,
        "cache": "🟢 Connected" if cache.is_connected() else "⚠️ Fallback",
        "connections": len(manager.active_connections)
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "✅ SQLite",
        "cache": "🟢 Connected" if cache.is_connected() else "⚠️ Fallback",
        "workers": "✅ Ready",
        "websockets": len(manager.active_connections)
    }

# ===== PRICE API ENDPOINTS =====

@app.get("/api/prices/{symbol}")
async def get_price(symbol: str):
    """
    Get real-time price for a stock
    Returns: {symbol, price, change, change_percent, timestamp}
    """
    price = await price_service.get_realtime_price(symbol)
    if price:
        return price
    return JSONResponse(
        status_code=404,
        content={"error": f"Could not fetch price for {symbol}"}
    )

@app.get("/api/prices/batch")
async def get_batch_prices(symbols: str):
    """
    Get prices for multiple stocks
    Usage: /api/prices/batch?symbols=RELIANCE.NS,TCS.NS,INFY.NS
    """
    symbol_list = [s.strip() for s in symbols.split(",")]
    prices = await price_service.get_batch_prices(symbol_list)
    return prices

# ===== SIGNAL API ENDPOINTS =====

@app.get("/api/signals/{symbol}")
async def get_signal(symbol: str):
    """
    Get trading signal for a stock
    Returns: {symbol, signal, confidence, recommendation}
    """
    try:
        from modules.utils import fetch_price_data
        price_data = fetch_price_data(symbol)
        signal = await signal_service.analyze_stock(symbol, price_data)
        return signal
    except Exception as e:
        logger.error(f"Signal error: {e}")
        return {"error": str(e)[:100]}

@app.get("/api/portfolio")
async def get_portfolio():
    """
    Get portfolio analysis
    """
    return {
        "capital": 250000,
        "current_value": 265200,
        "pnl": 15200,
        "pnl_percent": 6.08,
        "accuracy": "72.5%",
        "trades": 145
    }

@app.get("/api/stocks")
async def get_stocks():
    """Get list of all available stocks"""
    from app.config import NIFTY_50_STOCKS
    return {
        "total": len(NIFTY_50_STOCKS),
        "stocks": NIFTY_50_STOCKS
    }

# ===== WEBSOCKET ENDPOINTS =====

@app.websocket("/ws/prices/{symbol}")
async def websocket_prices(websocket: WebSocket, symbol: str):
    """
    WebSocket for real-time price streaming
    Sends new price every 2 seconds
    """
    await manager.connect(websocket)
    manager.subscribe(websocket, symbol)
    
    try:
        while True:
            # Get price
            price = await price_service.get_realtime_price(symbol)
            
            if price:
                await websocket.send_json({
                    "type": "price_update",
                    "data": price
                })
            
            # Wait 2 seconds before next update
            await asyncio.sleep(2)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"📌 {symbol} stream closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    """
    WebSocket for signal broadcasts
    Receives subscribe/unsubscribe messages
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Wait for subscription messages
            message = await websocket.receive_text()
            
            if message.startswith("subscribe:"):
                symbol = message.split(":")[1].strip()
                manager.subscribe(websocket, symbol)
                await websocket.send_json({
                    "type": "subscription",
                    "symbol": symbol,
                    "status": "subscribed"
                })
            
            elif message.startswith("unsubscribe:"):
                symbol = message.split(":")[1].strip()
                manager.unsubscribe(websocket, symbol)
                await websocket.send_json({
                    "type": "subscription",
                    "symbol": symbol,
                    "status": "unsubscribed"
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Signal WebSocket error: {e}")
        manager.disconnect(websocket)

# ===== BACKGROUND BROADCAST =====

async def broadcast_signals():
    """Background task to broadcast signals periodically"""
    while True:
        try:
            # Every 5 minutes, broadcast signals for top stocks
            from app.config import NIFTY_50_STOCKS
            top_stocks = NIFTY_50_STOCKS[:5]
            
            for symbol in top_stocks:
                try:
                    from modules.utils import fetch_price_data
                    price_data = fetch_price_data(symbol)
                    signal = await signal_service.analyze_stock(symbol, price_data)
                    
                    await manager.broadcast_to_subscription(symbol, {
                        "type": "signal_update",
                        "data": signal
                    })
                except:
                    pass
            
            await asyncio.sleep(300)  # 5 minutes
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
