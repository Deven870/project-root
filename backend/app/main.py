"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    NSEIQ v5.0 - API SERVER                                ║
║         Institutional NSE Stock Intelligence & Trading System               ║
╚════════════════════════════════════════════════════════════════════════════╝

FastAPI backend with:
  - 6-layer stock prediction engine
  - Portfolio generation & optimization
  - Google Sheets real-time logging
  - Backtesting & validation
  - Risk management & alerts
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import logging
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Background task for live predictions
background_tasks_set = set()

# ═════════════════════════════════════════════════════════════════════════════
# APP CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

APP_NAME = "NSEIQ v5.0"
APP_VERSION = "5.0.0"
APP_TITLE = "Institutional NSE Stock Intelligence System"
CORS_ORIGINS = ["*"]

# ═════════════════════════════════════════════════════════════════════════════
# CREATE FASTAPI APP
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_TITLE,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ═════════════════════════════════════════════════════════════════════════════
# MIDDLEWARE
# ═════════════════════════════════════════════════════════════════════════════

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═════════════════════════════════════════════════════════════════════════════
# ROOT & HEALTH ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Root endpoint - NSEIQ System Info"""
    return {
        "status": "🚀 LIVE",
        "app": APP_NAME,
        "version": APP_VERSION,
        "title": APP_TITLE,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predictions": "/api/v1/nseiq/predict",
            "portfolio": "/api/v1/nseiq/portfolio",
            "backtest": "/api/v1/nseiq/backtest",
            "pre_market": "/api/v1/nseiq/pre-market-brief",
        },
    }


@app.get("/health")
async def health():
    """Global health check"""
    return {
        "status": "healthy",
        "service": APP_NAME,
        "version": APP_VERSION,
    }


@app.get("/api/v1/health")
async def api_health():
    """API v1 health check"""
    return {
        "status": "healthy",
        "version": "v1",
        "app": APP_NAME,
    }


# ═════════════════════════════════════════════════════════════════════════════
# WEBSOCKET ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/predictions")
async def websocket_live_predictions(websocket: WebSocket):
    """
    WebSocket endpoint for real-time live predictions
    
    Clients connect here to receive continuous prediction updates
    """
    from .ws_manager import manager
    from .services.live_prediction_service import get_live_prediction_service
    
    try:
        await manager.connect(websocket)
        logger.info("📊 Client connected to live predictions feed")
        
        # Send initial predictions
        service = get_live_prediction_service()
        initial_data = {
            "type": "initial_predictions",
            "timestamp": datetime.now().isoformat(),
            "data": service.get_current_predictions(),
            "service_stats": service.get_stats()
        }
        await websocket.send_json(initial_data)
        
        # Keep connection alive
        while True:
            # Receive any client messages (for subscription management)
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30)
                
                if data.get("action") == "subscribe":
                    symbol = data.get("symbol")
                    manager.subscribe(websocket, symbol)
                    
                elif data.get("action") == "unsubscribe":
                    symbol = data.get("symbol")
                    manager.unsubscribe(websocket, symbol)
                    
                elif data.get("action") == "get_stats":
                    service = get_live_prediction_service()
                    stats = {
                        "type": "service_stats",
                        "data": service.get_stats()
                    }
                    await websocket.send_json(stats)
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                heartbeat = {
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send_json(heartbeat)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("📊 Client disconnected from live predictions feed")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        if websocket in manager.active_connections:
            manager.disconnect(websocket)


@app.websocket("/ws/stock/{symbol}")
async def websocket_stock_updates(websocket: WebSocket, symbol: str):
    """
    WebSocket endpoint for specific stock updates
    
    Connect to receive updates for a specific stock symbol
    """
    from .ws_manager import manager
    from .services.live_prediction_service import get_live_prediction_service
    
    symbol = symbol.upper()
    
    try:
        await manager.connect(websocket)
        manager.subscribe(websocket, symbol)
        logger.info(f"📌 Client subscribed to {symbol}")
        
        # Send initial data for stock
        service = get_live_prediction_service()
        stock_data = service.get_current_predictions(symbol)
        
        initial_msg = {
            "type": "stock_initial",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "data": stock_data
        }
        await websocket.send_json(initial_msg)
        
        # Keep connection alive
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=60)
            except asyncio.TimeoutError:
                # Send heartbeat
                heartbeat = {
                    "type": "heartbeat",
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send_json(heartbeat)
                
    except WebSocketDisconnect:
        manager.unsubscribe(websocket, symbol)
        manager.disconnect(websocket)
        logger.info(f"📌 Client unsubscribed from {symbol}")
    except Exception as e:
        logger.error(f"❌ Stock WebSocket error for {symbol}: {e}")
        if websocket in manager.active_connections:
            manager.disconnect(websocket)


# ═════════════════════════════════════════════════════════════════════════════
# LIVE SERVICE ENDPOINTS (HTTP API)
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/live/status")
async def live_service_status():
    """Get live prediction service status"""
    from .services.live_prediction_service import get_live_prediction_service
    
    service = get_live_prediction_service()
    return service.get_stats()


@app.get("/api/v1/live/predictions")
async def live_predictions(stock: str = None):
    """Get current live predictions (HTTP fallback)"""
    from .services.live_prediction_service import get_live_prediction_service
    
    service = get_live_prediction_service()
    
    if stock:
        return {
            "ticker": stock,
            "data": service.get_current_predictions(stock.upper())
        }
    
    return {
        "count": len(service.get_current_predictions()),
        "data": service.get_current_predictions()
    }


@app.post("/api/v1/live/refresh")
async def refresh_predictions_now():
    """Manually trigger prediction refresh"""
    from .services.live_prediction_service import get_live_prediction_service
    
    service = get_live_prediction_service()
    
    try:
        predictions = await service.fetch_batch_predictions()
        
        if predictions:
            service.current_predictions = predictions
            await service.broadcast_predictions(predictions)
            
        return {
            "status": "success",
            "predictions_updated": len(predictions)
        }
    except Exception as e:
        logger.error(f"❌ Manual refresh error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


# ═════════════════════════════════════════════════════════════════════════════
# INCLUDE NSEIQ ROUTER
# ═════════════════════════════════════════════════════════════════════════════

try:
    from .api.nseiq import router as nseiq_router
    app.include_router(nseiq_router)
    logger.info("✅ NSEIQ router registered")
except ImportError as e:
    logger.error(f"⚠️  Failed to import NSEIQ router: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# STARTUP & SHUTDOWN EVENTS
# ═════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🔌 NSEIQ API Server Starting...")
    logger.info(f"📊 {APP_NAME} v{APP_VERSION}")
    
    # Initialize WebSocket connection manager
    from .ws_manager import manager
    logger.info("✅ WebSocket manager initialized")
    
    # Initialize and start live prediction service
    try:
        from .services.live_prediction_service import create_live_prediction_service
        live_service = create_live_prediction_service(manager)
        
        # Create background task for live predictions
        task = asyncio.create_task(live_service.start())
        background_tasks_set.add(task)
        task.add_done_callback(background_tasks_set.discard)
        
        logger.info("🟢 Live Prediction Service STARTED")
        logger.info("📡 Real-time predictions enabled via WebSocket")
    except Exception as e:
        logger.error(f"⚠️  Failed to start live prediction service: {e}")
        logger.info("⏸️  Continuing with API-only mode (manual predictions)")
    
    logger.info("✅ All services initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 NSEIQ API Server Shutting Down...")
    
    # Stop live prediction service
    try:
        from .services.live_prediction_service import get_live_prediction_service
        service = get_live_prediction_service()
        await service.stop()
        logger.info("✅ Live Prediction Service stopped")
    except Exception as e:
        logger.error(f"⚠️  Error stopping live service: {e}")
    
    # Cancel background tasks
    for task in background_tasks_set:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


# ═════════════════════════════════════════════════════════════════════════════
# STARTUP
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"🚀 Starting {APP_NAME} v{APP_VERSION}")
    logger.info("📡 API Server: http://localhost:8000")
    logger.info("📖 Swagger UI: http://localhost:8000/docs")
    logger.info("📘 ReDoc: http://localhost:8000/redoc")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
