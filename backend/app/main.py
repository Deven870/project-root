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
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    logger.info("✅ All services initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 NSEIQ API Server Shutting Down...")


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
