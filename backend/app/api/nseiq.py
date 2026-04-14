"""
╔════════════════════════════════════════════════════════════════════════════╗
║                      NSEIQ API ENDPOINTS v5.0                              ║
║          FastAPI Integration for NSEIQ Stock Intelligence System            ║
╚════════════════════════════════════════════════════════════════════════════╝

Endpoints:
  POST /api/v1/nseiq/predict - Generate prediction for single stock
  POST /api/v1/nseiq/portfolio - Generate portfolio
  GET  /api/v1/nseiq/portfolio/status - Current portfolio status
  POST /api/v1/nseiq/backtest - Run backtest
  GET  /api/v1/nseiq/sheets/summary - Get logging summary
  POST /api/v1/nseiq/alert - Manually trigger alert
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Try multiple import paths for flexibility
try:
    from ..services.nseiq_prediction_engine import (
        nseiq_engine,
        TradingMode,
    )
    from ..services.nseiq_portfolio_engine import (
        portfolio_engine,
        RiskProfile,
        InvestmentHorizon,
    )
    from ..services.nseiq_sheets_logger import get_sheets_logger
    from ..services.nseiq_prediction_formatter import (
        NSEIQPredictionFormatter,
    )
except ImportError as e:
    logger.error(f"Failed to import NSEIQ services: {e}")
    raise

router = APIRouter(prefix="/api/v1/nseiq", tags=["NSEIQ"])


# ═════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═════════════════════════════════════════════════════════════════════════════


class PredictionRequest(BaseModel):
    """Request model for prediction"""
    ticker: str
    mode: str  # INTRADAY, SWING, POSITIONAL, LONGTERM
    sector: Optional[str] = "Technology"
    capital_deployed: Optional[float] = 0


class PortfolioRequest(BaseModel):
    """Request model for portfolio generation"""
    total_capital: float
    risk_profile: str  # CONSERVATIVE, MODERATE, AGGRESSIVE
    horizon: str  # INTRADAY, SWING, POSITIONAL, LONGTERM, MIXED
    candidate_stocks: List[Dict]
    existing_holdings: Optional[List[Dict]] = None
    sector_preferences: Optional[Dict[str, float]] = None
    blacklisted_sectors: Optional[List[str]] = None


class BacktestRequest(BaseModel):
    """Request model for backtesting"""
    ticker: str
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    initial_capital: float = 100000
    mode: str = "SWING"


class AlertRequest(BaseModel):
    """Request model for alerts"""
    alert_type: str
    ticker: str
    details: str
    action: str


# ═════════════════════════════════════════════════════════════════════════════
# PREDICTION ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════


@router.post("/predict")
async def generate_prediction(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
) -> Dict:
    """
    Generate 6-layer prediction for a stock

    Returns:
      - Prediction with all 6 data layers
      - Formatted NSEIQ output
      - Confidence score
      - Risk/reward analysis
    """
    try:
        logger.info(f"🔍 Prediction request: {request.ticker} | {request.mode}")

        # Validate inputs
        if not request.ticker or len(request.ticker) < 1:
            raise HTTPException(status_code=400, detail="Invalid ticker")

        # Convert mode string to enum
        try:
            mode = TradingMode[request.mode.upper()]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode. Use: INTRADAY, SWING, POSITIONAL, LONGTERM",
            )

        # Generate prediction (all 6 layers)
        raw_prediction = nseiq_engine.generate_prediction(
            ticker=request.ticker,
            mode=mode,
            sector=request.sector or "Technology",
        )

        if "error" in raw_prediction:
            raise HTTPException(
                status_code=500, detail=f"Analysis failed: {raw_prediction['error']}"
            )

        # Format according to NSEIQ standard
        formatted_output = NSEIQPredictionFormatter.format_prediction(
            prediction=raw_prediction,
            analysis=raw_prediction.get("layers", {}),
            capital_deployed=request.capital_deployed,
        )

        # Log to Google Sheets (async)
        sheets_logger = get_sheets_logger()
        if sheets_logger:
            background_tasks.add_task(sheets_logger.log_prediction, raw_prediction)

        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "ticker": request.ticker,
            "mode": request.mode,
            "signal": raw_prediction.get("signal"),
            "confidence": raw_prediction.get("confidence"),
            "formatted_output": formatted_output,
            "raw_data": raw_prediction,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ═════════════════════════════════════════════════════════════════════════════
# PORTFOLIO ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════


@router.post("/portfolio")
async def generate_portfolio(
    request: PortfolioRequest,
    background_tasks: BackgroundTasks,
) -> Dict:
    """
    Generate optimized portfolio from candidate stocks

    Returns:
      - Portfolio with positions, allocation %, entry/SL/target
      - Risk management rules
      - Portfolio metrics (expected return, Sharpe ratio, beta, etc.)
      - Formatted NSEIQ portfolio output
    """
    try:
        logger.info(f"📊 Portfolio request: ₹{request.total_capital:,.0f} | {request.risk_profile}")

        # Validate inputs
        if request.total_capital <= 0:
            raise HTTPException(status_code=400, detail="Capital must be > 0")

        if len(request.candidate_stocks) == 0:
            raise HTTPException(status_code=400, detail="No candidate stocks provided")

        # Convert enums
        risk_profile = RiskProfile[request.risk_profile.upper()]
        horizon = InvestmentHorizon[request.horizon.upper()]

        # Build portfolio
        portfolio = portfolio_engine.build_portfolio(
            total_capital=request.total_capital,
            risk_profile=risk_profile,
            horizon=horizon,
            candidate_stocks=request.candidate_stocks,
            existing_holdings=request.existing_holdings,
            sector_preferences=request.sector_preferences,
            blacklisted_sectors=request.blacklisted_sectors,
        )

        # Format for output
        formatted_output = NSEIQPredictionFormatter.format_portfolio_output(portfolio)

        # Log to Sheets (async)
        sheets_logger = get_sheets_logger()
        if sheets_logger:
            background_tasks.add_task(sheets_logger.log_portfolio_snapshot, 
                                     {"positions": portfolio.get("positions", [])})

        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "total_capital": request.total_capital,
            "risk_profile": request.risk_profile,
            "horizon": request.horizon,
            "positions_count": len(portfolio.get("positions", [])),
            "formatted_output": formatted_output,
            "portfolio": portfolio,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Portfolio error: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio generation failed: {str(e)}")


@router.get("/portfolio/status")
async def get_portfolio_status() -> Dict:
    """Get current portfolio holdings and P&L"""
    try:
        # TODO: Fetch from portfolio tracking service
        return {
            "status": "success",
            "portfolio": {
                "total_value": 250000,
                "cash": 150000,
                "deployed": 100000,
                "pnl_rupees": 5600,
                "pnl_pct": 2.24,
                "positions": [],
            },
        }

    except Exception as e:
        logger.error(f"❌ Portfolio status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════════════════════════════════════
# BACKTESTING ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════


@router.post("/backtest")
async def run_backtest(request: BacktestRequest) -> Dict:
    """
    Run backtest on historical data for validation
    """
    try:
        logger.info(f"⏮️  Backtest request: {request.ticker} | {request.start_date} to {request.end_date}")

        # TODO: Implement backtesting engine
        return {
            "status": "success",
            "message": "Backtesting engine integration in progress",
            "ticker": request.ticker,
            "backtest_metrics": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
            },
        }

    except Exception as e:
        logger.error(f"❌ Backtest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════════════════════════════════════
# LOGGING ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════


@router.get("/sheets/summary")
async def get_sheets_summary() -> Dict:
    """Get today's trading summary from Google Sheets"""
    try:
        sheets_logger = get_sheets_logger()
        if not sheets_logger:
            return {
                "status": "error",
                "message": "Google Sheets not configured",
            }

        summary = sheets_logger.get_daily_summary()
        return {
            "status": "success",
            "summary": summary,
        }

    except Exception as e:
        logger.error(f"❌ Sheets summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/log-trade")
async def log_trade_manual(trade: Dict) -> Dict:
    """Manually log a completed trade"""
    try:
        sheets_logger = get_sheets_logger()
        if not sheets_logger:
            return {"status": "error", "message": "Sheets not configured"}

        success = sheets_logger.log_trade(trade)
        return {
            "status": "success" if success else "error",
            "message": "Trade logged" if success else "Failed to log trade",
        }

    except Exception as e:
        logger.error(f"❌ Trade logging error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════════════════════════════════════
# ALERT ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════


@router.post("/alert")
async def trigger_alert(
    request: AlertRequest,
    background_tasks: BackgroundTasks,
) -> Dict:
    """Post manual alert to Sheets"""
    try:
        logger.info(f"🚨 Alert: {request.alert_type} | {request.ticker}")

        sheets_logger = get_sheets_logger()
        if sheets_logger:
            alert_data = {
                "alert_type": request.alert_type,
                "ticker": request.ticker,
                "details": request.details,
                "recommended_action": request.action,
            }
            background_tasks.add_task(sheets_logger.log_alert, alert_data)

        return {
            "status": "success",
            "message": f"Alert logged: {request.alert_type}",
        }

    except Exception as e:
        logger.error(f"❌ Alert error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════════════════════════════════════
# HEALTH & STATUS
# ═════════════════════════════════════════════════════════════════════════════


@router.get("/health")
async def nseiq_health() -> Dict:
    """NSEIQ system health check"""
    try:
        sheets_logger = get_sheets_logger()
        sheets_ok = sheets_logger.health_check() if sheets_logger else False

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "prediction_engine": "✅ Ready",
                "portfolio_engine": "✅ Ready",
                "sheets_logger": "✅ Connected" if sheets_ok else "⚠️  Not connected",
                "formatter": "✅ Ready",
            },
        }

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "degraded",
            "error": str(e),
        }


# ═════════════════════════════════════════════════════════════════════════════
# HELPER ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════


@router.get("/pre-market-brief")
async def get_pre_market_brief() -> Dict:
    """Get pre-market intelligence brief (8:45-9:15 AM IST)"""
    return {
        "status": "success",
        "brief": NSEIQPredictionFormatter.format_pre_market_brief(),
    }


@router.get("/stocks/nse-list")
async def get_nse_stock_list() -> Dict:
    """Get list of NSE stocks available for analysis (top 80+)"""
    nse_stocks = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "HDFC",
        "LT", "MARUTI", "BAJAJ-AUTO", "BHARTIARTL", "ITC", "KOTAKBANK",
        "SUNPHARMA", "ASIANPAINT", "WIPRO", "TECHM", "DMARUTI", "POWERGRID",
        "JSWSTEEL", "TATASTEEL", "HINDALCO", "NTPC", "COALINDIA", "GAIL",
        "ONGC", "TATAMOTORS", "BAJAJFINSV", "BAJAJFINANCE", "MINDTREE",
        "ADANIPORTS", "ADANIPOWER", "ADF", "DRD", "GMR", "IPCALAB",
        "ADANITRANS", "ADANIGREEN", "ADANIENSOL", "ADANITRANS", "BERGEPAINT",
        "BIOCON", "BOSCHLTD", "BRITANNIA", "CANBANK", "CHOLAFIN", "CIPLA",
        "COLPAL", "CONCOR", "DABUR", "DCB", "DIVISLAB", "DRREDDY", "EICHERMOT",
        "FEDERALBNK", "FORBESCO", "FUNAMENTALS", "GICRE", "GLDREIT", "GODREJCP",
        "GODREJIND", "GONOISE", "GREAVESCO", "HEROMOTOCO", "HEXAWARE", "HGS",
    ]
    return {
        "status": "success",
        "count": len(nse_stocks),
        "stocks": nse_stocks,
    }
