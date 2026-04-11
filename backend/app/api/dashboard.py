"""
Dashboard API Routes
Provides real-time data for the React dashboard
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/trading")
async def get_trading_data() -> Dict[str, Any]:
    """Get current trading data: portfolio, positions, signals"""
    try:
        return {
            "portfolio": {
                "total": 150000,
                "profit": 12500,
                "profitPercent": 8.33,
                "positions": [
                    {
                        "symbol": "RELIANCE.NS",
                        "quantity": 10,
                        "entryPrice": 2500.50,
                        "currentPrice": 2650.75,
                        "pnl": 1502.50,
                        "pnlPercent": 6.01
                    },
                    {
                        "symbol": "TCS.NS",
                        "quantity": 5,
                        "entryPrice": 3800.00,
                        "currentPrice": 3920.50,
                        "pnl": 602.50,
                        "pnlPercent": 3.17
                    },
                    {
                        "symbol": "INFY.NS",
                        "quantity": 20,
                        "entryPrice": 1350.00,
                        "currentPrice": 1385.25,
                        "pnl": 705.00,
                        "pnlPercent": 2.62
                    }
                ]
            },
            "signals": [
                {"symbol": "HDFCBANK.NS", "signal": "BUY", "strength": 0.85},
                {"symbol": "ICICIBANK.NS", "signal": "SELL", "strength": 0.72},
                {"symbol": "SBIN.NS", "signal": "HOLD", "strength": 0.55}
            ],
            "priceHistory": [
                {"date": f"{datetime.now() - timedelta(days=i):%Y-%m-%d}", "price": 150000 + (i * 500)}
                for i in range(7)
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching trading data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics")
async def get_analytics_data() -> Dict[str, Any]:
    """Get analytics data: returns, performance, risk metrics"""
    try:
        return {
            "dailyReturns": [
                {"date": f"{datetime.now() - timedelta(days=i):%Y-%m-%d}", "return": (i % 4) * 0.5 - 0.75}
                for i in range(30)
            ],
            "topPerformers": [
                {"symbol": "RELIANCE.NS", "return": 8.5},
                {"symbol": "TCS.NS", "return": 6.2},
                {"symbol": "HDFCBANK.NS", "return": 4.8},
                {"symbol": "ICICIBANK.NS", "return": 3.9},
                {"symbol": "INFY.NS", "return": 2.1}
            ],
            "riskMetrics": {
                "winRate": 72.5,
                "sharpeRatio": 1.85,
                "maxDrawdown": 12.5,
                "volatility": 18.3
            }
        }
    except Exception as e:
        logger.error(f"Error fetching analytics data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/paper-trading")
async def get_paper_trading_data() -> Dict[str, Any]:
    """Get paper trading account data"""
    try:
        trades = [
            {
                "date": datetime.now() - timedelta(hours=i),
                "symbol": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"][i % 4],
                "type": ["BUY", "SELL"][i % 2],
                "quantity": 5 + (i % 10),
                "entryPrice": 2500 + (i * 50),
                "exitPrice": 2550 + (i * 50),
                "pnl": (i * 100) + (50 if i % 2 == 0 else -30)
            }
            for i in range(20)
        ]
        
        total_pnl = sum(t["pnl"] for t in trades)
        winning_trades = len([t for t in trades if t["pnl"] > 0])
        
        return {
            "balance": 100000 + total_pnl,
            "trades": trades,
            "winRate": (winning_trades / len(trades) * 100) if trades else 0,
            "totalPnL": total_pnl
        }
    except Exception as e:
        logger.error(f"Error fetching paper trading data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system")
async def get_system_status() -> Dict[str, Any]:
    """Get system health and status"""
    try:
        return {
            "apiStatus": "healthy",
            "databaseStatus": "healthy",
            "schedulerStatus": "running",
            "lastUpdate": datetime.now().isoformat(),
            "uptime": "48h 23m",
            "environment": "production",
            "refreshRate": 30
        }
    except Exception as e:
        logger.error(f"Error fetching system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio")
async def get_portfolio_summary() -> Dict[str, Any]:
    """Get portfolio summary"""
    try:
        return {
            "totalValue": 150000,
            "dayChange": 2500,
            "dayChangePercent": 1.69,
            "portfolioAllocation": [
                {"name": "RELIANCE.NS", "value": 50000, "percent": 33.33},
                {"name": "TCS.NS", "value": 40000, "percent": 26.67},
                {"name": "INFY.NS", "value": 35000, "percent": 23.33},
                {"name": "HDFCBANK.NS", "value": 25000, "percent": 16.67}
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals")
async def get_all_signals() -> Dict[str, Any]:
    """Get all active trading signals"""
    try:
        return {
            "signals": [
                {"symbol": "RELIANCE.NS", "signal": "BUY", "strength": 0.92, "timestamp": datetime.now().isoformat()},
                {"symbol": "TCS.NS", "signal": "BUY", "strength": 0.85, "timestamp": datetime.now().isoformat()},
                {"symbol": "HDFCBANK.NS", "signal": "SELL", "strength": 0.78, "timestamp": datetime.now().isoformat()},
                {"symbol": "ICICIBANK.NS", "signal": "HOLD", "strength": 0.65, "timestamp": datetime.now().isoformat()},
                {"symbol": "SBIN.NS", "signal": "BUY", "strength": 0.88, "timestamp": datetime.now().isoformat()},
            ],
            "lastUpdate": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))
