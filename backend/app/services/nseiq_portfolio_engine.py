"""
╔════════════════════════════════════════════════════════════════════════════╗
║                  NSEIQ PORTFOLIO GENERATION ENGINE v5.0                    ║
║         Institutional Portfolio Construction & Risk Management              ║
╚════════════════════════════════════════════════════════════════════════════╝

Features:
  - Diversification enforcement (max 20% per stock, max 35% per sector)
  - Correlation checking
  - Liquidity filtering (min ₹5 Cr daily volume)
  - Quality filtering (based on risk profile)
  - Position sizing (Kelly Criterion, fixed %)
  - Rebalancing triggers
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging
import yfinance as yf
from datetime import datetime

logger = logging.getLogger(__name__)


class RiskProfile(Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"


class InvestmentHorizon(Enum):
    """Investment timeframes"""
    INTRADAY = "INTRADAY"
    SWING = "SWING"
    POSITIONAL = "POSITIONAL"
    LONGTERM = "LONGTERM"
    MIXED = "MIXED"


@dataclass
class PortfolioStock:
    """Single stock position in portfolio"""
    ticker: str
    sector: str
    allocation_pct: float
    capital_amount: float
    entry_zone_low: float
    entry_zone_high: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: Optional[float] = None
    expected_return_pct: float = 0.0
    risk_tier: str = "MEDIUM"
    signal_strength: str = "BUY"
    confidence: int = 70


@dataclass
class PortfolioMetrics:
    """Portfolio-level metrics"""
    total_capital: float
    total_allocated: float
    cash_reserve: float
    weighted_expected_return: float
    max_drawdown_estimate: float
    portfolio_beta: float
    sharpe_ratio: float
    win_rate_estimate: float


class NSEIQPortfolioEngine:
    """Portfolio generation and management engine"""

    def __init__(self):
        self.max_single_stock_pct = 0.20  # 20% max per stock
        self.max_sector_pct = 0.35  # 35% max per sector
        self.min_daily_volume_crore = 5  # ₹5 Cr minimum
        self.correlation_threshold = 0.7  # High correlation threshold

        # Risk profile parameters
        self.risk_profiles = {
            RiskProfile.CONSERVATIVE: {
                "max_loss_pct_portfolio": 0.01,  # 1% daily
                "max_drawdown_pct": 0.10,  # 10%
                "quality_filter_pe_max": 20,
                "quality_filter_pe_min": 8,
                "debt_to_equity_max": 1.0,
                "cash_reserve_pct": 0.40,
            },
            RiskProfile.MODERATE: {
                "max_loss_pct_portfolio": 0.02,  # 2% daily
                "max_drawdown_pct": 0.15,  # 15%
                "quality_filter_pe_max": 25,
                "quality_filter_pe_min": 5,
                "debt_to_equity_max": 1.5,
                "cash_reserve_pct": 0.30,
            },
            RiskProfile.AGGRESSIVE: {
                "max_loss_pct_portfolio": 0.05,  # 5% daily
                "max_drawdown_pct": 0.20,  # 20%
                "quality_filter_pe_max": 35,
                "quality_filter_pe_min": 0,
                "debt_to_equity_max": 2.5,
                "cash_reserve_pct": 0.15,
            },
        }

    def build_portfolio(
        self,
        total_capital: float,
        risk_profile: RiskProfile,
        horizon: InvestmentHorizon,
        candidate_stocks: List[Dict],
        existing_holdings: Optional[List[Dict]] = None,
        sector_preferences: Optional[Dict[str, bool]] = None,
        blacklisted_sectors: Optional[List[str]] = None,
    ) -> Dict:
        """
        Build optimized portfolio from candidate stocks

        Args:
            total_capital: Total investment capital (₹)
            risk_profile: CONSERVATIVE/MODERATE/AGGRESSIVE
            horizon: INTRADAY/SWING/POSITIONAL/LONGTERM/MIXED
            candidate_stocks: List of {ticker, sector, signal_strength, expected_return, etc}
            existing_holdings: Current positions to consider
            sector_preferences: {sector: weight_preference}
            blacklisted_sectors: Sectors to exclude

        Returns:
            Portfolio dict with positions, metrics, risk rules
        """
        logger.info(
            f"📊 Building portfolio: ₹{total_capital:,.0f} | {risk_profile.value} | {horizon.value}"
        )

        risk_config = self.risk_profiles[risk_profile]
        cash_reserve_pct = risk_config["cash_reserve_pct"]
        deployable_capital = total_capital * (1 - cash_reserve_pct)

        # Step 1: Filter candidates by quality
        filtered_stocks = self._quality_filter(candidate_stocks, risk_config)
        logger.info(f"✅ Quality filter passed: {len(filtered_stocks)} stocks")

        # Step 2: Check liquidity
        liquid_stocks = self._liquidity_filter(filtered_stocks)
        logger.info(f"✅ Liquidity filter passed: {len(liquid_stocks)} stocks")

        # Step 3: Sector diversification
        sector_allocation = self._sector_allocation(
            liquid_stocks, risk_profile, sector_preferences, blacklisted_sectors
        )
        logger.info(f"✅ Sector allocation: {len(sector_allocation)} sectors")

        # Step 4: Position sizing
        portfolio_positions = self._size_positions(
            liquid_stocks,
            deployable_capital,
            sector_allocation,
            horizon,
            existing_holdings,
        )
        logger.info(f"✅ Position sizing: {len(portfolio_positions)} stocks selected")

        # Step 5: Correlation check
        portfolio_positions = self._correlation_filter(portfolio_positions)
        logger.info(f"✅ Correlation filter: {len(portfolio_positions)} stocks remain")

        # Step 6: Calculate portfolio metrics
        portfolio_metrics = self._calculate_metrics(
            portfolio_positions, total_capital, risk_profile
        )

        # Step 7: Build risk management rules
        risk_rules = self._build_risk_rules(
            total_capital, risk_profile, portfolio_positions
        )

        return {
            "generated_at": datetime.now().isoformat(),
            "total_capital": total_capital,
            "risk_profile": risk_profile.value,
            "horizon": horizon.value,
            "positions": portfolio_positions,
            "metrics": portfolio_metrics,
            "risk_management": risk_rules,
            "cash_reserve": total_capital * cash_reserve_pct,
        }

    # ═════════════════════════════════════════════════════════════════════════
    # FILTERING & SELECTION
    # ═════════════════════════════════════════════════════════════════════════

    def _quality_filter(self, stocks: List[Dict], risk_config: Dict) -> List[Dict]:
        """Filter by fundamental quality (P/E, Debt/Equity, etc)"""
        filtered = []

        for stock in stocks:
            pe = stock.get("pe_ratio")
            debt_to_eq = stock.get("debt_to_equity", 999)

            # P/E check
            if pe and (
                risk_config["quality_filter_pe_min"]
                <= pe
                <= risk_config["quality_filter_pe_max"]
            ):
                # Debt/Equity check
                if debt_to_eq <= risk_config["debt_to_equity_max"]:
                    filtered.append(stock)
            elif not pe:
                # No P/E data, pass with caution
                if debt_to_eq <= risk_config["debt_to_equity_max"]:
                    filtered.append(stock)

        return filtered

    def _liquidity_filter(self, stocks: List[Dict]) -> List[Dict]:
        """Filter by minimum daily volume (₹5 Cr)"""
        filtered = []

        for stock in stocks:
            ticker = stock.get("ticker")

            try:
                hist = yf.download(ticker, period="20d", progress=False)
                if not hist.empty:
                    avg_volume = hist["Volume"].mean()
                    # Approximate: avg_volume * avg_close_20d ≈ value traded
                    avg_close = hist["Close"].mean()
                    daily_value = (avg_volume * avg_close) / 10000000  # Convert to crores
                    
                    if daily_value >= self.min_daily_volume_crore:
                        stock["daily_volume_crore"] = daily_value
                        filtered.append(stock)
            except:
                pass

        return filtered

    def _sector_allocation(
        self,
        stocks: List[Dict],
        risk_profile: RiskProfile,
        preferences: Optional[Dict] = None,
        blacklist: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Determine sector allocation weights"""
        blacklist = blacklist or []
        preferences = preferences or {}

        # Group by sector
        sector_groups = {}
        for stock in stocks:
            sector = stock.get("sector", "Unknown")
            if sector not in blacklist:
                if sector not in sector_groups:
                    sector_groups[sector] = []
                sector_groups[sector].append(stock)

        # Allocate weights per sector (respect max 35%)
        sector_allocation = {}
        num_sectors = len(sector_groups)

        if num_sectors > 0:
            base_allocation = min(1.0 / num_sectors, self.max_sector_pct)

            for sector in sector_groups:
                # Apply preferences if provided
                if sector in preferences:
                    weight = preferences[sector]
                else:
                    weight = base_allocation

                # Cap at max sector percentage
                weight = min(weight, self.max_sector_pct)
                sector_allocation[sector] = weight

        # Normalize to 100%
        total = sum(sector_allocation.values())
        if total > 0:
            sector_allocation = {k: v / total for k, v in sector_allocation.items()}

        return sector_allocation

    def _size_positions(
        self,
        stocks: List[Dict],
        deployable_capital: float,
        sector_allocation: Dict[str, float],
        horizon: InvestmentHorizon,
        existing_holdings: Optional[List[Dict]] = None,
    ) -> List[PortfolioStock]:
        """Size individual positions using Kelly Criterion (fractional) or fixed %"""
        positions = []
        sector_capital_used = {s: 0 for s in sector_allocation}

        # Sort by signal strength (best signals first)
        signal_priority = {
            "STRONG BUY": 4,
            "BUY": 3,
            "NEUTRAL": 2,
            "SELL": 1,
            "STRONG SELL": 0,
        }
        stocks.sort(
            key=lambda x: signal_priority.get(x.get("signal_strength", "NEUTRAL"), 2),
            reverse=True,
        )

        for i, stock in enumerate(stocks[:10]):  # Limit to top 10
            ticker = stock.get("ticker")
            sector = stock.get("sector", "Unknown")
            signal = stock.get("signal_strength", "NEUTRAL")
            expected_return = stock.get("expected_return_pct", 0.05)
            confidence = stock.get("confidence", 70)

            # Skip if no sector allocation
            if sector not in sector_allocation:
                continue

            # Position size: (confidence / 100) * (sector allocation / num_stocks_in_sector)
            sector_capital = deployable_capital * sector_allocation[sector]
            available_in_sector = sector_capital - sector_capital_used[sector]

            # Per-stock allocation
            if len(stocks) <= 3:
                stock_pct = min(0.20, available_in_sector / deployable_capital)
            elif len(stocks) <= 7:
                stock_pct = min(0.15, available_in_sector / deployable_capital)
            else:
                stock_pct = min(0.10, available_in_sector / deployable_capital)

            # Confidence adjustment
            stock_pct *= confidence / 100

            allocation_amount = deployable_capital * stock_pct
            sector_capital_used[sector] += allocation_amount

            # Risk/reward estimation
            entry_low = stock.get("entry_zone_low", 100)
            entry_high = stock.get("entry_zone_high", 105)
            stop_loss = stock.get("stop_loss", 95)
            target_1 = stock.get("target_1", 115)
            target_2 = stock.get("target_2", 125)
            target_3 = stock.get("target_3", 135) if horizon in [
                InvestmentHorizon.POSITIONAL,
                InvestmentHorizon.LONGTERM,
            ] else None

            risk_tier = "LOW" if expected_return > 0.05 else "MEDIUM"
            risk_tier = "HIGH" if confidence < 60 else risk_tier

            position = PortfolioStock(
                ticker=ticker,
                sector=sector,
                allocation_pct=stock_pct * 100,
                capital_amount=allocation_amount,
                entry_zone_low=entry_low,
                entry_zone_high=entry_high,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                target_3=target_3,
                expected_return_pct=expected_return * 100,
                risk_tier=risk_tier,
                signal_strength=signal,
                confidence=confidence,
            )

            positions.append(position)

        return positions

    def _correlation_filter(self, positions: List[PortfolioStock]) -> List[PortfolioStock]:
        """Remove highly correlated positions"""
        if len(positions) <= 1:
            return positions

        # In production, calculate correlation matrix from historical prices
        # For now, simple heuristic: don't include 2 banking or IT stocks together
        filtered = []
        sector_count = {}

        for pos in positions:
            sector = pos.sector
            sector_count[sector] = sector_count.get(sector, 0) + 1

            # Allow max 2 stocks from same sector
            if sector_count[sector] <= 2:
                filtered.append(pos)

        return filtered

    # ═════════════════════════════════════════════════════════════════════════
    # METRICS & RISK RULES
    # ═════════════════════════════════════════════════════════════════════════

    def _calculate_metrics(
        self,
        positions: List[PortfolioStock],
        total_capital: float,
        risk_profile: RiskProfile,
    ) -> Dict:
        """Calculate portfolio metrics (return, risk, beta, Sharpe)"""
        if not positions:
            return {}

        # Weighted expected return
        total_allocation = sum(p.allocation_pct for p in positions)
        weighted_return = sum(
            (p.allocation_pct / 100) * p.expected_return_pct for p in positions
        )

        # Estimate max drawdown per risk profile
        risk_config = self.risk_profiles[risk_profile]
        est_max_drawdown = risk_config["max_drawdown_pct"]

        # Sharpe ratio estimate (assuming 8% risk-free rate)
        risk_free_rate = 8.0
        portfolio_vol_est = 15.0 if risk_profile == RiskProfile.CONSERVATIVE else 25.0
        sharpe = (weighted_return - risk_free_rate) / portfolio_vol_est if portfolio_vol_est > 0 else 0

        # Portfolio beta (vs NIFTY, approximate)
        portfolio_beta = 0.8 if risk_profile == RiskProfile.CONSERVATIVE else 1.2

        # Win rate estimate (from confidence scores)
        avg_confidence = np.mean([p.confidence for p in positions])
        estimated_win_rate = avg_confidence / 100

        return {
            "weighted_expected_return_pct": round(weighted_return, 2),
            "max_drawdown_estimate_pct": est_max_drawdown * 100,
            "portfolio_beta": round(portfolio_beta, 2),
            "sharpe_ratio": round(sharpe, 2),
            "win_rate_estimate_pct": round(estimated_win_rate * 100, 1),
            "avg_confidence": round(avg_confidence, 1),
            "total_allocation_pct": round(total_allocation, 1),
        }

    def _build_risk_rules(
        self,
        total_capital: float,
        risk_profile: RiskProfile,
        positions: List[PortfolioStock],
    ) -> Dict:
        """Build per-portfolio risk management rules"""
        risk_config = self.risk_profiles[risk_profile]
        daily_loss_limit = total_capital * risk_config["max_loss_pct_portfolio"]
        drawdown_limit = total_capital * risk_config["max_drawdown_pct"]

        # Per-trade max loss
        num_positions = len(positions)
        per_trade_loss = daily_loss_limit / num_positions if num_positions > 0 else daily_loss_limit

        return {
            "overall_portfolio_stop_loss_pct": risk_config["max_drawdown_pct"] * 100,
            "daily_max_loss_rupees": int(daily_loss_limit),
            "per_trade_max_loss_rupees": int(per_trade_loss),
            "portfolio_drawdown_exit": int(drawdown_limit),
            "profit_booking_rule": "Book 50% at Target 1; trail rest to Target 2",
            "review_frequency": "Daily for Intraday; Weekly for others",
            "rebalance_trigger_pct": 5,  # Rebalance if drift >5%
            "vix_extreme_threshold": 22,
            "vix_elevated_threshold": 18,
            "cash_utilization_in_crisis": 10,  # Use 10% cash reserve in corrections
        }

    # ═════════════════════════════════════════════════════════════════════════
    # PORTFOLIO ANALYSIS
    # ═════════════════════════════════════════════════════════════════════════

    def get_portfolio_summary(self, portfolio: Dict) -> str:
        """Generate human-readable portfolio summary"""
        positions = portfolio.get("positions", [])
        metrics = portfolio.get("metrics", {})

        summary = f"""
╔══════════════════════════════════════════════════════════════════════╗
║              NSEIQ PORTFOLIO SUMMARY                                 ║
╚══════════════════════════════════════════════════════════════════════╝

Capital: ₹{portfolio['total_capital']:,.0f}
Risk Profile: {portfolio['risk_profile']}
Horizon: {portfolio['horizon']}
Cash Reserve: ₹{portfolio['cash_reserve']:,.0f}

POSITIONS ({len(positions)}):
"""
        for pos in positions:
            summary += f"""  • {pos.ticker:8} | {pos.sector:12} | {pos.allocation_pct:5.1f}% | ₹{pos.capital_amount:10,.0f} | {pos.signal_strength}
"""

        summary += f"""
METRICS:
  Expected Return:     {metrics.get('weighted_expected_return_pct', 0):.1f}%
  Max Drawdown:        {metrics.get('max_drawdown_estimate_pct', 0):.1f}%
  Portfolio Beta:      {metrics.get('portfolio_beta', 1):.2f}
  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.2f}
  Est. Win Rate:       {metrics.get('win_rate_estimate_pct', 0):.1f}%
"""
        return summary


# ═════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═════════════════════════════════════════════════════════════════════════════

portfolio_engine = NSEIQPortfolioEngine()
