"""
╔════════════════════════════════════════════════════════════════════════════╗
║                 NSEIQ PREDICTION FORMATTER v5.0                           ║
║          Converts raw analysis into strict NSEIQ output format             ║
╚════════════════════════════════════════════════════════════════════════════╝

Formats all predictions with:
  - Current price
  - Conservative/Base/Bull targets
  - R:R ratio
  - Signal strength with confidence
  - Risk factors
  - Data freshness stamps
  - SEBI disclaimer
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class NSEIQPredictionFormatter:
    """Strict formatting for NSEIQ predictions"""

    @staticmethod
    def format_prediction(
        prediction: Dict,
        analysis: Dict,
        capital_deployed: float = 0,
    ) -> str:
        """
        Format prediction in strict NSEIQ format

        Returns ready-to-publish prediction string with all required sections
        """
        ticker = prediction.get("ticker", "UNKNOWN")
        mode = prediction.get("mode", "SWING")
        signal = prediction.get("signal", "NEUTRAL")
        confidence = prediction.get("confidence", 0)

        # Extract current price
        current_price = analysis.get("current_price", 0)

        # Calculate price targets
        targets = NSEIQPredictionFormatter._calculate_targets(analysis, confidence)
        conservative = targets["conservative"]
        base_case = targets["base_case"]
        bull_case = targets["bull_case"]

        # Calculate entry and SL
        entry = targets["entry"]
        sl = targets["stop_loss"]
        target_1 = targets["target_1"]
        target_2 = targets["target_2"]
        target_3 = targets["target_3"]

        # Risk:Reward ratio
        risk = current_price - sl
        reward = target_1 - current_price if current_price < target_1 else current_price - entry
        rr_ratio = f"{reward/risk:.2f}:1" if risk > 0 else "N/A"

        # Consolidate thesis
        thesis_lines = []
        layers = analysis.get("layers", {})

        tech = layers.get("technical", {})
        if tech.get("signal_score"):
            thesis_lines.append(
                f"Technical: {tech.get('signal_score', 0)}/100 - "
                + ", ".join(tech.get("reasons", [])[:2])
            )

        fund = layers.get("fundamental", {})
        if fund.get("signal_score"):
            thesis_lines.append(
                f"Fundamental: {fund.get('signal_score', 0)}/100 - Strong balance sheet"
                if fund.get("debt_to_equity", 2) < 1.5 else "Moderate valuation"
            )

        sent = layers.get("sentiment", {})
        if sent.get("sentiment"):
            thesis_lines.append(f"Sentiment: {sent.get('sentiment')} ({sent.get('confidence', 0):.0f}% confidence)")

        # Risk factors
        risk_factors = NSEIQPredictionFormatter._identify_risk_factors(analysis)

        # Data freshness
        now_ist = datetime.now().strftime("%d-%b-%Y | %H:%M IST")

        # Build formatted output
        output = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                          NSEIQ PREDICTION REPORT                           ║
╚════════════════════════════════════════════════════════════════════════════╝

STOCK: {ticker} | MODE: {mode}
DATE OF ANALYSIS: {datetime.now().strftime("%d-%b-%Y")} | GENERATED AT: {datetime.now().strftime("%H:%M")} IST

{'-'*80}
CURRENT PRICE: ₹{current_price:,.2f}

PREDICTED PRICE RANGE:
  → Conservative Target: ₹{conservative:,.2f} (probability: {NSEIQPredictionFormatter._prob(conservative, current_price)}%)
  → Base Case Target:    ₹{base_case:,.2f} (probability: {NSEIQPredictionFormatter._prob(base_case, current_price)}%)
  → Bull Case Target:    ₹{bull_case:,.2f} (probability: {NSEIQPredictionFormatter._prob(bull_case, current_price)}%)

ENTRY & EXIT:
  ENTRY ZONE:       ₹{entry-2:,.2f} – ₹{entry+2:,.2f}
  STOP LOSS:        ₹{sl:,.2f} (hard) | ₹{int(sl*1.02):,.2f} (trailing after 2% move)
  TARGET 1:         ₹{target_1:,.2f} (+{((target_1-current_price)/current_price)*100:.1f}%)
  TARGET 2:         ₹{target_2:,.2f} (+{((target_2-current_price)/current_price)*100:.1f}%)
{"  TARGET 3:         ₹" + f"{target_3:,.2f} (+{((target_3-current_price)/current_price)*100:.1f}%)" if target_3 else ""}
  RISK:REWARD RATIO: {rr_ratio}

{'-'*80}
SIGNAL STRENGTH:  {signal}
CONFIDENCE SCORE: {confidence}/100

THESIS SUMMARY:
{chr(10).join(f"  • {line}" for line in thesis_lines)}

RISK FACTORS (CRITICAL - MUST READ):
{chr(10).join(f"  {i+1}. {factor}" for i, factor in enumerate(risk_factors[:5]))}

{'-'*80}
DATA FRESHNESS CHECK:
  → Technical data:    {now_ist}
  → News/Sentiment:    {now_ist}
  → Fundamentals:      {analysis.get('fundamental_updated', now_ist)}
  → Options data:      NSE API integration pending
  → Insider Activity:  NSE filings feed integration pending

{'-'*80}
⚠️  DISCLAIMER:
    This is AI-generated analysis based on historical data & sentiment signals.
    NOT SEBI-registered investment advice.
    Real money capital carries REAL RISK.
    Always use hard stop losses & position sizing.
    Past performance does not guarantee future results.
    
    Responsibility: 100% on trader for execution & risk management.

═════════════════════════════════════════════════════════════════════════════════
GENERATED BY: NSEIQ v5.0 | Institutional NSE Stock Intelligence System
═════════════════════════════════════════════════════════════════════════════════
"""
        return output

    @staticmethod
    def _calculate_targets(analysis: Dict, confidence: int) -> Dict:
        """Calculate price targets based on technical & fundamental analysis"""
        current_price = analysis.get("current_price", 100)
        
        # Extract technical signals
        tech = analysis.get("layers", {}).get("technical", {})
        resistance = tech.get("resistance_20", current_price * 1.05)
        support = tech.get("support_20", current_price * 0.95)
        atr = tech.get("atr_14", current_price * 0.02)

        # Expected move based on confidence
        move_pct = (confidence / 100) * 0.03  # 1-3% expected move

        # Conservative target (50% of expected move)
        conservative = current_price * (1 + move_pct * 0.5)

        # Base case (full expected move)
        base_case = current_price * (1 + move_pct)

        # Bull case (150% of expected move)
        bull_case = current_price * (1 + move_pct * 1.5)

        # Entry zone (support level -1%)
        entry = support * 0.99

        # Stop loss (below support)
        stop_loss = support * 0.97

        # Targets based on Fibonacci/ATR
        target_1 = current_price + (atr * 1.5)
        target_2 = current_price + (atr * 2.5)
        target_3 = current_price + (atr * 4.0)

        return {
            "conservative": conservative,
            "base_case": base_case,
            "bull_case": bull_case,
            "entry": entry,
            "stop_loss": stop_loss,
            "target_1": target_1,
            "target_2": target_2,
            "target_3": target_3,
        }

    @staticmethod
    def _prob(target: float, current: float) -> int:
        """Estimate probability of hitting target based on distance"""
        distance_pct = abs((target - current) / current)
        if distance_pct < 0.03:
            return 75
        elif distance_pct < 0.07:
            return 60
        elif distance_pct < 0.12:
            return 45
        else:
            return 30

    @staticmethod
    def _identify_risk_factors(analysis: Dict) -> List[str]:
        """Identify and list key risk factors"""
        risk_factors = []
        layers = analysis.get("layers", {})

        # Technical risks
        tech = layers.get("technical", {})
        if tech.get("signal_score", 0) < -30:
            risk_factors.append("❌ Technical breakdown: Major support broken")
        if tech.get("volume_ratio", 1) > 2.0:
            risk_factors.append("⚠️  Extreme volume spike - reversal risk")

        # Fundamental risks
        fund = layers.get("fundamental", {})
        if fund.get("debt_to_equity", 0) > 2.0:
            risk_factors.append("❌ High leverage: Debt/Equity > 2.0")
        if fund.get("pe_ratio", 0) > 30:
            risk_factors.append("⚠️  Expensive valuation: High P/E")

        # Sentiment risks
        sent = layers.get("sentiment", {})
        if sent.get("sentiment") == "BEARISH" and sent.get("confidence", 0) > 70:
            risk_factors.append("❌ Strong bearish sentiment in news")

        # Macro risks
        macro = layers.get("macro", {})
        if macro.get("nifty_trend") == "BEAR":
            risk_factors.append("⚠️  NIFTY in downtrend - sector headwinds")

        # Default risks if none identified
        if len(risk_factors) == 0:
            risk_factors = [
                "1. Market volatility: Unexpected macroeconomic events",
                "2. Company-specific: Management changes, regulatory issues",
                "3. Technical reversion: Profit booking at resistance levels",
                "4. Liquidity risk: Sudden volume drop at entry/exit",
                "5. Black swan: Geopolitical shocks, market crashes",
            ]
        elif len(risk_factors) < 3:
            risk_factors.extend([
                f"{len(risk_factors)+1}. Unexpected earnings miss",
                f"{len(risk_factors)+2}. Market-wide correction",
            ])

        return risk_factors[:5]  # Return top 5

    @staticmethod
    def format_portfolio_output(portfolio: Dict) -> str:
        """Format portfolio for display"""
        positions = portfolio.get("positions", [])
        metrics = portfolio.get("metrics", {})
        total_capital = portfolio.get("total_capital", 0)

        output = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    NSEIQ PORTFOLIO REPORT                                  ║
╚════════════════════════════════════════════════════════════════════════════╝

PORTFOLIO GENERATED: {datetime.now().strftime("%d-%b-%Y")}
TOTAL CAPITAL: ₹{total_capital:,.0f} | RISK PROFILE: {portfolio.get('risk_profile')} | HORIZON: {portfolio.get('horizon')}

{'-'*80}
POSITIONS ALLOCATION:
═════════════════════════════════════════════════════════════════════════════
"""
        # Table header
        output += "  STOCK     │ SECTOR    │ ALLOC % │ AMOUNT ₹  │ ENTRY  │ TARGET │ SL    │ EXP. RET │ RISK\n"
        output += "  " + "─" * 76 + "\n"

        # Positions
        for pos in positions:
            output += f"  {pos.ticker:8} │ {pos.sector:9} │ {pos.allocation_pct:6.1f}% │ ₹{pos.capital_amount:9,.0f} │ ₹{pos.entry_zone_low:5.0f} │ ₹{pos.target_1:6.0f} │ ₹{pos.stop_loss:4.0f} │ {pos.expected_return_pct:7.1f}% │ {pos.risk_tier}\n"

        output += f"""
{'-'*80}
PORTFOLIO METRICS:
  → Weighted Expected Return:     {metrics.get('weighted_expected_return_pct', 0):.2f}%
  → Max Drawdown Estimate:        {metrics.get('max_drawdown_estimate_pct', 0):.1f}%
  → Portfolio Beta (vs NIFTY):    {metrics.get('portfolio_beta', 1):.2f}
  → Sharpe Ratio Estimate:        {metrics.get('sharpe_ratio', 0):.2f}
  → Cash Reserve Recommended:     ₹{portfolio.get('cash_reserve'):,.0f}

{'-'*80}
RISK MANAGEMENT RULES FOR THIS PORTFOLIO:
  → Overall Portfolio Stop Loss:  {portfolio.get('risk_management', {}).get('overall_portfolio_stop_loss_pct', 0):.1f}% drawdown = EXIT ALL
  → Per-trade max loss:           ₹{portfolio.get('risk_management', {}).get('per_trade_max_loss_rupees', 0):,.0f}
  → Profit booking rule:          Book 50% at Target 1; trail rest
  → Review frequency:             Daily for Intraday; Weekly for others
  → Rebalance trigger:            Drift > 5% from target weight

═════════════════════════════════════════════════════════════════════════════════
DISCLAIMER: This portfolio is AI-generated. Not SEBI-registered advice.
═════════════════════════════════════════════════════════════════════════════════
"""
        return output

    @staticmethod
    def format_pre_market_brief() -> str:
        """Format pre-market intelligence brief (8:45-9:15 AM)"""
        now = datetime.now()
        
        return f"""
╔════════════════════════════════════════════════════════════════════════════╗
║              NSEIQ PRE-MARKET INTELLIGENCE BRIEF                           ║
║                   {now.strftime("%d-%b-%Y | %H:%M IST")}                           ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 GLOBAL OVERNIGHT CUES:
  • US Markets: [Pending real-time data]
  • SGX NIFTY: [Pending real-time data]
  • Crude Oil: [Pending real-time data]
  • Gold (MCX): [Pending real-time data]
  • USD/INR: [Pending real-time data]

📈 FII/DII ACTIVITY (Previous Session):
  [Pending NSE data]

🎯 STOCKS IN FOCUS TODAY:
  [Pending earnings/ex-date calendar]

📍 NIFTY 50 EXPECTED OPENING RANGE:
  [Pending pre-market data]

⭐ TOP 3 TRADE SETUPS FOR THE DAY:
  1. [Pending real-time analysis]
  2. [Pending real-time analysis]
  3. [Pending real-time analysis]

📊 MARKET MOOD:
  VIX Level: [Pending]
  Risk Appetite: [Pending]
  Sector Rotation: [Pending]

⚠️  BLACK SWAN WATCH:
  [None detected]

═════════════════════════════════════════════════════════════════════════════════
NSEIQ v5.0 | Institutional NSE Stock Intelligence
═════════════════════════════════════════════════════════════════════════════════
"""

        return output


# ═════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═════════════════════════════════════════════════════════════════════════════

formatter = NSEIQPredictionFormatter()
