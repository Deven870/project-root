#!/usr/bin/env python
"""
NSEIQ v5.0 - Integration Test
Tests all major system components
"""

from backend.app.services.nseiq_prediction_engine import nseiq_engine, TradingMode
from backend.app.services.nseiq_portfolio_engine import portfolio_engine, RiskProfile, InvestmentHorizon
from backend.app.services.nseiq_prediction_formatter import NSEIQPredictionFormatter
import sys

def test_prediction_engine():
    print("\n" + "="*80)
    print("TEST 1: PREDICTION ENGINE")
    print("="*80)
    try:
        print("🔍 Generating prediction for RELIANCE (SWING mode)...")
        result = nseiq_engine.generate_prediction('RELIANCE', TradingMode.SWING, 'Energy')
        
        if 'error' in result:
            print(f'❌ Error: {result.get("error")}')
            return False
        
        ticker = result.get('ticker', 'N/A')
        signal = result.get('signal', 'N/A')
        confidence = result.get('confidence', 0)
        score = result.get('aggregate_score', 0)
        
        print(f'✅ Prediction generated successfully!')
        print(f'   Ticker: {ticker}')
        print(f'   Signal: {signal}')
        print(f'   Confidence: {confidence}/100')
        print(f'   Aggregate Score: {score}/100')
        print(f'   Layers analyzed: {len(result.get("layers", {}))} layers')
        return True
        
    except Exception as e:
        print(f'❌ Exception: {str(e)[:200]}')
        return False


def test_portfolio_engine():
    print("\n" + "="*80)
    print("TEST 2: PORTFOLIO ENGINE")
    print("="*80)
    try:
        print("📊 Building portfolio (₹250,000 | MODERATE | SWING)...")
        
        candidate_stocks = [
            {"ticker": "RELIANCE", "sector": "Energy", "signal_strength": "BUY", "expected_return_pct": 5.0, "confidence": 75, "pe_ratio": 22, "debt_to_equity": 0.5},
            {"ticker": "INFY", "sector": "IT", "signal_strength": "BUY", "expected_return_pct": 3.0, "confidence": 70, "pe_ratio": 18, "debt_to_equity": 0.3},
        ]
        
        portfolio = portfolio_engine.build_portfolio(
            total_capital=250000,
            risk_profile=RiskProfile.MODERATE,
            horizon=InvestmentHorizon.SWING,
            candidate_stocks=candidate_stocks,
        )
        
        if 'positions' in portfolio:
            print(f'✅ Portfolio built successfully!')
            print(f'   Total Capital: ₹{portfolio.get("total_capital"):,.0f}')
            print(f'   Positions: {len(portfolio.get("positions", []))}')
            print(f'   Cash Reserve: ₹{portfolio.get("cash_reserve", 0):,.0f}')
            print(f'   Expected Return: {portfolio.get("metrics", {}).get("weighted_expected_return_pct", 0):.2f}%')
            return True
        else:
            print('❌ No positions generated')
            return False
            
    except Exception as e:
        print(f'❌ Exception: {str(e)[:200]}')
        return False


def test_formatter():
    print("\n" + "="*80)
    print("TEST 3: PREDICTION FORMATTER")
    print("="*80)
    try:
        print("📝 Testing formatter with sample data...")
        
        sample_analysis = {
            "current_price": 2650,
            "layers": {
                "technical": {
                    "signal_score": 35,
                    "reasons": ["Above 50-EMA", "RSI bullish"],
                    "resistance_20": 2700,
                    "support_20": 2600,
                    "atr_14": 50,
                },
                "fundamental": {
                    "signal_score": 40,
                    "debt_to_equity": 0.5,
                    "pe_ratio": 22,
                },
                "sentiment": {
                    "sentiment": "BULLISH",
                    "confidence": 75,
                    "sentiment_score": 30,
                },
                "macro": {
                    "signal_score": 15,
                    "nifty_trend": "BULL",
                },
            },
        }
        
        sample_prediction = {
            "ticker": "RELIANCE",
            "mode": "SWING",
            "signal": "BUY",
            "confidence": 75,
            "layers": sample_analysis["layers"],
        }
        
        formatted = NSEIQPredictionFormatter.format_prediction(
            sample_prediction, 
            sample_analysis
        )
        
        if formatted and len(formatted) > 500:
            print(f'✅ Formatter working! Generated {len(formatted)} chars of output')
            print(f'   Contains targets: {"Target" in formatted}')
            print(f'   Contains risk factors: {"Risk Factor" in formatted}')
            print(f'   Contains disclaimer: {"DISCLAIMER" in formatted}')
            return True
        else:
            print('❌ Formatter output too short')
            return False
            
    except Exception as e:
        print(f'❌ Exception: {str(e)[:200]}')
        return False


def main():
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "NSEIQ v5.0 - SYSTEM INTEGRATION TEST".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    results = []
    
    # Run tests
    results.append(("Prediction Engine", test_prediction_engine()))
    results.append(("Portfolio Engine", test_portfolio_engine()))
    results.append(("Formatter", test_formatter()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! NSEIQ v5.0 is ready for deployment.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
