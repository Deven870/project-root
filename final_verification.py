#!/usr/bin/env python3
"""
COMPREHENSIVE SYSTEM VERIFICATION & FIXES REPORT
Tests all major components after fixes
"""
import sys

print("\n" + "="*80)
print("DIGITRADER v4.0 - POST-FIX COMPREHENSIVE TEST")
print("="*80 + "\n")

# Test 1: Module imports
print("1. TESTING MODULE IMPORTS...")
try:
    from modules.nse_stock_list import get_all_nse_stocks, get_stock_options, SECTOR_STOCKS
    stocks = get_all_nse_stocks()
    print(f"   ✅ NSE Stock List: {len(stocks)} stocks loaded")
except Exception as e:
    print(f"   ❌ NSE Module: {str(e)[:50]}")
    sys.exit(1)

try:
    from modules.utils import fetch_price_data, get_nse_stock_list
    print("   ✅ Utils Module: OK")
except Exception as e:
    print(f"   ❌ Utils Module: {str(e)[:50]}")
    sys.exit(1)

try:
    from modules.precision_analyzer import EnhancedPrecisionAnalyzer
    analyzer = EnhancedPrecisionAnalyzer()
    print("   ✅ Precision Analyzer: Initialized")
except Exception as e:
    print(f"   ❌ Analyzer Init: {str(e)[:50]}")
    sys.exit(1)

# Test 2: Data fetching
print("\n2. TESTING DATA FETCHING...")
test_symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
fetch_results = {}

for sym in test_symbols:
    try:
        data = fetch_price_data(sym)
        if data is not None and not data.empty:
            fetch_results[sym] = "OK"
            print(f"   ✅ {sym}: {len(data)} data points")
        else:
            fetch_results[sym] = "EMPTY"
            print(f"   ⚠️  {sym}: Empty data")
    except Exception as e:
        fetch_results[sym] = "ERROR"
        print(f"   ❌ {sym}: {str(e)[:40]}")

# Test 3: Precision analysis
print("\n3. TESTING PRECISION ANALYSIS...")
analysis_results = {}

for sym in test_symbols:
    try:
        price_data = fetch_price_data(sym)
        if price_data is not None and not price_data.empty:
            result = analyzer.get_precision_analysis(sym, price_data, price_data)
            
            if result and all(k in result for k in ['signal', 'confidence', 'final_score']):
                analysis_results[sym] = "OK"
                print(f"   ✅ {sym}:")
                print(f"      Signal: {result['signal']}")
                print(f"      Score: {result['final_score']:.3f}")
                print(f"      Confidence: {result['confidence']:.1f}%")
                print(f"      Data Quality: {result['precision_metrics']['data_quality']}")
            else:
                analysis_results[sym] = "INCOMPLETE"
                print(f"   ⚠️  {sym}: Missing fields in result")
        else:
            analysis_results[sym] = "NO_DATA"
            print(f"   ⚠️  {sym}: No data available")
    except Exception as e:
        analysis_results[sym] = "ERROR"
        print(f"   ❌ {sym}: {str(e)[:50]}")

# Test 4: Component analysis
print("\n4. TESTING COMPONENT BREAKDOWN...")
test_stock = "INFY.NS"
try:
    price_data = fetch_price_data(test_stock)
    if price_data is not None and not price_data.empty:
        result = analyzer.get_precision_analysis(test_stock, price_data, price_data)
        
        print(f"   Testing {test_stock} component sources:")
        tech = result['components'].get('technical', {})
        finnhub = result['components'].get('finnhub', {})
        market = result['components'].get('market', {})
        
        print(f"      Technical: score={tech.get('score', 0):.3f}, "
              f"confidence={tech.get('confidence', 0):.0f}%, "
              f"signals={len(tech.get('signals', []))}")
        print(f"      Finnhub: score={finnhub.get('score', 0):.3f}, "
              f"confidence={finnhub.get('confidence', 0):.0f}%, "
              f"data_points={finnhub.get('data_points', 0)}")
        print(f"      Market: score={market.get('score', 0):.3f}, "
              f"confidence={market.get('confidence', 0):.0f}%, "
              f"data_points={market.get('data_points', 0)}")
except Exception as e:
    print(f"   ❌ Component test failed: {str(e)[:50]}")

# Test 5: Error handling
print("\n5. TESTING ERROR HANDLING...")
try:
    # Test with invalid stock
    result = analyzer.get_precision_analysis("INVALID.NS", None, None)
    print(f"   ✅ Invalid stock handling: Returns {result.get('signal', 'UNKNOWN')}")
except Exception as e:
    print(f"   ❌ Error handling failed: {str(e)[:50]}")

# Summary
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80 + "\n")

succ_fetch = sum(1 for v in fetch_results.values() if v == "OK")
succ_analysis = sum(1 for v in analysis_results.values() if v == "OK")

print(f"Data Fetch Success Rate: {succ_fetch}/{len(test_symbols)} ({100*succ_fetch//len(test_symbols)}%)")
print(f"Analysis Success Rate: {succ_analysis}/{len(test_symbols)} ({100*succ_analysis//len(test_symbols)}%)")

if succ_fetch >= 3 and succ_analysis >= 3:
    print("\n✅ SYSTEM READY FOR DEPLOYMENT")
    print("   - Core modules: Functional")
    print("   - Data fetching: Operational")
    print("   - Analysis engine: Generating signals")
    print("   - Error handling: Implemented")
    print("\n   Status: READY TO RUN 'streamlit run app.py'")
else:
    print("\n⚠️  SYSTEM PARTIALLY FUNCTIONAL")
    print("   - Some components may need review")
    print("   - Consider checking API keys")
    print("   - May still be usable with limited data")

print("\n" + "="*80)
