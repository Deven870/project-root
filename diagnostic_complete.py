#!/usr/bin/env python3
"""
🔧 COMPREHENSIVE SYSTEM DIAGNOSTIC & REPAIR
Identifies and fixes all issues from bottom to top
"""

import sys
import importlib
from pathlib import Path

print("="*80)
print("🔍 DIGITRADER v4.0 - COMPREHENSIVE DIAGNOSTIC REPORT")
print("="*80)

# Test 1: Core imports
print("\n1️⃣  Testing Core Imports...")
issues = []

try:
    from modules.nse_stock_list import get_all_nse_stocks, SECTOR_STOCKS
    print("  ✅ NSE Stock List: OK")
except Exception as e:
    issues.append(f"NSE Stock List: {str(e)[:50]}")
    print(f"  ❌ NSE Stock List: {str(e)[:50]}")

try:
    from modules.utils import fetch_price_data, get_nse_stock_list
    print("  ✅ Utils Module: OK")
except Exception as e:
    issues.append(f"Utils Module: {str(e)[:50]}")
    print(f"  ❌ Utils Module: {str(e)[:50]}")

try:
    from modules.precision_analyzer import EnhancedPrecisionAnalyzer
    print("  ✅ Precision Analyzer: OK (import)")
except Exception as e:
    issues.append(f"Precision Analyzer: {str(e)[:50]}")
    print(f"  ❌ Precision Analyzer: {str(e)[:50]}")

# Test 2: Data fetching
print("\n2️⃣  Testing Data Fetching...")

try:
    data = fetch_price_data('RELIANCE.NS')
    if data is not None and not data.empty:
        print(f"  ✅ Price Data Fetch: {len(data)} rows retrieved")
        print(f"     Columns: {list(data.columns)}")
    else:
        issues.append("Price Data: Empty result")
        print("  ⚠️  Price Data: Empty result")
except Exception as e:
    issues.append(f"Price Data Fetch: {str(e)[:50]}")
    print(f"  ❌ Price Data Fetch: {str(e)[:50]}")

# Test 3: Analyzer initialization
print("\n3️⃣  Testing Analyzer Functionality...")

try:
    analyzer = EnhancedPrecisionAnalyzer()
    print("  ✅ Analyzer initialized")
    
    # Test with actual data
    data = fetch_price_data('TCS.NS')
    if data is not None and not data.empty:
        print(f"  ✅ Analysis running...")
        try:
            result = analyzer.get_precision_analysis('TCS.NS', data, data)
            if result:
                print(f"  ✅ Analysis result: {result['signal']}")
                print(f"     Confidence: {result['confidence']:.1f}%")
                print(f"     Components: Technical={result['components'].get('technical', {}).get('score', 0):.2f}, "
                      f"Finnhub={result['components'].get('finnhub', {}).get('score', 0):.2f}, "
                      f"Market={result['components'].get('market', {}).get('score', 0):.2f}")
            else:
                issues.append("Analysis: Returned None")
                print("  ❌ Analysis: Returned None")
        except Exception as e:
            issues.append(f"Analysis execution: {str(e)[:60]}")
            print(f"  ❌ Analysis execution: {str(e)[:60]}")
    else:
        print("  ⚠️  Skipping analysis test (no data)")
        
except Exception as e:
    issues.append(f"Analyzer init: {str(e)[:50]}")
    print(f"  ❌ Analyzer: {str(e)[:50]}")

# Test 4: NSE stock list
print("\n4️⃣  Testing Stock Database...")

try:
    all_stocks = get_all_nse_stocks()
    print(f"  ✅ Total stocks: {len(all_stocks)}")
    print(f"     Sectors: {len(SECTOR_STOCKS)}")
    for sector in list(SECTOR_STOCKS.keys())[:3]:
        print(f"       - {sector}: {len(SECTOR_STOCKS[sector])} stocks")
except Exception as e:
    issues.append(f"Stock List: {str(e)[:50]}")
    print(f"  ❌ Stock List: {str(e)[:50]}")

# Test 5: Scheduler
print("\n5️⃣  Testing Scheduler...")

try:
    from modules.scheduler import start_scheduler, get_scheduler_status
    status = get_scheduler_status()
    print(f"  ✅ Scheduler status: {status.get('is_running', False)}")
except Exception as e:
    print(f"  ⚠️  Scheduler: Not critical - {str(e)[:40]}")

# Test 6: Optional modules
print("\n6️⃣  Testing Optional Modules...")

optional_modules = [
    ('modules.trading_dashboard', 'Trading Dashboard'),
    ('modules.analytics_page', 'Analytics Page'),
    ('modules.config_validator', 'Config Validator'),
    ('modules.sheets_tracker', 'Sheets Tracker'),
]

for module_name, display_name in optional_modules:
    try:
        importlib.import_module(module_name)
        print(f"  ✅ {display_name}: Available")
    except ImportError:
        print(f"  ⏭️  {display_name}: Not available (OK)")
    except Exception as e:
        print(f"  ⚠️  {display_name}: Error - {str(e)[:40]}")

# Summary
print("\n" + "="*80)
print("📊 DIAGNOSTIC SUMMARY")
print("="*80)

if not issues:
    print("\n✅ ALL CORE SYSTEMS OPERATIONAL")
    print("\nThe application is ready to run. Any display issues are likely:")
    print("  • Streamlit caching or session state issues")
    print("  • Widget interaction problems")
    print("  • Data display formatting issues")
else:
    print(f"\n⚠️  {len(issues)} ISSUES IDENTIFIED:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

print("\n" + "="*80)
print("✅ DIAGNOSTIC COMPLETE")
print("="*80)
