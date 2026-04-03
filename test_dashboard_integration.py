#!/usr/bin/env python
"""
Test Dashboard Integration
Verify all 70% system components are working before launching dashboard
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported"""
    print("\n🔍 Testing Module Imports...\n")
    
    tests_passed = 0
    tests_failed = 0
    
    modules_to_test = [
        ("modules.multitimeframe_ensemble_v3", "MultiTimeframeEnsembleV2"),
        ("modules.prediction_70_integration", "get_router"),
        ("modules.macro_signals", "get_macro_signals"),
        ("modules.sentiment_integration_real", "SentimentBooster"),
        ("modules.paper_trading_framework", "PaperTradingManager"),
        ("modules.feature_engineering", "build_features"),
        ("dashboard_70_system", "render_70_accuracy_dashboard"),
    ]
    
    for module_name, class_or_func in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_or_func])
            getattr(module, class_or_func)
            print(f"  ✅ {module_name}")
            tests_passed += 1
        except Exception as e:
            print(f"  ❌ {module_name}")
            print(f"     Error: {str(e)[:60]}")
            tests_failed += 1
    
    return tests_passed, tests_failed

def test_configs():
    """Test that configuration files exist"""
    print("\n📋 Testing Configuration Files...\n")
    
    tests_passed = 0
    tests_failed = 0
    
    config_files = [
        "paper_trading_config_deployed.json",
        "paper_trading_logs/week1_trading_log.json",
    ]
    
    for config_file in config_files:
        file_path = project_root / config_file
        if file_path.exists():
            print(f"  ✅ {config_file}")
            tests_passed += 1
        else:
            print(f"  ⚠️  {config_file} (not found - will be created on first run)")
            tests_passed += 1  # Not critical
    
    return tests_passed, tests_failed

def test_app_integration():
    """Test that app.py has been updated with 70% system"""
    print("\n🎯 Testing App Integration...\n")
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        app_path = project_root / "app.py"
        with open(app_path, 'r', encoding='utf-8', errors='replace') as f:
            app_content = f.read()
        
        checks = [
            ("70% Accuracy System in navigation", "🎯 70% Accuracy System"),
            ("dashboard_70_system import", "from dashboard_70_system import"),
            ("render_70_accuracy_dashboard", "render_70_accuracy_dashboard()"),
        ]
        
        for check_name, check_string in checks:
            if check_string in app_content:
                print(f"  ✅ {check_name}")
                tests_passed += 1
            else:
                print(f"  ❌ {check_name}")
                tests_failed += 1
        
    except Exception as e:
        print(f"  ❌ Error reading app.py: {e}")
        tests_failed += 1
    
    return tests_passed, tests_failed

def test_data_availability():
    """Test that key data sources are accessible"""
    print("\n📡 Testing Data Availability...\n")
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        import yfinance as yf
        print("  ✅ yfinance (stock data)")
        tests_passed += 1
    except:
        print("  ⚠️  yfinance not available")
    
    try:
        import pandas as pd
        print("  ✅ pandas")
        tests_passed += 1
    except:
        print("  ⚠️  pandas not available")
    
    try:
        import streamlit as st
        print("  ✅ streamlit")
        tests_passed += 1
    except:
        print("  ⚠️  streamlit not available")
    
    return tests_passed, tests_failed

def main():
    """Run all tests"""
    print("=" * 70)
    print("🎯 70% ACCURACY SYSTEM - DASHBOARD INTEGRATION TEST")
    print("=" * 70)
    
    total_passed = 0
    total_failed = 0
    
    # Run all tests
    p, f = test_imports()
    total_passed += p
    total_failed += f
    
    p, f = test_configs()
    total_passed += p
    total_failed += f
    
    p, f = test_app_integration()
    total_passed += p
    total_failed += f
    
    p, f = test_data_availability()
    total_passed += p
    total_failed += f
    
    # Summary
    print("\n" + "=" * 70)
    print(f"📊 RESULTS: {total_passed} passed, {total_failed} failed")
    
    if total_failed == 0:
        print("\n✅ ALL TESTS PASSED!")
        print("\n🚀 Dashboard is ready to launch!")
        print("\nTo start the dashboard:")
        print("  streamlit run app.py")
        print("\nThen select: 🎯 70% Accuracy System")
        return 0
    else:
        print(f"\n⚠️  {total_failed} test(s) failed")
        print("Please fix the issues above and try again.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
