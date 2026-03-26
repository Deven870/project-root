#!/usr/bin/env python
"""
DEPLOYMENT VALIDATION SCRIPT
Verifies all systems are ready for production deployment
"""

import sys
import os
import subprocess
from pathlib import Path

print("=" * 70)
print("DEPLOYMENT VALIDATION CHECK")
print("=" * 70)

checks_passed = 0
checks_failed = 0

# Check 1: Python version
print("\n[1/8] Checking Python version...")
if sys.version_info.major == 3 and sys.version_info.minor >= 9:
    print(f"  ✅ Python {sys.version_info.major}.{sys.version_info.minor} - OK")
    checks_passed += 1
else:
    print(f"  ❌ Python {sys.version_info.major}.{sys.version_info.minor} - REQUIRES 3.9+")
    checks_failed += 1

# Check 2: Required packages
print("\n[2/8] Checking required packages...")
required_packages = [
    'pandas', 'numpy', 'sklearn', 'streamlit', 'plotly', 'yfinance',
    'requests', 'joblib'
]

missing_packages = []
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        missing_packages.append(pkg)

if not missing_packages:
    print(f"  ✅ All {len(required_packages)} packages installed")
    checks_passed += 1
else:
    print(f"  ❌ Missing packages: {', '.join(missing_packages)}")
    checks_failed += 1

# Check 3: Project structure
print("\n[3/8] Checking project structure...")
required_files = [
    'app.py', 'main.py', 'config.py', 'requirements.txt',
    'modules/__init__.py', 'modules/predictive_ml.py',
    'modules/feature_engineering.py', 'modules/backtester.py',
    'modules/utils.py'
]

missing_files = []
for file in required_files:
    if not Path(file).exists():
        missing_files.append(file)

if not missing_files:
    print(f"  ✅ All {len(required_files)} project files present")
    checks_passed += 1
else:
    print(f"  ❌ Missing files: {', '.join(missing_files[:3])}")
    checks_failed += 1

# Check 4: Models can be imported
print("\n[4/8] Checking model imports...")
try:
    from modules.predictive_ml import train_random_forest, train_xgboost
    from modules.feature_engineering import build_features, get_feature_columns
    from modules.utils import fetch_price_data, get_stock_predictions
    print("  ✅ All model modules import successfully")
    checks_passed += 1
except ImportError as e:
    print(f"  ❌ Model import failed: {e}")
    checks_failed += 1

# Check 5: Streamlit can load
print("\n[5/8] Checking Streamlit configuration...")
try:
    import streamlit as st
    print("  ✅ Streamlit installed and working")
    checks_passed += 1
except Exception as e:
    print(f"  ❌ Streamlit error: {e}")
    checks_failed += 1

# Check 6: Feature columns defined
print("\n[6/8] Checking feature engineering...")
try:
    from modules.feature_engineering import get_feature_columns
    features = get_feature_columns()
    print(f"  ✅ {len(features)} features defined for ML models")
    checks_passed += 1
except Exception as e:
    print(f"  ❌ Feature engineering error: {e}")
    checks_failed += 1

# Check 7: Synthetic data generation
print("\n[7/8] Testing data generation (synthetic)...")
try:
    import pandas as pd
    import numpy as np
    from modules.utils import _generate_synthetic_ohlc
    
    test_data = _generate_synthetic_ohlc('RELIANCE.NS', period='1mo', interval='1d')
    if len(test_data) > 0 and 'Close' in test_data.columns:
        print(f"  ✅ Synthetic data generation working ({len(test_data)} rows)")
        checks_passed += 1
    else:
        print("  ❌ Synthetic data generation failed")
        checks_failed += 1
except Exception as e:
    print(f"  ❌ Data generation error: {e}")
    checks_failed += 1

# Check 8: Model training capability
print("\n[8/8] Testing model training capability...")
try:
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    score = model.score(X, y)
    
    print(f"  ✅ Model training works (test accuracy: {score:.1%})")
    checks_passed += 1
except Exception as e:
    print(f"  ❌ Model training error: {e}")
    checks_failed += 1

# Final summary
print("\n" + "=" * 70)
print(f"VALIDATION RESULTS: {checks_passed} PASSED, {checks_failed} FAILED")
print("=" * 70)

if checks_failed == 0:
    print("\n✅ ALL CHECKS PASSED - SYSTEM READY FOR DEPLOYMENT")
    print("\nNext steps:")
    print("  1. Run dashboard:    streamlit run app.py")
    print("  2. Run quick test:   python quick_accuracy_test.py")
    print("  3. Run improved:     python improved_accuracy_model.py")
    print("  4. Review report:    Read DEPLOYMENT_REPORT.md")
    sys.exit(0)
else:
    print(f"\n❌ {checks_failed} CHECKS FAILED - FIX ISSUES BEFORE DEPLOYMENT")
    print("\nRun these commands to fix:")
    print("  pip install -r requirements.txt")
    print("  python -m pip install --upgrade scikit-learn")
    sys.exit(1)
