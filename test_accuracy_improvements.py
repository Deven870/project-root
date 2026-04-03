#!/usr/bin/env python3
"""
Accuracy Improvement Test Script
Tests all strategies to improve accuracy from 50% to 60%+
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.accuracy_optimizer import AccuracyOptimizer, test_accuracy_improvements
from modules.feature_engineering import build_features, get_feature_columns
from modules.utils import fetch_price_data


def main():
    print("\n" + "="*70)
    print("[ANALYSIS] ACCURACY IMPROVEMENT - COMPREHENSIVE TEST")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}\n")
    
    # Test stocks
    test_stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    
    overall_results = {
        "baseline_accuracy": [],
        "improved_accuracy": [],
        "improvements": []
    }
    
    for stock in test_stocks:
        print(f"\n[PROCESSING] {stock}...")
        
        try:
            # Fetch data
            data = fetch_price_data(stock, period="2y", interval="1d")
            
            if data is None or len(data) < 200:
                print(f"  [SKIP] Insufficient data for {stock}")
                continue
            
            # Build features
            features_df = build_features(data, sentiment_score=0.0)
            
            if features_df.empty or len(features_df) < 100:
                print(f"  [SKIP] Feature engineering failed for {stock}")
                continue
            
            # Prepare data
            feature_cols = [c for c in get_feature_columns() if c in features_df.columns]
            X = features_df[feature_cols].values
            y = features_df["target_direction"].values
            
            # Clean
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
            y = np.nan_to_num(y, nan=0, posinf=0, neginf=0).astype(int)
            
            # Train/test split (80/20)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                print(f"  [SKIP] Not enough classes for {stock}")
                continue
            
            # Test improvements
            results = test_accuracy_improvements(X_train, y_train, X_test, y_test, stock)
            
            overall_results["baseline_accuracy"].append(results["baseline"]["accuracy"])
            overall_results["improved_accuracy"].append(results["improved"]["best_accuracy"])
            overall_results["improvements"].append(results["improved"]["improvement_pct"])
        
        except Exception as e:
            print(f"  [ERROR] {stock}: {e}")
            continue
    
    # Summary
    print("\n" + "="*70)
    print("[SUMMARY] OVERALL ACCURACY IMPROVEMENT")
    print("="*70)
    
    if overall_results["baseline_accuracy"]:
        avg_baseline = np.mean(overall_results["baseline_accuracy"])
        avg_improved = np.mean(overall_results["improved_accuracy"])
        avg_improvement = np.mean(overall_results["improvements"])
        
        print(f"\nStocks Tested: {len(overall_results['baseline_accuracy'])}")
        print(f"Average Baseline: {avg_baseline:.1%}")
        print(f"Average Improved: {avg_improved:.1%}")
        print(f"Average Improvement: +{avg_improvement:.1f}%")
        
        print(f"\n[TECHNIQUES EFFECTIVENESS]:")
        print(f"  * Class weights: Handle imbalanced datasets")
        print(f"  * Feature selection: Remove noise, keep signal")
        print(f"  * XGBoost: Powerful gradient boosting")
        print(f"  * Threshold tuning: Optimize classification boundary")
        
        if avg_improved > avg_baseline + 0.05:
            print(f"\n[SUCCESS] Accuracy improved by {(avg_improved - avg_baseline)*100:.1f}%")
            print(f"[NEXT] Deploy improved models to production")
        else:
            print(f"\n[WARNING] Limited improvement - market is efficient")
            print(f"[NEXT] Focus on longer-term signals, not daily predictions")
    else:
        print("No sufficient test data available")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
