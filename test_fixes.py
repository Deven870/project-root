#!/usr/bin/env python3
"""Quick test of fixes"""
from modules.precision_analyzer import EnhancedPrecisionAnalyzer
from modules.utils import fetch_price_data

analyzer = EnhancedPrecisionAnalyzer()
data = fetch_price_data('INFY.NS')

result = analyzer.get_precision_analysis('INFY.NS', data, data)

print("TEST RESULTS:")
print(f"  Signal: {result['signal']}")
print(f"  Confidence: {result['confidence']:.1f}%")
print(f"  Finnhub Score: {result['components']['finnhub']['score']:.2f}")
print(f"  Technical Score: {result['components']['technical']['score']:.2f}")
print(f"  Market Score: {result['components']['market']['score']:.2f}")
print(f"  Data Quality: {result['precision_metrics']['data_quality']}")

if result['components']['finnhub']['score'] != 0:
    print("\nSUCCESS: Finnhub error FIXED!")
else:
    print("\nWARNING: Finnhub still returning 0")
