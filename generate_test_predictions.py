#!/usr/bin/env python
"""Generate test STRONG_BUY signals for immediate bot testing"""

import requests
import json
from datetime import datetime

# Sample STRONG_BUY predictions
test_predictions = {
    "RELIANCE": {
        "ticker": "RELIANCE",
        "signal": "STRONG_BUY",
        "current_price": 2450.00,
        "target_price": 2650.00,
        "stop_loss": 2350.00,
        "confidence": 85.5,
        "timestamp": datetime.now().isoformat(),
        "technical_score": 90,
        "fundamental_score": 80,
        "sentiment_score": 85,
    },
    "TCS": {
        "ticker": "TCS",
        "signal": "STRONG_BUY",
        "current_price": 3850.00,
        "target_price": 4200.00,
        "stop_loss": 3650.00,
        "confidence": 82.0,
        "timestamp": datetime.now().isoformat(),
        "technical_score": 88,
        "fundamental_score": 78,
        "sentiment_score": 80,
    },
    "HDFCBANK": {
        "ticker": "HDFCBANK",
        "signal": "STRONG_BUY",
        "current_price": 1680.00,
        "target_price": 1850.00,
        "stop_loss": 1580.00,
        "confidence": 80.0,
        "timestamp": datetime.now().isoformat(),
        "technical_score": 85,
        "fundamental_score": 82,
        "sentiment_score": 78,
    },
    "INFY": {
        "ticker": "INFY",
        "signal": "BUY",
        "current_price": 1950.00,
        "target_price": 2100.00,
        "stop_loss": 1850.00,
        "confidence": 75.0,
        "timestamp": datetime.now().isoformat(),
        "technical_score": 80,
        "fundamental_score": 75,
        "sentiment_score": 72,
    },
    "SBIN": {
        "ticker": "SBIN",
        "signal": "NEUTRAL",
        "current_price": 625.00,
        "target_price": 650.00,
        "stop_loss": 600.00,
        "confidence": 65.0,
        "timestamp": datetime.now().isoformat(),
        "technical_score": 70,
        "fundamental_score": 65,
        "sentiment_score": 62,
    },
}

print("\n" + "="*80)
print("🧪 GENERATING TEST PREDICTIONS FOR BOT TESTING")
print("="*80 + "\n")

print("📊 Test Predictions Summary:")
print(f"   STRONG_BUY: {len([p for p in test_predictions.values() if p['signal'] == 'STRONG_BUY'])}")
print(f"   BUY: {len([p for p in test_predictions.values() if p['signal'] == 'BUY'])}")
print(f"   NEUTRAL: {len([p for p in test_predictions.values() if p['signal'] == 'NEUTRAL'])}\n")

print("🎯 STRONG_BUY SIGNALS (Bot will trade these):\n")
for pred in [p for p in test_predictions.values() if p['signal'] == 'STRONG_BUY']:
    profit_pct = ((pred['target_price'] - pred['current_price']) / pred['current_price'] * 100)
    risk_pct = ((pred['current_price'] - pred['stop_loss']) / pred['current_price'] * 100)
    
    print(f"✅ {pred['ticker']}")
    print(f"   Entry: ₹{pred['current_price']:.2f}")
    print(f"   Target: ₹{pred['target_price']:.2f} (+{profit_pct:.1f}%)")
    print(f"   Stop Loss: ₹{pred['stop_loss']:.2f} (-{risk_pct:.1f}%)")
    print(f"   Confidence: {pred['confidence']:.1f}%\n")

print("="*80)
print("💾 INJECTING TEST PREDICTIONS INTO SYSTEM...")
print("="*80 + "\n")

# Inject into live prediction service
try:
    from backend.app.services.live_prediction_service import get_live_prediction_service
    
    service = get_live_prediction_service()
    service.current_predictions = test_predictions
    print("✅ Test predictions injected into live prediction service")
    print("✅ Bot will now see these STRONG_BUY signals")
    print("✅ Navigate to dashboard to see trading bot execute trades\n")
    
except Exception as e:
    print(f"Info: Direct injection not available ({e})")
    print("Bot will use real predictions once they're ready\n")

print("="*80)
print("🚀 Ready for testing! Open dashboard at http://localhost:8501")
print("="*80 + "\n")
