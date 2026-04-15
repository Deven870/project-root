#!/usr/bin/env python
"""Check live predictions and show STRONG_BUY signals for trading"""

import requests
import json
from datetime import datetime

API_URL = "http://localhost:8000/api/v1/live/predictions"

print("\n" + "="*80)
print("🚀 NSEIQ LIVE PREDICTIONS ANALYZER")
print("="*80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

try:
    print("📡 Fetching live predictions from API...\n")
    resp = requests.get(API_URL, timeout=10)
    
    if resp.status_code == 200:
        data = resp.json()
        
        # Handle response format: {'count': N, 'data': {symbol: pred_dict, ...}}
        predictions_dict = data.get('data', {})
        
        # Convert dict to list
        predictions = [p for p in predictions_dict.values()] if isinstance(predictions_dict, dict) else []
        
        print(f"✅ API CONNECTED - {len(predictions)} stocks analyzed\n")
        
        # Filter STRONG_BUY signals
        strong_buys = [p for p in predictions if p.get('signal') == 'STRONG_BUY']
        buys = [p for p in predictions if p.get('signal') == 'BUY']
        
        print(f"📊 PREDICTION SUMMARY:")
        print(f"   • STRONG_BUY: {len(strong_buys)} stocks 🎯")
        print(f"   • BUY: {len(buys)} stocks")
        print(f"   • NEUTRAL/HOLD: {len([p for p in predictions if p.get('signal') in ['NEUTRAL', 'HOLD']])} stocks")
        print(f"   • SELL/STRONG_SELL: {len([p for p in predictions if p.get('signal') in ['SELL', 'STRONG_SELL']])} stocks\n")
        
        if strong_buys:
            print("="*80)
            print("🎯 STRONG_BUY SIGNALS - YOUR BOT WILL AUTOMATICALLY TRADE THESE!")
            print("="*80 + "\n")
            
            for i, pred in enumerate(strong_buys, 1):
                symbol = pred.get('symbol', 'N/A')
                entry = pred.get('current_price', 0)
                target = pred.get('target_price', 0)
                sl = pred.get('stop_loss', 0)
                conf = pred.get('confidence', 0)
                
                profit_pct = ((target - entry) / entry * 100) if entry > 0 else 0
                risk_pct = ((entry - sl) / entry * 100) if entry > 0 else 0
                reward_ratio = profit_pct / risk_pct if risk_pct > 0 else 0
                
                print(f"#{i} {symbol}")
                print(f"    Entry Price:      ₹{entry:.2f}")
                print(f"    Target Price:     ₹{target:.2f} (Profit: +{profit_pct:.1f}%)")
                print(f"    Stop Loss:        ₹{sl:.2f} (Risk: -{risk_pct:.1f}%)")
                print(f"    Profit/Risk:      {reward_ratio:.2f}x")
                print(f"    Confidence:       {conf:.1f}%")
                print(f"    Signal Strength:  {'⭐' * int(conf/20)}")
                print()
        else:
            print("⏳ No STRONG_BUY signals yet (bot analyzing...)\n")
            print("📈 TOP OPPORTUNITIES BY CONFIDENCE:\n")
            
            top = sorted(predictions, key=lambda x: x.get('confidence', 0), reverse=True)[:5]
            for i, pred in enumerate(top, 1):
                symbol = pred.get('symbol', 'N/A')
                signal = pred.get('signal', 'N/A')
                conf = pred.get('confidence', 0)
                print(f"{i}. {symbol:12} | Signal: {signal:12} | Confidence: {conf:.1f}%")
        
        print("\n" + "="*80)
        print("📌 HOW YOUR BOT WORKS:")
        print("="*80)
        print("""
1. ✅ API constantly analyzes all 15 NSE stocks with 6-layer analysis
2. ✅ When STRONG_BUY signal found (>75% confidence), bot receives it
3. ✅ Bot validates: Risk/Reward ratio, Daily loss limit, Max positions
4. ✅ Bot auto-executes: Entry, Stop Loss, Target
5. ✅ Position tracked: Live P&L, Exit management, Trade journal
        
Capital: ₹300,000 | Risk/Trade: 8% (₹24,000) | Daily Limit: 7% (₹21,000)
        """)
        
    else:
        print(f"❌ API Error: {resp.status_code}")
        print("Make sure API server is running: python -m uvicorn backend.app.main:app --port 8000")
        
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to API at http://localhost:8000")
    print("Make sure API server is running on port 8000")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*80)
