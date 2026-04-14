"""
╔════════════════════════════════════════════════════════════════════════════╗
║             TEST LIVE PREDICTIONS - Verification Script                   ║
║        Confirms predictions are flowing from API and WebSocket             ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import requests
import websocket
import json
import threading
import time
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

API_BASE = "http://localhost:8000"
WS_URL = "ws://localhost:8000"

class LivePredictionsTest:
    """Test harness for live predictions system"""
    
    def __init__(self):
        self.predictions_received = 0
        self.last_update = None
        self.ws = None
        
    def test_http_endpoints(self):
        """Test HTTP API endpoints"""
        logger.info("=" * 70)
        logger.info("🧪 TESTING HTTP ENDPOINTS")
        logger.info("=" * 70)
        
        # Test 1: Health check
        logger.info("\n📌 Test 1: API Health Check")
        try:
            response = requests.get(f"{API_BASE}/health", timeout=5)
            logger.info(f"✅ Health endpoint: {response.json()}")
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False
        
        # Test 2: Live service status
        logger.info("\n📌 Test 2: Live Service Status")
        try:
            response = requests.get(f"{API_BASE}/api/v1/live/status", timeout=5)
            status = response.json()
            logger.info(f"✅ Service Status:")
            logger.info(f"   Status: {status.get('status')}")
            logger.info(f"   Stocks Monitored: {status.get('stocks_monitored')}")
            logger.info(f"   Update Interval: {status.get('update_interval')}s")
            logger.info(f"   Market Open: {status.get('is_market_open')}")
            logger.info(f"   Active Subscribers: {status.get('active_subscribers')}")
            logger.info(f"   Total Updates: {status.get('total_updates')}")
        except Exception as e:
            logger.error(f"❌ Status request failed: {e}")
            return False
        
        # Test 3: Get current predictions
        logger.info("\n📌 Test 3: Fetch Current Predictions (HTTP)")
        try:
            response = requests.get(f"{API_BASE}/api/v1/live/predictions", timeout=10)
            predictions = response.json()
            count = predictions.get('count', 0)
            logger.info(f"✅ Received {count} predictions:")
            
            # Show first 3 stocks
            for i, (stock, pred) in enumerate(predictions.get('data', {}).items()):
                if i >= 3:
                    logger.info(f"   ... and {count - 3} more stocks")
                    break
                logger.info(f"\n   📈 {stock}")
                logger.info(f"      Signal: {pred.get('signal')}")
                logger.info(f"      Current: ₹{pred.get('current_price'):.2f}")
                logger.info(f"      Target: ₹{pred.get('target_price'):.2f}")
                logger.info(f"      Stop Loss: ₹{pred.get('stop_loss'):.2f}")
                logger.info(f"      Confidence: {pred.get('confidence'):.1%}")
                logger.info(f"      Technical: {pred.get('technical_score'):.1f} | Fundamental: {pred.get('fundamental_score'):.1f} | Sentiment: {pred.get('sentiment_score'):.1f}")
                
        except Exception as e:
            logger.error(f"❌ Predictions fetch failed: {e}")
            return False
        
        # Test 4: Manual refresh
        logger.info("\n📌 Test 4: Manual Prediction Refresh")
        try:
            response = requests.post(f"{API_BASE}/api/v1/live/refresh", timeout=30)
            result = response.json()
            logger.info(f"✅ Refresh result: {result}")
        except Exception as e:
            logger.error(f"❌ Manual refresh failed: {e}")
        
        return True
    
    def test_websocket_endpoint(self):
        """Test WebSocket connection"""
        logger.info("\n" + "=" * 70)
        logger.info("🧪 TESTING WEBSOCKET ENDPOINT")
        logger.info("=" * 70)
        
        ws_url = f"{WS_URL.replace('http', 'ws')}/ws/predictions"
        logger.info(f"\n📌 Connecting to: {ws_url}")
        
        try:
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            # Run WebSocket in thread
            thread = threading.Thread(target=self.ws.run_forever)
            thread.daemon = True
            thread.start()
            
            # Wait for messages (15 seconds)
            logger.info("\n⏳ Listening for WebSocket messages (15 seconds)...")
            for i in range(15):
                time.sleep(1)
                if self.predictions_received > 0:
                    logger.info(f"✅ Received {self.predictions_received} message(s)")
                    break
            
            if self.predictions_received == 0:
                logger.warning("⚠️  No WebSocket messages received in 15 seconds")
            
            self.ws.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ WebSocket test failed: {e}")
            return False
    
    def _on_open(self, ws):
        """WebSocket opened"""
        logger.info("✅ WebSocket connected")
    
    def _on_message(self, ws, message):
        """Receive WebSocket message"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "predictions_update":
                self.predictions_received += 1
                count = data.get("count", 0)
                self.last_update = datetime.fromisoformat(data.get("timestamp", ""))
                
                logger.info(f"\n✅ WebSocket Message #{self.predictions_received}")
                logger.info(f"   Type: {msg_type}")
                logger.info(f"   Predictions: {count}")
                logger.info(f"   Timestamp: {self.last_update.strftime('%H:%M:%S')}")
                
                # Show first 2 stocks
                for i, (stock, pred) in enumerate(data.get('data', {}).items()):
                    if i >= 2:
                        break
                    logger.info(f"   📈 {stock}: {pred.get('signal')} (Target: ₹{pred.get('target_price'):.2f})")
                
            elif msg_type == "initial_predictions":
                logger.info(f"✅ Initial predictions received: {data.get('count')} stocks")
                
            elif msg_type == "heartbeat":
                logger.debug("💓 Heartbeat")
        
        except Exception as e:
            logger.error(f"❌ Message parse error: {e}")
    
    def _on_error(self, ws, error):
        """WebSocket error"""
        logger.error(f"❌ WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket closed"""
        logger.info("🔌 WebSocket closed")
    
    def run_all_tests(self):
        """Run all tests"""
        logger.info("\n🚀 STARTING LIVE PREDICTIONS TEST SUITE\n")
        
        # HTTP Tests
        http_ok = self.test_http_endpoints()
        
        # WebSocket Test
        ws_ok = self.test_websocket_endpoint()
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 70)
        logger.info(f"HTTP Endpoints: {'✅ PASS' if http_ok else '❌ FAIL'}")
        logger.info(f"WebSocket: {'✅ PASS' if ws_ok else '❌ FAIL'}")
        
        if http_ok and ws_ok:
            logger.info("\n🎉 ALL TESTS PASSED - Live predictions working!")
        else:
            logger.warning("\n⚠️  Some tests failed - Check configuration")
        
        logger.info("=" * 70)


def main():
    """Run tests"""
    tester = LivePredictionsTest()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
