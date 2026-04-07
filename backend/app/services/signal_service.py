"""
Signal generation service using precision analyzer
"""
import sys
import os
from typing import Optional, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Add modules path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

class SignalService:
    """Generate trading signals using precision analyzer"""
    
    def __init__(self):
        try:
            from modules.precision_analyzer import EnhancedPrecisionAnalyzer
            self.analyzer = EnhancedPrecisionAnalyzer()
            logger.info("✅ Signal analyzer loaded")
        except Exception as e:
            logger.error(f"Failed to load analyzer: {e}")
            self.analyzer = None
    
    async def analyze_stock(self, symbol: str, price_data) -> Optional[Dict]:
        """
        Analyze stock and generate signal
        Returns: {signal, confidence, technical_score, recommendation}
        """
        try:
            if not self.analyzer or price_data is None or price_data.empty:
                return {
                    "symbol": symbol,
                    "signal": "HOLD",
                    "confidence": 0,
                    "recommendation": "Insufficient data"
                }
            
            # Use existing precision analyzer
            analysis = self.analyzer.get_precision_analysis(
                symbol, 
                price_data, 
                price_data
            )
            
            return {
                "symbol": symbol,
                "signal": analysis.get("signal", "HOLD"),
                "confidence": analysis.get("confidence", 0),
                "technical_score": analysis.get("technical_score", 0),
                "sentiment_score": analysis.get("sentiment_score", 0),
                "recommendation": analysis.get("recommendation", "Wait for better setup"),
                "analysis_data": analysis
            }
        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {e}")
            return {
                "symbol": symbol,
                "signal": "HOLD",
                "confidence": 0,
                "recommendation": f"Analysis failed: {str(e)[:50]}"
            }

# Global signal service
signal_service = SignalService()
