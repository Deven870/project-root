"""
Real-time price service with caching
"""
import aiohttp
import asyncio
from datetime import datetime
from typing import Optional, Dict
import logging
from app.config import (
    ALPHA_VANTAGE_KEY, 
    PRICE_CACHE_TTL,
    NIFTY_50_STOCKS
)
from app.services.cache_service import cache

logger = logging.getLogger(__name__)

class PriceService:
    """Async price fetcher with caching"""
    
    def __init__(self):
        self.av_key = ALPHA_VANTAGE_KEY
        self.cache_ttl = PRICE_CACHE_TTL
    
    async def get_realtime_price(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time price with caching
        Try cache first, then fetch fresh
        """
        cache_key = f"price:{symbol}"
        
        # Try cache first
        cached = await cache.get(cache_key)
        if cached:
            return cached
        
        # Fetch fresh from API
        try:
            price_data = await self._fetch_from_alpha_vantage(symbol)
            if price_data:
                # Cache for 60 seconds
                await cache.set(cache_key, price_data, self.cache_ttl)
            return price_data
        except Exception as e:
            logger.error(f"Price fetch error for {symbol}: {e}")
            return None
    
    async def _fetch_from_alpha_vantage(self, symbol: str) -> Optional[Dict]:
        """Fetch from Alpha Vantage API"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.av_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
                    
                    if "Global Quote" in data:
                        quote = data["Global Quote"]
                        return {
                            "symbol": symbol,
                            "price": float(quote.get("05. price", 0)),
                            "change": float(quote.get("09. change", 0)),
                            "change_percent": float(quote.get("10. change percent", "0").rstrip("%")),
                            "timestamp": datetime.now().isoformat(),
                            "volume": int(quote.get("06. volume", 0))
                        }
        except Exception as e:
            logger.error(f"Alpha Vantage error: {e}")
        
        return None
    
    async def get_batch_prices(self, symbols: list) -> Dict:
        """Fetch multiple prices concurrently"""
        tasks = [self.get_realtime_price(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        
        return {
            symbol: price 
            for symbol, price in zip(symbols, results)
            if price is not None
        }
    
    async def stream_prices(self, symbol: str):
        """Stream prices every 2 seconds"""
        while True:
            price = await self.get_realtime_price(symbol)
            if price:
                yield price
            await asyncio.sleep(2)

# Global price service
price_service = PriceService()
