"""
Redis cache service for prices, sentiment, and analysis
"""
import redis
import json
import logging
from typing import Any, Optional
from ..config import REDIS_HOST, REDIS_PORT, REDIS_DB

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        try:
            self.redis = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
            )
            self.redis.ping()
            logger.info("✅ Redis connected")
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable: {e}. Using fallback.")
            self.redis = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if not self.redis:
                return None
            value = self.redis.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Cache GET error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache with TTL"""
        try:
            if not self.redis:
                return
            self.redis.setex(
                key,
                ttl,
                json.dumps(value)
            )
        except Exception as e:
            logger.error(f"Cache SET error: {e}")
    
    async def delete(self, key: str):
        """Delete key from cache"""
        try:
            if not self.redis:
                return
            self.redis.delete(key)
        except Exception as e:
            logger.error(f"Cache DELETE error: {e}")
    
    async def clear_pattern(self, pattern: str):
        """Clear keys matching pattern"""
        try:
            if not self.redis:
                return
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
        except Exception as e:
            logger.error(f"Cache CLEAR error: {e}")
    
    def is_connected(self) -> bool:
        """Check if Redis is connected"""
        return self.redis is not None

# Global cache instance
cache = CacheService()
