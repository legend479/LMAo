"""
Cache Management System
Multi-level caching with Redis backend and in-memory fallback
"""

import time
import hashlib
from typing import Any, Optional, Dict, List, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import pickle
import gzip

from src.shared.config import get_settings
from src.shared.logging import get_logger

logger = get_logger(__name__)


class CacheLevel(Enum):
    """Cache levels with different characteristics"""

    MEMORY = "memory"  # Fast, limited size, process-local
    REDIS = "redis"  # Fast, shared, persistent
    DISK = "disk"  # Slower, large capacity, persistent


@dataclass
class CacheEntry:
    """Cache entry with metadata"""

    key: str
    value: Any
    created_at: float
    expires_at: Optional[float]
    access_count: int = 0
    last_accessed: float = 0
    size_bytes: int = 0
    compressed: bool = False


class MemoryCache:
    """In-memory LRU cache with size limits"""

    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.current_memory = 0
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        async with self._lock:
            if key in self.cache:
                entry = self.cache[key]

                # Check expiration
                if entry.expires_at and time.time() > entry.expires_at:
                    await self._remove_entry(key)
                    return None

                # Update access statistics
                entry.access_count += 1
                entry.last_accessed = time.time()

                # Move to end of access order (most recently used)
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)

                return entry.value

            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache"""
        async with self._lock:
            # Calculate size
            try:
                serialized = pickle.dumps(value)
                size_bytes = len(serialized)
            except Exception:
                logger.warning("Failed to serialize value for caching", key=key)
                return False

            # Check if we need to compress large values
            compressed = False
            if size_bytes > 1024:  # Compress values larger than 1KB
                try:
                    compressed_data = gzip.compress(serialized)
                    if (
                        len(compressed_data) < size_bytes * 0.8
                    ):  # Only if compression saves 20%+
                        serialized = compressed_data
                        size_bytes = len(compressed_data)
                        compressed = True
                except Exception:
                    pass  # Use uncompressed if compression fails

            # Check memory limits
            if size_bytes > self.max_memory_bytes:
                logger.warning(
                    "Value too large for memory cache", key=key, size=size_bytes
                )
                return False

            # Make room if necessary
            while (
                len(self.cache) >= self.max_size
                or self.current_memory + size_bytes > self.max_memory_bytes
            ):
                if not self.access_order:
                    break

                lru_key = self.access_order[0]
                await self._remove_entry(lru_key)

            # Create cache entry
            expires_at = time.time() + ttl if ttl else None
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                expires_at=expires_at,
                access_count=1,
                last_accessed=time.time(),
                size_bytes=size_bytes,
                compressed=compressed,
            )

            # Remove existing entry if present
            if key in self.cache:
                await self._remove_entry(key)

            # Add new entry
            self.cache[key] = entry
            self.access_order.append(key)
            self.current_memory += size_bytes

            return True

    async def delete(self, key: str) -> bool:
        """Delete value from memory cache"""
        async with self._lock:
            if key in self.cache:
                await self._remove_entry(key)
                return True
            return False

    async def clear(self):
        """Clear all entries from memory cache"""
        async with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.current_memory = 0

    async def _remove_entry(self, key: str):
        """Remove entry and update memory tracking"""
        if key in self.cache:
            entry = self.cache[key]
            self.current_memory -= entry.size_bytes
            del self.cache[key]

        if key in self.access_order:
            self.access_order.remove(key)

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_access_count = sum(entry.access_count for entry in self.cache.values())

        return {
            "entries": len(self.cache),
            "max_size": self.max_size,
            "memory_usage_bytes": self.current_memory,
            "max_memory_bytes": self.max_memory_bytes,
            "memory_usage_percent": (self.current_memory / self.max_memory_bytes) * 100,
            "total_accesses": total_access_count,
            "avg_accesses_per_entry": total_access_count / max(len(self.cache), 1),
        }


class RedisCache:
    """Redis-based cache with serialization and compression"""

    def __init__(self):
        self.redis_client = None  # TODO: Initialize Redis connection
        self.enabled = False

    async def initialize(self):
        """Initialize Redis connection"""
        try:
            # TODO: Initialize actual Redis connection
            # self.redis_client = await aioredis.from_url(get_settings().redis_url)
            # await self.redis_client.ping()
            # self.enabled = True
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.warning("Failed to initialize Redis cache", error=str(e))
            self.enabled = False

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self.enabled:
            return None

        try:
            # TODO: Implement Redis get
            # data = await self.redis_client.get(key)
            # if data:
            #     return pickle.loads(data)
            return None
        except Exception as e:
            logger.error("Redis cache get error", key=key, error=str(e))
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        if not self.enabled:
            return False

        try:
            # TODO: Implement Redis set
            # serialized = pickle.dumps(value)
            # await self.redis_client.set(key, serialized, ex=ttl)
            return True
        except Exception as e:
            logger.error("Redis cache set error", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache"""
        if not self.enabled:
            return False

        try:
            # TODO: Implement Redis delete
            # result = await self.redis_client.delete(key)
            # return result > 0
            return True
        except Exception as e:
            logger.error("Redis cache delete error", key=key, error=str(e))
            return False

    async def clear(self):
        """Clear Redis cache"""
        if not self.enabled:
            return

        try:
            # TODO: Implement Redis clear
            # await self.redis_client.flushdb()
            pass
        except Exception as e:
            logger.error("Redis cache clear error", error=str(e))


class CacheManager:
    """Multi-level cache manager with intelligent routing"""

    def __init__(self):
        self.memory_cache = MemoryCache()
        self.redis_cache = RedisCache()
        self.settings = get_settings()

        # Cache statistics
        self.stats = {
            "hits": {"memory": 0, "redis": 0, "total": 0},
            "misses": {"memory": 0, "redis": 0, "total": 0},
            "sets": {"memory": 0, "redis": 0, "total": 0},
            "deletes": {"memory": 0, "redis": 0, "total": 0},
            "errors": {"memory": 0, "redis": 0, "total": 0},
        }

        # Cache policies
        self.default_ttl = 300  # 5 minutes
        self.memory_only_patterns = ["temp:", "session:", "rate_limit:"]
        self.redis_preferred_patterns = ["user:", "document:", "tool_result:"]

    async def initialize(self):
        """Initialize cache manager"""
        await self.redis_cache.initialize()
        logger.info("Cache manager initialized")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (tries memory first, then Redis)"""

        # Try memory cache first
        value = await self.memory_cache.get(key)
        if value is not None:
            self.stats["hits"]["memory"] += 1
            self.stats["hits"]["total"] += 1
            return value

        self.stats["misses"]["memory"] += 1

        # Try Redis cache if not memory-only
        if not self._is_memory_only(key):
            value = await self.redis_cache.get(key)
            if value is not None:
                self.stats["hits"]["redis"] += 1
                self.stats["hits"]["total"] += 1

                # Promote to memory cache for faster future access
                await self.memory_cache.set(key, value, ttl=self.default_ttl)

                return value

            self.stats["misses"]["redis"] += 1

        self.stats["misses"]["total"] += 1
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in appropriate cache levels"""

        if ttl is None:
            ttl = self.default_ttl

        success = False

        # Always try to set in memory cache
        if await self.memory_cache.set(key, value, ttl):
            self.stats["sets"]["memory"] += 1
            success = True

        # Set in Redis if not memory-only
        if not self._is_memory_only(key):
            if await self.redis_cache.set(key, value, ttl):
                self.stats["sets"]["redis"] += 1
                success = True

        if success:
            self.stats["sets"]["total"] += 1

        return success

    async def delete(self, key: str) -> bool:
        """Delete value from all cache levels"""

        memory_deleted = await self.memory_cache.delete(key)
        redis_deleted = await self.redis_cache.delete(key)

        if memory_deleted:
            self.stats["deletes"]["memory"] += 1

        if redis_deleted:
            self.stats["deletes"]["redis"] += 1

        if memory_deleted or redis_deleted:
            self.stats["deletes"]["total"] += 1
            return True

        return False

    async def clear(self, pattern: Optional[str] = None):
        """Clear cache entries (optionally matching pattern)"""

        if pattern:
            # TODO: Implement pattern-based clearing
            logger.warning(
                "Pattern-based cache clearing not implemented", pattern=pattern
            )
        else:
            await self.memory_cache.clear()
            await self.redis_cache.clear()

    def _is_memory_only(self, key: str) -> bool:
        """Check if key should only be stored in memory cache"""
        return any(key.startswith(pattern) for pattern in self.memory_only_patterns)

    def _is_redis_preferred(self, key: str) -> bool:
        """Check if key should be preferentially stored in Redis"""
        return any(key.startswith(pattern) for pattern in self.redis_preferred_patterns)

    def generate_key(self, *parts: Union[str, int, float]) -> str:
        """Generate cache key from parts"""
        key_string = ":".join(str(part) for part in parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get_statistics(self) -> Dict:
        """Get comprehensive cache statistics"""

        memory_stats = self.memory_cache.get_stats()

        # Calculate hit rates
        total_requests = self.stats["hits"]["total"] + self.stats["misses"]["total"]
        hit_rate = (self.stats["hits"]["total"] / max(total_requests, 1)) * 100

        memory_requests = self.stats["hits"]["memory"] + self.stats["misses"]["memory"]
        memory_hit_rate = (self.stats["hits"]["memory"] / max(memory_requests, 1)) * 100

        redis_requests = self.stats["hits"]["redis"] + self.stats["misses"]["redis"]
        redis_hit_rate = (self.stats["hits"]["redis"] / max(redis_requests, 1)) * 100

        return {
            "overall": {
                "total_requests": total_requests,
                "hit_rate_percent": hit_rate,
                "hits": self.stats["hits"]["total"],
                "misses": self.stats["misses"]["total"],
            },
            "memory_cache": {
                **memory_stats,
                "hit_rate_percent": memory_hit_rate,
                "hits": self.stats["hits"]["memory"],
                "misses": self.stats["misses"]["memory"],
            },
            "redis_cache": {
                "enabled": self.redis_cache.enabled,
                "hit_rate_percent": redis_hit_rate,
                "hits": self.stats["hits"]["redis"],
                "misses": self.stats["misses"]["redis"],
            },
            "operations": {
                "sets": self.stats["sets"],
                "deletes": self.stats["deletes"],
                "errors": self.stats["errors"],
            },
        }


# Global cache manager instance
cache_manager = CacheManager()


async def get_cache_manager() -> CacheManager:
    """Get cache manager instance"""
    return cache_manager


# Convenience functions for common caching patterns
async def cached_function(key: str, func, ttl: Optional[int] = None, *args, **kwargs):
    """Cache function result"""

    # Try to get from cache first
    result = await cache_manager.get(key)
    if result is not None:
        return result

    # Execute function and cache result
    try:
        result = (
            await func(*args, **kwargs)
            if asyncio.iscoroutinefunction(func)
            else func(*args, **kwargs)
        )
        await cache_manager.set(key, result, ttl)
        return result
    except Exception as e:
        logger.error("Cached function execution error", key=key, error=str(e))
        raise


def cache_key(*parts: Union[str, int, float]) -> str:
    """Generate cache key from parts"""
    return cache_manager.generate_key(*parts)
