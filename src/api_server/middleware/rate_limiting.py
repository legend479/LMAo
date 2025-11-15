"""
Advanced Rate Limiting Middleware
Token bucket algorithm with user-specific limits, request queuing, and performance optimization
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import asyncio
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib

from src.shared.config import get_settings
from src.shared.logging import get_logger

logger = get_logger(__name__)


class RateLimitType(Enum):
    """Types of rate limits"""

    GLOBAL = "global"
    PER_IP = "per_ip"
    PER_USER = "per_user"
    PER_ENDPOINT = "per_endpoint"


@dataclass
class RateLimitRule:
    """Rate limit rule configuration"""

    limit_type: RateLimitType
    requests_per_minute: int
    burst_capacity: int
    window_size: int = 60  # seconds
    priority: int = 1  # Higher number = higher priority


@dataclass
class RequestInfo:
    """Information about a request for rate limiting"""

    client_ip: str
    user_id: Optional[str]
    endpoint: str
    method: str
    timestamp: float
    priority: int = 1


class TokenBucket:
    """Enhanced token bucket implementation with burst capacity and priority"""

    def __init__(
        self, capacity: int, refill_rate: float, burst_capacity: Optional[int] = None
    ):
        self.capacity = capacity
        self.burst_capacity = burst_capacity or capacity * 2
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

        # Statistics
        self.total_requests = 0
        self.rejected_requests = 0
        self.last_rejection = None

    async def consume(self, tokens: int = 1, priority: int = 1) -> Tuple[bool, Dict]:
        """Try to consume tokens from the bucket with priority support"""
        async with self._lock:
            now = time.time()

            # Refill tokens based on time elapsed
            elapsed = now - self.last_refill
            tokens_to_add = elapsed * self.refill_rate

            # Allow burst capacity for high priority requests
            max_tokens = self.burst_capacity if priority > 1 else self.capacity
            self.tokens = min(max_tokens, self.tokens + tokens_to_add)
            self.last_refill = now

            self.total_requests += 1

            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True, {
                    "consumed": tokens,
                    "remaining": int(self.tokens),
                    "capacity": self.capacity,
                    "refill_rate": self.refill_rate,
                }
            else:
                self.rejected_requests += 1
                self.last_rejection = now

                # Calculate retry after time
                tokens_needed = tokens - self.tokens
                retry_after = tokens_needed / self.refill_rate

                return False, {
                    "consumed": 0,
                    "remaining": int(self.tokens),
                    "capacity": self.capacity,
                    "retry_after": retry_after,
                    "rejection_count": self.rejected_requests,
                }

    def get_stats(self) -> Dict:
        """Get bucket statistics"""
        return {
            "capacity": self.capacity,
            "burst_capacity": self.burst_capacity,
            "current_tokens": int(self.tokens),
            "refill_rate": self.refill_rate,
            "total_requests": self.total_requests,
            "rejected_requests": self.rejected_requests,
            "rejection_rate": self.rejected_requests / max(self.total_requests, 1),
            "last_rejection": self.last_rejection,
        }


class RequestQueue:
    """Priority queue for handling requests during high load"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue: List[Tuple[float, int, RequestInfo, asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self.processing = False

    async def enqueue(self, request_info: RequestInfo, timeout: float = 30.0) -> bool:
        """Enqueue a request with priority and timeout"""
        async with self._lock:
            if len(self.queue) >= self.max_size:
                return False

            future = asyncio.Future()
            priority_score = -request_info.priority  # Negative for max heap behavior

            # Insert in priority order
            item = (priority_score, request_info.timestamp, request_info, future)
            self.queue.append(item)
            self.queue.sort(
                key=lambda x: (x[0], x[1])
            )  # Sort by priority, then timestamp

            # Start processing if not already running
            if not self.processing:
                asyncio.create_task(self._process_queue())

            try:
                return await asyncio.wait_for(future, timeout=timeout)
            except asyncio.TimeoutError:
                # Remove from queue if timeout
                self.queue = [item for item in self.queue if item[3] != future]
                return False

    async def _process_queue(self):
        """Process queued requests"""
        self.processing = True

        try:
            while self.queue:
                async with self._lock:
                    if not self.queue:
                        break

                    _, _, request_info, future = self.queue.pop(0)

                # Simulate processing delay
                await asyncio.sleep(0.1)

                if not future.done():
                    future.set_result(True)

        finally:
            self.processing = False


class PerformanceMonitor:
    """Monitor and optimize performance metrics"""

    def __init__(self):
        self.request_times: List[float] = []
        self.error_counts: Dict[str, int] = {}
        self.endpoint_stats: Dict[str, Dict] = {}
        self.cache_stats = {"hits": 0, "misses": 0}

    def record_request(
        self, endpoint: str, method: str, duration: float, status_code: int
    ):
        """Record request performance metrics"""

        # Keep only recent request times (last 1000)
        self.request_times.append(duration)
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-1000:]

        # Track endpoint statistics
        endpoint_key = f"{method}:{endpoint}"
        if endpoint_key not in self.endpoint_stats:
            self.endpoint_stats[endpoint_key] = {
                "count": 0,
                "total_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
                "error_count": 0,
            }

        stats = self.endpoint_stats[endpoint_key]
        stats["count"] += 1
        stats["total_time"] += duration
        stats["min_time"] = min(stats["min_time"], duration)
        stats["max_time"] = max(stats["max_time"], duration)

        if status_code >= 400:
            stats["error_count"] += 1
            error_key = f"{status_code}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""

        if not self.request_times:
            return {"message": "No requests recorded yet"}

        # Calculate percentiles
        sorted_times = sorted(self.request_times)
        count = len(sorted_times)

        p50_idx = int(count * 0.5)
        p95_idx = int(count * 0.95)
        p99_idx = int(count * 0.99)

        return {
            "request_count": count,
            "avg_response_time": sum(self.request_times) / count,
            "min_response_time": min(self.request_times),
            "max_response_time": max(self.request_times),
            "p50_response_time": sorted_times[p50_idx],
            "p95_response_time": sorted_times[p95_idx],
            "p99_response_time": sorted_times[p99_idx],
            "error_counts": self.error_counts,
            "cache_hit_rate": self.cache_stats["hits"]
            / max(self.cache_stats["hits"] + self.cache_stats["misses"], 1),
            "endpoint_stats": {
                endpoint: {
                    **stats,
                    "avg_time": stats["total_time"] / stats["count"],
                    "error_rate": stats["error_count"] / stats["count"],
                }
                for endpoint, stats in self.endpoint_stats.items()
            },
        }

    def record_cache_hit(self):
        """Record cache hit"""
        self.cache_stats["hits"] += 1

    def record_cache_miss(self):
        """Record cache miss"""
        self.cache_stats["misses"] += 1


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting middleware with user-specific limits and performance optimization"""

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        enable_queuing: bool = True,
        enable_performance_monitoring: bool = True,
    ):
        super().__init__(app)
        self.settings = get_settings()

        # Rate limiting configuration
        self.default_requests_per_minute = requests_per_minute
        self.buckets: Dict[str, TokenBucket] = {}
        self.rate_limit_rules = self._initialize_rate_limit_rules()

        # Request queuing
        self.enable_queuing = enable_queuing
        self.request_queue = RequestQueue() if enable_queuing else None

        # Performance monitoring
        self.enable_monitoring = enable_performance_monitoring
        self.performance_monitor = (
            PerformanceMonitor() if enable_performance_monitoring else None
        )

        # Cache for frequently accessed data
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.cache_ttl = 300  # 5 minutes

        # Cleanup task (only in non-test environments)
        self._cleanup_task = None
        if self.settings.environment != "testing":
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def cleanup(self):
        """Clean up background tasks"""
        if hasattr(self, "_cleanup_task") and self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    def _initialize_rate_limit_rules(self) -> List[RateLimitRule]:
        """Initialize rate limiting rules"""

        return [
            # Global rate limit
            RateLimitRule(
                limit_type=RateLimitType.GLOBAL,
                requests_per_minute=self.settings.max_concurrent_requests * 60,
                burst_capacity=self.settings.max_concurrent_requests * 120,
                priority=1,
            ),
            # Per-IP rate limit
            RateLimitRule(
                limit_type=RateLimitType.PER_IP,
                requests_per_minute=self.default_requests_per_minute,
                burst_capacity=self.default_requests_per_minute * 2,
                priority=2,
            ),
            # Per-user rate limit (higher for authenticated users)
            RateLimitRule(
                limit_type=RateLimitType.PER_USER,
                requests_per_minute=self.default_requests_per_minute * 2,
                burst_capacity=self.default_requests_per_minute * 4,
                priority=3,
            ),
            # Per-endpoint rate limits
            RateLimitRule(
                limit_type=RateLimitType.PER_ENDPOINT,
                requests_per_minute=30,  # Lower for resource-intensive endpoints
                burst_capacity=60,
                priority=2,
            ),
        ]

    async def dispatch(self, request: Request, call_next):
        """Apply advanced rate limiting and performance optimization"""

        # Bypass rate limiting in testing environment for performance
        if self.settings.environment == "testing":
            return await call_next(request)

        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")

        try:
            # Extract request information
            request_info = self._extract_request_info(request)

            # Check rate limits
            rate_limit_result = await self._check_rate_limits(request_info)

            if not rate_limit_result["allowed"]:
                # Try queuing if enabled
                if self.enable_queuing and self.request_queue:
                    logger.info(
                        "Request queued due to rate limit",
                        request_id=request_id,
                        client_ip=request_info.client_ip,
                    )

                    queued = await self.request_queue.enqueue(request_info)
                    if not queued:
                        return self._create_rate_limit_response(
                            rate_limit_result, "Queue full"
                        )
                else:
                    return self._create_rate_limit_response(rate_limit_result)

            # Check cache for GET requests
            cache_key = None
            if request.method == "GET" and self._is_cacheable_endpoint(request):
                cache_key = self._generate_cache_key(request)
                cached_response = self._get_cached_response(cache_key)

                if cached_response:
                    if self.performance_monitor:
                        self.performance_monitor.record_cache_hit()

                    logger.debug(
                        "Cache hit", request_id=request_id, cache_key=cache_key
                    )
                    return cached_response

                if self.performance_monitor:
                    self.performance_monitor.record_cache_miss()

            # Process request
            response = await call_next(request)

            # Cache successful GET responses
            if cache_key and response.status_code == 200 and request.method == "GET":
                self._cache_response(cache_key, response)

            # Add rate limit headers
            self._add_rate_limit_headers(response, rate_limit_result)

            # Record performance metrics
            if self.performance_monitor:
                duration = time.time() - start_time
                self.performance_monitor.record_request(
                    endpoint=request.url.path,
                    method=request.method,
                    duration=duration,
                    status_code=response.status_code,
                )

            return response

        except Exception as e:
            logger.error(
                "Rate limiting middleware error", request_id=request_id, error=str(e)
            )

            # Don't block requests on middleware errors
            return await call_next(request)

    def _extract_request_info(self, request: Request) -> RequestInfo:
        """Extract request information for rate limiting"""

        client_ip = request.client.host if request.client else "unknown"

        # Try to get user ID from token (if available)
        user_id = None
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # TODO: Extract user ID from JWT token
            pass

        # Determine request priority
        priority = 1
        if request.url.path.startswith("/health"):
            priority = 3  # High priority for health checks
        elif request.url.path.startswith("/api/v1/chat"):
            priority = 2  # Medium priority for chat

        return RequestInfo(
            client_ip=client_ip,
            user_id=user_id,
            endpoint=request.url.path,
            method=request.method,
            timestamp=time.time(),
            priority=priority,
        )

    async def _check_rate_limits(self, request_info: RequestInfo) -> Dict:
        """Check all applicable rate limits"""

        results = []

        for rule in self.rate_limit_rules:
            bucket_key = self._get_bucket_key(rule, request_info)

            if bucket_key not in self.buckets:
                self.buckets[bucket_key] = TokenBucket(
                    capacity=rule.requests_per_minute,
                    refill_rate=rule.requests_per_minute / 60.0,
                    burst_capacity=rule.burst_capacity,
                )

            bucket = self.buckets[bucket_key]
            allowed, bucket_info = await bucket.consume(1, request_info.priority)

            results.append(
                {
                    "rule_type": rule.limit_type.value,
                    "allowed": allowed,
                    "bucket_info": bucket_info,
                    "bucket_key": bucket_key,
                }
            )

            # If any rule blocks the request, deny it
            if not allowed:
                return {
                    "allowed": False,
                    "blocking_rule": rule.limit_type.value,
                    "bucket_info": bucket_info,
                    "all_results": results,
                }

        return {"allowed": True, "all_results": results}

    def _get_bucket_key(self, rule: RateLimitRule, request_info: RequestInfo) -> str:
        """Generate bucket key based on rate limit rule type"""

        if rule.limit_type == RateLimitType.GLOBAL:
            return "global"
        elif rule.limit_type == RateLimitType.PER_IP:
            return f"ip:{request_info.client_ip}"
        elif rule.limit_type == RateLimitType.PER_USER:
            return f"user:{request_info.user_id or request_info.client_ip}"
        elif rule.limit_type == RateLimitType.PER_ENDPOINT:
            return f"endpoint:{request_info.method}:{request_info.endpoint}"
        else:
            return "unknown"

    def _create_rate_limit_response(
        self, rate_limit_result: Dict, additional_message: str = ""
    ) -> Response:
        """Create rate limit exceeded response"""

        bucket_info = rate_limit_result.get("bucket_info", {})
        retry_after = bucket_info.get("retry_after", 60)

        message = f"Rate limit exceeded for {rate_limit_result.get('blocking_rule', 'requests')}."
        if additional_message:
            message += f" {additional_message}"

        headers = {
            "Retry-After": str(int(retry_after)),
            "X-RateLimit-Limit": str(bucket_info.get("capacity", "unknown")),
            "X-RateLimit-Remaining": str(bucket_info.get("remaining", 0)),
            "X-RateLimit-Reset": str(int(time.time() + retry_after)),
        }

        return Response(content=message, status_code=429, headers=headers)

    def _add_rate_limit_headers(self, response: Response, rate_limit_result: Dict):
        """Add rate limit headers to response"""

        if rate_limit_result.get("allowed") and rate_limit_result.get("all_results"):
            # Use the most restrictive bucket for headers
            most_restrictive = min(
                rate_limit_result["all_results"],
                key=lambda x: x["bucket_info"].get("remaining", float("inf")),
            )

            bucket_info = most_restrictive["bucket_info"]

            response.headers["X-RateLimit-Limit"] = str(
                bucket_info.get("capacity", "unknown")
            )
            response.headers["X-RateLimit-Remaining"] = str(
                bucket_info.get("remaining", 0)
            )
            response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))

    def _is_cacheable_endpoint(self, request: Request) -> bool:
        """Check if endpoint response can be cached"""

        cacheable_paths = ["/health", "/api/v1/tools", "/info"]

        return any(request.url.path.startswith(path) for path in cacheable_paths)

    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key for request"""

        key_parts = [
            request.method,
            request.url.path,
            str(sorted(request.query_params.items())),
        ]

        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[Response]:
        """Get cached response if available and not expired"""

        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]

            if time.time() - timestamp < self.cache_ttl:
                return cached_data
            else:
                # Remove expired entry
                del self.cache[cache_key]

        return None

    def _cache_response(self, cache_key: str, response: Response):
        """Cache response for future use"""

        # Only cache small responses
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > 10240:  # 10KB limit
            return

        self.cache[cache_key] = (response, time.time())

        # Limit cache size
        if len(self.cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(self.cache.keys(), key=lambda k: self.cache[k][1])[
                :100
            ]

            for key in oldest_keys:
                del self.cache[key]

    async def _periodic_cleanup(self):
        """Periodic cleanup of expired buckets and cache entries"""

        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                current_time = time.time()

                # Clean up inactive buckets
                inactive_buckets = []
                for bucket_key, bucket in self.buckets.items():
                    if current_time - bucket.last_refill > 3600:  # 1 hour inactive
                        inactive_buckets.append(bucket_key)

                for bucket_key in inactive_buckets:
                    del self.buckets[bucket_key]

                # Clean up expired cache entries
                expired_cache_keys = []
                for cache_key, (_, timestamp) in self.cache.items():
                    if current_time - timestamp > self.cache_ttl:
                        expired_cache_keys.append(cache_key)

                for cache_key in expired_cache_keys:
                    del self.cache[cache_key]

                logger.debug(
                    "Rate limiting cleanup completed",
                    removed_buckets=len(inactive_buckets),
                    removed_cache_entries=len(expired_cache_keys),
                )

            except Exception as e:
                logger.error("Rate limiting cleanup error", error=str(e))

    def get_statistics(self) -> Dict:
        """Get rate limiting and performance statistics"""

        bucket_stats = {}
        for bucket_key, bucket in self.buckets.items():
            bucket_stats[bucket_key] = bucket.get_stats()

        stats = {
            "active_buckets": len(self.buckets),
            "cache_entries": len(self.cache),
            "bucket_statistics": bucket_stats,
        }

        if self.performance_monitor:
            stats["performance_metrics"] = (
                self.performance_monitor.get_performance_metrics()
            )

        if self.request_queue:
            stats["queue_size"] = len(self.request_queue.queue)
            stats["queue_processing"] = self.request_queue.processing

        return stats
