"""
Metrics Collection and Monitoring
Prometheus-compatible metrics for system monitoring
"""

from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
import time
import psutil
import asyncio
from functools import wraps

from .config import get_settings
from .logging import get_logger

logger = get_logger(__name__)

# Define metrics
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status_code"]
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

ACTIVE_CONNECTIONS = Gauge("active_connections", "Number of active connections")

TOOL_EXECUTION_COUNT = Counter(
    "tool_executions_total", "Total tool executions", ["tool_name", "status"]
)

TOOL_EXECUTION_DURATION = Histogram(
    "tool_execution_duration_seconds",
    "Tool execution duration in seconds",
    ["tool_name"],
)

DOCUMENT_PROCESSING_COUNT = Counter(
    "documents_processed_total", "Total documents processed", ["format", "status"]
)

DOCUMENT_PROCESSING_DURATION = Histogram(
    "document_processing_duration_seconds",
    "Document processing duration in seconds",
    ["format"],
)

SEARCH_QUERIES_COUNT = Counter(
    "search_queries_total", "Total search queries", ["query_type"]
)

SEARCH_QUERY_DURATION = Histogram(
    "search_query_duration_seconds", "Search query duration in seconds", ["query_type"]
)

# System metrics
CPU_USAGE = Gauge("cpu_usage_percent", "CPU usage percentage")
MEMORY_USAGE = Gauge("memory_usage_bytes", "Memory usage in bytes")
DISK_USAGE = Gauge("disk_usage_bytes", "Disk usage in bytes", ["path"])

# Application info
APP_INFO = Info("app_info", "Application information")


class MetricsCollector:
    """Centralized metrics collection and management"""

    def __init__(self):
        self.settings = get_settings()
        self._server_started = False
        self._collection_task = None

    async def start_metrics_server(self):
        """Start Prometheus metrics server"""
        if self._server_started:
            return

        if self.settings.enable_metrics:
            try:
                start_http_server(self.settings.metrics_port)
                self._server_started = True

                # Set application info
                APP_INFO.info(
                    {
                        "version": self.settings.version,
                        "environment": self.settings.environment,
                        "app_name": self.settings.app_name,
                    }
                )

                # Start system metrics collection
                self._collection_task = asyncio.create_task(
                    self._collect_system_metrics()
                )

                logger.info("Metrics server started", port=self.settings.metrics_port)

            except OSError as e:
                if e.errno == 98:  # Address already in use
                    logger.warning(
                        f"Metrics server port {self.settings.metrics_port} is busy (likely occupied by another worker). Skipping."
                    )
                    # Mark as started so we don't keep retrying and filling logs
                    self._server_started = True
                else:
                    logger.error("Failed to start metrics server", error=str(e))
            except Exception as e:
                logger.error("Failed to start metrics server", error=str(e))
        else:
            logger.info("Metrics collection disabled")

    async def stop_metrics_server(self):
        """Stop metrics collection"""
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass

        logger.info("Metrics collection stopped")

    async def _collect_system_metrics(self):
        """Collect system metrics periodically"""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                CPU_USAGE.set(cpu_percent)

                # Memory usage
                memory = psutil.virtual_memory()
                MEMORY_USAGE.set(memory.used)

                # Disk usage
                disk = psutil.disk_usage("/")
                DISK_USAGE.labels(path="/").set(disk.used)

                await asyncio.sleep(30)  # Collect every 30 seconds

            except Exception as e:
                logger.error("Error collecting system metrics", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error

    def record_request(
        self, method: str, endpoint: str, status_code: int, duration: float
    ):
        """Record HTTP request metrics"""
        REQUEST_COUNT.labels(
            method=method, endpoint=endpoint, status_code=str(status_code)
        ).inc()

        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

    def record_tool_execution(self, tool_name: str, status: str, duration: float):
        """Record tool execution metrics"""
        TOOL_EXECUTION_COUNT.labels(tool_name=tool_name, status=status).inc()

        TOOL_EXECUTION_DURATION.labels(tool_name=tool_name).observe(duration)

    def record_document_processing(self, format: str, status: str, duration: float):
        """Record document processing metrics"""
        DOCUMENT_PROCESSING_COUNT.labels(format=format, status=status).inc()

        DOCUMENT_PROCESSING_DURATION.labels(format=format).observe(duration)

    def record_search_query(self, query_type: str, duration: float):
        """Record search query metrics"""
        SEARCH_QUERIES_COUNT.labels(query_type=query_type).inc()

        SEARCH_QUERY_DURATION.labels(query_type=query_type).observe(duration)

    def increment_active_connections(self):
        """Increment active connections counter"""
        ACTIVE_CONNECTIONS.inc()

    def decrement_active_connections(self):
        """Decrement active connections counter"""
        ACTIVE_CONNECTIONS.dec()


# Global metrics collector instance
metrics_collector = MetricsCollector()


def track_request_metrics(endpoint: str = None):
    """Decorator to track request metrics"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status_code = 200
            method = "GET"  # Default, should be extracted from request

            try:
                # Try to extract method from request if available
                if args and hasattr(args[0], "method"):
                    method = args[0].method

                result = await func(*args, **kwargs)

                # Try to extract status code from response
                if hasattr(result, "status_code"):
                    status_code = result.status_code

                return result

            except Exception:
                status_code = 500
                raise

            finally:
                duration = time.time() - start_time
                endpoint_name = endpoint or func.__name__
                metrics_collector.record_request(
                    method, endpoint_name, status_code, duration
                )

        return wrapper

    return decorator


def track_tool_metrics(tool_name: str):
    """Decorator to track tool execution metrics"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            try:
                result = await func(*args, **kwargs)

                # Check if result indicates failure
                if hasattr(result, "success") and not result.success:
                    status = "failure"

                return result

            except Exception:
                status = "error"
                raise

            finally:
                duration = time.time() - start_time
                metrics_collector.record_tool_execution(tool_name, status, duration)

        return wrapper

    return decorator


def track_processing_metrics(format: str):
    """Decorator to track document processing metrics"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            try:
                result = await func(*args, **kwargs)
                return result

            except Exception:
                status = "error"
                raise

            finally:
                duration = time.time() - start_time
                metrics_collector.record_document_processing(format, status, duration)

        return wrapper

    return decorator


async def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance"""
    if not metrics_collector._server_started:
        await metrics_collector.start_metrics_server()
    return metrics_collector


# Export commonly used functions and classes
__all__ = [
    "MetricsCollector",
    "metrics_collector",
    "track_request_metrics",
    "track_tool_metrics",
    "track_processing_metrics",
    "get_metrics_collector",
]
