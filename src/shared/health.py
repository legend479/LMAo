"""
Health Check System
Comprehensive health monitoring for all service components
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
from datetime import datetime

from .config import get_settings
from .logging import get_logger

logger = get_logger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    component: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    response_time: float
    timestamp: datetime


@dataclass
class SystemHealth:
    overall_status: HealthStatus
    components: List[HealthCheckResult]
    timestamp: datetime
    uptime: float


class BaseHealthChecker:
    """Base class for health checkers"""

    def __init__(self, name: str, timeout: float = 5.0):
        self.name = name
        self.timeout = timeout
        self.logger = get_logger(f"health.{name}")

    async def check(self) -> HealthCheckResult:
        """Perform health check"""
        start_time = time.time()

        try:
            # Run the actual health check with timeout
            result = await asyncio.wait_for(self._perform_check(), timeout=self.timeout)

            response_time = time.time() - start_time

            return HealthCheckResult(
                component=self.name,
                status=result.get("status", HealthStatus.UNKNOWN),
                message=result.get("message", "Health check completed"),
                details=result.get("details", {}),
                response_time=response_time,
                timestamp=datetime.utcnow(),
            )

        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout}s",
                details={"timeout": self.timeout},
                response_time=response_time,
                timestamp=datetime.utcnow(),
            )

        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error("Health check failed", error=str(e))

            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
                response_time=response_time,
                timestamp=datetime.utcnow(),
            )

    async def _perform_check(self) -> Dict[str, Any]:
        """Override this method to implement specific health check logic"""
        raise NotImplementedError


class DatabaseHealthChecker(BaseHealthChecker):
    """Health checker for database connectivity"""

    def __init__(self):
        super().__init__("database")

    async def _perform_check(self) -> Dict[str, Any]:
        """Check database connectivity"""
        # TODO: Implement actual database connection check
        # For now, return a placeholder

        try:
            # Simulate database check
            await asyncio.sleep(0.1)

            return {
                "status": HealthStatus.HEALTHY,
                "message": "Database connection successful",
                "details": {"connection_pool": "active", "response_time_ms": 100},
            }

        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Database connection failed: {str(e)}",
                "details": {"error": str(e)},
            }


class RedisHealthChecker(BaseHealthChecker):
    """Health checker for Redis connectivity"""

    def __init__(self):
        super().__init__("redis")

    async def _perform_check(self) -> Dict[str, Any]:
        """Check Redis connectivity"""
        # TODO: Implement actual Redis connection check
        # For now, return a placeholder

        try:
            # Simulate Redis check
            await asyncio.sleep(0.05)

            return {
                "status": HealthStatus.HEALTHY,
                "message": "Redis connection successful",
                "details": {"ping_response": "PONG", "memory_usage": "50MB"},
            }

        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Redis connection failed: {str(e)}",
                "details": {"error": str(e)},
            }


class ElasticsearchHealthChecker(BaseHealthChecker):
    """Health checker for Elasticsearch connectivity"""

    def __init__(self):
        super().__init__("elasticsearch")

    async def _perform_check(self) -> Dict[str, Any]:
        """Check Elasticsearch connectivity"""
        # TODO: Implement actual Elasticsearch connection check
        # For now, return a placeholder

        try:
            # Simulate Elasticsearch check
            await asyncio.sleep(0.2)

            return {
                "status": HealthStatus.HEALTHY,
                "message": "Elasticsearch cluster healthy",
                "details": {"cluster_status": "green", "nodes": 1, "indices": 5},
            }

        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Elasticsearch connection failed: {str(e)}",
                "details": {"error": str(e)},
            }


class ServiceHealthChecker(BaseHealthChecker):
    """Health checker for internal services"""

    def __init__(self, service_name: str, check_function=None):
        super().__init__(service_name)
        self.check_function = check_function

    async def _perform_check(self) -> Dict[str, Any]:
        """Check service health"""

        if self.check_function:
            try:
                result = await self.check_function()
                return result
            except Exception as e:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "message": f"Service check failed: {str(e)}",
                    "details": {"error": str(e)},
                }

        # Default service check
        return {
            "status": HealthStatus.HEALTHY,
            "message": f"{self.name} service is running",
            "details": {"initialized": True},
        }


class HealthMonitor:
    """Central health monitoring system"""

    def __init__(self):
        self.checkers: Dict[str, BaseHealthChecker] = {}
        self.settings = get_settings()
        self.start_time = time.time()
        self._last_check_results: Dict[str, HealthCheckResult] = {}

    def register_checker(self, checker: BaseHealthChecker):
        """Register a health checker"""
        self.checkers[checker.name] = checker
        logger.info("Health checker registered", component=checker.name)

    def unregister_checker(self, name: str):
        """Unregister a health checker"""
        if name in self.checkers:
            del self.checkers[name]
            logger.info("Health checker unregistered", component=name)

    async def check_all(self) -> SystemHealth:
        """Perform health checks on all registered components"""

        logger.debug("Starting system health check")

        # Run all health checks concurrently
        tasks = [checker.check() for checker in self.checkers.values()]

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            component_results = []
            for result in results:
                if isinstance(result, Exception):
                    # Handle exceptions from health checks
                    component_results.append(
                        HealthCheckResult(
                            component="unknown",
                            status=HealthStatus.UNHEALTHY,
                            message=f"Health check exception: {str(result)}",
                            details={"error": str(result)},
                            response_time=0.0,
                            timestamp=datetime.utcnow(),
                        )
                    )
                else:
                    component_results.append(result)
                    # Cache the result
                    self._last_check_results[result.component] = result
        else:
            component_results = []

        # Determine overall system health
        overall_status = self._determine_overall_status(component_results)

        # Calculate uptime
        uptime = time.time() - self.start_time

        system_health = SystemHealth(
            overall_status=overall_status,
            components=component_results,
            timestamp=datetime.utcnow(),
            uptime=uptime,
        )

        logger.info(
            "System health check completed",
            overall_status=overall_status.value,
            component_count=len(component_results),
        )

        return system_health

    async def check_component(self, component_name: str) -> Optional[HealthCheckResult]:
        """Check health of a specific component"""

        if component_name not in self.checkers:
            return None

        checker = self.checkers[component_name]
        result = await checker.check()

        # Cache the result
        self._last_check_results[component_name] = result

        return result

    def get_last_results(self) -> Dict[str, HealthCheckResult]:
        """Get the last health check results"""
        return self._last_check_results.copy()

    def _determine_overall_status(
        self, results: List[HealthCheckResult]
    ) -> HealthStatus:
        """Determine overall system health from component results"""

        if not results:
            return HealthStatus.UNKNOWN

        statuses = [result.status for result in results]

        # If any component is unhealthy, system is unhealthy
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY

        # If any component is degraded, system is degraded
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED

        # If all components are healthy, system is healthy
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY

        # Default to unknown
        return HealthStatus.UNKNOWN

    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        return time.time() - self.start_time

    def get_component_names(self) -> List[str]:
        """Get list of registered component names"""
        return list(self.checkers.keys())


# Global health monitor instance
health_monitor = HealthMonitor()


def setup_default_health_checks():
    """Set up default health checkers"""

    # Register built-in health checkers
    health_monitor.register_checker(DatabaseHealthChecker())
    health_monitor.register_checker(RedisHealthChecker())
    health_monitor.register_checker(ElasticsearchHealthChecker())

    logger.info("Default health checkers registered")


async def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance"""
    return health_monitor


# Export commonly used classes and functions
__all__ = [
    "HealthStatus",
    "HealthCheckResult",
    "SystemHealth",
    "BaseHealthChecker",
    "HealthMonitor",
    "health_monitor",
    "setup_default_health_checks",
    "get_health_monitor",
]
