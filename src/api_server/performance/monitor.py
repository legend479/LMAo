"""
Performance Monitoring and Optimization System
Real-time performance tracking with optimization recommendations
"""

import time
import asyncio
import psutil
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque, defaultdict
import statistics

from src.shared.config import get_settings
from src.shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric"""

    name: str
    value: float
    timestamp: float
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System-level performance metrics"""

    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    timestamp: float


@dataclass
class RequestMetrics:
    """Request-level performance metrics"""

    endpoint: str
    method: str
    status_code: int
    duration: float
    timestamp: float
    user_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""

    category: str
    priority: str  # high, medium, low
    title: str
    description: str
    impact: str
    implementation: str
    metrics: Dict[str, float]


class MetricsCollector:
    """Collects and aggregates performance metrics"""

    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.request_metrics: deque = deque(maxlen=max_history)
        self.system_metrics: deque = deque(maxlen=max_history)
        self.custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Aggregated statistics
        self.endpoint_stats: Dict[str, Dict] = defaultdict(
            lambda: {
                "count": 0,
                "total_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
                "error_count": 0,
                "status_codes": defaultdict(int),
            }
        )

        # Real-time tracking
        self.active_requests = 0
        self.peak_concurrent_requests = 0
        self.start_time = time.time()

        # System monitoring
        self.system_monitor_running = False
        self.system_monitor_task = None

    def record_request(self, metrics: RequestMetrics):
        """Record request performance metrics"""

        self.request_metrics.append(metrics)

        # Update endpoint statistics
        endpoint_key = f"{metrics.method}:{metrics.endpoint}"
        stats = self.endpoint_stats[endpoint_key]

        stats["count"] += 1
        stats["total_time"] += metrics.duration
        stats["min_time"] = min(stats["min_time"], metrics.duration)
        stats["max_time"] = max(stats["max_time"], metrics.duration)
        stats["status_codes"][metrics.status_code] += 1

        if metrics.status_code >= 400:
            stats["error_count"] += 1

    def record_system_metrics(self, metrics: SystemMetrics):
        """Record system performance metrics"""
        self.system_metrics.append(metrics)

    def record_custom_metric(
        self, name: str, value: float, tags: Dict[str, str] = None
    ):
        """Record custom performance metric"""

        metric = PerformanceMetric(
            name=name, value=value, timestamp=time.time(), tags=tags or {}
        )

        self.custom_metrics[name].append(metric)

    def increment_active_requests(self):
        """Increment active request counter"""
        self.active_requests += 1
        self.peak_concurrent_requests = max(
            self.peak_concurrent_requests, self.active_requests
        )

    def decrement_active_requests(self):
        """Decrement active request counter"""
        self.active_requests = max(0, self.active_requests - 1)

    async def start_system_monitoring(self, interval: float = 30.0):
        """Start system metrics collection"""

        if self.system_monitor_running:
            return

        self.system_monitor_running = True
        self.system_monitor_task = asyncio.create_task(
            self._system_monitor_loop(interval)
        )

        logger.info("System monitoring started", interval=interval)

    async def stop_system_monitoring(self):
        """Stop system metrics collection"""

        self.system_monitor_running = False

        if self.system_monitor_task:
            self.system_monitor_task.cancel()
            try:
                await self.system_monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("System monitoring stopped")

    async def _system_monitor_loop(self, interval: float):
        """System monitoring loop"""

        while self.system_monitor_running:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage("/")
                network = psutil.net_io_counters()

                metrics = SystemMetrics(
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_mb=memory.used / (1024 * 1024),
                    memory_available_mb=memory.available / (1024 * 1024),
                    disk_usage_percent=disk.percent,
                    disk_free_gb=disk.free / (1024 * 1024 * 1024),
                    network_bytes_sent=network.bytes_sent,
                    network_bytes_recv=network.bytes_recv,
                    active_connections=self.active_requests,
                    timestamp=time.time(),
                )

                self.record_system_metrics(metrics)

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error("System monitoring error", error=str(e))
                await asyncio.sleep(interval)

    def get_request_statistics(self, time_window: Optional[int] = None) -> Dict:
        """Get request performance statistics"""

        if not self.request_metrics:
            return {"message": "No request data available"}

        # Filter by time window if specified
        cutoff_time = time.time() - time_window if time_window else 0
        filtered_metrics = [
            m for m in self.request_metrics if m.timestamp >= cutoff_time
        ]

        if not filtered_metrics:
            return {"message": "No data in specified time window"}

        # Calculate statistics
        durations = [m.duration for m in filtered_metrics]
        status_codes = defaultdict(int)
        error_count = 0

        for metric in filtered_metrics:
            status_codes[metric.status_code] += 1
            if metric.status_code >= 400:
                error_count += 1

        return {
            "total_requests": len(filtered_metrics),
            "avg_response_time": statistics.mean(durations),
            "median_response_time": statistics.median(durations),
            "min_response_time": min(durations),
            "max_response_time": max(durations),
            "p95_response_time": self._percentile(durations, 95),
            "p99_response_time": self._percentile(durations, 99),
            "error_rate": (error_count / len(filtered_metrics)) * 100,
            "status_code_distribution": dict(status_codes),
            "requests_per_second": len(filtered_metrics) / max(time_window or 3600, 1),
        }

    def get_system_statistics(self, time_window: Optional[int] = None) -> Dict:
        """Get system performance statistics"""

        if not self.system_metrics:
            return {"message": "No system data available"}

        # Filter by time window if specified
        cutoff_time = time.time() - time_window if time_window else 0
        filtered_metrics = [
            m for m in self.system_metrics if m.timestamp >= cutoff_time
        ]

        if not filtered_metrics:
            return {"message": "No system data in specified time window"}

        # Calculate averages
        cpu_values = [m.cpu_percent for m in filtered_metrics]
        memory_values = [m.memory_percent for m in filtered_metrics]

        return {
            "avg_cpu_percent": statistics.mean(cpu_values),
            "max_cpu_percent": max(cpu_values),
            "avg_memory_percent": statistics.mean(memory_values),
            "max_memory_percent": max(memory_values),
            "current_memory_mb": filtered_metrics[-1].memory_used_mb,
            "available_memory_mb": filtered_metrics[-1].memory_available_mb,
            "disk_usage_percent": filtered_metrics[-1].disk_usage_percent,
            "disk_free_gb": filtered_metrics[-1].disk_free_gb,
            "active_connections": self.active_requests,
            "peak_concurrent_requests": self.peak_concurrent_requests,
            "uptime_seconds": time.time() - self.start_time,
        }

    def get_endpoint_statistics(self) -> Dict:
        """Get per-endpoint performance statistics"""

        endpoint_stats = {}

        for endpoint, stats in self.endpoint_stats.items():
            if stats["count"] > 0:
                endpoint_stats[endpoint] = {
                    "request_count": stats["count"],
                    "avg_response_time": stats["total_time"] / stats["count"],
                    "min_response_time": stats["min_time"],
                    "max_response_time": stats["max_time"],
                    "error_count": stats["error_count"],
                    "error_rate": (stats["error_count"] / stats["count"]) * 100,
                    "status_codes": dict(stats["status_codes"]),
                }

        return endpoint_stats

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        index = min(index, len(sorted_data) - 1)

        return sorted_data[index]


class PerformanceOptimizer:
    """Analyzes performance metrics and provides optimization recommendations"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.settings = get_settings()

    def analyze_performance(self) -> List[OptimizationRecommendation]:
        """Analyze current performance and generate recommendations"""

        recommendations = []

        # Get current statistics
        request_stats = self.metrics_collector.get_request_statistics(
            time_window=3600
        )  # Last hour
        system_stats = self.metrics_collector.get_system_statistics(time_window=3600)
        endpoint_stats = self.metrics_collector.get_endpoint_statistics()

        # Analyze response times
        recommendations.extend(self._analyze_response_times(request_stats))

        # Analyze system resources
        recommendations.extend(self._analyze_system_resources(system_stats))

        # Analyze endpoint performance
        recommendations.extend(self._analyze_endpoint_performance(endpoint_stats))

        # Analyze error rates
        recommendations.extend(self._analyze_error_rates(request_stats, endpoint_stats))

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x.priority, 3))

        return recommendations

    def _analyze_response_times(self, stats: Dict) -> List[OptimizationRecommendation]:
        """Analyze response time performance"""

        recommendations = []

        if "avg_response_time" not in stats:
            return recommendations

        avg_time = stats["avg_response_time"]
        p95_time = stats.get("p95_response_time", 0)
        p99_time = stats.get("p99_response_time", 0)

        # High average response time
        if avg_time > 2.0:  # 2 seconds
            recommendations.append(
                OptimizationRecommendation(
                    category="response_time",
                    priority="high",
                    title="High Average Response Time",
                    description=f"Average response time is {avg_time:.2f}s, which exceeds the recommended 2s threshold.",
                    impact="Poor user experience and potential timeouts",
                    implementation="Consider implementing caching, database optimization, or async processing",
                    metrics={"avg_response_time": avg_time},
                )
            )

        # High P95 response time
        if p95_time > 5.0:  # 5 seconds
            recommendations.append(
                OptimizationRecommendation(
                    category="response_time",
                    priority="medium",
                    title="High P95 Response Time",
                    description=f"95th percentile response time is {p95_time:.2f}s, indicating performance issues for some requests.",
                    impact="Inconsistent user experience",
                    implementation="Identify and optimize slow endpoints, implement request queuing",
                    metrics={"p95_response_time": p95_time},
                )
            )

        return recommendations

    def _analyze_system_resources(
        self, stats: Dict
    ) -> List[OptimizationRecommendation]:
        """Analyze system resource usage"""

        recommendations = []

        if "avg_cpu_percent" not in stats:
            return recommendations

        avg_cpu = stats["avg_cpu_percent"]
        max_cpu = stats.get("max_cpu_percent", 0)
        avg_memory = stats.get("avg_memory_percent", 0)
        max_memory = stats.get("max_memory_percent", 0)

        # High CPU usage
        if avg_cpu > 70:
            recommendations.append(
                OptimizationRecommendation(
                    category="system_resources",
                    priority="high" if avg_cpu > 85 else "medium",
                    title="High CPU Usage",
                    description=f"Average CPU usage is {avg_cpu:.1f}%, which may impact performance.",
                    impact="Slower response times and potential service degradation",
                    implementation="Consider scaling horizontally, optimizing algorithms, or upgrading hardware",
                    metrics={"avg_cpu_percent": avg_cpu, "max_cpu_percent": max_cpu},
                )
            )

        # High memory usage
        if avg_memory > 80:
            recommendations.append(
                OptimizationRecommendation(
                    category="system_resources",
                    priority="high" if avg_memory > 90 else "medium",
                    title="High Memory Usage",
                    description=f"Average memory usage is {avg_memory:.1f}%, approaching system limits.",
                    impact="Risk of out-of-memory errors and system instability",
                    implementation="Implement memory optimization, increase cache limits, or add more RAM",
                    metrics={
                        "avg_memory_percent": avg_memory,
                        "max_memory_percent": max_memory,
                    },
                )
            )

        return recommendations

    def _analyze_endpoint_performance(
        self, endpoint_stats: Dict
    ) -> List[OptimizationRecommendation]:
        """Analyze per-endpoint performance"""

        recommendations = []

        # Find slowest endpoints
        slow_endpoints = []
        for endpoint, stats in endpoint_stats.items():
            if stats["avg_response_time"] > 1.0:  # 1 second threshold
                slow_endpoints.append((endpoint, stats))

        # Sort by response time
        slow_endpoints.sort(key=lambda x: x[1]["avg_response_time"], reverse=True)

        # Recommend optimization for top 3 slowest endpoints
        for endpoint, stats in slow_endpoints[:3]:
            recommendations.append(
                OptimizationRecommendation(
                    category="endpoint_performance",
                    priority="medium",
                    title=f"Slow Endpoint: {endpoint}",
                    description=f"Endpoint {endpoint} has average response time of {stats['avg_response_time']:.2f}s",
                    impact="Poor user experience for specific functionality",
                    implementation="Profile and optimize endpoint logic, add caching, or implement async processing",
                    metrics={
                        "avg_response_time": stats["avg_response_time"],
                        "request_count": stats["request_count"],
                    },
                )
            )

        return recommendations

    def _analyze_error_rates(
        self, request_stats: Dict, endpoint_stats: Dict
    ) -> List[OptimizationRecommendation]:
        """Analyze error rates"""

        recommendations = []

        # Overall error rate
        overall_error_rate = request_stats.get("error_rate", 0)
        if overall_error_rate > 5:  # 5% threshold
            recommendations.append(
                OptimizationRecommendation(
                    category="error_rate",
                    priority="high" if overall_error_rate > 10 else "medium",
                    title="High Overall Error Rate",
                    description=f"Overall error rate is {overall_error_rate:.1f}%, indicating system issues.",
                    impact="Poor user experience and potential data loss",
                    implementation="Investigate error logs, improve error handling, and fix underlying issues",
                    metrics={"error_rate": overall_error_rate},
                )
            )

        # High error rate endpoints
        for endpoint, stats in endpoint_stats.items():
            if (
                stats["error_rate"] > 10 and stats["request_count"] > 10
            ):  # 10% threshold, min 10 requests
                recommendations.append(
                    OptimizationRecommendation(
                        category="error_rate",
                        priority="medium",
                        title=f"High Error Rate: {endpoint}",
                        description=f"Endpoint {endpoint} has error rate of {stats['error_rate']:.1f}%",
                        impact="Functionality issues for specific features",
                        implementation="Debug endpoint logic, improve input validation, and enhance error handling",
                        metrics={
                            "error_rate": stats["error_rate"],
                            "error_count": stats["error_count"],
                            "request_count": stats["request_count"],
                        },
                    )
                )

        return recommendations


class PerformanceMonitor:
    """Main performance monitoring system"""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.optimizer = PerformanceOptimizer(self.metrics_collector)
        self.monitoring_active = False

    async def start_monitoring(self):
        """Start performance monitoring"""

        if self.monitoring_active:
            return

        self.monitoring_active = True
        await self.metrics_collector.start_system_monitoring()

        logger.info("Performance monitoring started")

    async def stop_monitoring(self):
        """Stop performance monitoring"""

        self.monitoring_active = False
        await self.metrics_collector.stop_system_monitoring()

        logger.info("Performance monitoring stopped")

    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float,
        user_id: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """Record request performance metrics"""

        metrics = RequestMetrics(
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            duration=duration,
            timestamp=time.time(),
            user_id=user_id,
            error=error,
        )

        self.metrics_collector.record_request(metrics)

    def get_performance_report(self, time_window: Optional[int] = None) -> Dict:
        """Get comprehensive performance report"""

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "time_window_seconds": time_window,
            "request_statistics": self.metrics_collector.get_request_statistics(
                time_window
            ),
            "system_statistics": self.metrics_collector.get_system_statistics(
                time_window
            ),
            "endpoint_statistics": self.metrics_collector.get_endpoint_statistics(),
            "optimization_recommendations": [
                {
                    "category": rec.category,
                    "priority": rec.priority,
                    "title": rec.title,
                    "description": rec.description,
                    "impact": rec.impact,
                    "implementation": rec.implementation,
                    "metrics": rec.metrics,
                }
                for rec in self.optimizer.analyze_performance()
            ],
        }


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


async def get_performance_monitor() -> PerformanceMonitor:
    """Get performance monitor instance"""
    return performance_monitor
