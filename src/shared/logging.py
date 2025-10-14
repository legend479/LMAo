"""
Structured Logging Configuration
Centralized logging setup with different levels for each service component
"""

import structlog
import logging
import sys
from typing import Any, Dict
from datetime import datetime

from .config import get_settings


def configure_logging():
    """Configure structured logging for the application"""

    settings = get_settings()

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )

    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Add appropriate renderer based on format
    if settings.log_format.lower() == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a configured logger instance"""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""

    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger for this class"""
        return get_logger(self.__class__.__name__)


class RequestLogger:
    """Logger for HTTP requests with structured data"""

    def __init__(self):
        self.logger = get_logger("request")

    def log_request(
        self,
        method: str,
        url: str,
        headers: Dict[str, Any] = None,
        body: Any = None,
        user_id: str = None,
    ):
        """Log incoming request"""

        log_data = {
            "event": "request_received",
            "method": method,
            "url": url,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if user_id:
            log_data["user_id"] = user_id

        if headers:
            # Log selected headers (avoid sensitive data)
            safe_headers = {
                k: v
                for k, v in headers.items()
                if k.lower() not in ["authorization", "cookie", "x-api-key"]
            }
            log_data["headers"] = safe_headers

        self.logger.info("Request received", **log_data)

    def log_response(
        self,
        status_code: int,
        processing_time: float,
        response_size: int = None,
        error: str = None,
    ):
        """Log response details"""

        log_data = {
            "event": "request_completed",
            "status_code": status_code,
            "processing_time": processing_time,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if response_size:
            log_data["response_size"] = response_size

        if error:
            log_data["error"] = error

        if status_code >= 400:
            self.logger.error("Request failed", **log_data)
        else:
            self.logger.info("Request completed", **log_data)


class PerformanceLogger:
    """Logger for performance metrics and monitoring"""

    def __init__(self):
        self.logger = get_logger("performance")

    def log_execution_time(
        self, operation: str, execution_time: float, metadata: Dict[str, Any] = None
    ):
        """Log operation execution time"""

        log_data = {
            "event": "operation_completed",
            "operation": operation,
            "execution_time": execution_time,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if metadata:
            log_data.update(metadata)

        # Log as warning if operation is slow
        if execution_time > 5.0:
            self.logger.warning("Slow operation detected", **log_data)
        else:
            self.logger.info("Operation completed", **log_data)

    def log_resource_usage(
        self,
        cpu_percent: float = None,
        memory_mb: float = None,
        disk_usage: Dict[str, Any] = None,
    ):
        """Log system resource usage"""

        log_data = {
            "event": "resource_usage",
            "timestamp": datetime.utcnow().isoformat(),
        }

        if cpu_percent is not None:
            log_data["cpu_percent"] = cpu_percent

        if memory_mb is not None:
            log_data["memory_mb"] = memory_mb

        if disk_usage:
            log_data["disk_usage"] = disk_usage

        self.logger.info("Resource usage", **log_data)


class SecurityLogger:
    """Logger for security-related events"""

    def __init__(self):
        self.logger = get_logger("security")

    def log_authentication_attempt(
        self,
        user_id: str = None,
        success: bool = True,
        ip_address: str = None,
        user_agent: str = None,
    ):
        """Log authentication attempts"""

        log_data = {
            "event": "authentication_attempt",
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if user_id:
            log_data["user_id"] = user_id

        if ip_address:
            log_data["ip_address"] = ip_address

        if user_agent:
            log_data["user_agent"] = user_agent

        if success:
            self.logger.info("Authentication successful", **log_data)
        else:
            self.logger.warning("Authentication failed", **log_data)

    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        metadata: Dict[str, Any] = None,
    ):
        """Log security events"""

        log_data = {
            "event": "security_event",
            "event_type": event_type,
            "severity": severity,
            "description": description,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if metadata:
            log_data.update(metadata)

        if severity.lower() in ["high", "critical"]:
            self.logger.error("Security event", **log_data)
        elif severity.lower() == "medium":
            self.logger.warning("Security event", **log_data)
        else:
            self.logger.info("Security event", **log_data)


class AuditLogger:
    """Logger for audit trail and compliance"""

    def __init__(self):
        self.logger = get_logger("audit")

    def log_user_action(
        self,
        user_id: str,
        action: str,
        resource: str = None,
        result: str = "success",
        metadata: Dict[str, Any] = None,
    ):
        """Log user actions for audit trail"""

        log_data = {
            "event": "user_action",
            "user_id": user_id,
            "action": action,
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if resource:
            log_data["resource"] = resource

        if metadata:
            log_data.update(metadata)

        self.logger.info("User action", **log_data)

    def log_data_access(
        self, user_id: str, data_type: str, operation: str, record_count: int = None
    ):
        """Log data access for compliance"""

        log_data = {
            "event": "data_access",
            "user_id": user_id,
            "data_type": data_type,
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if record_count is not None:
            log_data["record_count"] = record_count

        self.logger.info("Data access", **log_data)


# Global logger instances
request_logger = RequestLogger()
performance_logger = PerformanceLogger()
security_logger = SecurityLogger()
audit_logger = AuditLogger()


# Initialize logging on module import
configure_logging()


# Export commonly used loggers and functions
__all__ = [
    "configure_logging",
    "get_logger",
    "LoggerMixin",
    "RequestLogger",
    "PerformanceLogger",
    "SecurityLogger",
    "AuditLogger",
    "request_logger",
    "performance_logger",
    "security_logger",
    "audit_logger",
]
