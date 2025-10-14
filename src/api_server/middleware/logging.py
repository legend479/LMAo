"""
Logging Middleware
Structured logging for API requests and responses
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time
import structlog

logger = structlog.get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response logging"""

    async def dispatch(self, request: Request, call_next):
        """Log request and response details"""
        start_time = time.time()

        # Log request
        request_data = {
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }

        logger.info("Request started", **request_data)

        # Process request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Log response
        response_data = {
            **request_data,
            "status_code": response.status_code,
            "process_time": round(process_time, 4),
        }

        if response.status_code >= 400:
            logger.error("Request failed", **response_data)
        else:
            logger.info("Request completed", **response_data)

        # Add process time header
        response.headers["X-Process-Time"] = str(process_time)

        return response
