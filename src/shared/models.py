"""
Shared Data Models
Common data structures used across services
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum


class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class HealthCheck(BaseModel):
    """Health check response model"""

    status: ServiceStatus
    timestamp: datetime
    version: str
    services: Dict[str, Any] = {}
    uptime_seconds: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None


class APIResponse(BaseModel):
    """Standard API response wrapper"""

    success: bool
    data: Any = None
    message: str = ""
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class PaginationParams(BaseModel):
    """Pagination parameters"""

    page: int = Field(default=1, ge=1)
    size: int = Field(default=20, ge=1, le=100)

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size


class PaginatedResponse(BaseModel):
    """Paginated response wrapper"""

    items: List[Any]
    total: int
    page: int
    size: int
    pages: int

    @classmethod
    def create(cls, items: List[Any], total: int, pagination: PaginationParams):
        pages = (total + pagination.size - 1) // pagination.size
        return cls(
            items=items,
            total=total,
            page=pagination.page,
            size=pagination.size,
            pages=pages,
        )


class ServiceInfo(BaseModel):
    """Service information model"""

    name: str
    version: str
    status: ServiceStatus
    host: str
    port: int
    health_endpoint: str
    last_check: Optional[datetime] = None
    response_time_ms: Optional[float] = None


class MetricPoint(BaseModel):
    """Single metric data point"""

    name: str
    value: Union[int, float]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = {}
    unit: Optional[str] = None


class LogEntry(BaseModel):
    """Structured log entry"""

    level: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    logger_name: str
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    extra_data: Dict[str, Any] = {}
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ConfigurationItem(BaseModel):
    """Configuration item model"""

    key: str
    value: Any
    description: Optional[str] = None
    is_secret: bool = False
    source: str = "default"  # default, env, file, override
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class ValidationError(BaseModel):
    """Validation error details"""

    field: str
    message: str
    invalid_value: Any = None
    constraint: Optional[str] = None
