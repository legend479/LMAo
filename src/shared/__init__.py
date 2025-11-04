# Shared Utilities Module
# Common utilities, models, and configurations used across services

# Core services and clients
from .services import (
    ServiceClient,
    AgentServiceClient,
    RAGServiceClient,
    ServiceRegistry,
    get_service_registry,
    get_agent_client,
    get_rag_client,
)
from .session_manager import SessionManager, get_session_manager

# Database components
from .database import (
    initialize_database,
    get_database_session,
    database_session_scope,
    DatabaseManager,
    database_health_check,
    close_database_connections,
)

# Configuration and settings
from .config import (
    Settings,
    DevelopmentSettings,
    ProductionSettings,
    TestingSettings,
    get_settings,
    validate_config,
    get_database_config,
    get_redis_config,
    get_elasticsearch_config,
    get_cors_config,
)

# Common models
from .models import (
    ServiceStatus,
    HealthCheck,
    APIResponse,
    ErrorResponse,
    PaginationParams,
    PaginatedResponse,
    ServiceInfo,
    MetricPoint,
    LogEntry,
    ConfigurationItem,
    ValidationError as ModelValidationError,
)

# Logging utilities
from .logging import (
    get_logger,
    configure_logging,
    LoggerMixin,
    RequestLogger,
    PerformanceLogger,
    SecurityLogger,
    AuditLogger,
)

# Health monitoring
from .health import (
    HealthStatus,
    HealthCheckResult,
    SystemHealth,
    BaseHealthChecker,
    DatabaseHealthChecker,
    RedisHealthChecker,
    ElasticsearchHealthChecker,
    ServiceHealthChecker,
    HealthMonitor,
    setup_default_health_checks,
    get_health_monitor,
)

# Metrics and monitoring
from .metrics import MetricsCollector, get_metrics_collector

# Validation utilities
from .validation import (
    ValidationResult,
    ConfigValidator,
    validate_configuration,
    print_validation_results,
)

# Startup utilities
from .startup import (
    StartupManager,
    APIServerStartup,
    AgentServerStartup,
    RAGPipelineStartup,
    create_startup_manager,
    initialize_application,
    run_with_startup,
)

# LLM integration
from . import llm

__all__ = [
    # Services and clients
    "ServiceClient",
    "AgentServiceClient",
    "RAGServiceClient",
    "ServiceRegistry",
    "get_service_registry",
    "get_agent_client",
    "get_rag_client",
    # Session management
    "SessionManager",
    "get_session_manager",
    # Database
    "initialize_database",
    "get_database_session",
    "database_session_scope",
    "DatabaseManager",
    "database_health_check",
    "close_database_connections",
    # Configuration
    "Settings",
    "DevelopmentSettings",
    "ProductionSettings",
    "TestingSettings",
    "get_settings",
    "validate_config",
    "get_database_config",
    "get_redis_config",
    "get_elasticsearch_config",
    "get_cors_config",
    # Models
    "ServiceStatus",
    "HealthCheck",
    "APIResponse",
    "ErrorResponse",
    "PaginationParams",
    "PaginatedResponse",
    "ServiceInfo",
    "MetricPoint",
    "LogEntry",
    "ConfigurationItem",
    "ModelValidationError",
    # Logging
    "get_logger",
    "configure_logging",
    "LoggerMixin",
    "RequestLogger",
    "PerformanceLogger",
    "SecurityLogger",
    "AuditLogger",
    # Health monitoring
    "HealthStatus",
    "HealthCheckResult",
    "SystemHealth",
    "BaseHealthChecker",
    "DatabaseHealthChecker",
    "RedisHealthChecker",
    "ElasticsearchHealthChecker",
    "ServiceHealthChecker",
    "HealthMonitor",
    "setup_default_health_checks",
    "get_health_monitor",
    # Metrics
    "MetricsCollector",
    "get_metrics_collector",
    # Validation
    "ValidationResult",
    "ConfigValidator",
    "validate_configuration",
    "print_validation_results",
    # Startup/shutdown
    "StartupManager",
    "APIServerStartup",
    "AgentServerStartup",
    "RAGPipelineStartup",
    "create_startup_manager",
    "initialize_application",
    "run_with_startup",
    # LLM submodule
    "llm",
]
