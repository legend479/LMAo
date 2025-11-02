"""
Database package for SQLAlchemy models and connection management
"""

from .models import (
    Base,
    User,
    Session,
    Document,
    ToolExecution,
    WorkflowExecution,
    SystemMetric,
)
from .connection import (
    DatabaseManager,
    get_database_session,
    database_session_scope,
    initialize_database,
    database_health_check,
    close_database_connections,
)
from .operations import (
    UserOperations,
    SessionOperations,
    DocumentOperations,
    ToolExecutionOperations,
    MetricsOperations,
)

__all__ = [
    # Models
    "Base",
    "User",
    "Session",
    "Document",
    "ToolExecution",
    "WorkflowExecution",
    "SystemMetric",
    # Connection management
    "DatabaseManager",
    "get_database_session",
    "database_session_scope",
    "initialize_database",
    "database_health_check",
    "close_database_connections",
    # Operations
    "UserOperations",
    "SessionOperations",
    "DocumentOperations",
    "ToolExecutionOperations",
    "MetricsOperations",
]
