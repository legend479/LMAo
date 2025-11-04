# API Server Module
# Main FastAPI server for handling external requests

from .main import APIServer
from .websocket_auth import WebSocketTokenAuth, WebSocketAuthError
from . import routers, middleware, auth, cache, performance

# Import key classes from submodules
from .auth import (
    JWTManager,
    TokenPayload,
    RBACManager,
    Permission,
    Role,
    RoleDefinition,
    get_jwt_manager,
    get_rbac_manager,
    require_permission,
)
from .cache import CacheManager
from .performance import PerformanceMonitor

__all__ = [
    # Main server
    "APIServer",
    "WebSocketTokenAuth",
    "WebSocketAuthError",
    # Submodules
    "routers",
    "middleware",
    "auth",
    "cache",
    "performance",
    # Auth components
    "JWTManager",
    "TokenPayload",
    "RBACManager",
    "Permission",
    "Role",
    "RoleDefinition",
    "get_jwt_manager",
    "get_rbac_manager",
    "require_permission",
    # Other components
    "CacheManager",
    "PerformanceMonitor",
]
