"""
Authentication and Authorization Module
Comprehensive JWT-based authentication with RBAC
"""

from .jwt_manager import JWTManager, TokenPayload, jwt_manager, get_jwt_manager
from .rbac import (
    RBACManager,
    Permission,
    Role,
    RoleDefinition,
    rbac_manager,
    get_rbac_manager,
    require_permission,
    require_any_permission,
    require_all_permissions,
    require_resource_access,
)

__all__ = [
    "JWTManager",
    "TokenPayload",
    "jwt_manager",
    "get_jwt_manager",
    "RBACManager",
    "Permission",
    "Role",
    "RoleDefinition",
    "rbac_manager",
    "get_rbac_manager",
    "require_permission",
    "require_any_permission",
    "require_all_permissions",
    "require_resource_access",
]
