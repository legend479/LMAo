"""
Role-Based Access Control (RBAC) System
Comprehensive permission management with hierarchical roles and granular permissions
"""

from typing import Dict, List, Set, Optional
from enum import Enum
from dataclasses import dataclass

from ...shared.logging import get_logger

logger = get_logger(__name__)


class Permission(Enum):
    """System permissions"""

    # Chat permissions
    CHAT_READ = "chat:read"
    CHAT_WRITE = "chat:write"
    CHAT_HISTORY = "chat:history"
    CHAT_DELETE = "chat:delete"

    # Document permissions
    DOCUMENT_READ = "document:read"
    DOCUMENT_WRITE = "document:write"
    DOCUMENT_DELETE = "document:delete"
    DOCUMENT_GENERATE = "document:generate"
    DOCUMENT_UPLOAD = "document:upload"

    # Tool permissions
    TOOL_LIST = "tool:list"
    TOOL_EXECUTE = "tool:execute"
    TOOL_MANAGE = "tool:manage"

    # User management permissions
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_DELETE = "user:delete"
    USER_MANAGE_ROLES = "user:manage_roles"

    # System administration permissions
    SYSTEM_READ = "system:read"
    SYSTEM_WRITE = "system:write"
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_METRICS = "system:metrics"
    SYSTEM_LOGS = "system:logs"
    SYSTEM_CONFIG = "system:config"

    # API permissions
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_ADMIN = "api:admin"

    # Special permissions
    ALL_PERMISSIONS = "*"


class Role(Enum):
    """System roles with hierarchical structure"""

    # Basic user role
    USER = "user"

    # Power user with additional capabilities
    POWER_USER = "power_user"

    # Moderator with content management capabilities
    MODERATOR = "moderator"

    # Administrator with system management capabilities
    ADMIN = "admin"

    # Super administrator with all permissions
    SUPER_ADMIN = "super_admin"

    # Service account for system-to-system communication
    SERVICE = "service"

    # Read-only access for monitoring/auditing
    READONLY = "readonly"


@dataclass
class RoleDefinition:
    """Role definition with permissions and metadata"""

    name: str
    display_name: str
    description: str
    permissions: Set[Permission]
    inherits_from: Optional[List[Role]] = None
    is_system_role: bool = False
    max_sessions: int = 10


class RBACManager:
    """Role-Based Access Control Manager"""

    def __init__(self):
        self.role_definitions = self._initialize_role_definitions()
        self.permission_cache: Dict[str, Set[Permission]] = {}

    def _initialize_role_definitions(self) -> Dict[Role, RoleDefinition]:
        """Initialize default role definitions"""

        return {
            Role.USER: RoleDefinition(
                name="user",
                display_name="User",
                description="Basic user with chat and document access",
                permissions={
                    Permission.CHAT_READ,
                    Permission.CHAT_WRITE,
                    Permission.CHAT_HISTORY,
                    Permission.DOCUMENT_READ,
                    Permission.DOCUMENT_GENERATE,
                    Permission.TOOL_LIST,
                    Permission.TOOL_EXECUTE,
                    Permission.API_READ,
                },
                max_sessions=5,
            ),
            Role.POWER_USER: RoleDefinition(
                name="power_user",
                display_name="Power User",
                description="Advanced user with additional document and tool capabilities",
                permissions={
                    Permission.DOCUMENT_WRITE,
                    Permission.DOCUMENT_UPLOAD,
                    Permission.CHAT_DELETE,
                    Permission.API_WRITE,
                },
                inherits_from=[Role.USER],
                max_sessions=10,
            ),
            Role.MODERATOR: RoleDefinition(
                name="moderator",
                display_name="Moderator",
                description="Content moderator with user management capabilities",
                permissions={
                    Permission.USER_READ,
                    Permission.DOCUMENT_DELETE,
                    Permission.SYSTEM_READ,
                },
                inherits_from=[Role.POWER_USER],
                max_sessions=15,
            ),
            Role.ADMIN: RoleDefinition(
                name="admin",
                display_name="Administrator",
                description="System administrator with management capabilities",
                permissions={
                    Permission.USER_WRITE,
                    Permission.USER_DELETE,
                    Permission.USER_MANAGE_ROLES,
                    Permission.TOOL_MANAGE,
                    Permission.SYSTEM_WRITE,
                    Permission.SYSTEM_METRICS,
                    Permission.SYSTEM_LOGS,
                    Permission.SYSTEM_CONFIG,
                    Permission.API_ADMIN,
                },
                inherits_from=[Role.MODERATOR],
                is_system_role=True,
                max_sessions=20,
            ),
            Role.SUPER_ADMIN: RoleDefinition(
                name="super_admin",
                display_name="Super Administrator",
                description="Super administrator with all system permissions",
                permissions={Permission.ALL_PERMISSIONS},
                is_system_role=True,
                max_sessions=50,
            ),
            Role.SERVICE: RoleDefinition(
                name="service",
                display_name="Service Account",
                description="Service account for system-to-system communication",
                permissions={
                    Permission.API_READ,
                    Permission.API_WRITE,
                    Permission.TOOL_EXECUTE,
                    Permission.SYSTEM_READ,
                },
                is_system_role=True,
                max_sessions=100,
            ),
            Role.READONLY: RoleDefinition(
                name="readonly",
                display_name="Read Only",
                description="Read-only access for monitoring and auditing",
                permissions={
                    Permission.CHAT_READ,
                    Permission.DOCUMENT_READ,
                    Permission.TOOL_LIST,
                    Permission.USER_READ,
                    Permission.SYSTEM_READ,
                    Permission.API_READ,
                },
                max_sessions=5,
            ),
        }

    def get_user_permissions(self, roles: List[str]) -> Set[Permission]:
        """Get all permissions for a user based on their roles"""

        # Create cache key
        cache_key = "|".join(sorted(roles))

        # Check cache
        if cache_key in self.permission_cache:
            return self.permission_cache[cache_key]

        all_permissions = set()

        for role_name in roles:
            try:
                role = Role(role_name)
                permissions = self._get_role_permissions(role)
                all_permissions.update(permissions)
            except ValueError:
                logger.warning("Unknown role", role=role_name)
                continue

        # Cache result
        self.permission_cache[cache_key] = all_permissions

        return all_permissions

    def _get_role_permissions(self, role: Role) -> Set[Permission]:
        """Get all permissions for a role (including inherited)"""

        if role not in self.role_definitions:
            logger.warning("Role definition not found", role=role.value)
            return set()

        role_def = self.role_definitions[role]
        permissions = set(role_def.permissions)

        # Handle special ALL_PERMISSIONS
        if Permission.ALL_PERMISSIONS in permissions:
            return set(Permission)

        # Add inherited permissions
        if role_def.inherits_from:
            for parent_role in role_def.inherits_from:
                parent_permissions = self._get_role_permissions(parent_role)
                permissions.update(parent_permissions)

        return permissions

    def has_permission(
        self, user_roles: List[str], required_permission: Permission
    ) -> bool:
        """Check if user has a specific permission"""

        user_permissions = self.get_user_permissions(user_roles)

        # Check for ALL_PERMISSIONS wildcard
        if Permission.ALL_PERMISSIONS in user_permissions:
            return True

        return required_permission in user_permissions

    def has_any_permission(
        self, user_roles: List[str], required_permissions: List[Permission]
    ) -> bool:
        """Check if user has any of the required permissions"""

        user_permissions = self.get_user_permissions(user_roles)

        # Check for ALL_PERMISSIONS wildcard
        if Permission.ALL_PERMISSIONS in user_permissions:
            return True

        return any(perm in user_permissions for perm in required_permissions)

    def has_all_permissions(
        self, user_roles: List[str], required_permissions: List[Permission]
    ) -> bool:
        """Check if user has all required permissions"""

        user_permissions = self.get_user_permissions(user_roles)

        # Check for ALL_PERMISSIONS wildcard
        if Permission.ALL_PERMISSIONS in user_permissions:
            return True

        return all(perm in user_permissions for perm in required_permissions)

    def can_access_resource(
        self, user_roles: List[str], resource: str, action: str
    ) -> bool:
        """Check if user can perform action on resource"""

        # Map resource:action to permission
        permission_map = {
            "chat:read": Permission.CHAT_READ,
            "chat:write": Permission.CHAT_WRITE,
            "chat:delete": Permission.CHAT_DELETE,
            "document:read": Permission.DOCUMENT_READ,
            "document:write": Permission.DOCUMENT_WRITE,
            "document:delete": Permission.DOCUMENT_DELETE,
            "document:generate": Permission.DOCUMENT_GENERATE,
            "tool:list": Permission.TOOL_LIST,
            "tool:execute": Permission.TOOL_EXECUTE,
            "tool:manage": Permission.TOOL_MANAGE,
            "user:read": Permission.USER_READ,
            "user:write": Permission.USER_WRITE,
            "user:delete": Permission.USER_DELETE,
            "system:read": Permission.SYSTEM_READ,
            "system:write": Permission.SYSTEM_WRITE,
            "system:admin": Permission.SYSTEM_ADMIN,
            "api:read": Permission.API_READ,
            "api:write": Permission.API_WRITE,
            "api:admin": Permission.API_ADMIN,
        }

        resource_action = f"{resource}:{action}"
        required_permission = permission_map.get(resource_action)

        if not required_permission:
            logger.warning("Unknown resource:action", resource=resource, action=action)
            return False

        return self.has_permission(user_roles, required_permission)

    def get_role_definition(self, role: Role) -> Optional[RoleDefinition]:
        """Get role definition"""
        return self.role_definitions.get(role)

    def get_available_roles(
        self, include_system_roles: bool = False
    ) -> List[RoleDefinition]:
        """Get list of available roles"""

        roles = []
        for role_def in self.role_definitions.values():
            if include_system_roles or not role_def.is_system_role:
                roles.append(role_def)

        return roles

    def validate_roles(self, roles: List[str]) -> List[str]:
        """Validate and filter valid roles"""

        valid_roles = []
        for role_name in roles:
            try:
                role = Role(role_name)
                if role in self.role_definitions:
                    valid_roles.append(role_name)
                else:
                    logger.warning("Role not in definitions", role=role_name)
            except ValueError:
                logger.warning("Invalid role", role=role_name)

        return valid_roles

    def get_max_sessions(self, roles: List[str]) -> int:
        """Get maximum allowed sessions for user roles"""

        max_sessions = 0
        for role_name in roles:
            try:
                role = Role(role_name)
                role_def = self.role_definitions.get(role)
                if role_def:
                    max_sessions = max(max_sessions, role_def.max_sessions)
            except ValueError:
                continue

        return max_sessions or 5  # Default to 5 if no valid roles

    def clear_permission_cache(self):
        """Clear permission cache"""
        self.permission_cache.clear()
        logger.info("Permission cache cleared")


# Global RBAC manager instance
rbac_manager = RBACManager()


def get_rbac_manager() -> RBACManager:
    """Get RBAC manager instance"""
    return rbac_manager


# Convenience functions for common permission checks
def require_permission(permission: Permission):
    """Decorator factory for requiring specific permission"""

    def decorator(func):
        func._required_permission = permission
        return func

    return decorator


def require_any_permission(*permissions: Permission):
    """Decorator factory for requiring any of the specified permissions"""

    def decorator(func):
        func._required_any_permissions = list(permissions)
        return func

    return decorator


def require_all_permissions(*permissions: Permission):
    """Decorator factory for requiring all specified permissions"""

    def decorator(func):
        func._required_all_permissions = list(permissions)
        return func

    return decorator


def require_resource_access(resource: str, action: str):
    """Decorator factory for requiring resource access"""

    def decorator(func):
        func._required_resource = resource
        func._required_action = action
        return func

    return decorator
