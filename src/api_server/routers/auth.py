"""
Authentication Router
JWT-based authentication and authorization endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext

from ...shared.config import get_settings
from ...shared.logging import get_logger
from ..auth import jwt_manager, rbac_manager, Permission

logger = get_logger(__name__)
router = APIRouter()

# Security setup
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    role: str = "user"


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    email: Optional[str] = None
    user_id: Optional[str] = None
    roles: List[str] = []


class User(BaseModel):
    id: str
    email: EmailStr
    full_name: Optional[str] = None
    roles: List[str] = []
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None


class RefreshTokenRequest(BaseModel):
    refresh_token: str


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    settings = get_settings()
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.access_token_expire_minutes
        )

    to_encode.update({"exp": expire, "type": "access"})

    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm="HS256")
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token"""
    settings = get_settings()
    to_encode = data.copy()

    # Refresh tokens expire in 7 days
    expire = datetime.utcnow() + timedelta(days=7)
    to_encode.update({"exp": expire, "type": "refresh"})

    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm="HS256")
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> Optional[TokenData]:
    """Verify and decode JWT token"""
    settings = get_settings()

    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])

        # Check token type
        if payload.get("type") != token_type:
            return None

        email: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        roles: List[str] = payload.get("roles", [])

        if email is None:
            return None

        return TokenData(email=email, user_id=user_id, roles=roles)

    except jwt.PyJWTError as e:
        logger.warning("Token verification failed", error=str(e))
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Use JWT manager for token verification
    token_payload = await jwt_manager.verify_token(credentials.credentials, "access")
    if token_payload is None:
        raise credentials_exception

    # TODO: Get user from database
    # For now, return a mock user
    user = User(
        id=token_payload.user_id,
        email=token_payload.email,
        full_name="Mock User",
        roles=token_payload.roles,
        is_active=True,
        created_at=datetime.utcnow(),
        last_login=datetime.utcnow(),
    )

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def require_roles(required_roles: List[str]):
    """Dependency to require specific roles"""

    def role_checker(current_user: User = Depends(get_current_active_user)):
        if not any(role in current_user.roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions"
            )
        return current_user

    return role_checker


def require_permission(permission: Permission):
    """Dependency to require specific permission"""

    def permission_checker(current_user: User = Depends(get_current_active_user)):
        if not rbac_manager.has_permission(current_user.roles, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission.value}",
            )
        return current_user

    return permission_checker


def require_resource_access(resource: str, action: str):
    """Dependency to require resource access"""

    def access_checker(current_user: User = Depends(get_current_active_user)):
        if not rbac_manager.can_access_resource(current_user.roles, resource, action):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied to {resource}:{action}",
            )
        return current_user

    return access_checker


@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate):
    """Register a new user"""

    # TODO: Check if user already exists
    # TODO: Store user in database

    # Hash password
    hashed_password = get_password_hash(user_data.password)

    # Create user (mock implementation)
    user = User(
        id="user_" + str(hash(user_data.email))[-8:],
        email=user_data.email,
        full_name=user_data.full_name,
        roles=[user_data.role],
        is_active=True,
        created_at=datetime.utcnow(),
    )

    logger.info("User registered", user_id=user.id, email=user.email)

    return user


@router.post("/login", response_model=Token)
async def login_user(user_data: UserLogin):
    """Authenticate user and return tokens"""

    # TODO: Get user from database and verify password
    # For now, mock authentication

    if user_data.email == "admin@example.com" and user_data.password == "admin123":
        user_id = "admin_user"
        roles = ["admin", "user"]
    elif user_data.password == "user123":
        user_id = "regular_user"
        roles = ["user"]
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    # Get user permissions
    permissions = list(rbac_manager.get_user_permissions(roles))
    permission_strings = [p.value for p in permissions]

    # Create token pair using JWT manager
    token_pair = await jwt_manager.create_token_pair(
        user_id=user_id,
        email=user_data.email,
        roles=roles,
        permissions=permission_strings,
    )

    logger.info("User logged in", user_id=user_id, email=user_data.email, roles=roles)

    return Token(
        access_token=token_pair["access_token"],
        refresh_token=token_pair["refresh_token"],
        expires_in=token_pair["expires_in"],
    )


@router.post("/refresh", response_model=Token)
async def refresh_access_token(refresh_data: RefreshTokenRequest):
    """Refresh access token using refresh token"""

    # Use JWT manager for token refresh
    token_result = await jwt_manager.refresh_access_token(refresh_data.refresh_token)

    if token_result is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )

    return Token(
        access_token=token_result["access_token"],
        refresh_token=refresh_data.refresh_token,  # Keep same refresh token
        expires_in=token_result["expires_in"],
    )


@router.get("/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return current_user


@router.post("/logout")
async def logout_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    current_user: User = Depends(get_current_active_user),
):
    """Logout user (invalidate tokens)"""

    # Revoke current token
    await jwt_manager.revoke_token(credentials.credentials)

    logger.info("User logged out", user_id=current_user.id, email=current_user.email)

    return {"message": "Successfully logged out"}


@router.get("/verify-token")
async def verify_user_token(current_user: User = Depends(get_current_active_user)):
    """Verify if current token is valid"""
    return {
        "valid": True,
        "user_id": current_user.id,
        "email": current_user.email,
        "roles": current_user.roles,
    }


# Admin endpoints
@router.get("/users", dependencies=[Depends(require_roles(["admin"]))])
async def list_users():
    """List all users (admin only)"""
    # TODO: Get users from database
    return {"users": [], "total": 0}


@router.put("/users/{user_id}/roles", dependencies=[Depends(require_roles(["admin"]))])
async def update_user_roles(user_id: str, roles: List[str]):
    """Update user roles (admin only)"""
    # TODO: Update user roles in database
    return {"message": f"Roles updated for user {user_id}", "roles": roles}


@router.delete("/users/{user_id}", dependencies=[Depends(require_roles(["admin"]))])
async def delete_user(user_id: str):
    """Delete user (admin only)"""
    # TODO: Delete user from database
    return {"message": f"User {user_id} deleted"}
