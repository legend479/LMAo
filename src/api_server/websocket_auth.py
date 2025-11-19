"""
WebSocket Authentication System
Handles WebSocket connection authentication and authorization
"""

import jwt
from typing import Optional, Dict, Any
from fastapi import WebSocket, HTTPException, status
from urllib.parse import parse_qs
from types import SimpleNamespace
import uuid

from src.shared.config import get_settings
from src.shared.logging import get_logger
from src.shared.database.operations import UserOperations
from src.shared.database.models import User

logger = get_logger(__name__)


class WebSocketAuthError(Exception):
    """WebSocket authentication error"""

    pass


async def authenticate_websocket(websocket: WebSocket) -> Optional[User]:
    """Authenticate WebSocket connection using token from query parameters"""

    try:
        # Get token from query parameters
        query_params = parse_qs(str(websocket.url.query))
        token = query_params.get("token", [None])[0]

        if not token:
            logger.warning("WebSocket connection attempted without token")
            return None

        # Development/testing shortcut: accept a demo token
        settings = get_settings()
        if token == "mock-jwt-token" and settings.environment != "production":
            logger.info("Using development mock token for WebSocket connection")
            mock_user = SimpleNamespace(
                id=uuid.UUID(int=1),
                username="devuser",
                email="devuser@example.com",
                full_name="Development User",
                is_active=True,
            )
            return mock_user

        # Decode JWT token
        settings = get_settings()
        try:
            payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
            user_id = payload.get("sub")

            if not user_id:
                logger.warning("WebSocket token missing user ID")
                return None

        except jwt.ExpiredSignatureError:
            logger.warning("WebSocket token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"WebSocket token invalid: {e}")
            return None

        # Get user from database
        user = UserOperations.get_user_by_id(user_id)
        if not user:
            logger.warning(f"WebSocket user not found: {user_id}")
            return None

        if not user.is_active:
            logger.warning(f"WebSocket user inactive: {user_id}")
            return None

        logger.info(f"WebSocket authenticated user: {user.id}")
        return user

    except Exception as e:
        logger.error(f"WebSocket authentication error: {e}")
        return None


async def authenticate_websocket_or_reject(websocket: WebSocket) -> User:
    """Authenticate WebSocket or close connection with error"""

    user = await authenticate_websocket(websocket)

    if not user:
        # Send error message before closing
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION, reason="Authentication required"
        )
        raise WebSocketAuthError("Authentication failed")

    return user


class WebSocketTokenAuth:
    """WebSocket token-based authentication middleware"""

    def __init__(self):
        self.settings = get_settings()

    async def authenticate(self, websocket: WebSocket) -> Optional[Dict[str, Any]]:
        """Authenticate WebSocket connection and return user info"""

        try:
            user = await authenticate_websocket(websocket)

            if user:
                return {
                    "user_id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "is_active": user.is_active,
                }

            return None

        except Exception as e:
            logger.error(f"WebSocket authentication middleware error: {e}")
            return None

    async def require_auth(self, websocket: WebSocket) -> Dict[str, Any]:
        """Require authentication or close connection"""

        user_info = await self.authenticate(websocket)

        if not user_info:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION, reason="Authentication required"
            )
            raise WebSocketAuthError("Authentication required")

        return user_info


# Global WebSocket auth instance
websocket_auth = WebSocketTokenAuth()
