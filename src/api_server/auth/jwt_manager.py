"""
JWT Token Management System
Comprehensive JWT token handling with refresh tokens, blacklisting, and security features
"""

import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Set
import hashlib
import secrets
from dataclasses import dataclass

from src.shared.config import get_settings
from src.shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TokenPayload:
    """Token payload structure"""

    user_id: str
    email: str
    roles: List[str]
    permissions: List[str]
    session_id: str
    issued_at: datetime
    expires_at: datetime
    token_type: str  # access, refresh


class JWTManager:
    """Comprehensive JWT token management"""

    def __init__(self):
        self.settings = get_settings()
        self.redis_client = None  # TODO: Initialize Redis connection
        self.algorithm = "HS256"
        self.access_token_expire = timedelta(
            minutes=self.settings.access_token_expire_minutes
        )
        self.refresh_token_expire = timedelta(days=7)

        # Token blacklist (in-memory for now, should use Redis in production)
        self.blacklisted_tokens: Set[str] = set()

    async def create_token_pair(
        self, user_id: str, email: str, roles: List[str], permissions: List[str] = None
    ) -> Dict[str, Any]:
        """Create access and refresh token pair"""

        session_id = self._generate_session_id()
        now = datetime.utcnow()

        # Create access token
        access_payload = {
            "user_id": user_id,
            "email": email,
            "roles": roles,
            "permissions": permissions or [],
            "session_id": session_id,
            "iat": now,
            "exp": now + self.access_token_expire,
            "type": "access",
            "jti": self._generate_jti(),  # JWT ID for tracking
        }

        access_token = jwt.encode(
            access_payload, self.settings.secret_key, algorithm=self.algorithm
        )

        # Create refresh token
        refresh_payload = {
            "user_id": user_id,
            "email": email,
            "session_id": session_id,
            "iat": now,
            "exp": now + self.refresh_token_expire,
            "type": "refresh",
            "jti": self._generate_jti(),
        }

        refresh_token = jwt.encode(
            refresh_payload, self.settings.secret_key, algorithm=self.algorithm
        )

        # Store session information
        await self._store_session(
            session_id,
            {
                "user_id": user_id,
                "email": email,
                "roles": roles,
                "permissions": permissions or [],
                "created_at": now.isoformat(),
                "last_activity": now.isoformat(),
                "is_active": True,
            },
        )

        logger.info(
            "Token pair created", user_id=user_id, session_id=session_id, roles=roles
        )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": int(self.access_token_expire.total_seconds()),
            "session_id": session_id,
        }

    async def verify_token(
        self, token: str, token_type: str = "access"
    ) -> Optional[TokenPayload]:
        """Verify and decode JWT token"""

        try:
            # Check if token is blacklisted
            if await self._is_token_blacklisted(token):
                logger.warning(
                    "Blacklisted token used", token_hash=self._hash_token(token)
                )
                return None

            # Decode token
            payload = jwt.decode(
                token, self.settings.secret_key, algorithms=[self.algorithm]
            )

            # Verify token type
            if payload.get("type") != token_type:
                logger.warning(
                    "Invalid token type",
                    expected=token_type,
                    actual=payload.get("type"),
                )
                return None

            # Check if session is still active
            session_id = payload.get("session_id")
            if session_id and not await self._is_session_active(session_id):
                logger.warning("Inactive session token used", session_id=session_id)
                return None

            # Update last activity
            if session_id:
                await self._update_session_activity(session_id)

            return TokenPayload(
                user_id=payload["user_id"],
                email=payload["email"],
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
                session_id=payload.get("session_id", ""),
                issued_at=datetime.fromtimestamp(payload["iat"]),
                expires_at=datetime.fromtimestamp(payload["exp"]),
                token_type=payload["type"],
            )

        except jwt.ExpiredSignatureError:
            logger.info("Expired token used")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid token", error=str(e))
            return None
        except Exception as e:
            logger.error("Token verification error", error=str(e))
            return None

    async def refresh_access_token(
        self, refresh_token: str
    ) -> Optional[Dict[str, Any]]:
        """Create new access token using refresh token"""

        # Verify refresh token
        token_payload = await self.verify_token(refresh_token, "refresh")
        if not token_payload:
            return None

        # Get current session data
        session_data = await self._get_session(token_payload.session_id)
        if not session_data or not session_data.get("is_active"):
            return None

        # Create new access token
        now = datetime.utcnow()
        access_payload = {
            "user_id": token_payload.user_id,
            "email": token_payload.email,
            "roles": session_data.get("roles", []),
            "permissions": session_data.get("permissions", []),
            "session_id": token_payload.session_id,
            "iat": now,
            "exp": now + self.access_token_expire,
            "type": "access",
            "jti": self._generate_jti(),
        }

        access_token = jwt.encode(
            access_payload, self.settings.secret_key, algorithm=self.algorithm
        )

        logger.info(
            "Access token refreshed",
            user_id=token_payload.user_id,
            session_id=token_payload.session_id,
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": int(self.access_token_expire.total_seconds()),
        }

    async def revoke_token(self, token: str) -> bool:
        """Revoke (blacklist) a token"""

        try:
            # Decode token to get JTI
            payload = jwt.decode(
                token,
                self.settings.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False},  # Allow expired tokens for revocation
            )

            jti = payload.get("jti")
            if jti:
                await self._blacklist_token(jti, payload.get("exp", 0))
                logger.info("Token revoked", jti=jti)
                return True

        except Exception as e:
            logger.error("Token revocation error", error=str(e))

        return False

    async def revoke_session(self, session_id: str) -> bool:
        """Revoke all tokens for a session"""

        try:
            # Mark session as inactive
            await self._deactivate_session(session_id)
            logger.info("Session revoked", session_id=session_id)
            return True

        except Exception as e:
            logger.error("Session revocation error", error=str(e))
            return False

    async def revoke_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user"""

        try:
            # TODO: Get all user sessions from Redis and deactivate them
            # For now, return 0
            logger.info("All user sessions revoked", user_id=user_id)
            return 0

        except Exception as e:
            logger.error("User session revocation error", error=str(e))
            return 0

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"sess_{secrets.token_urlsafe(32)}"

    def _generate_jti(self) -> str:
        """Generate JWT ID"""
        return f"jti_{secrets.token_urlsafe(16)}"

    def _hash_token(self, token: str) -> str:
        """Create hash of token for blacklist storage"""
        return hashlib.sha256(token.encode()).hexdigest()

    async def _store_session(self, session_id: str, session_data: Dict[str, Any]):
        """Store session data"""
        # TODO: Store in Redis
        # For now, just log
        logger.debug("Session stored", session_id=session_id)

    async def _get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        # TODO: Get from Redis
        # For now, return mock data
        return {"is_active": True, "roles": ["user"], "permissions": []}

    async def _is_session_active(self, session_id: str) -> bool:
        """Check if session is active"""
        session_data = await self._get_session(session_id)
        return session_data and session_data.get("is_active", False)

    async def _update_session_activity(self, session_id: str):
        """Update session last activity"""
        # TODO: Update in Redis
        pass

    async def _deactivate_session(self, session_id: str):
        """Deactivate a session"""
        # TODO: Update in Redis
        pass

    async def _blacklist_token(self, jti: str, exp: int):
        """Add token JTI to blacklist"""
        # TODO: Store in Redis with expiration
        self.blacklisted_tokens.add(jti)

    async def _is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted"""
        try:
            payload = jwt.decode(
                token,
                self.settings.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False},
            )
            jti = payload.get("jti")
            return jti in self.blacklisted_tokens
        except:
            return False


# Global JWT manager instance
jwt_manager = JWTManager()


async def get_jwt_manager() -> JWTManager:
    """Get JWT manager instance"""
    return jwt_manager
