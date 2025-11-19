"""
Session Management System
Handles user sessions, WebSocket authentication, and session persistence
"""

import uuid
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from .database import (
    SessionOperations,
    UserOperations,
    database_session_scope,
    Session as DBSession,
)
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class SessionInfo:
    """Session information data class"""

    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    session_type: str = "conversation"
    context: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    is_active: bool = True

    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.metadata is None:
            self.metadata = {}


class SessionManager:
    """Manages user sessions and WebSocket connections"""

    def __init__(self):
        self.active_sessions: Dict[str, SessionInfo] = {}
        self.user_sessions: Dict[str, List[str]] = {}  # user_id -> [session_ids]
        self.session_cleanup_interval = 3600  # 1 hour

    async def create_session(
        self,
        user_id: str,
        session_type: str = "conversation",
        title: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> SessionInfo:
        """Create a new session"""

        session_id = session_id or str(uuid.uuid4())
        now = datetime.utcnow()

        # Create session info
        session_info = SessionInfo(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            session_type=session_type,
            context=context or {},
            metadata={"title": title} if title else {},
        )

        # Store in memory
        self.active_sessions[session_id] = session_info

        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session_id)

        # Persist to database
        try:
            db_session = SessionOperations.create_session(
                user_id=user_id,
                title=title,
                session_type=session_type,
                context=context or {},
                session_id=session_id,
            )

            # Update session_id to match database
            if db_session and db_session.id != session_id:
                # Update our in-memory tracking
                old_session_id = session_id
                session_id = db_session.id
                session_info.session_id = session_id

                # Update dictionaries
                self.active_sessions[session_id] = self.active_sessions.pop(
                    old_session_id
                )
                self.user_sessions[user_id] = [
                    session_id if s == old_session_id else s
                    for s in self.user_sessions[user_id]
                ]

            logger.info(
                "Session created",
                session_id=session_id,
                user_id=user_id,
                session_type=session_type,
            )

        except Exception as e:
            logger.error(f"Failed to persist session to database: {e}")
            # Continue with in-memory session

        return session_info

    async def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session by ID"""

        # Check memory first
        if session_id in self.active_sessions:
            session_info = self.active_sessions[session_id]
            # Update last activity
            session_info.last_activity = datetime.utcnow()
            await self.update_session_activity(session_id)
            return session_info

        # Try to load from database
        try:
            db_session = SessionOperations.get_session_by_id(session_id)
            if db_session and db_session.is_active:
                # Load into memory
                session_info = SessionInfo(
                    session_id=db_session.id,
                    user_id=db_session.user_id,
                    created_at=db_session.created_at,
                    last_activity=db_session.last_activity,
                    session_type=db_session.session_type,
                    context=db_session.context or {},
                    metadata={"title": db_session.title} if db_session.title else {},
                    is_active=db_session.is_active,
                )

                self.active_sessions[session_id] = session_info

                if session_info.user_id not in self.user_sessions:
                    self.user_sessions[session_info.user_id] = []
                if session_id not in self.user_sessions[session_info.user_id]:
                    self.user_sessions[session_info.user_id].append(session_id)

                return session_info

        except Exception as e:
            logger.error(f"Failed to load session from database: {e}")

        return None

    async def update_session_activity(self, session_id: str) -> bool:
        """Update session last activity timestamp"""

        if session_id in self.active_sessions:
            self.active_sessions[session_id].last_activity = datetime.utcnow()

            # Update in database
            try:
                SessionOperations.update_session_activity(session_id)
                return True
            except Exception as e:
                logger.error(f"Failed to update session activity in database: {e}")
                return False

        return False

    async def update_session_context(
        self, session_id: str, context: Dict[str, Any]
    ) -> bool:
        """Update session context"""

        if session_id in self.active_sessions:
            # Merge into in-memory context first
            self.active_sessions[session_id].context.update(context)

            # Update in database with full context
            try:
                SessionOperations.update_session_context(
                    session_id, self.active_sessions[session_id].context
                )
                return True
            except Exception as e:
                logger.error(f"Failed to update session context in database: {e}")
                return False

        return False

    async def get_user_sessions(
        self, user_id: str, active_only: bool = True
    ) -> List[SessionInfo]:
        """Get all sessions for a user"""

        sessions = []

        # Get from memory
        if user_id in self.user_sessions:
            for session_id in self.user_sessions[user_id]:
                if session_id in self.active_sessions:
                    session_info = self.active_sessions[session_id]
                    if not active_only or session_info.is_active:
                        sessions.append(session_info)

        # Also check database for any sessions not in memory
        try:
            db_sessions = SessionOperations.get_user_sessions(
                user_id=user_id, active_only=active_only, limit=50
            )

            for db_session in db_sessions:
                if db_session.id not in self.active_sessions:
                    session_info = SessionInfo(
                        session_id=db_session.id,
                        user_id=db_session.user_id,
                        created_at=db_session.created_at,
                        last_activity=db_session.last_activity,
                        session_type=db_session.session_type,
                        context=db_session.context or {},
                        metadata=(
                            {"title": db_session.title} if db_session.title else {}
                        ),
                        is_active=db_session.is_active,
                    )
                    sessions.append(session_info)

        except Exception as e:
            logger.error(f"Failed to load user sessions from database: {e}")

        # Sort by last activity
        sessions.sort(key=lambda s: s.last_activity, reverse=True)
        return sessions

    async def end_session(self, session_id: str) -> bool:
        """End a session"""

        if session_id in self.active_sessions:
            session_info = self.active_sessions[session_id]
            session_info.is_active = False

            # Remove from active tracking
            user_id = session_info.user_id
            if user_id in self.user_sessions:
                self.user_sessions[user_id] = [
                    s for s in self.user_sessions[user_id] if s != session_id
                ]
                if not self.user_sessions[user_id]:
                    del self.user_sessions[user_id]

            del self.active_sessions[session_id]

            # Update database
            try:
                with database_session_scope() as db_session:
                    db_session_obj = (
                        db_session.query(DBSession).filter_by(id=session_id).first()
                    )

                    if db_session_obj:
                        db_session_obj.is_active = False
                        db_session_obj.updated_at = datetime.utcnow()

                logger.info("Session ended", session_id=session_id, user_id=user_id)
                return True

            except Exception as e:
                logger.error(f"Failed to end session in database: {e}")
                return False

        return False

    async def cleanup_inactive_sessions(self, max_age_hours: int = 24):
        """Clean up old inactive sessions"""

        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        sessions_to_remove = []

        for session_id, session_info in self.active_sessions.items():
            if session_info.last_activity < cutoff_time:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            await self.end_session(session_id)

        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} inactive sessions")

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""

        total_sessions = len(self.active_sessions)
        total_users = len(self.user_sessions)

        # Calculate session types
        session_types = {}
        for session_info in self.active_sessions.values():
            session_type = session_info.session_type
            session_types[session_type] = session_types.get(session_type, 0) + 1

        return {
            "total_active_sessions": total_sessions,
            "total_active_users": total_users,
            "session_types": session_types,
            "timestamp": datetime.utcnow().isoformat(),
        }


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get global session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
