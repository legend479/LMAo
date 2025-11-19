"""
Chat Router
Handles conversation endpoints and WebSocket connections with comprehensive real-time support
"""

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    HTTPException,
)
from pydantic import BaseModel, Field, field_serializer, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import asyncio
import uuid

from .auth import get_current_active_user, User
from src.shared.logging import get_logger
import src.shared.services as services
from src.shared.session_manager import get_session_manager

logger = get_logger(__name__)
router = APIRouter()


async def get_agent_client():
    """Wrapper around shared get_agent_client for easier testing and patching."""
    return await services.get_agent_client()


class ChatMessage(BaseModel):
    message: str = Field(
        ..., min_length=1, max_length=10000, description="Chat message content"
    )
    session_id: Optional[str] = Field(
        None, max_length=100, description="Session identifier"
    )
    user_id: Optional[str] = Field(None, max_length=100, description="User identifier")
    message_type: str = Field(
        default="user", description="Type of message: user, system, agent"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate and sanitize message content"""
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")

        # Check for potentially dangerous content
        dangerous_patterns = [
            "<script>",
            "</script>",
            "javascript:",
            "onerror=",
            "onclick=",
        ]
        message_lower = v.lower()
        for pattern in dangerous_patterns:
            if pattern in message_lower:
                raise ValueError(
                    f"Message contains potentially dangerous content: {pattern}"
                )

        # Strip excessive whitespace
        return v.strip()

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate session ID format"""
        if v is not None:
            # Remove any non-alphanumeric characters except hyphens and underscores
            import re

            if not re.match(r"^[a-zA-Z0-9_-]+$", v):
                raise ValueError("Session ID contains invalid characters")
        return v


class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_id: str
    timestamp: datetime
    message_type: str = "agent"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time: Optional[float] = None

    @field_serializer("timestamp")
    def serialize_timestamp(self, value: datetime) -> str:
        return value.isoformat()


class WebSocketMessage(BaseModel):
    type: str  # message, status, error, typing, etc.
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer("timestamp")
    def serialize_timestamp(self, value: datetime) -> str:
        return value.isoformat()


class ChatSession(BaseModel):
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    message_count: int = 0
    status: str = "active"  # active, inactive, ended


class ConnectionManager:
    """Manages WebSocket connections for real-time chat"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_users: Dict[str, str] = {}  # session_id -> user_id
        self.user_sessions: Dict[str, List[str]] = {}  # user_id -> [session_ids]

    async def connect(self, websocket: WebSocket, session_id: str, user_id: str):
        """Accept WebSocket connection and register session"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_users[session_id] = user_id

        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session_id)

        logger.info(
            "WebSocket connected",
            session_id=session_id,
            user_id=user_id,
            total_connections=len(self.active_connections),
        )

    def disconnect(self, session_id: str):
        """Remove WebSocket connection"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]

        if session_id in self.session_users:
            user_id = self.session_users[session_id]
            del self.session_users[session_id]

            # Remove session from user's session list
            if user_id in self.user_sessions:
                self.user_sessions[user_id] = [
                    s for s in self.user_sessions[user_id] if s != session_id
                ]
                if not self.user_sessions[user_id]:
                    del self.user_sessions[user_id]

        logger.info(
            "WebSocket disconnected",
            session_id=session_id,
            total_connections=len(self.active_connections),
        )

    async def send_personal_message(self, message: str, session_id: str):
        """Send message to specific session"""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(
                    "Failed to send WebSocket message",
                    session_id=session_id,
                    error=str(e),
                )
                self.disconnect(session_id)

    async def send_json_message(self, data: Dict[str, Any], session_id: str):
        """Send JSON message to specific session"""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_json(data)
            except Exception as e:
                logger.error(
                    "Failed to send WebSocket JSON", session_id=session_id, error=str(e)
                )
                self.disconnect(session_id)

    async def broadcast_to_user(self, message: Dict[str, Any], user_id: str):
        """Broadcast message to all sessions of a user"""
        if user_id in self.user_sessions:
            for session_id in self.user_sessions[user_id]:
                await self.send_json_message(message, session_id)

    async def send_typing_indicator(self, session_id: str, is_typing: bool):
        """Send typing indicator to session"""
        message = {
            "type": "typing",
            "data": {"is_typing": is_typing},
            "timestamp": datetime.utcnow().isoformat(),
        }
        await self.send_json_message(message, session_id)

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        return list(self.active_connections.keys())

    def get_session_count(self) -> int:
        """Get total number of active sessions"""
        return len(self.active_connections)


# Global connection manager
manager = ConnectionManager()


@router.post("/message", response_model=ChatResponse)
async def send_message(
    message: ChatMessage, current_user: User = Depends(get_current_active_user)
):
    """Send a message to the agent via REST API"""

    start_time = datetime.utcnow()
    message_id = str(uuid.uuid4())
    session_id = (
        message.session_id
        or f"session_{current_user.id}_{int(datetime.utcnow().timestamp())}"
    )

    logger.info(
        "Chat message received",
        user_id=current_user.id,
        session_id=session_id,
        message_id=message_id,
        message_length=len(message.message),
    )

    try:
        # Process message through agent server via HTTP client
        try:
            agent_client = await get_agent_client()

            # Process the message through the agent service
            agent_response = await agent_client.process_message(
                message=message.message, session_id=session_id, user_id=current_user.id
            )

            response_text = agent_response.get(
                "response",
                "I apologize, but I couldn't process your message at this time.",
            )

            # Add metadata from agent response
            agent_metadata = agent_response.get("metadata", {})

        except Exception as e:
            logger.error(f"Agent processing failed: {str(e)}")
            response_text = f"Hello {current_user.full_name or current_user.email}! I'm the SE SME Agent. I'm experiencing some technical difficulties but I'm here to help with software engineering questions. Please try again in a moment."
            agent_metadata = {"error": True, "error_message": str(e)}

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        response = ChatResponse(
            response=response_text,
            session_id=session_id,
            message_id=message_id,
            timestamp=datetime.utcnow(),
            processing_time=processing_time,
            metadata={
                "user_id": current_user.id,
                "original_message_length": len(message.message),
                **agent_metadata,
            },
        )

        # Persist messages to session history
        session_manager = get_session_manager()

        # Ensure session exists and is owned by the current user
        session_info = await session_manager.get_session(session_id)
        if not session_info:
            session_info = await session_manager.create_session(
                user_id=current_user.id,
                session_type="conversation",
                title=message.message[:80],
                context={},
                session_id=session_id,
            )

        if session_info.user_id != current_user.id:
            logger.warning(
                "Session user mismatch when storing chat message",
                session_id=session_id,
                session_user_id=session_info.user_id,
                current_user_id=current_user.id,
            )
        else:
            history = session_info.context.get("messages", [])

            user_entry = {
                "id": message_id,
                "role": "user",
                "content": message.message,
                "timestamp": start_time.isoformat(),
                "source": "rest",
            }

            agent_entry = {
                "id": f"agent_{message_id}",
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "rest",
                "metadata": agent_metadata,
            }

            history.extend([user_entry, agent_entry])

            await session_manager.update_session_context(
                session_id, {"messages": history}
            )

        # Send response to WebSocket if session is connected
        if session_id in manager.active_connections:
            await manager.send_json_message(
                {
                    "type": "message",
                    "data": response.model_dump(),
                    "timestamp": datetime.utcnow().isoformat(),
                },
                session_id,
            )

        logger.info(
            "Chat message processed",
            user_id=current_user.id,
            session_id=session_id,
            message_id=message_id,
            processing_time=processing_time,
        )

        return response

    except Exception as e:
        logger.error(
            "Chat message processing failed",
            user_id=current_user.id,
            session_id=session_id,
            message_id=message_id,
            error=str(e),
        )

        raise HTTPException(status_code=500, detail="Failed to process message")


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat with authentication and comprehensive features"""

    try:
        # Authenticate WebSocket connection
        from ..websocket_auth import websocket_auth
        from src.shared.session_manager import get_session_manager

        user_info = await websocket_auth.require_auth(websocket)
        user_id = user_info["user_id"]

        # Get or create session
        session_manager = get_session_manager()
        session_info = await session_manager.get_session(session_id)

        if not session_info:
            # Create new session if it doesn't exist
            session_info = await session_manager.create_session(
                user_id=user_id, session_type="websocket_conversation"
            )
            session_id = session_info.session_id
        elif session_info.user_id != user_id:
            # User doesn't own this session
            await websocket.close(code=1008, reason="Session access denied")
            return

        await manager.connect(websocket, session_id, user_id)

        # Send welcome message
        welcome_message = {
            "type": "system",
            "data": {
                "message": "Connected to SE SME Agent",
                "session_id": session_id,
                "capabilities": [
                    "Real-time conversation",
                    "Document generation",
                    "Code assistance",
                    "Technical guidance",
                ],
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
        await manager.send_json_message(welcome_message, session_id)

        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()

                try:
                    message_data = json.loads(data)
                except json.JSONDecodeError:
                    # Handle plain text messages
                    message_data = {"type": "message", "data": {"message": data}}

                logger.debug(
                    "WebSocket message received",
                    session_id=session_id,
                    message_type=message_data.get("type", "unknown"),
                )

                # Handle different message types
                message_type = message_data.get("type", "message")

                if message_type == "message":
                    await handle_chat_message(
                        websocket, session_id, message_data, user_id
                    )
                elif message_type == "typing":
                    await handle_typing_indicator(session_id, message_data)
                elif message_type == "ping":
                    await handle_ping(websocket, session_id)
                else:
                    logger.warning(
                        "Unknown WebSocket message type",
                        session_id=session_id,
                        message_type=message_type,
                    )

            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected", session_id=session_id)
                break
            except Exception as e:
                logger.error(
                    "WebSocket message handling error",
                    session_id=session_id,
                    error=str(e),
                )

                # Send error message to client
                error_message = {
                    "type": "error",
                    "data": {
                        "message": "An error occurred processing your message",
                        "error_id": str(uuid.uuid4()),
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                }
                await manager.send_json_message(error_message, session_id)

    except Exception as e:
        logger.error("WebSocket connection error", session_id=session_id, error=str(e))
    finally:
        manager.disconnect(session_id)


async def handle_chat_message(
    websocket: WebSocket, session_id: str, message_data: Dict, user_id: str
):
    """Handle chat message from WebSocket"""

    message_content = message_data.get("data", {}).get("message", "")
    message_id = str(uuid.uuid4())

    # Send typing indicator
    await manager.send_typing_indicator(session_id, True)

    try:
        # Process message through agent server via HTTP client
        try:
            agent_client = await get_agent_client()

            # Process the message through the agent service
            agent_response = await agent_client.process_message(
                message=message_content, session_id=session_id, user_id=user_id
            )

            response_text = agent_response.get(
                "response",
                "I apologize, but I couldn't process your message at this time.",
            )
            response_metadata = agent_response.get("metadata", {})

        except Exception as e:
            logger.error(f"Agent processing failed in WebSocket: {str(e)}")
            response_text = "I'm experiencing some technical difficulties. Please try again in a moment."
            response_metadata = {"error": True, "error_message": str(e)}

        # Persist messages to session history
        session_manager = get_session_manager()
        session_info = await session_manager.get_session(session_id)
        if not session_info:
            session_info = await session_manager.create_session(
                user_id=user_id,
                session_type="websocket_conversation",
                title=message_content[:80],
                context={},
                session_id=session_id,
            )

        history = session_info.context.get("messages", [])

        user_entry = {
            "id": message_id,
            "role": "user",
            "content": message_content,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "websocket",
        }

        agent_entry = {
            "id": f"agent_{message_id}",
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "websocket",
            "metadata": response_metadata,
        }

        history.extend([user_entry, agent_entry])

        await session_manager.update_session_context(session_id, {"messages": history})

        # Send response
        response_message = {
            "type": "message",
            "data": {
                "message_id": message_id,
                "response": response_text,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "message_type": "agent",
                "metadata": response_metadata,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        await manager.send_json_message(response_message, session_id)

    finally:
        # Stop typing indicator
        await manager.send_typing_indicator(session_id, False)


async def handle_typing_indicator(session_id: str, message_data: Dict):
    """Handle typing indicator from client"""
    # Echo typing indicator back (for multi-user scenarios)
    pass


async def handle_ping(websocket: WebSocket, session_id: str):
    """Handle ping message for connection health check"""
    pong_message = {
        "type": "pong",
        "data": {"timestamp": datetime.utcnow().isoformat()},
        "timestamp": datetime.utcnow().isoformat(),
    }
    await manager.send_json_message(pong_message, session_id)


@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify response handling"""
    return {"message": "test response"}


@router.get("/sessions", response_model=List[ChatSession])
async def list_user_sessions(current_user: User = Depends(get_current_active_user)):
    """List chat sessions for current user"""

    from src.shared.session_manager import get_session_manager

    session_manager = get_session_manager()
    session_infos = await session_manager.get_user_sessions(
        user_id=current_user.id, active_only=True
    )

    sessions = []
    for session_info in session_infos:
        message_history = session_info.context.get("messages", [])
        sessions.append(
            ChatSession(
                session_id=session_info.session_id,
                user_id=session_info.user_id,
                created_at=session_info.created_at,
                last_activity=session_info.last_activity,
                message_count=len(message_history),
                status="active" if session_info.is_active else "inactive",
            )
        )

    return sessions


@router.get("/sessions/{session_id}")
async def get_session_info(
    session_id: str, current_user: User = Depends(get_current_active_user)
):
    """Get information about a specific chat session"""

    from src.shared.session_manager import get_session_manager

    session_manager = get_session_manager()
    session_info = await session_manager.get_session(session_id)

    if not session_info:
        raise HTTPException(status_code=404, detail="Session not found")

    if session_info.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Session access denied")

    message_history = session_info.context.get("messages", [])

    return {
        "session_id": session_info.session_id,
        "user_id": session_info.user_id,
        "created_at": session_info.created_at,
        "last_activity": session_info.last_activity,
        "message_count": len(message_history),
        "status": "active" if session_info.is_active else "inactive",
        "is_connected": session_id in manager.active_connections,
        "session_type": session_info.session_type,
        "title": session_info.metadata.get("title"),
    }


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None


class UpdateSessionRequest(BaseModel):
    title: Optional[str] = None
    is_active: Optional[bool] = None


@router.post("/sessions", response_model=ChatSession)
async def create_session(
    request: CreateSessionRequest, current_user: User = Depends(get_current_active_user)
):
    """Create a new chat session for the current user"""

    session_manager = get_session_manager()
    session_info = await session_manager.create_session(
        user_id=current_user.id, session_type="conversation", title=request.title
    )

    return ChatSession(
        session_id=session_info.session_id,
        user_id=session_info.user_id,
        created_at=session_info.created_at,
        last_activity=session_info.last_activity,
        message_count=len(session_info.context.get("messages", [])),
        status="active" if session_info.is_active else "inactive",
    )


@router.patch("/sessions/{session_id}")
async def update_session(
    session_id: str,
    update: UpdateSessionRequest,
    current_user: User = Depends(get_current_active_user),
):
    """Update session metadata (title, active flag)"""

    session_manager = get_session_manager()
    session_info = await session_manager.get_session(session_id)

    if not session_info:
        raise HTTPException(status_code=404, detail="Session not found")

    if session_info.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Session access denied")

    # Update in-memory metadata
    if update.title:
        session_info.metadata["title"] = update.title

    if update.is_active is not None:
        session_info.is_active = update.is_active

    # Persist title change to database (if present)
    try:
        from src.shared.database.connection import database_session_scope
        from src.shared.database.models import Session as DBSession

        with database_session_scope() as db_session:
            db_obj = db_session.query(DBSession).filter_by(id=session_id).first()
            if db_obj:
                if update.title is not None:
                    db_obj.title = update.title
                if update.is_active is not None:
                    db_obj.is_active = update.is_active
    except Exception as e:
        logger.error(f"Failed to persist session update: {e}")

    return {
        "message": "Session updated",
        "session_id": session_id,
    }


@router.get("/sessions/{session_id}/history")
async def get_chat_history(
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_active_user),
):
    """Get chat history for a session"""

    session_manager = get_session_manager()
    session_info = await session_manager.get_session(session_id)

    if not session_info:
        raise HTTPException(status_code=404, detail="Session not found")

    if session_info.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Session access denied")

    messages = session_info.context.get("messages", [])
    total = len(messages)

    # Apply offset/limit safely
    if offset < 0:
        offset = 0
    end = offset + limit
    messages_slice = messages[offset:end]

    return {
        "session_id": session_id,
        "messages": messages_slice,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str, current_user: User = Depends(get_current_active_user)
):
    """Delete a chat session"""

    session_manager = get_session_manager()
    session_info = await session_manager.get_session(session_id)

    if not session_info:
        raise HTTPException(status_code=404, detail="Session not found")

    if session_info.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Session access denied")

    # Disconnect WebSocket if active
    if session_id in manager.active_connections:
        manager.disconnect(session_id)

    # Mark session as ended
    await session_manager.end_session(session_id)

    logger.info("Chat session deleted", session_id=session_id, user_id=current_user.id)

    return {"message": "Session deleted successfully"}


@router.get("/stats")
async def get_chat_statistics(current_user: User = Depends(get_current_active_user)):
    """Get chat statistics for current user"""

    session_manager = get_session_manager()
    sessions = await session_manager.get_user_sessions(
        user_id=current_user.id, active_only=False
    )

    total_sessions = len(sessions)
    total_messages = sum(len(s.context.get("messages", [])) for s in sessions)
    active_sessions = len([s for s in sessions if s.is_active])

    avg_session_length = total_messages / total_sessions if total_sessions > 0 else 0.0

    # Determine most active day based on last_activity
    from collections import Counter

    if sessions:
        day_counts = Counter(s.last_activity.strftime("%A") for s in sessions)
        most_active_day = day_counts.most_common(1)[0][0]
    else:
        most_active_day = None

    return {
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "active_sessions": active_sessions,
        "avg_session_length": avg_session_length,
        "most_active_day": most_active_day,
    }


@router.get("/ws/status")
async def websocket_status():
    """Get WebSocket connection status"""

    return {
        "active_connections": manager.get_session_count(),
        "active_sessions": manager.get_active_sessions(),
        "total_users": len(manager.user_sessions),
        "server_time": datetime.utcnow().isoformat(),
    }
