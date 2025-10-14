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
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import asyncio
import uuid

from .auth import get_current_active_user, User
from ...shared.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    message_type: str = Field(
        default="user", description="Type of message: user, system, agent"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_id: str
    timestamp: datetime
    message_type: str = "agent"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time: Optional[float] = None


class WebSocketMessage(BaseModel):
    type: str  # message, status, error, typing, etc.
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


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
        # TODO: Process message through agent server
        # For now, return a mock response

        # Simulate processing time
        await asyncio.sleep(0.1)

        response_text = f"Hello {current_user.full_name or current_user.email}! I'm the SE SME Agent. I received your message: '{message.message[:50]}...'. I'm currently being set up to provide comprehensive software engineering assistance."

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        response = ChatResponse(
            response=response_text,
            session_id=session_id,
            message_id=message_id,
            timestamp=datetime.utcnow(),
            processing_time=processing_time,
            metadata={
                "status": "setup_mode",
                "user_id": current_user.id,
                "original_message_length": len(message.message),
            },
        )

        # Send response to WebSocket if session is connected
        if session_id in manager.active_connections:
            await manager.send_json_message(
                {
                    "type": "message",
                    "data": response.dict(),
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
        # TODO: Implement WebSocket authentication
        # For now, use a mock user ID
        user_id = "websocket_user"

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
        # TODO: Process message through agent server
        # Simulate processing
        await asyncio.sleep(0.5)

        response_text = f"WebSocket response to: {message_content[:50]}..."

        # Send response
        response_message = {
            "type": "message",
            "data": {
                "message_id": message_id,
                "response": response_text,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "message_type": "agent",
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


@router.get("/sessions", response_model=List[ChatSession])
async def list_user_sessions(current_user: User = Depends(get_current_active_user)):
    """List chat sessions for current user"""

    # TODO: Get actual sessions from database
    # For now, return mock sessions
    sessions = [
        ChatSession(
            session_id=f"session_{current_user.id}_1",
            user_id=current_user.id,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            message_count=5,
            status="active",
        )
    ]

    return sessions


@router.get("/sessions/{session_id}")
async def get_session_info(
    session_id: str, current_user: User = Depends(get_current_active_user)
):
    """Get information about a specific chat session"""

    # TODO: Verify user owns this session
    # TODO: Get actual session from database

    return {
        "session_id": session_id,
        "user_id": current_user.id,
        "created_at": datetime.utcnow(),
        "last_activity": datetime.utcnow(),
        "message_count": 0,
        "status": "active",
        "is_connected": session_id in manager.active_connections,
    }


@router.get("/sessions/{session_id}/history")
async def get_chat_history(
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_active_user),
):
    """Get chat history for a session"""

    # TODO: Verify user owns this session
    # TODO: Get actual history from database

    return {
        "session_id": session_id,
        "messages": [],
        "total": 0,
        "limit": limit,
        "offset": offset,
    }


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str, current_user: User = Depends(get_current_active_user)
):
    """Delete a chat session"""

    # TODO: Verify user owns this session
    # TODO: Delete session from database

    # Disconnect WebSocket if active
    if session_id in manager.active_connections:
        manager.disconnect(session_id)

    logger.info("Chat session deleted", session_id=session_id, user_id=current_user.id)

    return {"message": "Session deleted successfully"}


@router.get("/stats")
async def get_chat_statistics(current_user: User = Depends(get_current_active_user)):
    """Get chat statistics for current user"""

    # TODO: Get actual statistics from database
    return {
        "total_sessions": 1,
        "total_messages": 5,
        "active_sessions": len(
            [s for s in manager.user_sessions.get(current_user.id, [])]
        ),
        "avg_session_length": 10.5,
        "most_active_day": "Monday",
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
