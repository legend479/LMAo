"""
Integration tests for API server and Agent server communication
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient
from datetime import datetime

from src.api_server.main import create_app
from src.agent_server.main import AgentServer
from src.api_server.routers.auth import get_current_active_user, User


# Mock user for testing
def get_mock_user():
    """Return a mock authenticated user for testing"""
    return User(
        id="test_user_123",
        email="test@example.com",
        full_name="Test User",
        roles=["user"],
        is_active=True,
        created_at=datetime.utcnow(),
        last_login=datetime.utcnow(),
    )


@pytest.fixture
async def authenticated_client(app) -> AsyncClient:
    """Create test client with authentication mocked"""
    # Override the authentication dependency
    app.dependency_overrides[get_current_active_user] = get_mock_user

    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

    # Clean up
    app.dependency_overrides.clear()


@pytest.mark.integration
@pytest.mark.asyncio
class TestAPIAgentIntegration:
    """Test integration between API server and Agent server."""

    async def test_chat_message_flow(self, authenticated_client: AsyncClient):
        """Test complete chat message flow from API to Agent."""

        # Mock agent server response
        mock_agent_response = {
            "response": "Hello! I'm the SE SME Agent. How can I help you with software engineering?",
            "session_id": "test_session_123",
            "timestamp": "2024-01-01T10:00:00Z",
            "metadata": {"processing_time": 0.5, "tools_used": [], "confidence": 0.9},
        }

        with patch(
            "src.api_server.routers.chat.get_agent_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_agent = AsyncMock()
            mock_agent.process_message = AsyncMock(return_value=mock_agent_response)
            mock_get_client.return_value = mock_agent

            response = await authenticated_client.post(
                "/api/v1/chat/message",
                json={
                    "message": "Hello, what can you help me with?",
                    "session_id": "test_session_123",
                },
            )

            # Verify the mock was called
            mock_get_client.assert_called_once()
            mock_agent.process_message.assert_called_once()

            assert response.status_code == 200
            data = response.json()
            assert data["response"] == mock_agent_response["response"]
            assert data["session_id"] == "test_session_123"

    async def test_tool_execution_flow(self, authenticated_client: AsyncClient):
        """Test tool execution flow from API to Agent."""

        mock_tool_response = {
            "tool_name": "knowledge_retrieval",
            "result": {
                "query": "Python programming",
                "results": [
                    {
                        "content": "Python is a high-level programming language...",
                        "score": 0.95,
                        "source": "python_docs.md",
                    }
                ],
                "total_results": 1,
            },
            "status": "success",
            "execution_time": 1.2,
        }

        # Note: Tools router doesn't use agent_client yet, so this test needs adjustment
        # For now, we'll test the endpoint directly
        response = await authenticated_client.post(
            "/api/v1/tools/execute",
            json={
                "tool_name": "knowledge_retrieval",
                "parameters": {"query": "Python programming", "max_results": 5},
                "session_id": "test_session_123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "tool_name" in data
        assert "status" in data

    async def test_agent_server_unavailable(self, authenticated_client: AsyncClient):
        """Test API behavior when agent server is unavailable."""

        with patch(
            "src.api_server.routers.chat.get_agent_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_agent = AsyncMock()
            mock_agent.process_message = AsyncMock(
                side_effect=ConnectionError("Agent server unavailable")
            )
            mock_get_client.return_value = mock_agent

            response = await authenticated_client.post(
                "/api/v1/chat/message",
                json={"message": "Hello", "session_id": "test_session_123"},
            )

        # API should handle the error gracefully and return 200 with error metadata
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["metadata"].get("error") is True

    async def test_websocket_agent_integration(self):
        """Test WebSocket integration with agent server."""

        app = create_app()

        mock_agent_response = {
            "response": "WebSocket response from agent",
            "session_id": "ws_session_123",
            "timestamp": "2024-01-01T10:00:00Z",
            "metadata": {},
        }

        with patch(
            "src.api_server.routers.chat.get_agent_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_agent = AsyncMock()
            mock_agent.process_message = AsyncMock(return_value=mock_agent_response)
            mock_get_client.return_value = mock_agent

            async with AsyncClient(app=app, base_url="http://test") as client:
                # Note: WebSocket testing requires more complex setup
                # This is a simplified test structure
                pass

    async def test_concurrent_requests_to_agent(
        self, authenticated_client: AsyncClient
    ):
        """Test concurrent requests to agent server."""

        async def mock_process_message(message, session_id, user_id=None):
            # Simulate processing time
            await asyncio.sleep(0.1)
            return {
                "response": f"Processed: {message}",
                "session_id": session_id,
                "timestamp": "2024-01-01T10:00:00Z",
                "metadata": {"processing_time": 0.1},
            }

        with patch(
            "src.api_server.routers.chat.get_agent_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_agent = AsyncMock()
            mock_agent.process_message = mock_process_message
            mock_get_client.return_value = mock_agent

            # Send multiple concurrent requests
            tasks = []
            for i in range(5):
                task = authenticated_client.post(
                    "/api/v1/chat/message",
                    json={"message": f"Message {i}", "session_id": f"session_{i}"},
                )
                tasks.append(task)

            responses = await asyncio.gather(*tasks)

        # All requests should succeed
        for i, response in enumerate(responses):
            assert response.status_code == 200
            data = response.json()
            assert f"Message {i}" in data["response"]

    async def test_agent_error_handling(self, authenticated_client: AsyncClient):
        """Test error handling when agent server returns errors."""

        mock_error_response = {
            "response": "I apologize, but I encountered an error processing your request.",
            "session_id": "error_session",
            "timestamp": "2024-01-01T10:00:00Z",
            "metadata": {"error": True, "error_type": "ProcessingError"},
        }

        with patch(
            "src.api_server.routers.chat.get_agent_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_agent = AsyncMock()
            mock_agent.process_message = AsyncMock(return_value=mock_error_response)
            mock_get_client.return_value = mock_agent

            response = await authenticated_client.post(
                "/api/v1/chat/message",
                json={"message": "Cause an error", "session_id": "error_session"},
            )

        assert response.status_code == 200  # API should handle agent errors gracefully
        data = response.json()
        assert data["metadata"]["error"] is True

    async def test_session_management_integration(
        self, authenticated_client: AsyncClient
    ):
        """Test session management between API and Agent."""

        session_id = "persistent_session_123"

        # First message
        mock_response_1 = {
            "response": "Hello! How can I help you?",
            "session_id": session_id,
            "timestamp": "2024-01-01T10:00:00Z",
            "metadata": {"message_count": 1},
        }

        # Second message in same session
        mock_response_2 = {
            "response": "I remember our previous conversation.",
            "session_id": session_id,
            "timestamp": "2024-01-01T10:01:00Z",
            "metadata": {"message_count": 2},
        }

        with patch(
            "src.api_server.routers.chat.get_agent_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_agent = AsyncMock()
            mock_agent.process_message = AsyncMock(
                side_effect=[mock_response_1, mock_response_2]
            )
            mock_get_client.return_value = mock_agent

            # Send first message
            response1 = await authenticated_client.post(
                "/api/v1/chat/message",
                json={"message": "Hello", "session_id": session_id},
            )

            # Send second message
            response2 = await authenticated_client.post(
                "/api/v1/chat/message",
                json={"message": "Do you remember me?", "session_id": session_id},
            )

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        assert data1["session_id"] == session_id
        assert data2["session_id"] == session_id
        assert data2["metadata"]["message_count"] == 2


@pytest.mark.integration
@pytest.mark.requires_redis
class TestAPIAgentWithRedis:
    """Test API-Agent integration with Redis for session management."""

    async def test_session_persistence_with_redis(
        self, authenticated_client: AsyncClient, redis_client
    ):
        """Test session persistence using Redis."""

        session_id = "redis_session_123"

        # Mock agent server to store session data
        async def mock_process_with_redis(message, session_id, user_id=None):
            # Simulate storing session data in Redis
            await redis_client.set(f"session:{session_id}", message)
            return {
                "response": f"Stored message: {message}",
                "session_id": session_id,
                "timestamp": "2024-01-01T10:00:00Z",
                "metadata": {},
            }

        with patch(
            "src.api_server.routers.chat.get_agent_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_agent = AsyncMock()
            mock_agent.process_message = mock_process_with_redis
            mock_get_client.return_value = mock_agent

            response = await authenticated_client.post(
                "/api/v1/chat/message",
                json={"message": "Store this message", "session_id": session_id},
            )

        assert response.status_code == 200

        # Verify data was stored in Redis
        stored_message = await redis_client.get(f"session:{session_id}")
        # Redis returns bytes, so decode it
        if isinstance(stored_message, bytes):
            stored_message = stored_message.decode("utf-8")
        assert stored_message == "Store this message"


@pytest.mark.integration
@pytest.mark.slow
class TestAPIAgentPerformance:
    """Test performance aspects of API-Agent integration."""

    async def test_response_time_under_load(self, authenticated_client: AsyncClient):
        """Test response times under concurrent load."""

        import time

        async def mock_process_with_delay(message, session_id, user_id=None):
            # Simulate realistic processing time
            await asyncio.sleep(0.05)
            return {
                "response": f"Processed: {message}",
                "session_id": session_id,
                "timestamp": "2024-01-01T10:00:00Z",
                "metadata": {"processing_time": 0.05},
            }

        with patch(
            "src.api_server.routers.chat.get_agent_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_agent = AsyncMock()
            mock_agent.process_message = mock_process_with_delay
            mock_get_client.return_value = mock_agent

            start_time = time.time()

            # Send 10 concurrent requests
            tasks = []
            for i in range(10):
                task = authenticated_client.post(
                    "/api/v1/chat/message",
                    json={
                        "message": f"Load test message {i}",
                        "session_id": f"load_session_{i}",
                    },
                )
                tasks.append(task)

            responses = await asyncio.gather(*tasks)

            end_time = time.time()
            total_time = end_time - start_time

        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)

        # Should complete in reasonable time (allowing for concurrency)
        assert (
            total_time < 2.0
        )  # Should be much faster than 10 * 0.05 = 0.5s due to concurrency

    async def test_memory_usage_stability(self, authenticated_client: AsyncClient):
        """Test memory usage remains stable under repeated requests."""

        import gc
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        async def mock_process_simple(message, session_id, user_id=None):
            return {
                "response": "Simple response",
                "session_id": session_id,
                "timestamp": "2024-01-01T10:00:00Z",
                "metadata": {},
            }

        with patch(
            "src.api_server.routers.chat.get_agent_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_agent = AsyncMock()
            mock_agent.process_message = mock_process_simple
            mock_get_client.return_value = mock_agent

            # Send requests to test memory stability (reduced to avoid rate limiting)
            for i in range(20):
                response = await authenticated_client.post(
                    "/api/v1/chat/message",
                    json={
                        "message": f"Memory test {i}",
                        "session_id": f"memory_session_{i}",
                    },
                )
                assert response.status_code == 200

                # Force garbage collection every 5 requests
                if i % 5 == 0:
                    gc.collect()

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024
