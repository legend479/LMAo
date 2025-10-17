"""
Integration tests for API server and Agent server communication
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient

from src.api_server.main import create_app
from src.agent_server.main import AgentServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestAPIAgentIntegration:
    """Test integration between API server and Agent server."""

    async def test_chat_message_flow(self, client: AsyncClient):
        """Test complete chat message flow from API to Agent."""

        # Mock agent server response
        mock_agent_response = {
            "response": "Hello! I'm the SE SME Agent. How can I help you with software engineering?",
            "session_id": "test_session_123",
            "timestamp": "2024-01-01T10:00:00Z",
            "metadata": {"processing_time": 0.5, "tools_used": [], "confidence": 0.9},
        }

        with patch("src.api_server.routers.chat.agent_server") as mock_agent:
            mock_agent.process_message = AsyncMock(return_value=mock_agent_response)

            response = await client.post(
                "/api/v1/chat/message",
                json={
                    "message": "Hello, what can you help me with?",
                    "session_id": "test_session_123",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == mock_agent_response["response"]
        assert data["session_id"] == "test_session_123"

    async def test_tool_execution_flow(self, client: AsyncClient):
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

        with patch("src.api_server.routers.tools.agent_server") as mock_agent:
            mock_agent.execute_tool = AsyncMock(return_value=mock_tool_response)

            response = await client.post(
                "/api/v1/tools/knowledge_retrieval/execute",
                json={
                    "parameters": {"query": "Python programming", "max_results": 5},
                    "session_id": "test_session_123",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["tool_name"] == "knowledge_retrieval"
        assert "result" in data

    async def test_agent_server_unavailable(self, client: AsyncClient):
        """Test API behavior when agent server is unavailable."""

        with patch("src.api_server.routers.chat.agent_server") as mock_agent:
            mock_agent.process_message = AsyncMock(
                side_effect=ConnectionError("Agent server unavailable")
            )

            response = await client.post(
                "/api/v1/chat/message",
                json={"message": "Hello", "session_id": "test_session_123"},
            )

        assert response.status_code == 500
        data = response.json()
        assert "error" in data

    async def test_websocket_agent_integration(self):
        """Test WebSocket integration with agent server."""

        app = create_app()

        mock_agent_response = {
            "response": "WebSocket response from agent",
            "session_id": "ws_session_123",
            "timestamp": "2024-01-01T10:00:00Z",
            "metadata": {},
        }

        with patch("src.api_server.routers.chat.agent_server") as mock_agent:
            mock_agent.process_message = AsyncMock(return_value=mock_agent_response)

            async with AsyncClient(app=app, base_url="http://test") as client:
                # Note: WebSocket testing requires more complex setup
                # This is a simplified test structure
                pass

    async def test_concurrent_requests_to_agent(self, client: AsyncClient):
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

        with patch("src.api_server.routers.chat.agent_server") as mock_agent:
            mock_agent.process_message = mock_process_message

            # Send multiple concurrent requests
            tasks = []
            for i in range(5):
                task = client.post(
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

    async def test_agent_error_handling(self, client: AsyncClient):
        """Test error handling when agent server returns errors."""

        mock_error_response = {
            "response": "I apologize, but I encountered an error processing your request.",
            "session_id": "error_session",
            "timestamp": "2024-01-01T10:00:00Z",
            "metadata": {"error": True, "error_type": "ProcessingError"},
        }

        with patch("src.api_server.routers.chat.agent_server") as mock_agent:
            mock_agent.process_message = AsyncMock(return_value=mock_error_response)

            response = await client.post(
                "/api/v1/chat/message",
                json={"message": "Cause an error", "session_id": "error_session"},
            )

        assert response.status_code == 200  # API should handle agent errors gracefully
        data = response.json()
        assert data["metadata"]["error"] is True

    async def test_session_management_integration(self, client: AsyncClient):
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

        with patch("src.api_server.routers.chat.agent_server") as mock_agent:
            mock_agent.process_message = AsyncMock(
                side_effect=[mock_response_1, mock_response_2]
            )

            # Send first message
            response1 = await client.post(
                "/api/v1/chat/message",
                json={"message": "Hello", "session_id": session_id},
            )

            # Send second message
            response2 = await client.post(
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
        self, client: AsyncClient, redis_client
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

        with patch("src.api_server.routers.chat.agent_server") as mock_agent:
            mock_agent.process_message = mock_process_with_redis

            response = await client.post(
                "/api/v1/chat/message",
                json={"message": "Store this message", "session_id": session_id},
            )

        assert response.status_code == 200

        # Verify data was stored in Redis
        stored_message = await redis_client.get(f"session:{session_id}")
        assert stored_message == "Store this message"


@pytest.mark.integration
@pytest.mark.slow
class TestAPIAgentPerformance:
    """Test performance aspects of API-Agent integration."""

    async def test_response_time_under_load(self, client: AsyncClient):
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

        with patch("src.api_server.routers.chat.agent_server") as mock_agent:
            mock_agent.process_message = mock_process_with_delay

            start_time = time.time()

            # Send 10 concurrent requests
            tasks = []
            for i in range(10):
                task = client.post(
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

    async def test_memory_usage_stability(self, client: AsyncClient):
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

        with patch("src.api_server.routers.chat.agent_server") as mock_agent:
            mock_agent.process_message = mock_process_simple

            # Send many requests to test memory stability
            for i in range(100):
                response = await client.post(
                    "/api/v1/chat/message",
                    json={
                        "message": f"Memory test {i}",
                        "session_id": f"memory_session_{i}",
                    },
                )
                assert response.status_code == 200

                # Force garbage collection every 10 requests
                if i % 10 == 0:
                    gc.collect()

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024
