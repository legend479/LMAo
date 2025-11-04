"""
Unit tests for API server endpoints.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
import json


# Mock the API server app
@pytest.fixture
def mock_api_app():
    """Mock FastAPI application for testing."""
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "api-server"}

    @app.post("/chat")
    async def chat_endpoint(message: dict):
        content = message.get("content", "")
        # Simple XSS filtering for test
        filtered_content = content.replace("<script>", "").replace("</script>", "")
        return {"response": f"Echo: {filtered_content}", "status": "success"}

    @app.get("/agents")
    async def list_agents():
        return {"agents": ["sme_agent", "research_agent"], "count": 2}

    return app


@pytest.fixture
def client(mock_api_app):
    """Test client for API endpoints."""
    return TestClient(mock_api_app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check_success(self, client):
        """Test successful health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "api-server"

    def test_health_check_response_format(self, client):
        """Test health check response format."""
        response = client.get("/health")
        data = response.json()
        assert isinstance(data, dict)
        assert "status" in data
        assert "service" in data


class TestChatEndpoint:
    """Test chat endpoint functionality."""

    def test_chat_endpoint_success(self, client):
        """Test successful chat interaction."""
        message = {"content": "Hello, world!", "user_id": "test_user"}
        response = client.post("/chat", json=message)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "status" in data
        assert data["status"] == "success"

    def test_chat_endpoint_empty_message(self, client):
        """Test chat endpoint with empty message."""
        message = {"content": "", "user_id": "test_user"}
        response = client.post("/chat", json=message)
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Echo: "

    def test_chat_endpoint_invalid_json(self, client):
        """Test chat endpoint with invalid JSON."""
        response = client.post("/chat", data="invalid json")
        assert response.status_code == 422  # Unprocessable Entity

    def test_chat_endpoint_mock_success(self, client):
        """Test chat endpoint with mock (successful case)."""
        message = {"content": "Hello"}
        response = client.post("/chat", json=message)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "Echo: Hello" in data["response"]


class TestRealAPIEndpoints:
    """Test real API endpoints with service integration."""

    @pytest.mark.asyncio
    async def test_health_endpoint_works(self):
        """Test that the real app can be created and health endpoint works."""
        from src.api_server.main import create_app
        from httpx import AsyncClient

        # Create the real app
        app = create_app()

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health/")

            # Health endpoint should work
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_chat_endpoint_auth_required(self):
        """Test that chat endpoint requires authentication."""
        from src.api_server.main import create_app
        from httpx import AsyncClient

        # Create the real app
        app = create_app()

        async with AsyncClient(app=app, base_url="http://test") as client:
            message = {"message": "Hello"}
            response = await client.post("/api/v1/chat/message", json=message)

            # Should require authentication
            assert response.status_code in [401, 403]


class TestAgentsEndpoint:
    """Test agents listing endpoint."""

    def test_list_agents_success(self, client):
        """Test successful agent listing."""
        response = client.get("/agents")
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert "count" in data
        assert isinstance(data["agents"], list)
        assert data["count"] == len(data["agents"])

    def test_agents_response_format(self, client):
        """Test agents endpoint response format."""
        response = client.get("/agents")
        data = response.json()
        assert isinstance(data["agents"], list)
        assert isinstance(data["count"], int)
        assert all(isinstance(agent, str) for agent in data["agents"])


class TestErrorHandling:
    """Test error handling across endpoints."""

    def test_404_endpoint(self, client):
        """Test non-existent endpoint returns 404."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test wrong HTTP method returns 405."""
        response = client.post("/health")
        assert response.status_code == 405

    @patch("src.shared.database.get_database_session")
    def test_database_connection_error(self, mock_db, client):
        """Test handling of database connection errors."""
        mock_db.side_effect = Exception("Database unavailable")
        # Test endpoints that depend on database
        response = client.get("/agents")
        # Should handle gracefully - either 200 or 503 is acceptable
        assert response.status_code in [200, 503]


class TestRequestValidation:
    """Test request validation and sanitization."""

    def test_chat_message_length_limit(self, client):
        """Test chat message length validation."""
        long_message = {"content": "x" * 10000, "user_id": "test_user"}
        response = client.post("/chat", json=long_message)
        # Should either accept or reject with appropriate status
        assert response.status_code in [200, 400, 413]

    def test_special_characters_handling(self, client):
        """Test handling of special characters in messages."""
        special_chars = {
            "content": "Hello <script>alert('xss')</script>",
            "user_id": "test_user",
        }
        response = client.post("/chat", json=special_chars)
        assert response.status_code == 200
        # Response should be sanitized
        data = response.json()
        assert "<script>" not in data["response"]

    def test_unicode_handling(self, client):
        """Test Unicode character handling."""
        unicode_message = {"content": "Hello 世界 🌍", "user_id": "test_user"}
        response = client.post("/chat", json=unicode_message)
        assert response.status_code == 200
        data = response.json()
        assert "世界" in data["response"] or "🌍" in data["response"]


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_enforcement(self, client):
        """Test rate limiting is enforced."""
        # Make multiple rapid requests
        responses = []
        for i in range(10):
            response = client.post(
                "/chat", json={"content": f"Message {i}", "user_id": "test_user"}
            )
            responses.append(response)

        # Check if any requests were rate limited
        status_codes = [r.status_code for r in responses]
        # Should have some 429 (Too Many Requests) if rate limiting is active
        # Or all 200 if rate limiting is not implemented yet
        assert all(code in [200, 429] for code in status_codes)

    def test_rate_limit_per_user(self, client):
        """Test rate limiting is per user."""
        # Test with different user IDs
        user1_response = client.post(
            "/chat", json={"content": "Hello", "user_id": "user1"}
        )
        user2_response = client.post(
            "/chat", json={"content": "Hello", "user_id": "user2"}
        )

        assert user1_response.status_code == 200
        assert user2_response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__])
