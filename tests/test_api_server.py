"""
API Server Tests
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    """Test health check endpoint."""
    response = await client.get("/health/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_readiness_check(client: AsyncClient):
    """Test readiness check endpoint."""
    response = await client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"


@pytest.mark.asyncio
async def test_liveness_check(client: AsyncClient):
    """Test liveness check endpoint."""
    response = await client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"


@pytest.mark.asyncio
async def test_chat_message(client: AsyncClient):
    """Test chat message endpoint."""
    message_data = {"message": "Hello, SE SME Agent!", "session_id": "test_session"}

    response = await client.post("/api/v1/chat/message", json=message_data)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["session_id"] == "test_session"


@pytest.mark.asyncio
async def test_list_tools(client: AsyncClient):
    """Test list tools endpoint."""
    response = await client.get("/api/v1/tools/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

    # Should have at least the knowledge retrieval tool
    tool_names = [tool["name"] for tool in data]
    assert "knowledge_retrieval" in tool_names
