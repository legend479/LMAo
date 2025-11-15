"""Simple test to debug chat endpoint"""

import asyncio
from httpx import AsyncClient
from src.api_server.main import create_app
from unittest.mock import patch, AsyncMock
from datetime import datetime
import os


async def test_simple():
    """Simple test"""
    # Set testing environment
    os.environ["ENVIRONMENT"] = "testing"

    # Clear settings cache
    from src.shared.config import get_settings

    get_settings.cache_clear()

    app = create_app()

    # Mock user
    def get_mock_user():
        from src.api_server.routers.auth import User

        return User(
            id="test_user_123",
            email="test@example.com",
            full_name="Test User",
            roles=["user"],
            is_active=True,
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow(),
        )

    # Override auth
    from src.api_server.routers.auth import get_current_active_user

    app.dependency_overrides[get_current_active_user] = get_mock_user

    # Mock agent response
    mock_response = {
        "response": "Hello!",
        "session_id": "test_123",
        "timestamp": "2024-01-01T10:00:00Z",
        "metadata": {},
    }

    with patch(
        "src.api_server.routers.chat.get_agent_client", new_callable=AsyncMock
    ) as mock_get_client:
        mock_agent = AsyncMock()
        mock_agent.process_message = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_agent

        async with AsyncClient(
            app=app, base_url="http://test", follow_redirects=True
        ) as client:
            response = await client.post(
                "/api/v1/chat/message",
                json={"message": "Hello", "session_id": "test_123"},
            )

            print(f"Status: {response.status_code}")
            print(f"Headers: {response.headers}")

            # Try reading response different ways
            print(f"\n=== Trying response.read() ===")
            try:
                content = await response.aread()
                print(f"aread() content: {content}")
            except Exception as e:
                print(f"aread() error: {e}")

            print(f"\n=== Trying response.content ===")
            print(f"Content: {response.content}")
            print(f"Content length: {len(response.content)}")

            print(f"\n=== Trying response.text ===")
            print(f"Text: '{response.text}'")

            print(f"\n=== Trying response.json() ===")
            try:
                data = response.json()
                print(f"JSON: {data}")
            except Exception as e:
                print(f"JSON error: {e}")


if __name__ == "__main__":
    asyncio.run(test_simple())
