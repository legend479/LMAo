import asyncio
import httpx
from datetime import datetime
from src.api_server.auth.jwt_manager import jwt_manager


async def test_chat_response():
    # Create test user
    test_user = {
        "id": "test_user_123",
        "email": "test@example.com",
        "full_name": "Test User",
        "roles": ["user"],
        "is_active": True,
        "created_at": datetime.utcnow(),
        "last_login": datetime.utcnow(),
    }

    # Create JWT token
    token_result = await jwt_manager.create_token_pair(
        user_id=test_user["id"], email=test_user["email"], roles=test_user["roles"]
    )

    headers = {
        "Authorization": f"Bearer {token_result['access_token']}",
        "Content-Type": "application/json",
    }

    # Create test client
    from src.api_server.main import create_app

    app = create_app()

    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        # Test the simple test endpoint first
        test_resp = await client.get("/api/v1/chat/test")
        print(f"Test endpoint status: {test_resp.status_code}")
        print(f"Test endpoint content: {test_resp.text}")

        response = await client.post(
            "/api/v1/chat/message",
            json={"message": "Hello test", "session_id": "test_session"},
            headers=headers,
        )

        print(f"Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Content length: {len(response.content)}")
        print(f"Raw content: {response.content}")
        print(f"Text: {response.text}")

        if response.text:
            try:
                data = response.json()
                print(f"Parsed JSON: {data}")
            except Exception as e:
                print(f"JSON decode error: {e}")
        else:
            print("Empty response text!")


if __name__ == "__main__":
    asyncio.run(test_chat_response())
