"""
Python Code Example - User Management System
"""

import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class User:
    """User data model"""

    id: int
    name: str
    email: str
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "is_active": self.is_active,
        }


class UserService:
    """Service for managing users"""

    def __init__(self):
        self.users: Dict[int, User] = {}
        self._next_id = 1

    async def create_user(self, name: str, email: str) -> User:
        """Create a new user"""
        user = User(id=self._next_id, name=name, email=email)
        self.users[user.id] = user
        self._next_id += 1
        return user

    async def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)

    async def list_users(self) -> list:
        """List all active users"""
        return [user for user in self.users.values() if user.is_active]


async def main():
    """Demo the user service"""
    service = UserService()

    # Create users
    user1 = await service.create_user("Alice", "alice@example.com")
    user2 = await service.create_user("Bob", "bob@example.com")

    print(f"Created: {user1.name}, {user2.name}")

    # List users
    users = await service.list_users()
    print(f"Total users: {len(users)}")


if __name__ == "__main__":
    asyncio.run(main())
