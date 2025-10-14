"""
Example Python code for testing code-aware chunking
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class User:
    """User data model"""

    id: int
    name: str
    email: str
    is_active: bool = True

    def __post_init__(self):
        if not self.email or "@" not in self.email:
            raise ValueError("Invalid email address")

    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary"""
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

    async def list_users(self, active_only: bool = True) -> List[User]:
        """List all users"""
        users = list(self.users.values())

        if active_only:
            users = [user for user in users if user.is_active]

        return users

    async def update_user(self, user_id: int, **kwargs) -> Optional[User]:
        """Update user information"""
        user = self.users.get(user_id)
        if not user:
            return None

        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)

        return user

    async def delete_user(self, user_id: int) -> bool:
        """Delete a user"""
        if user_id in self.users:
            del self.users[user_id]
            return True
        return False


async def main():
    """Main function demonstrating user service"""
    service = UserService()

    # Create some users
    user1 = await service.create_user("Alice Smith", "alice@example.com")
    user2 = await service.create_user("Bob Johnson", "bob@example.com")

    print(f"Created users: {user1.name}, {user2.name}")

    # List users
    users = await service.list_users()
    print(f"Total active users: {len(users)}")

    # Update user
    updated_user = await service.update_user(user1.id, name="Alice Brown")
    if updated_user:
        print(f"Updated user name to: {updated_user.name}")

    # Get specific user
    user = await service.get_user(user2.id)
    if user:
        print(f"Found user: {user.to_dict()}")


if __name__ == "__main__":
    asyncio.run(main())
