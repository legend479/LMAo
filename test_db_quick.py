#!/usr/bin/env python3

import sys
import os

sys.path.append("src")

from src.shared.database.operations import UserOperations, MetricsOperations
from src.shared.database.connection import DatabaseManager
from unittest.mock import patch, MagicMock
import tempfile


def test_quick():
    # Create temporary database
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)

    try:
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.debug = False

        mock_db_config = {"url": f"sqlite:///{db_path}", "echo": False}

        with (
            patch(
                "src.shared.database.connection.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "src.shared.database.connection.get_database_config",
                return_value=mock_db_config,
            ),
        ):

            # Initialize database
            from src.shared.database.connection import get_database_manager

            manager = get_database_manager()
            import asyncio

            asyncio.run(manager.initialize())

            # Test user operations
            print("Testing user operations...")
            user = UserOperations.create_user(
                username="testuser", email="test@example.com", full_name="Test User"
            )
            print(f"Created user: {user.username}, {user.email}")

            # Test retrieval
            retrieved_user = UserOperations.get_user_by_id(user.id)
            print(f"Retrieved user: {retrieved_user.username}, {retrieved_user.email}")

            # Test metrics
            print("Testing metrics operations...")
            metric = MetricsOperations.record_metric(
                metric_name="test_metric", value=42.0, metric_type="gauge", unit="count"
            )
            print(f"Created metric: {metric.metric_name} = {metric.value}")

            # Get metrics
            metrics = MetricsOperations.get_metrics("test_metric")
            print(f"Retrieved {len(metrics)} metrics")
            if metrics:
                print(f"First metric value: {metrics[0].value}")

            print("All tests passed!")

    finally:
        os.unlink(db_path)


if __name__ == "__main__":
    test_quick()
