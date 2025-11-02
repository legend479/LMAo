"""
Minimal test configuration for orchestrator tests to avoid circular imports
"""

import pytest
import sys
from unittest.mock import Mock, AsyncMock

# Mock the problematic RAG pipeline imports to avoid circular dependencies
sys.modules["src.rag_pipeline.main"] = Mock()
sys.modules["src.rag_pipeline.document_processor"] = Mock()
sys.modules["src.rag_pipeline.chunking_strategies"] = Mock()


@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    settings = Mock()
    settings.REDIS_URL = "redis://localhost:6379"
    settings.DATABASE_URL = "sqlite:///test.db"
    return settings


@pytest.fixture
def mock_redis_client():
    """Mock Redis client"""
    client = AsyncMock()
    client.ping.return_value = True
    client.hset.return_value = True
    client.hgetall.return_value = {}
    client.delete.return_value = True
    return client


@pytest.fixture
def mock_checkpointer():
    """Mock LangGraph checkpointer"""
    checkpointer = AsyncMock()
    return checkpointer


@pytest.fixture
def mock_llm_integration():
    """Mock LLM integration"""
    llm = AsyncMock()
    llm.generate_response.return_value = "Mock response"
    return llm
