"""
End-to-end tests for complete system workflows
"""

import pytest
import asyncio
import tempfile
import os
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock

from src.api_server.main import create_app


@pytest.mark.e2e
@pytest.mark.asyncio
class TestCompleteWorkflows:
    """Test complete end-to-end workflows."""

    async def test_complete_chat_workflow(self, client: AsyncClient):
        """Test complete chat workflow from start to finish."""

        session_id = "e2e_chat_session"

        # Mock agent responses for a complete conversation
        mock_responses = [
            {
                "response": "Hello! I'm the SE SME Agent. I can help you with software engineering questions, code generation, and document creation. What would you like to work on?",
                "session_id": session_id,
                "timestamp": "2024-01-01T10:00:00Z",
                "metadata": {"message_count": 1},
            },
            {
                "response": "I'll help you create a Python function for calculating Fibonacci numbers. Let me break this down into steps: 1) Analyze requirements, 2) Generate the function, 3) Add documentation and tests.",
                "session_id": session_id,
                "timestamp": "2024-01-01T10:00:05Z",
                "metadata": {"message_count": 2, "plan_created": True},
            },
            {
                "response": """Here's a Python function to calculate Fibonacci numbers:

```python
def fibonacci(n):
    \"\"\"
    Calculate the nth Fibonacci number using dynamic programming.
    
    Args:
        n (int): The position in the Fibonacci sequence (0-indexed)
        
    Returns:
        int: The nth Fibonacci number
        
    Raises:
        ValueError: If n is negative
    \"\"\"
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    
    # Use dynamic programming for efficiency
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Example usage and tests
if __name__ == "__main__":
    # Test cases
    test_cases = [0, 1, 2, 3, 4, 5, 10]
    for i in test_cases:
        print(f"fibonacci({i}) = {fibonacci(i)}")
```

This implementation uses dynamic programming for O(n) time complexity and O(1) space complexity. Would you like me to explain any part of the code or help you with testing it?""",
                "session_id": session_id,
                "timestamp": "2024-01-01T10:00:10Z",
                "metadata": {"message_count": 3, "code_generated": True},
            },
        ]

        with patch("src.api_server.routers.chat.agent_server") as mock_agent:
            mock_agent.process_message = AsyncMock(side_effect=mock_responses)

            # Step 1: Initial greeting
            response1 = await client.post(
                "/api/v1/chat/message",
                json={
                    "message": "Hello, I need help with programming",
                    "session_id": session_id,
                },
            )

            assert response1.status_code == 200
            data1 = response1.json()
            assert "SE SME Agent" in data1["response"]

            # Step 2: Request specific help
            response2 = await client.post(
                "/api/v1/chat/message",
                json={
                    "message": "Can you help me create a Python function to calculate Fibonacci numbers?",
                    "session_id": session_id,
                },
            )

            assert response2.status_code == 200
            data2 = response2.json()
            assert "Fibonacci" in data2["response"]
            assert data2["metadata"]["plan_created"] is True

            # Step 3: Get the generated code
            response3 = await client.post(
                "/api/v1/chat/message",
                json={
                    "message": "Please generate the code now",
                    "session_id": session_id,
                },
            )

            assert response3.status_code == 200
            data3 = response3.json()
            assert "def fibonacci" in data3["response"]
            assert "```python" in data3["response"]
            assert data3["metadata"]["code_generated"] is True

    async def test_document_processing_workflow(self, client: AsyncClient, temp_dir):
        """Test complete document processing workflow."""

        # Create a sample document
        doc_path = os.path.join(temp_dir, "sample_doc.txt")
        with open(doc_path, "w") as f:
            f.write(
                """
            # Python Programming Guide
            
            Python is a versatile programming language used for:
            - Web development with Django and Flask
            - Data science with pandas and numpy
            - Machine learning with scikit-learn and TensorFlow
            
            ## Basic Syntax
            
            Variables in Python are dynamically typed:
            ```python
            name = "Alice"
            age = 30
            is_student = False
            ```
            """
            )

        # Mock RAG pipeline responses
        mock_ingest_response = {
            "document_id": "doc_123",
            "chunks_processed": 5,
            "processing_time": 1.2,
            "status": "success",
        }

        mock_search_response = {
            "query": "Python web development",
            "results": [
                {
                    "content": "Python is a versatile programming language used for web development with Django and Flask",
                    "score": 0.95,
                    "chunk_id": "chunk_1",
                    "document_id": "doc_123",
                }
            ],
            "total_results": 1,
            "processing_time": 0.3,
        }

        with patch("src.api_server.routers.documents.rag_pipeline") as mock_rag:
            mock_rag.ingest_document = AsyncMock(return_value=mock_ingest_response)
            mock_rag.search = AsyncMock(return_value=mock_search_response)

            # Step 1: Upload document
            with open(doc_path, "rb") as f:
                response1 = await client.post(
                    "/api/v1/documents/upload",
                    files={"file": ("sample_doc.txt", f, "text/plain")},
                    data={"metadata": '{"category": "tutorial"}'},
                )

            assert response1.status_code == 200
            data1 = response1.json()
            assert data1["status"] == "success"
            assert data1["document_id"] == "doc_123"

            # Step 2: Search the uploaded document
            response2 = await client.post(
                "/api/v1/documents/search",
                json={"query": "Python web development", "max_results": 5},
            )

            assert response2.status_code == 200
            data2 = response2.json()
            assert len(data2["results"]) == 1
            assert "Django and Flask" in data2["results"][0]["content"]

    async def test_tool_execution_workflow(self, client: AsyncClient):
        """Test complete tool execution workflow."""

        # Mock tool responses
        mock_knowledge_response = {
            "tool_name": "knowledge_retrieval",
            "result": {
                "query": "Python design patterns",
                "results": [
                    {
                        "content": "The Singleton pattern ensures a class has only one instance...",
                        "score": 0.92,
                        "source": "design_patterns.md",
                    },
                    {
                        "content": "The Factory pattern creates objects without specifying exact classes...",
                        "score": 0.88,
                        "source": "factory_pattern.md",
                    },
                ],
                "total_results": 2,
            },
            "status": "success",
            "execution_time": 0.8,
        }

        mock_doc_gen_response = {
            "tool_name": "document_generation",
            "result": {
                "filename": "design_patterns_guide.docx",
                "file_path": "/tmp/design_patterns_guide.docx",
                "format": "docx",
                "content_analysis": {
                    "sections": 3,
                    "word_count": 1250,
                    "complexity": "medium",
                },
            },
            "status": "success",
            "execution_time": 2.1,
        }

        with patch("src.api_server.routers.tools.agent_server") as mock_agent:
            mock_agent.execute_tool = AsyncMock(
                side_effect=[mock_knowledge_response, mock_doc_gen_response]
            )

            # Step 1: Retrieve knowledge about design patterns
            response1 = await client.post(
                "/api/v1/tools/knowledge_retrieval/execute",
                json={
                    "parameters": {"query": "Python design patterns", "max_results": 5},
                    "session_id": "tool_session_123",
                },
            )

            assert response1.status_code == 200
            data1 = response1.json()
            assert data1["status"] == "success"
            assert len(data1["result"]["results"]) == 2
            assert "Singleton" in data1["result"]["results"][0]["content"]

            # Step 2: Generate document based on retrieved knowledge
            response2 = await client.post(
                "/api/v1/tools/document_generation/execute",
                json={
                    "parameters": {
                        "content": "# Python Design Patterns Guide\n\nBased on retrieved knowledge...",
                        "format": "docx",
                        "template": "default",
                    },
                    "session_id": "tool_session_123",
                },
            )

            assert response2.status_code == 200
            data2 = response2.json()
            assert data2["status"] == "success"
            assert data2["result"]["format"] == "docx"
            assert "design_patterns_guide.docx" in data2["result"]["filename"]

    async def test_error_recovery_workflow(self, client: AsyncClient):
        """Test system behavior during error conditions and recovery."""

        session_id = "error_recovery_session"

        # Mock responses including errors and recovery
        mock_responses = [
            {
                "response": "I'll help you with that task.",
                "session_id": session_id,
                "timestamp": "2024-01-01T10:00:00Z",
                "metadata": {"message_count": 1},
            },
            # Simulate an error response
            {
                "response": "I apologize, but I encountered an error processing your request. Let me try a different approach.",
                "session_id": session_id,
                "timestamp": "2024-01-01T10:00:05Z",
                "metadata": {
                    "message_count": 2,
                    "error": True,
                    "error_type": "ToolExecutionError",
                    "recovery_attempted": True,
                },
            },
            # Recovery response
            {
                "response": "I've successfully recovered and can now help you with your request. Here's the information you needed...",
                "session_id": session_id,
                "timestamp": "2024-01-01T10:00:10Z",
                "metadata": {
                    "message_count": 3,
                    "recovered": True,
                    "fallback_used": True,
                },
            },
        ]

        with patch("src.api_server.routers.chat.agent_server") as mock_agent:
            mock_agent.process_message = AsyncMock(side_effect=mock_responses)

            # Step 1: Initial request
            response1 = await client.post(
                "/api/v1/chat/message",
                json={
                    "message": "Help me with a complex task",
                    "session_id": session_id,
                },
            )

            assert response1.status_code == 200

            # Step 2: Request that causes error
            response2 = await client.post(
                "/api/v1/chat/message",
                json={
                    "message": "Execute a problematic operation",
                    "session_id": session_id,
                },
            )

            assert response2.status_code == 200
            data2 = response2.json()
            assert data2["metadata"]["error"] is True
            assert data2["metadata"]["recovery_attempted"] is True

            # Step 3: Verify recovery
            response3 = await client.post(
                "/api/v1/chat/message",
                json={"message": "Try again please", "session_id": session_id},
            )

            assert response3.status_code == 200
            data3 = response3.json()
            assert data3["metadata"]["recovered"] is True
            assert "successfully recovered" in data3["response"]

    async def test_multi_user_concurrent_workflow(self, client: AsyncClient):
        """Test concurrent workflows from multiple users."""

        # Define multiple user sessions
        users = [
            {"session_id": "user1_session", "name": "Alice"},
            {"session_id": "user2_session", "name": "Bob"},
            {"session_id": "user3_session", "name": "Charlie"},
        ]

        async def user_workflow(user_info):
            """Simulate a complete workflow for one user."""
            session_id = user_info["session_id"]
            name = user_info["name"]

            # Mock responses for this user
            mock_responses = [
                {
                    "response": f"Hello {name}! How can I help you today?",
                    "session_id": session_id,
                    "timestamp": "2024-01-01T10:00:00Z",
                    "metadata": {"user": name},
                },
                {
                    "response": f"I'll help you with that, {name}. Let me process your request.",
                    "session_id": session_id,
                    "timestamp": "2024-01-01T10:00:05Z",
                    "metadata": {"user": name, "processing": True},
                },
            ]

            with patch("src.api_server.routers.chat.agent_server") as mock_agent:
                mock_agent.process_message = AsyncMock(side_effect=mock_responses)

                # User greeting
                response1 = await client.post(
                    "/api/v1/chat/message",
                    json={"message": f"Hello, I'm {name}", "session_id": session_id},
                )

                # User request
                response2 = await client.post(
                    "/api/v1/chat/message",
                    json={
                        "message": "Help me with a programming task",
                        "session_id": session_id,
                    },
                )

                return [response1, response2]

        # Run workflows concurrently for all users
        tasks = [user_workflow(user) for user in users]
        results = await asyncio.gather(*tasks)

        # Verify all workflows completed successfully
        for user_results in results:
            for response in user_results:
                assert response.status_code == 200

        # Verify responses are user-specific
        for i, user_results in enumerate(results):
            user_name = users[i]["name"]
            for response in user_results:
                data = response.json()
                assert user_name in data["response"]
                assert data["metadata"]["user"] == user_name


@pytest.mark.e2e
@pytest.mark.slow
class TestPerformanceWorkflows:
    """Test performance aspects of complete workflows."""

    async def test_high_throughput_workflow(self, client: AsyncClient):
        """Test system performance under high throughput."""

        import time

        async def mock_fast_response(message, session_id, user_id=None):
            return {
                "response": f"Quick response to: {message[:20]}...",
                "session_id": session_id,
                "timestamp": "2024-01-01T10:00:00Z",
                "metadata": {"fast_mode": True},
            }

        with patch("src.api_server.routers.chat.agent_server") as mock_agent:
            mock_agent.process_message = mock_fast_response

            start_time = time.time()

            # Send 50 requests concurrently
            tasks = []
            for i in range(50):
                task = client.post(
                    "/api/v1/chat/message",
                    json={
                        "message": f"High throughput test message {i}",
                        "session_id": f"throughput_session_{i}",
                    },
                )
                tasks.append(task)

            responses = await asyncio.gather(*tasks)

            end_time = time.time()
            total_time = end_time - start_time

        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)

        # Calculate throughput
        throughput = len(responses) / total_time

        # Should handle at least 10 requests per second
        assert throughput >= 10.0

    async def test_large_document_workflow(self, client: AsyncClient, temp_dir):
        """Test workflow with large document processing."""

        # Create a large document
        large_doc_path = os.path.join(temp_dir, "large_doc.txt")
        with open(large_doc_path, "w") as f:
            # Write 10MB of content
            content = "This is a large document for testing. " * 1000
            for i in range(250):  # Approximately 10MB
                f.write(f"Section {i}: {content}\n")

        mock_ingest_response = {
            "document_id": "large_doc_123",
            "chunks_processed": 500,
            "processing_time": 15.2,
            "status": "success",
            "file_size_mb": 10.5,
        }

        with patch("src.api_server.routers.documents.rag_pipeline") as mock_rag:
            mock_rag.ingest_document = AsyncMock(return_value=mock_ingest_response)

            start_time = time.time()

            with open(large_doc_path, "rb") as f:
                response = await client.post(
                    "/api/v1/documents/upload",
                    files={"file": ("large_doc.txt", f, "text/plain")},
                    data={"metadata": '{"category": "large_test"}'},
                )

            end_time = time.time()
            processing_time = end_time - start_time

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["chunks_processed"] == 500

        # Should complete within reasonable time (30 seconds)
        assert processing_time < 30.0


@pytest.mark.e2e
@pytest.mark.requires_db
@pytest.mark.requires_redis
@pytest.mark.requires_es
class TestFullSystemWorkflows:
    """Test workflows requiring full system stack."""

    async def test_complete_system_integration(self, client: AsyncClient):
        """Test complete system integration with all components."""

        # This test would require actual database, Redis, and Elasticsearch
        # For now, we'll mock the components but structure the test
        # as if they were real

        session_id = "full_system_session"

        # Mock all system components
        with patch("src.api_server.routers.chat.agent_server") as mock_agent:
            with patch("src.api_server.routers.documents.rag_pipeline") as mock_rag:
                with patch("src.api_server.routers.tools.agent_server") as mock_tools:

                    # Setup comprehensive mocks
                    mock_agent.process_message = AsyncMock(
                        return_value={
                            "response": "System integration test successful",
                            "session_id": session_id,
                            "timestamp": "2024-01-01T10:00:00Z",
                            "metadata": {"system_test": True},
                        }
                    )

                    # Test the complete flow
                    response = await client.post(
                        "/api/v1/chat/message",
                        json={
                            "message": "Test full system integration",
                            "session_id": session_id,
                        },
                    )

        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["system_test"] is True
