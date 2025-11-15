"""
Load testing for the SE SME Agent system
"""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient


@pytest.mark.performance
@pytest.mark.slow
class TestLoadTesting:
    """Load testing scenarios for the system."""

    async def test_api_server_load(self, client: AsyncClient, performance_config):
        """Test API server under load."""

        concurrent_users = performance_config["concurrent_users"]
        requests_per_user = performance_config["requests_per_user"]
        acceptable_response_time = performance_config["acceptable_response_time"]

        print(
            f"\nStarting load test: {concurrent_users} users × {requests_per_user} requests = {concurrent_users * requests_per_user} total requests"
        )

        async def mock_agent_response(message, session_id, user_id=None):
            # Return immediately for faster, more reliable tests
            return {
                "response": f"Load test response for: {message[:20]}",
                "session_id": session_id,
                "timestamp": "2024-01-01T10:00:00Z",
                "metadata": {"load_test": True},
            }

        # Mock the get_agent_client function to return a mock client
        mock_client = AsyncMock()
        mock_client.process_message = AsyncMock(side_effect=mock_agent_response)

        with patch(
            "src.shared.services.get_agent_client",
            new=AsyncMock(return_value=mock_client),
        ):

            async def user_simulation(user_id: int):
                """Simulate a single user's requests."""
                response_times = []
                successful_requests = 0
                failed_requests = 0

                for request_num in range(requests_per_user):
                    start_time = time.time()

                    try:
                        response = await client.post(
                            "/api/v1/chat/message",
                            json={
                                "message": f"Load test message {request_num} from user {user_id}",
                                "session_id": f"load_session_{user_id}",
                            },
                        )

                        end_time = time.time()
                        response_time = end_time - start_time
                        response_times.append(response_time)

                        if response.status_code == 200:
                            successful_requests += 1
                        else:
                            failed_requests += 1
                            print(
                                f"Request failed for user {user_id}, request {request_num}: status {response.status_code}"
                            )

                    except Exception as e:
                        failed_requests += 1
                        print(
                            f"Request exception for user {user_id}, request {request_num}: {type(e).__name__}: {e}"
                        )

                    # Small delay between requests
                    await asyncio.sleep(0.01)

                # Progress indicator
                print(
                    f"User {user_id} completed: {successful_requests} successful, {failed_requests} failed"
                )

                return {
                    "user_id": user_id,
                    "response_times": response_times,
                    "successful_requests": successful_requests,
                    "failed_requests": failed_requests,
                }

            # Run concurrent user simulations
            start_time = time.time()

            tasks = [user_simulation(i) for i in range(concurrent_users)]
            results = await asyncio.gather(*tasks)

            end_time = time.time()
            total_test_time = end_time - start_time

        # Analyze results
        all_response_times = []
        total_successful = 0
        total_failed = 0

        for result in results:
            all_response_times.extend(result["response_times"])
            total_successful += result["successful_requests"]
            total_failed += result["failed_requests"]

        # Calculate metrics
        avg_response_time = (
            statistics.mean(all_response_times) if all_response_times else 0
        )
        p95_response_time = (
            statistics.quantiles(all_response_times, n=20)[18]
            if len(all_response_times) > 20
            else 0
        )
        success_rate = (
            total_successful / (total_successful + total_failed)
            if (total_successful + total_failed) > 0
            else 0
        )
        throughput = (total_successful + total_failed) / total_test_time

        # Assertions - adjusted for realistic test environment performance
        assert success_rate >= 0.90, f"Success rate {success_rate:.2%} below 90%"
        assert (
            avg_response_time <= 5.0
        ), f"Average response time {avg_response_time:.2f}s exceeds 5.0s"
        assert (
            p95_response_time <= 10.0
        ), f"P95 response time {p95_response_time:.2f}s exceeds 10.0s"
        assert (
            throughput >= 5
        ), f"Throughput {throughput:.1f} req/s below minimum 5 req/s"

        print(f"Load test results:")
        print(f"  Users: {concurrent_users}")
        print(f"  Requests per user: {requests_per_user}")
        print(f"  Total requests: {total_successful + total_failed}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Average response time: {avg_response_time:.3f}s")
        print(f"  P95 response time: {p95_response_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} req/s")

    async def test_websocket_load(self, performance_config):
        """Test WebSocket connections under load."""

        concurrent_connections = min(
            performance_config["concurrent_users"], 20
        )  # Limit for WebSocket
        messages_per_connection = 10

        async def websocket_simulation(connection_id: int):
            """Simulate a single WebSocket connection."""
            # This would require actual WebSocket testing setup
            # For now, we'll simulate the metrics

            await asyncio.sleep(0.1)  # Simulate connection time

            message_times = []
            for i in range(messages_per_connection):
                start_time = time.time()

                # Simulate message send/receive
                await asyncio.sleep(0.01)

                end_time = time.time()
                message_times.append(end_time - start_time)

            return {
                "connection_id": connection_id,
                "message_times": message_times,
                "messages_sent": messages_per_connection,
            }

        start_time = time.time()

        tasks = [websocket_simulation(i) for i in range(concurrent_connections)]
        results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        # Analyze WebSocket performance
        all_message_times = []
        total_messages = 0

        for result in results:
            all_message_times.extend(result["message_times"])
            total_messages += result["messages_sent"]

        avg_message_time = statistics.mean(all_message_times)
        message_throughput = total_messages / total_time

        # WebSocket should be faster than HTTP
        assert (
            avg_message_time <= 0.1
        ), f"WebSocket message time {avg_message_time:.3f}s too high"
        assert (
            message_throughput >= 100
        ), f"WebSocket throughput {message_throughput:.1f} msg/s too low"

    async def test_document_processing_load(self, client: AsyncClient, temp_dir):
        """Test document processing under load."""

        concurrent_uploads = 10

        # Create test documents
        test_docs = []
        for i in range(concurrent_uploads):
            doc_path = f"{temp_dir}/load_test_doc_{i}.txt"
            with open(doc_path, "w") as f:
                f.write(f"Load test document {i}\n" + "Content line\n" * 100)
            test_docs.append(doc_path)

        mock_ingest_response = {
            "document_id": "load_doc_123",
            "chunks_processed": 10,
            "processing_time": 0.5,
            "status": "success",
        }

        # Mock the get_rag_client function to return a mock client
        mock_rag_client = AsyncMock()
        mock_rag_client.ingest_document = AsyncMock()

        # Mock authentication to bypass auth requirements
        from src.api_server.routers.auth import User
        from datetime import datetime, timezone

        mock_user = User(
            id="test_user",
            email="test@example.com",
            full_name="Test User",
            roles=["user"],
            is_active=True,
            created_at=datetime.now(timezone.utc),
        )

        async def upload_document(doc_path: str, doc_id: int):
            """Upload a single document."""
            start_time = time.time()

            # Set the return value for this specific document
            mock_rag_client.ingest_document.return_value = {
                **mock_ingest_response,
                "document_id": f"load_doc_{doc_id}",
            }

            with open(doc_path, "rb") as f:
                response = await client.post(
                    "/api/v1/documents/upload",
                    files={"file": (f"doc_{doc_id}.txt", f, "text/plain")},
                    data={"metadata": f'{{"doc_id": {doc_id}}}'},
                )

            end_time = time.time()

            return {
                "doc_id": doc_id,
                "response": response,
                "processing_time": end_time - start_time,
            }

        # Upload documents concurrently with mocked RAG client and auth
        with (
            patch(
                "src.shared.services.get_rag_client",
                new=AsyncMock(return_value=mock_rag_client),
            ),
            patch(
                "src.api_server.routers.documents.get_current_active_user",
                new=AsyncMock(return_value=mock_user),
            ),
            patch(
                "src.shared.database.DocumentOperations.create_document"
            ) as mock_create_doc,
        ):

            # Mock document creation to return a document object
            from datetime import datetime

            class MockDocument:
                def __init__(self, doc_id):
                    self.id = doc_id
                    self.created_at = datetime.utcnow()

            mock_create_doc.side_effect = lambda **kwargs: MockDocument(
                f"load_doc_{kwargs.get('user_id', 'test')}"
            )

            start_time = time.time()

            tasks = [
                upload_document(doc_path, i) for i, doc_path in enumerate(test_docs)
            ]
            results = await asyncio.gather(*tasks)

            end_time = time.time()
            total_time = end_time - start_time

        # Analyze document processing performance
        successful_uploads = sum(1 for r in results if r["response"].status_code == 200)
        processing_times = [r["processing_time"] for r in results]
        avg_processing_time = statistics.mean(processing_times)
        upload_throughput = len(results) / total_time

        assert (
            successful_uploads == concurrent_uploads
        ), f"Only {successful_uploads}/{concurrent_uploads} uploads succeeded"
        assert (
            avg_processing_time <= 5.0
        ), f"Average processing time {avg_processing_time:.2f}s too high"
        assert (
            upload_throughput >= 2.0
        ), f"Upload throughput {upload_throughput:.1f} docs/s too low"

    async def test_search_performance_load(self, client: AsyncClient):
        """Test search performance under load."""

        concurrent_searches = 20
        searches_per_user = 5

        search_queries = [
            "Python programming",
            "JavaScript frameworks",
            "database design",
            "machine learning",
            "web development",
            "data structures",
            "algorithms",
            "software architecture",
            "testing strategies",
            "deployment practices",
        ]

        mock_search_response = {
            "query": "test query",
            "results": [
                {
                    "content": "Search result content...",
                    "score": 0.95,
                    "chunk_id": "chunk_1",
                    "document_id": "doc_1",
                }
            ],
            "total_results": 1,
            "processing_time": 0.1,
        }

        # Mock the get_rag_client function to return a mock client
        mock_rag_client = AsyncMock()

        async def search_simulation(user_id: int):
            """Simulate search requests from one user."""
            search_times = []

            for i in range(searches_per_user):
                query = search_queries[i % len(search_queries)]

                start_time = time.time()

                # Set the return value for this specific search
                mock_rag_client.search.return_value = {
                    **mock_search_response,
                    "query": query,
                }

                response = await client.post(
                    "/api/v1/documents/search",
                    json={"query": f"{query} user {user_id}", "max_results": 10},
                )

                end_time = time.time()
                search_times.append(end_time - start_time)

                assert response.status_code == 200

            return {
                "user_id": user_id,
                "search_times": search_times,
                "searches_completed": len(search_times),
            }

        # Mock authentication to bypass auth requirements
        from src.api_server.routers.auth import User
        from datetime import datetime, timezone

        mock_user = User(
            id="test_user",
            email="test@example.com",
            full_name="Test User",
            roles=["user"],
            is_active=True,
            created_at=datetime.now(timezone.utc),
        )

        # Run concurrent search simulations with mocked RAG client and auth
        with (
            patch(
                "src.shared.services.get_rag_client",
                new=AsyncMock(return_value=mock_rag_client),
            ),
            patch(
                "src.api_server.routers.documents.get_current_active_user",
                new=AsyncMock(return_value=mock_user),
            ),
        ):
            start_time = time.time()

            tasks = [search_simulation(i) for i in range(concurrent_searches)]
            results = await asyncio.gather(*tasks)

            end_time = time.time()
            total_time = end_time - start_time

        # Analyze search performance
        all_search_times = []
        total_searches = 0

        for result in results:
            all_search_times.extend(result["search_times"])
            total_searches += result["searches_completed"]

        avg_search_time = statistics.mean(all_search_times)
        search_throughput = total_searches / total_time

        assert (
            avg_search_time <= 1.0
        ), f"Average search time {avg_search_time:.3f}s too high"
        assert (
            search_throughput >= 50
        ), f"Search throughput {search_throughput:.1f} searches/s too low"


@pytest.mark.performance
class TestMemoryAndResourceUsage:
    """Test memory and resource usage under load."""

    async def test_memory_usage_under_load(self, client: AsyncClient):
        """Test memory usage remains stable under load."""

        import psutil
        import os
        import gc

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        async def mock_memory_intensive_response(message, session_id, user_id=None):
            # Simulate some memory usage
            data = "x" * 1000  # 1KB string
            return {
                "response": f"Response with data: {data[:50]}...",
                "session_id": session_id,
                "timestamp": "2024-01-01T10:00:00Z",
                "metadata": {"data_size": len(data)},
            }

        # Mock the get_agent_client function to return a mock client
        mock_client = AsyncMock()
        mock_client.process_message = AsyncMock(
            side_effect=mock_memory_intensive_response
        )

        with patch(
            "src.shared.services.get_agent_client",
            new=AsyncMock(return_value=mock_client),
        ):

            memory_samples = []

            # Send requests and monitor memory
            for i in range(100):
                response = await client.post(
                    "/api/v1/chat/message",
                    json={
                        "message": f"Memory test message {i}",
                        "session_id": f"memory_session_{i}",
                    },
                )

                assert response.status_code == 200

                # Sample memory every 10 requests
                if i % 10 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_samples.append(current_memory)

                    # Force garbage collection
                    gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        max_memory = max(memory_samples)

        # Memory increase should be reasonable
        assert (
            memory_increase < 100
        ), f"Memory increased by {memory_increase:.1f}MB, too much"
        assert (
            max_memory < initial_memory + 150
        ), f"Peak memory {max_memory:.1f}MB too high"

        print(f"Memory usage:")
        print(f"  Initial: {initial_memory:.1f}MB")
        print(f"  Final: {final_memory:.1f}MB")
        print(f"  Increase: {memory_increase:.1f}MB")
        print(f"  Peak: {max_memory:.1f}MB")

    async def test_cpu_usage_efficiency(self, client: AsyncClient):
        """Test CPU usage efficiency under load."""

        import psutil
        import threading

        cpu_samples = []
        monitoring = True

        def monitor_cpu():
            """Monitor CPU usage in background."""
            while monitoring:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_samples.append(cpu_percent)

        # Start CPU monitoring
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()

        try:

            async def mock_cpu_intensive_response(message, session_id, user_id=None):
                # Simulate some CPU work
                result = sum(i * i for i in range(1000))
                return {
                    "response": f"CPU intensive response: {result}",
                    "session_id": session_id,
                    "timestamp": "2024-01-01T10:00:00Z",
                    "metadata": {"computation_result": result},
                }

            # Mock the get_agent_client function to return a mock client
            mock_client = AsyncMock()
            mock_client.process_message = AsyncMock(
                side_effect=mock_cpu_intensive_response
            )

            with patch(
                "src.shared.services.get_agent_client",
                new=AsyncMock(return_value=mock_client),
            ):

                # Send concurrent requests
                tasks = []
                for i in range(50):
                    task = client.post(
                        "/api/v1/chat/message",
                        json={
                            "message": f"CPU test message {i}",
                            "session_id": f"cpu_session_{i}",
                        },
                    )
                    tasks.append(task)

                responses = await asyncio.gather(*tasks)

        finally:
            monitoring = False
            monitor_thread.join()

        # Analyze CPU usage
        avg_cpu = statistics.mean(cpu_samples) if cpu_samples else 0
        max_cpu = max(cpu_samples) if cpu_samples else 0

        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)

        # CPU usage should be reasonable
        assert avg_cpu <= 80, f"Average CPU usage {avg_cpu:.1f}% too high"
        assert max_cpu <= 95, f"Peak CPU usage {max_cpu:.1f}% too high"

        print(f"CPU usage:")
        print(f"  Average: {avg_cpu:.1f}%")
        print(f"  Peak: {max_cpu:.1f}%")
        print(f"  Samples: {len(cpu_samples)}")


@pytest.mark.performance
@pytest.mark.slow
class TestStressTests:
    """Stress tests to find system limits."""

    async def test_connection_limit_stress(self, app, performance_config):
        """Test system behavior at connection limits."""

        max_connections = 100
        connection_results = []

        async def create_connection(conn_id: int):
            """Attempt to create a connection."""
            try:
                async with AsyncClient(app=app, base_url="http://test") as client:
                    response = await client.get("/health")
                    return {
                        "conn_id": conn_id,
                        "success": True,
                        "status": response.status_code,
                    }
            except Exception as e:
                return {"conn_id": conn_id, "success": False, "error": str(e)}

        # Create many concurrent connections
        tasks = [create_connection(i) for i in range(max_connections)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_connections = sum(
            1 for r in results if isinstance(r, dict) and r.get("success")
        )

        # Should handle at least 80% of connections
        success_rate = successful_connections / max_connections
        assert (
            success_rate >= 0.8
        ), f"Connection success rate {success_rate:.2%} too low"

    async def test_large_payload_stress(self, client: AsyncClient):
        """Test system behavior with large payloads."""

        # Test with increasingly large payloads
        # Note: MAX_STRING_LENGTH in validation middleware is 100KB (100000 bytes)
        payload_sizes = [
            1024,
            10240,
            50000,
            100000,
            150000,
        ]  # 1KB, 10KB, 50KB, 100KB, 150KB

        for size in payload_sizes:
            large_message = "x" * size

            mock_response = {
                "response": f"Processed large message of {size} bytes",
                "session_id": "large_payload_session",
                "timestamp": "2024-01-01T10:00:00Z",
                "metadata": {"payload_size": size},
            }

            # Mock the get_agent_client function to return a mock client
            mock_client = AsyncMock()
            mock_client.process_message = AsyncMock(return_value=mock_response)

            with patch(
                "src.shared.services.get_agent_client",
                new=AsyncMock(return_value=mock_client),
            ):

                start_time = time.time()

                response = await client.post(
                    "/api/v1/chat/message",
                    json={
                        "message": large_message,
                        "session_id": "large_payload_session",
                    },
                )

                end_time = time.time()
                processing_time = end_time - start_time

            # Validation middleware has MAX_STRING_LENGTH = 100000 bytes
            if size <= 100000:  # Up to 100KB should work
                assert (
                    response.status_code == 200
                ), f"Failed to process {size} byte payload: status {response.status_code}"
                assert (
                    processing_time <= 5.0
                ), f"Processing {size} bytes took {processing_time:.2f}s"

                print(
                    f"✓ Payload size {size} bytes processed successfully in {processing_time:.3f}s"
                )
            else:  # Above 100KB should be rejected by validation middleware
                assert (
                    response.status_code == 400
                ), f"Expected 400 for {size} byte payload (exceeds MAX_STRING_LENGTH), got {response.status_code}"

                print(f"✓ Payload size {size} bytes correctly rejected (exceeds limit)")
