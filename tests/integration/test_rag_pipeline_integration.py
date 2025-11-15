"""
RAG Pipeline Integration Tests
Tests the complete RAG pipeline with resolved dependencies and component interactions
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.rag_pipeline.main import RAGPipeline
from src.rag_pipeline.models import DocumentMetadata, ProcessedDocument, Chunk


@pytest.fixture
def temp_document():
    """Create a temporary test document with sufficient content for chunking"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        # Write substantial content to ensure chunks are created
        content = """# Software Engineering Best Practices

This is a comprehensive test document for the RAG pipeline integration testing.
It contains substantial content to ensure proper chunking and processing.

## Section 1: Introduction to Software Engineering

Software engineering is a systematic approach to developing software applications.
It involves various phases including requirements gathering, design, implementation,
testing, and maintenance. The field has evolved significantly over the past decades,
incorporating new methodologies and practices to improve software quality and development efficiency.

Modern software engineering emphasizes collaboration, automation, and continuous improvement.
Teams work together using version control systems, continuous integration pipelines, and
automated testing frameworks to deliver high-quality software products.

## Section 2: Design Patterns and Architecture

Design patterns are reusable solutions to common problems in software design. They represent
best practices evolved over time by experienced software developers. Some fundamental patterns include:

### Creational Patterns
- Singleton: Ensures a class has only one instance and provides a global point of access to it
- Factory Method: Defines an interface for creating objects but lets subclasses decide which class to instantiate
- Builder: Separates the construction of a complex object from its representation

### Structural Patterns
- Adapter: Allows incompatible interfaces to work together
- Decorator: Adds new functionality to objects dynamically
- Facade: Provides a simplified interface to a complex subsystem

### Behavioral Patterns
- Observer: Defines a one-to-many dependency between objects
- Strategy: Defines a family of algorithms and makes them interchangeable
- Command: Encapsulates a request as an object

## Section 3: Testing Methodologies

Testing is a critical phase in software development that ensures the software meets specified
requirements and functions correctly. Different types of testing serve different purposes:

Unit Testing: Tests individual components or functions in isolation. Unit tests are fast,
focused, and help catch bugs early in the development process. They form the foundation
of a comprehensive testing strategy.

Integration Testing: Verifies that different modules or services work together correctly.
Integration tests catch issues that arise from the interaction between components, such as
incorrect API contracts or data format mismatches.

System Testing: Tests the complete integrated system to verify it meets specified requirements.
This includes functional testing, performance testing, security testing, and usability testing.

Acceptance Testing: Validates the system against business requirements and determines whether
it's ready for deployment. This is often performed by end users or stakeholders.

## Section 4: Version Control and Collaboration

Version control systems like Git enable teams to collaborate effectively by tracking changes
to source code over time. They provide features such as branching, merging, and conflict
resolution that facilitate parallel development.

Best practices for version control include:
- Commit frequently with meaningful commit messages
- Use feature branches for new development
- Review code through pull requests before merging
- Maintain a clean commit history
- Tag releases for easy reference

## Section 5: Continuous Integration and Deployment

Continuous Integration (CI) and Continuous Deployment (CD) practices automate the process
of building, testing, and deploying software. This leads to faster and more reliable releases.

CI/CD pipelines typically include:
- Automated builds triggered by code commits
- Automated test execution at multiple levels
- Code quality checks and static analysis
- Security scanning for vulnerabilities
- Automated deployment to staging and production environments

## Section 6: Code Quality and Maintainability

Writing clean, maintainable code is essential for long-term project success. Key principles include:

SOLID Principles:
- Single Responsibility: A class should have only one reason to change
- Open/Closed: Software entities should be open for extension but closed for modification
- Liskov Substitution: Subtypes must be substitutable for their base types
- Interface Segregation: Clients should not depend on interfaces they don't use
- Dependency Inversion: Depend on abstractions, not concretions

Code should be self-documenting with clear naming conventions, appropriate comments,
and comprehensive documentation. Regular refactoring helps maintain code quality and
prevents technical debt from accumulating.

## Conclusion

This document provides comprehensive test content for validating the RAG pipeline
functionality across different content types and structures. It includes sufficient
text to generate multiple chunks during processing, enabling thorough testing of
document ingestion, embedding generation, and search capabilities.

The content covers various aspects of software engineering, providing diverse
vocabulary and concepts that can be used to test semantic search and retrieval
functionality. This ensures the RAG pipeline can handle real-world documentation
and technical content effectively.
"""
        f.write(content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_rag_pipeline():
    """Create a RAG pipeline with mocked dependencies"""
    pipeline = RAGPipeline()

    # Mock external dependencies
    with patch("src.rag_pipeline.embedding_manager.EmbeddingManager") as mock_embedding:
        with patch(
            "src.rag_pipeline.vector_store.ElasticsearchStore"
        ) as mock_vector_store:
            with patch(
                "src.rag_pipeline.search_engine.HybridSearchEngine"
            ) as mock_search_engine:

                # Setup mock embedding manager
                mock_embedding_instance = AsyncMock()
                mock_embedding_instance.initialize.return_value = None
                mock_embedding_instance.generate_embeddings.return_value = Mock(
                    general_embedding=[0.1] * 384, domain_embedding=[0.2] * 384
                )
                mock_embedding_instance.health_check.return_value = {
                    "status": "healthy"
                }
                mock_embedding.return_value = mock_embedding_instance

                # Setup mock vector store
                mock_vector_store_instance = AsyncMock()
                mock_vector_store_instance.initialize.return_value = None
                mock_vector_store_instance.store_document.return_value = "doc_123"
                mock_vector_store_instance.health_check.return_value = {
                    "status": "healthy"
                }
                mock_vector_store.return_value = mock_vector_store_instance

                # Setup mock search engine
                mock_search_engine_instance = AsyncMock()
                mock_search_engine_instance.initialize.return_value = None
                mock_search_engine_instance.search.return_value = {
                    "results": [
                        {
                            "chunk_id": "chunk_1",
                            "content": "Sample search result",
                            "score": 0.95,
                            "metadata": {"source": "test_doc"},
                        }
                    ],
                    "total_results": 1,
                    "search_time": 0.1,
                }
                mock_search_engine_instance.health_check.return_value = {
                    "status": "healthy"
                }
                mock_search_engine.return_value = mock_search_engine_instance

                yield pipeline


class TestRAGPipelineInitialization:
    """Test RAG pipeline initialization and component setup"""

    @pytest.mark.asyncio
    async def test_pipeline_initialization_success(self, mock_rag_pipeline):
        """Test successful pipeline initialization"""
        pipeline = mock_rag_pipeline

        # Test initialization
        await pipeline.initialize()

        assert pipeline._initialized is True
        assert pipeline.document_processor is not None
        assert pipeline.embedding_manager is not None
        assert pipeline.vector_store is not None
        assert pipeline.search_engine is not None
        assert pipeline.ingestion_service is not None

    @pytest.mark.asyncio
    async def test_pipeline_lazy_initialization(self, mock_rag_pipeline):
        """Test that pipeline initializes lazily when needed"""
        pipeline = mock_rag_pipeline

        # Pipeline should not be initialized initially
        assert pipeline._initialized is False

        # Calling a method should trigger initialization
        await pipeline.health_check()

        assert pipeline._initialized is True

    @pytest.mark.asyncio
    async def test_pipeline_initialization_with_custom_configs(self):
        """Test pipeline initialization with custom configurations"""
        custom_configs = {
            "embedding_config": {"model_name": "custom-model"},
            "elasticsearch_config": {"host": "custom-host"},
            "search_config": {"max_results": 20},
            "ingestion_config": {"max_concurrent_files": 10},
        }

        with patch("src.rag_pipeline.embedding_manager.EmbeddingManager"):
            with patch("src.rag_pipeline.vector_store.ElasticsearchStore"):
                with patch("src.rag_pipeline.search_engine.HybridSearchEngine"):
                    pipeline = RAGPipeline(**custom_configs)

                    # Configs should be stored for lazy initialization
                    assert (
                        pipeline.embedding_config == custom_configs["embedding_config"]
                    )
                    assert (
                        pipeline.elasticsearch_config
                        == custom_configs["elasticsearch_config"]
                    )
                    assert pipeline.search_config == custom_configs["search_config"]
                    assert (
                        pipeline.ingestion_config == custom_configs["ingestion_config"]
                    )


class TestDocumentIngestion:
    """Test document ingestion functionality"""

    @pytest.mark.asyncio
    async def test_single_document_ingestion(self, mock_rag_pipeline, temp_document):
        """Test ingesting a single document"""
        pipeline = mock_rag_pipeline

        # Test document ingestion
        result = await pipeline.ingest_document(temp_document)

        assert result["status"] == "success"
        assert result["document_id"] is not None
        assert result["chunks_processed"] > 0
        assert result["embeddings_generated"] is True
        assert result["stored_in_vector_db"] is True

    @pytest.mark.asyncio
    async def test_document_ingestion_with_metadata(
        self, mock_rag_pipeline, temp_document
    ):
        """Test document ingestion with custom metadata"""
        pipeline = mock_rag_pipeline

        metadata = {
            "author": "Test Author",
            "category": "test",
            "tags": ["integration", "test"],
        }

        result = await pipeline.ingest_document(temp_document, metadata)

        assert result["status"] == "success"
        assert result["document_id"] is not None

    @pytest.mark.asyncio
    async def test_batch_document_ingestion(self, mock_rag_pipeline, temp_document):
        """Test batch document ingestion"""
        pipeline = mock_rag_pipeline

        # Create multiple test files
        file_paths = [temp_document]  # Using same file for simplicity

        with patch.object(pipeline, "document_processor") as mock_processor:
            mock_processor.process_batch.return_value = Mock(
                total_documents=1,
                successful=1,
                failed=0,
                processing_time=1.0,
                results=[{"document_id": "doc_1", "status": "success"}],
                errors=[],
            )

            result = await pipeline.ingest_batch(file_paths)

            assert result["total_documents"] == 1
            assert result["successful"] == 1
            assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_document_ingestion_error_handling(self, mock_rag_pipeline):
        """Test error handling during document ingestion"""
        pipeline = mock_rag_pipeline

        # Test with non-existent file
        result = await pipeline.ingest_document("non_existent_file.txt")

        assert result["status"] == "error"
        assert "error" in result
        assert result["document_id"] is None


class TestSearchFunctionality:
    """Test search functionality"""

    @pytest.mark.asyncio
    async def test_basic_search(self, mock_rag_pipeline):
        """Test basic search functionality"""
        pipeline = mock_rag_pipeline

        query = "test document content"
        result = await pipeline.search(query)

        assert "results" in result
        assert "total_results" in result
        assert "search_time" in result
        assert len(result["results"]) > 0

    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_rag_pipeline):
        """Test search with filters"""
        pipeline = mock_rag_pipeline

        query = "test content"
        filters = {"category": "test", "author": "Test Author"}

        result = await pipeline.search(query, filters=filters, max_results=5)

        assert "results" in result
        assert "total_results" in result

    @pytest.mark.asyncio
    async def test_search_different_types(self, mock_rag_pipeline):
        """Test different search types"""
        pipeline = mock_rag_pipeline

        query = "test query"

        # Test hybrid search (default)
        result_hybrid = await pipeline.search(query, search_type="hybrid")
        assert "results" in result_hybrid

        # Test vector search
        result_vector = await pipeline.search(query, search_type="vector")
        assert "results" in result_vector

        # Test keyword search
        result_keyword = await pipeline.search(query, search_type="keyword")
        assert "results" in result_keyword


class TestHealthChecks:
    """Test health check functionality"""

    @pytest.mark.asyncio
    async def test_pipeline_health_check(self, mock_rag_pipeline):
        """Test pipeline health check"""
        pipeline = mock_rag_pipeline

        health_status = await pipeline.health_check()

        assert health_status["pipeline"] == "healthy"
        assert "components" in health_status
        assert "document_processor" in health_status["components"]
        assert "embedding_manager" in health_status["components"]
        assert "vector_store" in health_status["components"]
        assert "search_engine" in health_status["components"]

    @pytest.mark.asyncio
    async def test_component_health_checks(self, mock_rag_pipeline):
        """Test individual component health checks"""
        pipeline = mock_rag_pipeline
        await pipeline.initialize()

        # Test each component health check
        components = [
            pipeline.document_processor,
            pipeline.embedding_manager,
            pipeline.vector_store,
            pipeline.search_engine,
        ]

        for component in components:
            if hasattr(component, "health_check"):
                health = await component.health_check()
                assert "status" in health


class TestErrorHandling:
    """Test error handling and recovery"""

    @pytest.mark.asyncio
    async def test_initialization_error_handling(self):
        """Test error handling during initialization"""
        with patch(
            "src.rag_pipeline.embedding_manager.EmbeddingManager"
        ) as mock_embedding:
            mock_embedding.side_effect = Exception("Initialization failed")

            pipeline = RAGPipeline()

            with pytest.raises(Exception, match="Initialization failed"):
                await pipeline.initialize()

    @pytest.mark.asyncio
    async def test_search_error_handling(self, mock_rag_pipeline):
        """Test error handling during search"""
        pipeline = mock_rag_pipeline

        # Mock search engine to raise an exception
        with patch.object(pipeline, "search_engine") as mock_search:
            mock_search.search.side_effect = Exception("Search failed")

            # Initialize pipeline first
            await pipeline.initialize()
            pipeline.search_engine = mock_search

            result = await pipeline.search("test query")

            # Should handle error gracefully
            assert "error" in result or "results" in result

    @pytest.mark.asyncio
    async def test_component_failure_recovery(self, mock_rag_pipeline):
        """Test recovery when individual components fail"""
        pipeline = mock_rag_pipeline

        # Test with embedding manager failure
        with patch.object(pipeline, "embedding_manager") as mock_embedding:
            mock_embedding.health_check.side_effect = Exception("Component failed")

            await pipeline.initialize()
            pipeline.embedding_manager = mock_embedding

            health_status = await pipeline.health_check()

            # Pipeline should still report overall status
            assert "pipeline" in health_status
            assert "components" in health_status


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""

    @pytest.mark.asyncio
    async def test_complete_document_workflow(self, mock_rag_pipeline, temp_document):
        """Test complete workflow: ingest -> search -> retrieve"""
        pipeline = mock_rag_pipeline

        # Step 1: Ingest document
        ingest_result = await pipeline.ingest_document(temp_document)
        assert ingest_result["status"] == "success"

        # Step 2: Search for content
        search_result = await pipeline.search("test document")
        assert len(search_result["results"]) > 0

        # Step 3: Verify health
        health_status = await pipeline.health_check()
        assert health_status["pipeline"] == "healthy"

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_rag_pipeline, temp_document):
        """Test concurrent pipeline operations"""
        pipeline = mock_rag_pipeline

        # Run multiple operations concurrently
        tasks = [
            pipeline.ingest_document(temp_document),
            pipeline.search("test query 1"),
            pipeline.search("test query 2"),
            pipeline.health_check(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All operations should complete successfully
        for result in results:
            assert not isinstance(result, Exception)

    @pytest.mark.asyncio
    async def test_pipeline_state_consistency(self, mock_rag_pipeline):
        """Test that pipeline maintains consistent state across operations"""
        pipeline = mock_rag_pipeline

        # Initialize pipeline
        await pipeline.initialize()
        initial_state = pipeline._initialized

        # Perform various operations
        await pipeline.health_check()
        await pipeline.search("test query")

        # State should remain consistent
        assert pipeline._initialized == initial_state
        assert pipeline._initialized is True


@pytest.mark.integration
class TestRealComponentIntegration:
    """Integration tests with real components (when available)"""

    @pytest.mark.asyncio
    async def test_document_processor_integration(self, temp_document):
        """Test integration with real document processor"""
        try:
            from src.rag_pipeline.document_processor import DocumentProcessor

            processor = DocumentProcessor()
            await processor.initialize()

            # Test document processing
            result = await processor.process_document(temp_document)

            assert result is not None
            assert hasattr(result, "chunks")
            assert len(result.chunks) > 0

        except ImportError:
            pytest.skip("Document processor not available for integration test")

    @pytest.mark.asyncio
    async def test_embedding_manager_integration(self):
        """Test integration with real embedding manager"""
        try:
            from src.rag_pipeline.embedding_manager import (
                EmbeddingManager,
                EmbeddingConfig,
            )

            config = EmbeddingConfig()
            manager = EmbeddingManager(config)

            # Test embedding generation
            result = await manager.generate_embeddings("test text")

            assert result is not None
            assert hasattr(result, "general_embedding")
            assert len(result.general_embedding) > 0

        except ImportError:
            pytest.skip("Embedding manager not available for integration test")
