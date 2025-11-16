"""
RAG Pipeline Integration Tests - Corrected Version
Tests the corrected RAG pipeline with resolved dependencies
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
from datetime import datetime

from src.rag_pipeline.main import RAGPipeline
from src.rag_pipeline.models import (
    DocumentMetadata,
    ProcessedDocument,
    Chunk,
    DocumentType,
)

# Configure pytest for async tests
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture
def temp_directory():
    """Create temporary directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_document_content():
    """Sample document content for testing"""
    return """
# Software Engineering Best Practices

## Introduction
This document outlines best practices for software engineering projects.

## Code Quality
- Write clean, readable code
- Use meaningful variable names
- Add comprehensive comments
- Follow coding standards

## Testing
- Write unit tests for all functions
- Implement integration tests
- Use test-driven development
- Maintain high test coverage
"""


class TestRAGPipelineIntegration:
    """Test RAG pipeline integration with resolved dependencies"""

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self):
        """Test RAG pipeline initialization with lazy imports"""
        pipeline = RAGPipeline()

        # Should not be initialized yet
        assert not pipeline._initialized
        assert pipeline.document_processor is None

        # Initialize pipeline with proper lazy import mocking
        with (
            patch(
                "src.rag_pipeline.document_processor.DocumentProcessor"
            ) as mock_doc_processor,
            patch(
                "src.rag_pipeline.embedding_manager.EmbeddingManager"
            ) as mock_embedding_manager,
            patch(
                "src.rag_pipeline.vector_store.ElasticsearchStore"
            ) as mock_vector_store,
            patch(
                "src.rag_pipeline.search_engine.HybridSearchEngine"
            ) as mock_search_engine,
            patch(
                "src.rag_pipeline.document_ingestion.DocumentIngestionService"
            ) as mock_ingestion_service,
            patch(
                "src.rag_pipeline.document_ingestion.IngestionConfig"
            ) as mock_ingestion_config,
            patch(
                "src.rag_pipeline.vector_store.ElasticsearchConfig"
            ) as mock_es_config,
            patch("src.rag_pipeline.search_engine.SearchConfig") as mock_search_config,
            patch(
                "src.rag_pipeline.embedding_manager.EmbeddingConfig"
            ) as mock_embedding_config,
        ):
            # Mock config classes
            mock_ingestion_config.return_value = Mock()
            mock_es_config.return_value = Mock()
            mock_search_config.return_value = Mock()
            mock_embedding_config.return_value = Mock()

            # Mock component initialization
            mock_doc_processor.return_value.initialize = AsyncMock()
            mock_embedding_manager.return_value.initialize = AsyncMock()
            mock_vector_store.return_value.initialize = AsyncMock()
            mock_search_engine.return_value.initialize = AsyncMock()
            mock_ingestion_service.return_value.initialize = AsyncMock()

            await pipeline.initialize()

            # Verify initialization
            assert pipeline._initialized
            assert pipeline.document_processor is not None
            assert pipeline.embedding_manager is not None
            assert pipeline.vector_store is not None
            assert pipeline.search_engine is not None
            assert pipeline.ingestion_service is not None

    @pytest.mark.asyncio
    async def test_document_ingestion_workflow(
        self, temp_directory, sample_document_content
    ):
        """Test complete document ingestion workflow"""
        # Create test document
        test_file = temp_directory / "test_document.md"
        test_file.write_text(sample_document_content)

        pipeline = RAGPipeline()

        with (
            patch(
                "src.rag_pipeline.document_processor.DocumentProcessor"
            ) as mock_doc_processor,
            patch(
                "src.rag_pipeline.embedding_manager.EmbeddingManager"
            ) as mock_embedding_manager,
            patch(
                "src.rag_pipeline.vector_store.ElasticsearchStore"
            ) as mock_vector_store,
            patch(
                "src.rag_pipeline.search_engine.HybridSearchEngine"
            ) as mock_search_engine,
            patch(
                "src.rag_pipeline.document_ingestion.DocumentIngestionService"
            ) as mock_ingestion_service,
            patch(
                "src.rag_pipeline.document_ingestion.IngestionConfig"
            ) as mock_ingestion_config,
            patch(
                "src.rag_pipeline.vector_store.ElasticsearchConfig"
            ) as mock_es_config,
            patch("src.rag_pipeline.search_engine.SearchConfig") as mock_search_config,
            patch(
                "src.rag_pipeline.embedding_manager.EmbeddingConfig"
            ) as mock_embedding_config,
        ):
            # Mock config classes
            mock_ingestion_config.return_value = Mock()
            mock_es_config.return_value = Mock()
            mock_search_config.return_value = Mock()
            mock_embedding_config.return_value = Mock()

            # Mock component initialization
            mock_doc_processor.return_value.initialize = AsyncMock()
            mock_embedding_manager.return_value.initialize = AsyncMock()
            mock_vector_store.return_value.initialize = AsyncMock()
            mock_search_engine.return_value.initialize = AsyncMock()
            mock_ingestion_service.return_value.initialize = AsyncMock()

            # Create proper mock objects for the workflow
            mock_chunk = Chunk(
                content="# Software Engineering Best Practices",
                chunk_id="chunk_1",
                document_id="test_doc_1",
                chunk_index=0,
                start_char=0,
                end_char=50,
                metadata=DocumentMetadata(
                    source_path=str(test_file),
                    file_name="test_document.md",
                    file_size=len(sample_document_content),
                    document_type=DocumentType.MD,
                    created_at=datetime.fromisoformat("2024-01-01T00:00:00"),
                    modified_at=datetime.fromisoformat("2024-01-01T00:00:00"),
                    content_hash="test_hash",
                ),
            )

            mock_processed_doc = ProcessedDocument(
                document_id="test_doc_1",
                original_path=str(test_file),
                content=sample_document_content,
                chunks=[mock_chunk],
                metadata=DocumentMetadata(
                    source_path=str(test_file),
                    file_name="test_document.md",
                    file_size=len(sample_document_content),
                    document_type=DocumentType.MD,
                    created_at=datetime.fromisoformat("2024-01-01T00:00:00"),
                    modified_at=datetime.fromisoformat("2024-01-01T00:00:00"),
                    content_hash="test_hash",
                ),
                processing_time=1.5,
                content_hash="test_hash",
                total_chunks=1,
                total_characters=len(sample_document_content),
            )

            # Mock the async methods properly
            mock_ingestion_service.return_value.ingest_single_file = AsyncMock(
                return_value=mock_processed_doc
            )

            # Mock embedding result
            mock_embedding_result = Mock()
            mock_embedding_result.general_embedding = [0.1, 0.2, 0.3]
            mock_embedding_result.domain_embedding = [0.4, 0.5, 0.6]
            mock_embedding_manager.return_value.generate_embeddings = AsyncMock(
                return_value=mock_embedding_result
            )

            mock_vector_store.return_value.store_document = AsyncMock(
                return_value="stored_doc_id"
            )

            # Test document ingestion
            result = await pipeline.ingest_document(str(test_file))

            # Verify ingestion workflow
            assert result["status"] == "success"
            assert result["document_id"] == "test_doc_1"
            assert result["chunks_processed"] == 1

            # Verify component interactions
            mock_ingestion_service.return_value.ingest_single_file.assert_called_once()
            mock_embedding_manager.return_value.generate_embeddings.assert_called_once()
            mock_vector_store.return_value.store_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_functionality(self):
        """Test search functionality with resolved dependencies"""
        pipeline = RAGPipeline()

        with (
            patch(
                "src.rag_pipeline.document_processor.DocumentProcessor"
            ) as mock_doc_processor,
            patch(
                "src.rag_pipeline.embedding_manager.EmbeddingManager"
            ) as mock_embedding_manager,
            patch(
                "src.rag_pipeline.vector_store.ElasticsearchStore"
            ) as mock_vector_store,
            patch(
                "src.rag_pipeline.search_engine.HybridSearchEngine"
            ) as mock_search_engine,
            patch(
                "src.rag_pipeline.document_ingestion.DocumentIngestionService"
            ) as mock_ingestion_service,
            patch(
                "src.rag_pipeline.document_ingestion.IngestionConfig"
            ) as mock_ingestion_config,
            patch(
                "src.rag_pipeline.vector_store.ElasticsearchConfig"
            ) as mock_es_config,
            patch("src.rag_pipeline.search_engine.SearchConfig") as mock_search_config,
            patch(
                "src.rag_pipeline.embedding_manager.EmbeddingConfig"
            ) as mock_embedding_config,
        ):
            # Mock config classes
            mock_ingestion_config.return_value = Mock()
            mock_es_config.return_value = Mock()
            mock_search_config.return_value = Mock()
            mock_embedding_config.return_value = Mock()

            # Mock component initialization
            mock_doc_processor.return_value.initialize = AsyncMock()
            mock_embedding_manager.return_value.initialize = AsyncMock()
            mock_vector_store.return_value.initialize = AsyncMock()
            mock_search_engine.return_value.initialize = AsyncMock()
            mock_ingestion_service.return_value.initialize = AsyncMock()

            # Mock search response with proper structure
            mock_search_result = Mock()
            mock_search_result.chunk_id = "chunk_1"
            mock_search_result.content = (
                "Software engineering best practices include writing clean code"
            )
            mock_search_result.score = 0.95
            mock_search_result.document_id = "doc_1"
            mock_search_result.chunk_type = "text"
            mock_search_result.parent_chunk_id = None
            mock_search_result.highlights = []
            mock_search_result.metadata = {"document_title": "Best Practices Guide"}

            mock_search_response = Mock()
            mock_search_response.results = [mock_search_result]
            mock_search_response.total_hits = 1
            mock_search_response.max_score = 0.95
            mock_search_response.took_ms = 45.2
            mock_search_response.search_type = "hybrid"

            mock_search_engine.return_value.search = AsyncMock(
                return_value=mock_search_response
            )

            # Test search
            results = await pipeline.search("software engineering best practices")

            # Verify search results
            assert results["total_results"] == 1
            assert len(results["results"]) == 1
            assert results["results"][0]["score"] == 0.95

            # Verify search engine was called
            mock_search_engine.return_value.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_functionality(self):
        """Test health check with component status reporting"""
        pipeline = RAGPipeline()

        with (
            patch(
                "src.rag_pipeline.document_processor.DocumentProcessor"
            ) as mock_doc_processor,
            patch(
                "src.rag_pipeline.embedding_manager.EmbeddingManager"
            ) as mock_embedding_manager,
            patch(
                "src.rag_pipeline.vector_store.ElasticsearchStore"
            ) as mock_vector_store,
            patch(
                "src.rag_pipeline.search_engine.HybridSearchEngine"
            ) as mock_search_engine,
            patch(
                "src.rag_pipeline.document_ingestion.DocumentIngestionService"
            ) as mock_ingestion_service,
            patch(
                "src.rag_pipeline.document_ingestion.IngestionConfig"
            ) as mock_ingestion_config,
            patch(
                "src.rag_pipeline.vector_store.ElasticsearchConfig"
            ) as mock_es_config,
            patch("src.rag_pipeline.search_engine.SearchConfig") as mock_search_config,
            patch(
                "src.rag_pipeline.embedding_manager.EmbeddingConfig"
            ) as mock_embedding_config,
        ):
            # Mock config classes
            mock_ingestion_config.return_value = Mock()
            mock_es_config.return_value = Mock()
            mock_search_config.return_value = Mock()
            mock_embedding_config.return_value = Mock()

            # Mock component initialization and health checks
            mock_doc_processor.return_value.initialize = AsyncMock()
            mock_doc_processor.return_value.health_check = AsyncMock(
                return_value={"status": "healthy"}
            )

            mock_embedding_manager.return_value.initialize = AsyncMock()
            mock_embedding_manager.return_value.health_check = AsyncMock(
                return_value={"status": "healthy"}
            )

            mock_vector_store.return_value.initialize = AsyncMock()
            mock_vector_store.return_value.health_check = AsyncMock(
                return_value={"status": "healthy"}
            )

            mock_search_engine.return_value.initialize = AsyncMock()
            mock_search_engine.return_value.health_check = AsyncMock(
                return_value={"status": "healthy"}
            )

            mock_ingestion_service.return_value.initialize = AsyncMock()
            mock_ingestion_service.return_value.get_ingestion_status = AsyncMock(
                return_value={"ingested_files_count": 5}
            )

            # Test health check
            health_status = await pipeline.health_check()

            # Verify health check results
            assert health_status["pipeline"] == "healthy"
            assert "components" in health_status
            assert (
                health_status["components"]["document_processor"]["status"] == "healthy"
            )
            assert (
                health_status["components"]["embedding_manager"]["status"] == "healthy"
            )
            assert health_status["components"]["vector_store"]["status"] == "healthy"
            assert health_status["components"]["search_engine"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_error_handling_during_ingestion(
        self, temp_directory, sample_document_content
    ):
        """Test error handling during document ingestion"""
        # Create test document
        test_file = temp_directory / "test_document.md"
        test_file.write_text(sample_document_content)

        pipeline = RAGPipeline()

        with (
            patch(
                "src.rag_pipeline.document_processor.DocumentProcessor"
            ) as mock_doc_processor,
            patch(
                "src.rag_pipeline.embedding_manager.EmbeddingManager"
            ) as mock_embedding_manager,
            patch(
                "src.rag_pipeline.vector_store.ElasticsearchStore"
            ) as mock_vector_store,
            patch(
                "src.rag_pipeline.search_engine.HybridSearchEngine"
            ) as mock_search_engine,
            patch(
                "src.rag_pipeline.document_ingestion.DocumentIngestionService"
            ) as mock_ingestion_service,
            patch(
                "src.rag_pipeline.document_ingestion.IngestionConfig"
            ) as mock_ingestion_config,
            patch(
                "src.rag_pipeline.vector_store.ElasticsearchConfig"
            ) as mock_es_config,
            patch("src.rag_pipeline.search_engine.SearchConfig") as mock_search_config,
            patch(
                "src.rag_pipeline.embedding_manager.EmbeddingConfig"
            ) as mock_embedding_config,
        ):
            # Mock config classes
            mock_ingestion_config.return_value = Mock()
            mock_es_config.return_value = Mock()
            mock_search_config.return_value = Mock()
            mock_embedding_config.return_value = Mock()

            # Mock component initialization
            mock_doc_processor.return_value.initialize = AsyncMock()
            mock_embedding_manager.return_value.initialize = AsyncMock()
            mock_vector_store.return_value.initialize = AsyncMock()
            mock_search_engine.return_value.initialize = AsyncMock()
            mock_ingestion_service.return_value.initialize = AsyncMock()

            # Mock ingestion service to raise an error
            mock_ingestion_service.return_value.ingest_single_file = AsyncMock(
                side_effect=Exception("File processing failed")
            )

            # Test error handling during ingestion
            result = await pipeline.ingest_document(str(test_file))

            # Verify error handling
            assert result["status"] == "error"
            assert "error" in result
            assert result["document_id"] is None


class TestCircularImportResolution:
    """Test that circular import issues are resolved"""

    def test_import_rag_pipeline_main(self):
        """Test that RAG pipeline main module can be imported without circular import errors"""
        try:
            from src.rag_pipeline.main import RAGPipeline

            assert RAGPipeline is not None
        except ImportError as e:
            pytest.fail(f"Circular import detected: {e}")

    def test_import_document_processor(self):
        """Test that document processor can be imported without circular import errors"""
        try:
            from src.rag_pipeline.document_processor import DocumentProcessor

            assert DocumentProcessor is not None
        except ImportError as e:
            pytest.fail(f"Circular import detected: {e}")

    def test_import_chunking_strategies(self):
        """Test that chunking strategies can be imported without circular import errors"""
        try:
            from src.rag_pipeline.chunking_strategies import ChunkingManager

            assert ChunkingManager is not None
        except ImportError as e:
            pytest.fail(f"Circular import detected: {e}")

    def test_import_all_rag_components(self):
        """Test that all RAG components can be imported together"""
        try:
            from src.rag_pipeline.main import RAGPipeline
            from src.rag_pipeline.document_processor import DocumentProcessor
            from src.rag_pipeline.document_ingestion import DocumentIngestionService
            from src.rag_pipeline.vector_store import ElasticsearchStore
            from src.rag_pipeline.search_engine import HybridSearchEngine
            from src.rag_pipeline.embedding_manager import EmbeddingManager
            from src.rag_pipeline.chunking_strategies import ChunkingManager

            # Verify all imports successful
            assert all(
                [
                    RAGPipeline,
                    DocumentProcessor,
                    DocumentIngestionService,
                    ElasticsearchStore,
                    HybridSearchEngine,
                    EmbeddingManager,
                    ChunkingManager,
                ]
            )
        except ImportError as e:
            pytest.fail(f"Circular import detected: {e}")

    def test_rag_pipeline_instantiation(self):
        """Test that RAG pipeline can be instantiated without import errors"""
        try:
            from src.rag_pipeline.main import RAGPipeline

            pipeline = RAGPipeline()
            assert pipeline is not None
            assert not pipeline._initialized
        except Exception as e:
            pytest.fail(f"RAG pipeline instantiation failed: {e}")


class TestComponentDependencyResolution:
    """Test that component dependencies are properly resolved"""

    @pytest.mark.asyncio
    async def test_lazy_import_resolution(self):
        """Test that lazy imports work correctly during initialization"""
        pipeline = RAGPipeline()

        # Components should be None before initialization
        assert pipeline.document_processor is None
        assert pipeline.embedding_manager is None
        assert pipeline.vector_store is None
        assert pipeline.search_engine is None
        assert pipeline.ingestion_service is None

        # Mock all the imports that happen during initialization
        with (
            patch(
                "src.rag_pipeline.document_processor.DocumentProcessor"
            ) as mock_doc_processor,
            patch(
                "src.rag_pipeline.embedding_manager.EmbeddingManager"
            ) as mock_embedding_manager,
            patch(
                "src.rag_pipeline.vector_store.ElasticsearchStore"
            ) as mock_vector_store,
            patch(
                "src.rag_pipeline.search_engine.HybridSearchEngine"
            ) as mock_search_engine,
            patch(
                "src.rag_pipeline.document_ingestion.DocumentIngestionService"
            ) as mock_ingestion_service,
            patch("src.rag_pipeline.document_ingestion.IngestionConfig"),
            patch("src.rag_pipeline.vector_store.ElasticsearchConfig"),
            patch("src.rag_pipeline.search_engine.SearchConfig"),
            patch("src.rag_pipeline.embedding_manager.EmbeddingConfig"),
        ):
            # Mock component initialization
            for mock_component in [
                mock_doc_processor,
                mock_embedding_manager,
                mock_vector_store,
                mock_search_engine,
                mock_ingestion_service,
            ]:
                mock_component.return_value.initialize = AsyncMock()

            # Initialize should work without import errors
            await pipeline.initialize()

            # Components should be initialized
            assert pipeline._initialized
            assert pipeline.document_processor is not None
            assert pipeline.embedding_manager is not None
            assert pipeline.vector_store is not None
            assert pipeline.search_engine is not None
            assert pipeline.ingestion_service is not None
