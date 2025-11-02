"""
Test RAG Pipeline Import Resolution
Tests that circular import issues have been resolved
"""

import pytest
import sys
from unittest.mock import patch, Mock


class TestImportResolution:
    """Test that RAG pipeline imports work correctly"""

    def test_rag_pipeline_main_import(self):
        """Test that RAG pipeline main module can be imported"""
        try:
            from src.rag_pipeline.main import RAGPipeline

            assert RAGPipeline is not None
        except ImportError as e:
            pytest.fail(f"Failed to import RAGPipeline: {e}")

    def test_document_processor_import(self):
        """Test that document processor can be imported"""
        try:
            from src.rag_pipeline.document_processor import DocumentProcessor

            assert DocumentProcessor is not None
        except ImportError as e:
            pytest.fail(f"Failed to import DocumentProcessor: {e}")

    def test_chunking_strategies_import(self):
        """Test that chunking strategies can be imported"""
        try:
            from src.rag_pipeline.chunking_strategies import ChunkingManager

            assert ChunkingManager is not None
        except ImportError as e:
            pytest.fail(f"Failed to import ChunkingManager: {e}")

    def test_models_import(self):
        """Test that models can be imported"""
        try:
            from src.rag_pipeline.models import (
                Chunk,
                DocumentMetadata,
                ProcessedDocument,
            )

            assert Chunk is not None
            assert DocumentMetadata is not None
            assert ProcessedDocument is not None
        except ImportError as e:
            pytest.fail(f"Failed to import models: {e}")

    def test_embedding_manager_import(self):
        """Test that embedding manager can be imported"""
        try:
            from src.rag_pipeline.embedding_manager import EmbeddingManager

            assert EmbeddingManager is not None
        except ImportError as e:
            pytest.fail(f"Failed to import EmbeddingManager: {e}")

    def test_vector_store_import(self):
        """Test that vector store can be imported"""
        try:
            from src.rag_pipeline.vector_store import ElasticsearchStore

            assert ElasticsearchStore is not None
        except ImportError as e:
            pytest.fail(f"Failed to import ElasticsearchStore: {e}")

    def test_search_engine_import(self):
        """Test that search engine can be imported"""
        try:
            from src.rag_pipeline.search_engine import HybridSearchEngine

            assert HybridSearchEngine is not None
        except ImportError as e:
            pytest.fail(f"Failed to import HybridSearchEngine: {e}")

    def test_document_ingestion_import(self):
        """Test that document ingestion can be imported"""
        try:
            from src.rag_pipeline.document_ingestion import DocumentIngestionService

            assert DocumentIngestionService is not None
        except ImportError as e:
            pytest.fail(f"Failed to import DocumentIngestionService: {e}")

    def test_no_circular_imports(self):
        """Test that there are no circular import issues"""
        # Clear any previously imported modules
        modules_to_clear = [
            "src.rag_pipeline.main",
            "src.rag_pipeline.document_processor",
            "src.rag_pipeline.chunking_strategies",
            "src.rag_pipeline.models",
        ]

        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]

        # Try importing in different orders to detect circular dependencies
        try:
            # Order 1: main -> document_processor -> chunking_strategies
            from src.rag_pipeline.main import RAGPipeline
            from src.rag_pipeline.document_processor import DocumentProcessor
            from src.rag_pipeline.chunking_strategies import ChunkingManager

            # Clear modules again
            for module in modules_to_clear:
                if module in sys.modules:
                    del sys.modules[module]

            # Order 2: chunking_strategies -> models -> document_processor
            from src.rag_pipeline.chunking_strategies import ChunkingManager
            from src.rag_pipeline.models import Chunk
            from src.rag_pipeline.document_processor import DocumentProcessor

            # If we get here, no circular imports
            assert True

        except ImportError as e:
            if "circular import" in str(e).lower():
                pytest.fail(f"Circular import detected: {e}")
            else:
                pytest.fail(f"Import error: {e}")

    def test_all_components_import_together(self):
        """Test that all RAG components can be imported together without conflicts"""
        try:
            # Import all components at once
            from src.rag_pipeline.main import RAGPipeline
            from src.rag_pipeline.document_processor import DocumentProcessor
            from src.rag_pipeline.document_ingestion import DocumentIngestionService
            from src.rag_pipeline.vector_store import ElasticsearchStore
            from src.rag_pipeline.search_engine import HybridSearchEngine
            from src.rag_pipeline.embedding_manager import EmbeddingManager
            from src.rag_pipeline.chunking_strategies import ChunkingManager
            from src.rag_pipeline.models import (
                Chunk,
                DocumentMetadata,
                ProcessedDocument,
            )

            # Verify all imports successful
            components = [
                RAGPipeline,
                DocumentProcessor,
                DocumentIngestionService,
                ElasticsearchStore,
                HybridSearchEngine,
                EmbeddingManager,
                ChunkingManager,
                Chunk,
                DocumentMetadata,
                ProcessedDocument,
            ]

            assert all(component is not None for component in components)

        except ImportError as e:
            pytest.fail(f"Failed to import all components together: {e}")

    def test_rag_pipeline_instantiation(self):
        """Test that RAG pipeline can be instantiated without import errors"""
        try:
            from src.rag_pipeline.main import RAGPipeline

            pipeline = RAGPipeline()
            assert pipeline is not None
            assert not pipeline._initialized
            assert pipeline.document_processor is None
            assert pipeline.embedding_manager is None
            assert pipeline.vector_store is None
            assert pipeline.search_engine is None
            assert pipeline.ingestion_service is None
        except Exception as e:
            pytest.fail(f"RAG pipeline instantiation failed: {e}")

    def test_chunking_config_initialization(self):
        """Test that ChunkingConfig can be initialized without circular import issues"""
        try:
            from src.rag_pipeline.models import ChunkingConfig

            config = ChunkingConfig(strategy="hierarchical")
            assert config is not None
            assert config.strategy == "hierarchical"
        except Exception as e:
            pytest.fail(f"ChunkingConfig initialization failed: {e}")

    def test_document_metadata_creation(self):
        """Test that DocumentMetadata can be created without import issues"""
        try:
            from src.rag_pipeline.models import DocumentMetadata, DocumentType
            from datetime import datetime

            metadata = DocumentMetadata(
                source_path="/test/path",
                file_name="test.txt",
                file_size=1000,
                document_type=DocumentType.TXT,
                created_at=datetime.now(),
                modified_at=datetime.now(),
                content_hash="test_hash",
            )

            assert metadata is not None
            assert metadata.file_name == "test.txt"
            assert metadata.document_type == DocumentType.TXT

        except Exception as e:
            pytest.fail(f"DocumentMetadata creation failed: {e}")

    def test_chunk_creation(self):
        """Test that Chunk objects can be created without import issues"""
        try:
            from src.rag_pipeline.models import Chunk, DocumentMetadata, DocumentType
            from datetime import datetime

            metadata = DocumentMetadata(
                source_path="/test/path",
                file_name="test.txt",
                file_size=1000,
                document_type=DocumentType.TXT,
                created_at=datetime.now(),
                modified_at=datetime.now(),
                content_hash="test_hash",
            )

            chunk = Chunk(
                content="Test chunk content",
                chunk_id="chunk_1",
                document_id="doc_1",
                chunk_index=0,
                start_char=0,
                end_char=18,
                metadata=metadata,
            )

            assert chunk is not None
            assert chunk.content == "Test chunk content"
            assert chunk.chunk_id == "chunk_1"
            assert chunk.child_chunk_ids == []  # Should be initialized by __post_init__

        except Exception as e:
            pytest.fail(f"Chunk creation failed: {e}")

    def test_processed_document_creation(self):
        """Test that ProcessedDocument can be created without import issues"""
        try:
            from src.rag_pipeline.models import (
                ProcessedDocument,
                Chunk,
                DocumentMetadata,
                DocumentType,
            )
            from datetime import datetime

            metadata = DocumentMetadata(
                source_path="/test/path",
                file_name="test.txt",
                file_size=1000,
                document_type=DocumentType.TXT,
                created_at=datetime.now(),
                modified_at=datetime.now(),
                content_hash="test_hash",
            )

            chunk = Chunk(
                content="Test chunk content",
                chunk_id="chunk_1",
                document_id="doc_1",
                chunk_index=0,
                start_char=0,
                end_char=18,
                metadata=metadata,
            )

            processed_doc = ProcessedDocument(
                document_id="doc_1",
                original_path="/test/path",
                content="Test document content",
                chunks=[chunk],
                metadata=metadata,
                processing_time=1.5,
                content_hash="test_hash",
                total_chunks=1,
                total_characters=20,
            )

            assert processed_doc is not None
            assert processed_doc.document_id == "doc_1"
            assert len(processed_doc.chunks) == 1
            assert processed_doc.total_chunks == 1

        except Exception as e:
            pytest.fail(f"ProcessedDocument creation failed: {e}")


class TestLazyImportResolution:
    """Test that lazy imports work correctly in RAG pipeline"""

    def test_lazy_import_in_initialize(self):
        """Test that lazy imports in initialize method work correctly"""
        try:
            from src.rag_pipeline.main import RAGPipeline

            # Create pipeline - should not trigger imports yet
            pipeline = RAGPipeline()
            assert not pipeline._initialized

            # Components should be None before initialization
            assert pipeline.document_processor is None
            assert pipeline.embedding_manager is None

        except Exception as e:
            pytest.fail(f"Lazy import test failed: {e}")

    def test_lazy_import_in_document_processor(self):
        """Test that lazy imports in document processor work correctly"""
        try:
            from src.rag_pipeline.document_processor import DocumentProcessor

            # Should be able to create without triggering chunking imports
            processor = DocumentProcessor()
            assert processor is not None

        except Exception as e:
            pytest.fail(f"Document processor lazy import test failed: {e}")

    def test_config_class_imports(self):
        """Test that config classes can be imported independently"""
        try:
            # These should all be importable without circular dependencies
            from src.rag_pipeline.models import ChunkingConfig
            from src.rag_pipeline.document_ingestion import IngestionConfig

            # Test basic instantiation
            chunking_config = ChunkingConfig(strategy="hierarchical")
            ingestion_config = IngestionConfig(
                source_directories=["/test"], supported_extensions=[".txt"]
            )

            assert chunking_config is not None
            assert ingestion_config is not None

        except Exception as e:
            pytest.fail(f"Config class import test failed: {e}")


class TestComponentIntegration:
    """Test that components can work together without import issues"""

    def test_chunking_manager_with_config(self):
        """Test that ChunkingManager can be created with config"""
        try:
            from src.rag_pipeline.chunking_strategies import ChunkingManager
            from src.rag_pipeline.models import ChunkingConfig

            config = ChunkingConfig(strategy="hierarchical")
            manager = ChunkingManager(config)

            assert manager is not None
            assert manager.config is not None

        except Exception as e:
            pytest.fail(f"ChunkingManager integration test failed: {e}")

    def test_document_processor_with_chunking(self):
        """Test that DocumentProcessor can work with chunking components"""
        try:
            from src.rag_pipeline.document_processor import DocumentProcessor
            from src.rag_pipeline.models import ChunkingConfig

            config = ChunkingConfig(strategy="hierarchical")
            processor = DocumentProcessor(chunking_config=config)

            assert processor is not None
            assert processor.chunking_config is not None

        except Exception as e:
            pytest.fail(f"DocumentProcessor chunking integration test failed: {e}")

    def test_models_enum_usage(self):
        """Test that enum classes from models work correctly"""
        try:
            from src.rag_pipeline.models import DocumentType

            # Test enum values
            assert DocumentType.PDF == "pdf"
            assert DocumentType.TXT == "txt"
            assert DocumentType.MD == "md"

            # Test enum usage
            doc_type = DocumentType.PDF
            assert doc_type.value == "pdf"

        except Exception as e:
            pytest.fail(f"Models enum usage test failed: {e}")


class TestErrorHandling:
    """Test error handling in import resolution"""

    def test_missing_optional_dependencies(self):
        """Test that missing optional dependencies don't break imports"""
        try:
            # These imports should work even if some optional dependencies are missing
            from src.rag_pipeline.main import RAGPipeline
            from src.rag_pipeline.models import Chunk, DocumentMetadata

            # Basic functionality should work
            pipeline = RAGPipeline()
            assert pipeline is not None

        except ImportError as e:
            # Only fail if it's a circular import, not a missing dependency
            if "circular import" in str(e).lower():
                pytest.fail(f"Circular import detected: {e}")
            # Otherwise, missing dependencies are acceptable for this test

    def test_import_error_messages(self):
        """Test that import errors provide clear messages"""
        try:
            # Try importing all components to check for clear error messages
            from src.rag_pipeline.main import RAGPipeline
            from src.rag_pipeline.document_processor import DocumentProcessor
            from src.rag_pipeline.chunking_strategies import ChunkingManager
            from src.rag_pipeline.models import (
                Chunk,
                DocumentMetadata,
                ProcessedDocument,
            )

            # If we get here, all imports succeeded
            assert True

        except ImportError as e:
            # Check that error message is informative
            error_msg = str(e).lower()

            # These would indicate circular import issues
            problematic_patterns = [
                "circular import",
                "partially initialized module",
                "cannot import name",
                "most likely due to a circular import",
            ]

            for pattern in problematic_patterns:
                if pattern in error_msg:
                    pytest.fail(f"Import resolution issue detected: {e}")

            # If it's just a missing dependency, that's acceptable
            pass
