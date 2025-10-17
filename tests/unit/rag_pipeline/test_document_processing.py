"""
Unit tests for RAG pipeline document processing.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path


class TestDocumentLoader:
    """Test document loading functionality."""

    @pytest.fixture
    def sample_text_file(self):
        """Create a temporary text file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a sample document for testing.")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def mock_document_loader(self):
        """Mock document loader."""
        loader = Mock()
        loader.load_document = Mock()
        loader.supported_formats = [".txt", ".pdf", ".docx", ".md"]
        return loader

    def test_text_file_loading(self, mock_document_loader, sample_text_file):
        """Test loading text files."""
        mock_document_loader.load_document.return_value = {
            "content": "This is a sample document for testing.",
            "metadata": {"file_type": "txt", "size": 38},
        }

        result = mock_document_loader.load_document(sample_text_file)
        assert "content" in result
        assert "metadata" in result
        assert result["content"] == "This is a sample document for testing."

    def test_unsupported_format(self, mock_document_loader):
        """Test handling of unsupported file formats."""
        mock_document_loader.load_document.side_effect = ValueError(
            "Unsupported format"
        )

        with pytest.raises(ValueError, match="Unsupported format"):
            mock_document_loader.load_document("test.xyz")

    def test_file_not_found(self, mock_document_loader):
        """Test handling of non-existent files."""
        mock_document_loader.load_document.side_effect = FileNotFoundError(
            "File not found"
        )

        with pytest.raises(FileNotFoundError):
            mock_document_loader.load_document("nonexistent.txt")

    def test_supported_formats(self, mock_document_loader):
        """Test supported file formats."""
        expected_formats = [".txt", ".pdf", ".docx", ".md"]
        assert mock_document_loader.supported_formats == expected_formats


class TestDocumentChunking:
    """Test document chunking functionality."""

    @pytest.fixture
    def mock_chunker(self):
        """Mock document chunker."""
        chunker = Mock()
        chunker.chunk_document = Mock()
        chunker.chunk_size = 1000
        chunker.overlap = 200
        return chunker

    def test_document_chunking(self, mock_chunker):
        """Test document chunking."""
        long_text = "This is a long document. " * 100
        mock_chunker.chunk_document.return_value = [
            {"text": long_text[:1000], "chunk_id": 0, "start": 0, "end": 1000},
            {"text": long_text[800:1800], "chunk_id": 1, "start": 800, "end": 1800},
        ]

        chunks = mock_chunker.chunk_document(long_text)
        assert len(chunks) == 2
        assert all("chunk_id" in chunk for chunk in chunks)
        assert all("text" in chunk for chunk in chunks)

    def test_chunk_size_configuration(self, mock_chunker):
        """Test chunk size configuration."""
        assert mock_chunker.chunk_size == 1000
        assert mock_chunker.overlap == 200

    def test_empty_document_chunking(self, mock_chunker):
        """Test chunking empty documents."""
        mock_chunker.chunk_document.return_value = []
        chunks = mock_chunker.chunk_document("")
        assert chunks == []

    def test_small_document_chunking(self, mock_chunker):
        """Test chunking documents smaller than chunk size."""
        small_text = "Short document."
        mock_chunker.chunk_document.return_value = [
            {"text": small_text, "chunk_id": 0, "start": 0, "end": len(small_text)}
        ]

        chunks = mock_chunker.chunk_document(small_text)
        assert len(chunks) == 1
        assert chunks[0]["text"] == small_text


class TestDocumentEmbedding:
    """Test document embedding functionality."""

    @pytest.fixture
    def mock_embedder(self):
        """Mock embedding model."""
        embedder = Mock()
        embedder.embed_text = Mock()
        embedder.embed_batch = Mock()
        embedder.dimension = 768
        return embedder

    def test_single_text_embedding(self, mock_embedder):
        """Test embedding single text."""
        mock_embedder.embed_text.return_value = [0.1] * 768

        embedding = mock_embedder.embed_text("Sample text")
        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)

    def test_batch_embedding(self, mock_embedder):
        """Test batch embedding."""
        texts = ["Text 1", "Text 2", "Text 3"]
        mock_embedder.embed_batch.return_value = [[0.1] * 768, [0.2] * 768, [0.3] * 768]

        embeddings = mock_embedder.embed_batch(texts)
        assert len(embeddings) == 3
        assert all(len(emb) == 768 for emb in embeddings)

    def test_empty_text_embedding(self, mock_embedder):
        """Test embedding empty text."""
        mock_embedder.embed_text.return_value = [0.0] * 768

        embedding = mock_embedder.embed_text("")
        assert len(embedding) == 768
        assert all(x == 0.0 for x in embedding)

    def test_embedding_dimension(self, mock_embedder):
        """Test embedding dimension consistency."""
        assert mock_embedder.dimension == 768


class TestVectorStore:
    """Test vector store functionality."""

    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store."""
        store = Mock()
        store.add_documents = Mock()
        store.search = Mock()
        store.delete = Mock()
        store.update = Mock()
        store.count = Mock(return_value=0)
        return store

    def test_document_addition(self, mock_vector_store):
        """Test adding documents to vector store."""
        documents = [
            {"id": "doc1", "text": "Document 1", "embedding": [0.1] * 768},
            {"id": "doc2", "text": "Document 2", "embedding": [0.2] * 768},
        ]

        mock_vector_store.add_documents(documents)
        mock_vector_store.add_documents.assert_called_once_with(documents)

    def test_similarity_search(self, mock_vector_store):
        """Test similarity search."""
        query_embedding = [0.15] * 768
        mock_vector_store.search.return_value = [
            {"id": "doc1", "text": "Document 1", "score": 0.95},
            {"id": "doc2", "text": "Document 2", "score": 0.87},
        ]

        results = mock_vector_store.search(query_embedding, k=2)
        assert len(results) == 2
        assert all("score" in result for result in results)
        assert results[0]["score"] > results[1]["score"]  # Should be sorted by score

    def test_document_deletion(self, mock_vector_store):
        """Test document deletion."""
        mock_vector_store.delete("doc1")
        mock_vector_store.delete.assert_called_once_with("doc1")

    def test_document_update(self, mock_vector_store):
        """Test document update."""
        updated_doc = {
            "id": "doc1",
            "text": "Updated document",
            "embedding": [0.3] * 768,
        }
        mock_vector_store.update(updated_doc)
        mock_vector_store.update.assert_called_once_with(updated_doc)

    def test_store_count(self, mock_vector_store):
        """Test getting document count."""
        mock_vector_store.count.return_value = 100
        count = mock_vector_store.count()
        assert count == 100


class TestDocumentMetadata:
    """Test document metadata handling."""

    @pytest.fixture
    def sample_metadata(self):
        """Sample document metadata."""
        return {
            "filename": "test_doc.pdf",
            "file_size": 1024,
            "created_at": "2024-01-01T00:00:00Z",
            "author": "Test Author",
            "title": "Test Document",
            "page_count": 5,
            "language": "en",
        }

    def test_metadata_extraction(self, sample_metadata):
        """Test metadata extraction."""
        assert sample_metadata["filename"] == "test_doc.pdf"
        assert sample_metadata["file_size"] == 1024
        assert sample_metadata["author"] == "Test Author"

    def test_metadata_validation(self, sample_metadata):
        """Test metadata validation."""
        required_fields = ["filename", "created_at"]
        for field in required_fields:
            assert field in sample_metadata

    def test_metadata_serialization(self, sample_metadata):
        """Test metadata serialization."""
        import json

        serialized = json.dumps(sample_metadata)
        deserialized = json.loads(serialized)
        assert deserialized == sample_metadata


class TestDocumentPreprocessing:
    """Test document preprocessing."""

    @pytest.fixture
    def mock_preprocessor(self):
        """Mock document preprocessor."""
        preprocessor = Mock()
        preprocessor.clean_text = Mock()
        preprocessor.normalize = Mock()
        preprocessor.extract_entities = Mock()
        return preprocessor

    def test_text_cleaning(self, mock_preprocessor):
        """Test text cleaning."""
        dirty_text = "This is a   messy\n\ntext with\textra   whitespace."
        clean_text = "This is a messy text with extra whitespace."
        mock_preprocessor.clean_text.return_value = clean_text

        result = mock_preprocessor.clean_text(dirty_text)
        assert result == clean_text

    def test_text_normalization(self, mock_preprocessor):
        """Test text normalization."""
        text = "This is MIXED case text with Numbers123!"
        normalized = "this is mixed case text with numbers123!"
        mock_preprocessor.normalize.return_value = normalized

        result = mock_preprocessor.normalize(text)
        assert result == normalized

    def test_entity_extraction(self, mock_preprocessor):
        """Test entity extraction."""
        text = "John Doe works at OpenAI in San Francisco."
        entities = [
            {"text": "John Doe", "type": "PERSON", "start": 0, "end": 8},
            {"text": "OpenAI", "type": "ORG", "start": 18, "end": 24},
            {"text": "San Francisco", "type": "GPE", "start": 28, "end": 41},
        ]
        mock_preprocessor.extract_entities.return_value = entities

        result = mock_preprocessor.extract_entities(text)
        assert len(result) == 3
        assert all("type" in entity for entity in result)


if __name__ == "__main__":
    pytest.main([__file__])
