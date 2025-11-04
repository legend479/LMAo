# RAG Pipeline Module
# Specialized retrieval system optimized for software engineering content

from .main import RAGPipeline
from .document_processor import DocumentProcessor
from .search_engine import (
    HybridSearchEngine,
    SearchConfig,
    RRFConfig,
    FusedSearchResult,
)
from .vector_store import (
    ElasticsearchStore,
    SearchResult,
    SearchResponse,
    ElasticsearchConfig,
)
from .embedding_manager import EmbeddingManager
from .document_ingestion import DocumentIngestionService
from .reranker import BGEReranker, RerankerConfig
from .chunking_strategies import (
    ChunkingStrategy,
    ChunkingConfigAdvanced,
    ChunkQuality,
    BaseChunkingStrategy,
    HierarchicalChunkingStrategy,
    ChunkingManager,
)
from .models import (
    DocumentType,
    DocumentMetadata,
    Chunk,
    ProcessedDocument,
    ChunkingConfig,
    IngestionResult,
)

__all__ = [
    # Main pipeline
    "RAGPipeline",
    # Core components
    "DocumentProcessor",
    "HybridSearchEngine",
    "ElasticsearchStore",
    "EmbeddingManager",
    "DocumentIngestionService",
    "BGEReranker",
    # Chunking strategies
    "ChunkingStrategy",
    "ChunkingConfigAdvanced",
    "ChunkQuality",
    "BaseChunkingStrategy",
    "HierarchicalChunkingStrategy",
    "ChunkingManager",
    # Configuration classes
    "SearchConfig",
    "RRFConfig",
    "ElasticsearchConfig",
    "RerankerConfig",
    "ChunkingConfig",
    # Data models
    "DocumentType",
    "DocumentMetadata",
    "Chunk",
    "ProcessedDocument",
    "IngestionResult",
    "SearchResult",
    "SearchResponse",
    "FusedSearchResult",
]
