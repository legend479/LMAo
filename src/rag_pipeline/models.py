"""
RAG Pipeline Data Models
Shared data classes and models for the RAG pipeline
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types"""

    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    TXT = "txt"
    MD = "md"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    UNKNOWN = "unknown"


@dataclass
class DocumentMetadata:
    """Document metadata"""

    source_path: str = ""
    file_name: str = ""
    file_size: int = 0
    document_type: DocumentType = DocumentType.UNKNOWN
    created_at: datetime = None
    modified_at: datetime = None
    content_hash: str = ""
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    language: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    mime_type: Optional[str] = None
    subject: Optional[str] = None
    keywords: List[str] = None
    tags: List[str] = None
    encoding: Optional[str] = None
    category: Optional[str] = None
    custom_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_metadata is None:
            self.custom_metadata = {}
        if self.keywords is None:
            self.keywords = []
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.modified_at is None:
            self.modified_at = datetime.now()


@dataclass
class Chunk:
    """Document chunk with metadata"""

    content: str
    chunk_id: str
    document_id: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]  # Changed from DocumentMetadata to Dict for flexibility
    chunk_type: str = "text"
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = None
    embeddings: Optional[Dict[str, Any]] = None  # Dual embeddings (general + domain)
    quality_score: Optional[float] = None
    semantic_density: Optional[float] = None

    def __post_init__(self):
        if self.child_chunk_ids is None:
            self.child_chunk_ids = []


@dataclass
class ProcessedDocument:
    """Processed document with chunks"""

    document_id: str
    original_path: str
    content: str
    chunks: List[Chunk]
    metadata: DocumentMetadata
    processing_time: float
    content_hash: str
    total_chunks: int
    total_characters: int
    processing_errors: List[str] = None

    def __post_init__(self):
        if self.processing_errors is None:
            self.processing_errors = []
        # Auto-calculate totals if not provided
        if self.total_chunks == 0:
            self.total_chunks = len(self.chunks)
        if self.total_characters == 0:
            self.total_characters = len(self.content)


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies"""

    strategy: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    preserve_structure: bool = True
    quality_threshold: float = 0.5
    semantic_similarity_threshold: float = 0.8
    custom_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}


@dataclass
class IngestionResult:
    """Result of document ingestion process"""

    success: bool
    documents_processed: int
    total_chunks: int
    processing_time: float
    errors: List[str] = None
    processed_documents: List[ProcessedDocument] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.processed_documents is None:
            self.processed_documents = []
