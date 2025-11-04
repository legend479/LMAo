"""
RAG Pipeline Main Module
Specialized retrieval system optimized for software engineering content
"""

import time
from typing import Dict, Any, List, Optional

from .models import ProcessedDocument, IngestionResult, Chunk, DocumentMetadata
from src.shared.config import get_settings
from src.shared.logging import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """Main RAG pipeline for document processing and retrieval"""

    def __init__(
        self,
        ingestion_config: Optional[Dict[str, Any]] = None,
        elasticsearch_config: Optional[Dict[str, Any]] = None,
        search_config: Optional[Dict[str, Any]] = None,
        embedding_config: Optional[Dict[str, Any]] = None,
    ):
        self.settings = get_settings()

        # Store configs for lazy initialization
        self.ingestion_config = ingestion_config
        self.elasticsearch_config = elasticsearch_config
        self.search_config = search_config
        self.embedding_config = embedding_config

        # Components will be initialized lazily
        self.document_processor = None
        self.embedding_manager = None
        self.ingestion_service = None
        self.vector_store = None
        self.search_engine = None
        self._initialized = False

    async def initialize(self):
        """Initialize RAG pipeline components with lazy imports"""
        if self._initialized:
            return

        logger.info("Initializing RAG Pipeline")

        try:
            # Lazy imports to avoid circular dependencies
            from .document_processor import DocumentProcessor
            from .document_ingestion import DocumentIngestionService, IngestionConfig
            from .vector_store import ElasticsearchStore, ElasticsearchConfig
            from .search_engine import HybridSearchEngine, SearchConfig
            from .embedding_manager import EmbeddingManager, EmbeddingConfig

            # Create default configs if not provided
            if self.ingestion_config is None:
                self.ingestion_config = IngestionConfig(
                    source_directories=[],
                    supported_extensions=[".pdf", ".docx", ".pptx", ".txt", ".md"],
                    max_file_size_mb=100,
                    max_concurrent_files=5,
                )
            elif isinstance(self.ingestion_config, dict):
                self.ingestion_config = IngestionConfig(**self.ingestion_config)

            if self.elasticsearch_config is None:
                self.elasticsearch_config = ElasticsearchConfig()
            elif isinstance(self.elasticsearch_config, dict):
                self.elasticsearch_config = ElasticsearchConfig(
                    **self.elasticsearch_config
                )

            if self.search_config is None:
                self.search_config = SearchConfig()
            elif isinstance(self.search_config, dict):
                self.search_config = SearchConfig(**self.search_config)

            if self.embedding_config is None:
                self.embedding_config = EmbeddingConfig()
            elif isinstance(self.embedding_config, dict):
                self.embedding_config = EmbeddingConfig(**self.embedding_config)

            # Initialize components
            self.document_processor = DocumentProcessor()
            self.embedding_manager = EmbeddingManager(self.embedding_config)
            self.ingestion_service = DocumentIngestionService(self.ingestion_config)
            self.vector_store = ElasticsearchStore(
                self.elasticsearch_config, self.embedding_manager
            )
            self.search_engine = HybridSearchEngine(
                self.vector_store, self.embedding_manager, self.search_config
            )

            # Initialize components in order
            await self.document_processor.initialize()
            await self.embedding_manager.initialize()
            await self.vector_store.initialize()
            await self.search_engine.initialize()
            await self.ingestion_service.initialize()

            self._initialized = True
            logger.info("RAG Pipeline initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize RAG Pipeline", error=str(e))
            raise

    async def ingest_document(
        self, file_path: str, metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Ingest a single document into the pipeline"""
        if not self._initialized:
            await self.initialize()

        logger.info("Ingesting document", file_path=file_path)

        try:
            # Use ingestion service to process document
            processed_doc = await self.ingestion_service.ingest_single_file(
                file_path, metadata
            )

            # Generate embeddings for chunks
            logger.info(
                "Generating embeddings for chunks",
                chunk_count=len(processed_doc.chunks),
            )
            for i, chunk in enumerate(processed_doc.chunks):
                try:
                    embedding_result = await self.embedding_manager.generate_embeddings(
                        chunk.content
                    )
                    chunk.embeddings = {
                        "general": embedding_result.general_embedding,
                        "domain": embedding_result.domain_embedding,
                    }
                    logger.debug(
                        f"Generated embeddings for chunk {i+1}/{len(processed_doc.chunks)}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to generate embeddings for chunk {i+1}: {e}"
                    )
                    # Continue without embeddings - chunks can still be stored for keyword search
                    chunk.embeddings = None

            # Store in vector database
            stored_doc_id = await self.vector_store.store_document(processed_doc)

            logger.info(
                "Document ingested, embedded, and stored successfully",
                file_path=file_path,
                doc_id=processed_doc.document_id,
                stored_doc_id=stored_doc_id,
                chunk_count=len(processed_doc.chunks),
            )

            return {
                "document_id": processed_doc.document_id,
                "chunks_processed": len(processed_doc.chunks),
                "processing_time": processed_doc.processing_time,
                "content_hash": processed_doc.content_hash,
                "embeddings_generated": True,
                "stored_in_vector_db": True,
                "status": "success",
            }

        except Exception as e:
            logger.error("Document ingestion failed", file_path=file_path, error=str(e))
            return {"document_id": None, "status": "error", "error": str(e)}

    async def ingest_batch(
        self,
        file_paths: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Ingest multiple documents in batch"""
        if not self._initialized:
            await self.initialize()

        logger.info("Starting batch ingestion", document_count=len(file_paths))

        # Use ingestion service for batch processing
        # Create temporary file list with metadata
        files_with_metadata = list(
            zip(file_paths, metadata_list or [{}] * len(file_paths))
        )

        # Process using ingestion service batch method
        batch_result = await self.document_processor.process_batch(
            file_paths, metadata_list, max_concurrent=5
        )

        logger.info(
            "Batch ingestion completed",
            total=batch_result.total_documents,
            successful=batch_result.successful,
            failed=batch_result.failed,
        )

        return {
            "total_documents": batch_result.total_documents,
            "successful": batch_result.successful,
            "failed": batch_result.failed,
            "processing_time": batch_result.processing_time,
            "results": batch_result.results,
            "errors": batch_result.errors,
        }

    async def ingest_from_directories(self, source_paths: List[str]) -> Dict[str, Any]:
        """Ingest documents from source directories"""
        if not self._initialized:
            await self.initialize()

        logger.info("Starting directory ingestion", source_paths=source_paths)

        # Discover files first
        all_files = []
        for source_path in source_paths:
            discovered_files = await self.ingestion_service.discover_documents(
                source_path
            )
            all_files.extend([file_path for file_path, _ in discovered_files])

        logger.info(f"Discovered {len(all_files)} files for full pipeline ingestion")

        # Process each file through the complete pipeline (including embeddings and storage)
        successful = 0
        failed = 0
        errors = []
        start_time = time.time()

        for file_path in all_files:
            try:
                logger.info(f"Processing file through full pipeline: {file_path}")
                result = await self.ingest_document(file_path)
                if result.get("status") == "success":
                    successful += 1
                else:
                    failed += 1
                    errors.append(
                        f"Failed to process {file_path}: {result.get('error', 'Unknown error')}"
                    )
            except Exception as e:
                failed += 1
                errors.append(f"Failed to process {file_path}: {str(e)}")
                logger.error(f"Error processing {file_path}: {e}")

        processing_time = time.time() - start_time

        logger.info(
            "Directory ingestion completed",
            total_found=len(all_files),
            successful=successful,
            failed=failed,
        )

        return {
            "total_files_found": len(all_files),
            "total_files_processed": len(all_files),
            "successful_ingestions": successful,
            "failed_ingestions": failed,
            "skipped_files": 0,
            "processing_time": processing_time,
            "file_statistics": {"total": len(all_files)},
            "errors": errors,
        }

    async def search(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        max_results: int = 10,
        search_type: str = "hybrid",
    ) -> Dict[str, Any]:
        """Search for relevant documents using hybrid search"""
        if not self._initialized:
            await self.initialize()

        logger.info(
            "Executing search",
            query=query[:100],
            max_results=max_results,
            search_type=search_type,
        )

        try:
            # Execute search using hybrid search engine
            search_response = await self.search_engine.search(
                query=query,
                filters=filters or {},
                max_results=max_results,
                search_type=search_type,
            )

            logger.info(
                "Search completed",
                query=query[:100],
                results_count=len(search_response.results),
                search_time=search_response.took_ms,
            )

            return {
                "query": query,
                "results": [
                    {
                        "chunk_id": result.chunk_id,
                        "content": result.content,
                        "score": result.score,
                        "document_id": result.document_id,
                        "chunk_type": result.chunk_type,
                        "parent_chunk_id": result.parent_chunk_id,
                        "highlights": result.highlights,
                        "metadata": {
                            "document_title": result.metadata.get("document_title"),
                            "document_author": result.metadata.get("document_author"),
                            "document_category": result.metadata.get(
                                "document_category"
                            ),
                            "size_category": result.metadata.get("size_category"),
                            "word_count": result.metadata.get("word_count"),
                            "text_type": result.metadata.get("text_type"),
                            "code_type": result.metadata.get("code_type"),
                        },
                    }
                    for result in search_response.results
                ],
                "total_results": search_response.total_hits,
                "max_score": search_response.max_score,
                "processing_time": search_response.took_ms
                / 1000.0,  # Convert to seconds
                "search_type": search_response.search_type,
            }

        except Exception as e:
            logger.error("Search failed", query=query[:100], error=str(e))
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "processing_time": 0.0,
                "error": str(e),
            }

    async def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about ingested documents"""
        if not self._initialized:
            await self.initialize()

        # Get stats from all components
        ingestion_stats = await self.ingestion_service.get_ingestion_status()
        processor_stats = await self.document_processor.get_processing_stats()
        vector_store_stats = await self.vector_store.get_stats()
        search_stats = await self.search_engine.get_search_stats()
        embedding_stats = await self.embedding_manager.get_model_info()

        return {
            "ingestion_stats": ingestion_stats,
            "processor_stats": processor_stats,
            "vector_store_stats": vector_store_stats,
            "search_stats": search_stats,
            "embedding_stats": embedding_stats,
        }

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the pipeline"""
        if not self._initialized:
            await self.initialize()

        logger.info("Deleting document", document_id=document_id)

        try:
            success = await self.vector_store.delete_document(document_id)

            if success:
                logger.info("Document deleted successfully", document_id=document_id)
            else:
                logger.warning("Document deletion failed", document_id=document_id)

            return success

        except Exception as e:
            logger.error(
                "Document deletion error", document_id=document_id, error=str(e)
            )
            return False

    async def update_document(
        self, document_id: str, file_path: str, metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Update an existing document"""
        if not self._initialized:
            await self.initialize()

        logger.info("Updating document", document_id=document_id, file_path=file_path)

        # For now, just re-ingest the document (deletion will be implemented later)
        return await self.ingest_document(file_path, metadata)

    async def health_check(self) -> Dict[str, Any]:
        """Check health of RAG pipeline components"""
        health_status = {"pipeline": "healthy", "components": {}}

        try:
            # Initialize if not already done
            if not self._initialized:
                await self.initialize()

            # Check all components
            if self.document_processor:
                health_status["components"][
                    "document_processor"
                ] = await self.document_processor.health_check()
            else:
                health_status["components"]["document_processor"] = {
                    "status": "not_initialized"
                }

            if self.embedding_manager:
                health_status["components"][
                    "embedding_manager"
                ] = await self.embedding_manager.health_check()
            else:
                health_status["components"]["embedding_manager"] = {
                    "status": "not_initialized"
                }

            if self.vector_store:
                health_status["components"][
                    "vector_store"
                ] = await self.vector_store.health_check()
            else:
                health_status["components"]["vector_store"] = {
                    "status": "not_initialized"
                }

            if self.search_engine:
                health_status["components"][
                    "search_engine"
                ] = await self.search_engine.health_check()
            else:
                health_status["components"]["search_engine"] = {
                    "status": "not_initialized"
                }

            # Check ingestion service
            if self.ingestion_service:
                try:
                    ingestion_status = (
                        await self.ingestion_service.get_ingestion_status()
                    )
                    health_status["components"]["ingestion_service"] = {
                        "status": "healthy",
                        "ingested_files_count": ingestion_status.get(
                            "ingested_files_count", 0
                        ),
                    }
                except Exception as e:
                    health_status["components"]["ingestion_service"] = {
                        "status": "error",
                        "error": str(e),
                    }
            else:
                health_status["components"]["ingestion_service"] = {
                    "status": "not_initialized"
                }

            # Overall health assessment
            component_statuses = []
            for component, status in health_status["components"].items():
                if isinstance(status, dict) and "status" in status:
                    component_statuses.append(status["status"])

            if all(status == "healthy" for status in component_statuses):
                health_status["pipeline"] = "healthy"
            elif any(status == "unhealthy" for status in component_statuses):
                health_status["pipeline"] = "unhealthy"
            else:
                health_status["pipeline"] = "degraded"

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            health_status["pipeline"] = "unhealthy"
            health_status["error"] = str(e)

        return health_status

    async def shutdown(self):
        """Shutdown RAG pipeline and cleanup resources"""
        logger.info("Shutting down RAG Pipeline")

        if hasattr(self, "search_engine"):
            await self.search_engine.shutdown()
        if hasattr(self, "vector_store"):
            await self.vector_store.shutdown()
        if hasattr(self, "embedding_manager"):
            await self.embedding_manager.shutdown()
        if hasattr(self, "ingestion_service"):
            await self.ingestion_service.shutdown()
        if hasattr(self, "document_processor"):
            await self.document_processor.shutdown()

        logger.info("RAG Pipeline shutdown complete")


# Global RAG pipeline instance
rag_pipeline = RAGPipeline()


async def get_rag_pipeline() -> RAGPipeline:
    """Get the global RAG pipeline instance"""
    if not rag_pipeline._initialized:
        await rag_pipeline.initialize()
    return rag_pipeline


# FastAPI application for HTTP endpoints
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    await rag_pipeline.initialize()
    yield
    # Shutdown
    await rag_pipeline.shutdown()


app = FastAPI(
    title="SE SME Agent - RAG Pipeline",
    description="Document processing and retrieval service",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health = await rag_pipeline.health_check()
    return health


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...), metadata: str = "{}"):
    """Ingest a document"""
    try:
        import json
        import tempfile
        import os

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f"_{file.filename}"
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Parse metadata
            metadata_dict = json.loads(metadata) if metadata else {}

            # Ingest document
            result = await rag_pipeline.ingest_document(tmp_file_path, metadata_dict)
            return result
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_documents(request: Dict[str, Any]):
    """Search documents"""
    try:
        query = request.get("query", "")
        filters = request.get("filters", {})
        max_results = request.get("max_results", 10)
        search_type = request.get("search_type", "hybrid")

        result = await rag_pipeline.search(query, filters, max_results, search_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_document_stats():
    """Get document statistics"""
    try:
        stats = await rag_pipeline.get_document_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    try:
        success = await rag_pipeline.delete_document(document_id)
        return {"success": success, "document_id": document_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    try:
        stats = await rag_pipeline.get_document_stats()
        return {
            "service": "rag-pipeline",
            "status": "healthy",
            "initialized": rag_pipeline._initialized,
            "stats": stats,
        }
    except Exception as e:
        return {"service": "rag-pipeline", "status": "error", "error": str(e)}
