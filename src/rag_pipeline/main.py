"""
RAG Pipeline Main Module
Specialized retrieval system optimized for software engineering content
"""

from typing import Dict, Any, List, Optional

from .document_processor import DocumentProcessor
from .document_ingestion import DocumentIngestionService, IngestionConfig
from .vector_store import ElasticsearchStore, ElasticsearchConfig
from .search_engine import HybridSearchEngine, SearchConfig
from .embedding_manager import EmbeddingManager, EmbeddingConfig
from src.shared.config import get_settings
from src.shared.logging import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """Main RAG pipeline for document processing and retrieval"""

    def __init__(
        self,
        ingestion_config: Optional[IngestionConfig] = None,
        elasticsearch_config: Optional[ElasticsearchConfig] = None,
        search_config: Optional[SearchConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
    ):
        self.settings = get_settings()

        # Create default configs if not provided
        if ingestion_config is None:
            ingestion_config = IngestionConfig(
                source_directories=[],
                supported_extensions=[".pdf", ".docx", ".pptx", ".txt", ".md"],
                max_file_size_mb=100,
                max_concurrent_files=5,
            )

        if elasticsearch_config is None:
            elasticsearch_config = ElasticsearchConfig()

        if search_config is None:
            search_config = SearchConfig()

        if embedding_config is None:
            embedding_config = EmbeddingConfig()

        # Initialize components
        self.document_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager(embedding_config)
        self.ingestion_service = DocumentIngestionService(ingestion_config)
        self.vector_store = ElasticsearchStore(
            elasticsearch_config, self.embedding_manager
        )
        self.search_engine = HybridSearchEngine(
            self.vector_store, self.embedding_manager, search_config
        )
        self._initialized = False

    async def initialize(self):
        """Initialize RAG pipeline components"""
        if self._initialized:
            return

        logger.info("Initializing RAG Pipeline")

        # Initialize components in order
        await self.document_processor.initialize()
        await self.embedding_manager.initialize()
        await self.vector_store.initialize()
        await self.search_engine.initialize()
        await self.ingestion_service.initialize()

        self._initialized = True
        logger.info("RAG Pipeline initialized successfully")

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
            for chunk in processed_doc.chunks:
                embedding_result = await self.embedding_manager.generate_embeddings(
                    chunk.content
                )
                chunk.embeddings = {
                    "general": embedding_result.general_embedding,
                    "domain": embedding_result.domain_embedding,
                }

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

        # Use ingestion service to process directories
        result = await self.ingestion_service.ingest_documents(source_paths)

        logger.info(
            "Directory ingestion completed",
            total_found=result.total_files_found,
            successful=result.successful_ingestions,
            failed=result.failed_ingestions,
        )

        return {
            "total_files_found": result.total_files_found,
            "total_files_processed": result.total_files_processed,
            "successful_ingestions": result.successful_ingestions,
            "failed_ingestions": result.failed_ingestions,
            "skipped_files": result.skipped_files,
            "processing_time": result.processing_time,
            "file_statistics": result.file_statistics,
            "errors": result.errors,
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
            # Check all components
            health_status["components"][
                "document_processor"
            ] = await self.document_processor.health_check()
            health_status["components"][
                "embedding_manager"
            ] = await self.embedding_manager.health_check()
            health_status["components"][
                "vector_store"
            ] = await self.vector_store.health_check()
            health_status["components"][
                "search_engine"
            ] = await self.search_engine.health_check()

            # Check ingestion service
            ingestion_status = await self.ingestion_service.get_ingestion_status()
            health_status["components"]["ingestion_service"] = {
                "status": "healthy",
                "ingested_files_count": ingestion_status["ingested_files_count"],
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

app = FastAPI(
    title="SE SME Agent - RAG Pipeline",
    description="Document processing and retrieval service",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup"""
    await rag_pipeline.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await rag_pipeline.shutdown()


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
