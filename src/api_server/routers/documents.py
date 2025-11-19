"""
Documents Router
Handles document generation, upload, and management endpoints with RAG integration
"""

import os
import uuid
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from enum import Enum

from .auth import get_current_active_user, User
from src.shared.logging import get_logger
from src.shared.config import get_settings
import src.shared.services as services

logger = get_logger(__name__)
router = APIRouter()


class DocumentFormat(str, Enum):
    DOCX = "docx"
    PDF = "pdf"
    PPT = "ppt"


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    TXT = "txt"
    MD = "md"
    HTML = "html"
    JSON = "json"
    CSV = "csv"


class DocumentGenerationRequest(BaseModel):
    content: str
    format: DocumentFormat
    template: Optional[str] = "default"
    title: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    file_size: int
    document_type: str
    upload_status: str
    processing_status: str
    created_at: datetime


class DocumentResponse(BaseModel):
    document_id: str
    download_url: str
    format: DocumentFormat
    status: str


class DocumentInfo(BaseModel):
    document_id: str
    original_filename: str
    file_size: int
    document_type: str
    processing_status: str
    created_at: datetime
    processing_time: Optional[float] = None
    total_chunks: int = 0
    indexed_at: Optional[datetime] = None


@router.post("/generate", response_model=DocumentResponse)
async def generate_document(
    request: DocumentGenerationRequest,
    current_user: User = Depends(get_current_active_user),
):
    """Generate a document in the specified format"""
    # TODO: Integrate with document generation tool

    settings = get_settings()

    try:
        # Call document_generation tool via Agent service
        agent_client = await services.get_agent_client()

        params: Dict[str, Any] = {
            "content": request.content,
            "format": request.format.value,
            "template": request.template or "default",
            "validate": True,
        }

        if request.title:
            params["filename"] = request.title

        session_id = f"docgen_{current_user.id}_{int(datetime.utcnow().timestamp())}"

        agent_response = await agent_client.execute_tool(
            "document_generation",
            params,
            session_id,
        )

        if agent_response.get("status") != "success":
            logger.error(
                "Document generation tool returned non-success status",
                status=agent_response.get("status"),
                metadata=agent_response.get("metadata"),
            )
            raise HTTPException(status_code=500, detail="Document generation failed")

        result = agent_response.get("result") or {}
        filename = result.get("filename")
        file_path = result.get("file_path")
        format_str = result.get("format", request.format.value)

        if not file_path or not os.path.exists(file_path):
            logger.error(
                "Generated document file not found",
                file_path=file_path,
            )
            raise HTTPException(
                status_code=500, detail="Generated document file not found"
            )

        # Compute file size and content hash
        import hashlib

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        file_size = len(file_bytes)
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        from src.shared.database import DocumentOperations

        document = DocumentOperations.create_document(
            user_id=current_user.id,
            original_filename=filename or f"generated_document.{format_str}",
            file_path=file_path,
            content_hash=content_hash,
            file_size=file_size,
            document_type=format_str,
            processing_status="completed",
        )

        download_url = f"/api/v1/documents/download/{document.id}"

        return DocumentResponse(
            document_id=document.id,
            download_url=download_url,
            format=request.format,
            status="completed",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Document generation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Document generation failed")


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
):
    """Upload a document for processing and RAG ingestion"""

    settings = get_settings()

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Check file size
    if file.size and file.size > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_file_size / (1024*1024):.1f}MB",
        )

    # Determine document type
    file_extension = Path(file.filename).suffix.lower()
    document_type_mapping = {
        ".pdf": DocumentType.PDF,
        ".docx": DocumentType.DOCX,
        ".pptx": DocumentType.PPTX,
        ".txt": DocumentType.TXT,
        ".md": DocumentType.MD,
        ".html": DocumentType.HTML,
        ".json": DocumentType.JSON,
        ".csv": DocumentType.CSV,
    }

    document_type = document_type_mapping.get(file_extension)
    if not document_type:
        raise HTTPException(
            status_code=400, detail=f"Unsupported file type: {file_extension}"
        )

    try:
        # Generate document ID
        document_id = str(uuid.uuid4())

        # Create upload directory if it doesn't exist
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save file to disk
        file_path = upload_dir / f"{document_id}_{file.filename}"

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        file_size = len(content)

        logger.info(
            "Document uploaded",
            document_id=document_id,
            filename=file.filename,
            file_size=file_size,
            user_id=current_user.id,
        )

        # Create document record in database
        from src.shared.database import DocumentOperations
        import hashlib

        content_hash = hashlib.sha256(content).hexdigest()

        document = DocumentOperations.create_document(
            user_id=current_user.id,
            original_filename=file.filename,
            file_path=str(file_path),
            content_hash=content_hash,
            file_size=file_size,
            document_type=document_type.value,
            processing_status="pending",
        )

        # Schedule background processing
        background_tasks.add_task(
            process_document_for_rag,
            document_id=document.id,
            file_path=str(file_path),
            user_id=current_user.id,
        )

        return DocumentUploadResponse(
            document_id=document.id,
            filename=file.filename,
            file_size=file_size,
            document_type=document_type.value,
            upload_status="completed",
            processing_status="pending",
            created_at=document.created_at,
        )

    except Exception as e:
        logger.error(f"Document upload failed: {e}")

        # Clean up file if it was created
        if "file_path" in locals() and file_path.exists():
            file_path.unlink()

        raise HTTPException(status_code=500, detail="Document upload failed")


async def process_document_for_rag(document_id: str, file_path: str, user_id: str):
    """Background task to process document through RAG pipeline"""

    try:
        logger.info(f"Starting RAG processing for document {document_id}")

        # Update processing status
        from src.shared.database import DocumentOperations

        DocumentOperations.update_document_processing_status(document_id, "processing")

        # Process through RAG pipeline
        from src.shared.services import get_rag_client

        rag_client = await get_rag_client()

        # Ingest document
        result = await rag_client.ingest_document(
            file_path=file_path,
            metadata={
                "user_id": user_id,
                "document_id": document_id,
                "uploaded_at": datetime.utcnow().isoformat(),
            },
        )

        if result.get("status") == "success":
            # Update document status
            processing_time = result.get("processing_time", 0.0)
            DocumentOperations.update_document_processing_status(
                document_id, "completed", processing_time
            )

            logger.info(
                f"Document {document_id} processed successfully",
                chunks_processed=result.get("chunks_processed", 0),
                processing_time=processing_time,
            )
        else:
            # Mark as failed
            DocumentOperations.update_document_processing_status(document_id, "failed")
            logger.error(
                f"Document {document_id} processing failed: {result.get('error')}"
            )

    except Exception as e:
        logger.error(f"RAG processing failed for document {document_id}: {e}")

        # Mark as failed
        try:
            from src.shared.database import DocumentOperations

            DocumentOperations.update_document_processing_status(document_id, "failed")
        except:
            pass


@router.get("/", response_model=List[DocumentInfo])
async def list_documents(
    current_user: User = Depends(get_current_active_user),
    status: Optional[str] = None,
    limit: int = 50,
):
    """List user's documents"""

    try:
        from src.shared.database import DocumentOperations

        documents = DocumentOperations.get_user_documents(
            user_id=current_user.id, status=status, limit=limit
        )

        return [
            DocumentInfo(
                document_id=doc.id,
                original_filename=doc.original_filename,
                file_size=doc.file_size,
                document_type=doc.document_type,
                processing_status=doc.processing_status,
                created_at=doc.created_at,
                processing_time=doc.processing_time,
                total_chunks=doc.total_chunks,
                indexed_at=doc.indexed_at,
            )
            for doc in documents
        ]

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")


@router.get("/{document_id}", response_model=DocumentInfo)
async def get_document_info(
    document_id: str, current_user: User = Depends(get_current_active_user)
):
    """Get information about a specific document"""

    try:
        from src.shared.database import DocumentOperations
        from src.shared.services import get_rag_client

        document = DocumentOperations.get_document_by_id(document_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        if document.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Document access denied")

        return DocumentInfo(
            document_id=document.id,
            original_filename=document.original_filename,
            file_size=document.file_size,
            document_type=document.document_type,
            processing_status=document.processing_status,
            created_at=document.created_at,
            processing_time=document.processing_time,
            total_chunks=document.total_chunks,
            indexed_at=document.indexed_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document")


@router.delete("/{document_id}")
async def delete_document(
    document_id: str, current_user: User = Depends(get_current_active_user)
):
    """Delete a document"""

    try:
        from src.shared.database import DocumentOperations

        document = DocumentOperations.get_document_by_id(document_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        if document.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Document access denied")

        # Delete file from disk
        file_path = Path(document.file_path)
        if file_path.exists():
            file_path.unlink()

        # Remove from RAG pipeline/vector store
        try:
            rag_client = await get_rag_client()
            rag_result = await rag_client.delete_document(document_id)

            if not rag_result.get("success", False):
                logger.warning(
                    "RAG document deletion reported failure",
                    document_id=document_id,
                    error=rag_result.get("error"),
                )
        except Exception as e:
            logger.error(
                "Failed to delete document from RAG pipeline",
                document_id=document_id,
                error=str(e),
            )

        # Delete from database
        deleted = DocumentOperations.delete_document(document_id)
        if not deleted:
            logger.warning(
                "Document deletion from database reported failure",
                document_id=document_id,
            )

        logger.info(
            "Document deleted", document_id=document_id, user_id=current_user.id
        )

        return {"message": "Document deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    max_results: int = Field(
        default=10, ge=1, le=100, description="Maximum number of results to return"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional filters for search"
    )
    search_type: str = Field(
        default="hybrid", description="Search type: semantic, keyword, or hybrid"
    )


class SearchResult(BaseModel):
    content: str = Field(..., description="Content of the search result")
    score: float = Field(..., description="Relevance score")
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    document_id: str = Field(..., description="Document ID this chunk belongs to")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )


class SearchResponse(BaseModel):
    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(..., description="List of search results")
    total_results: int = Field(..., description="Total number of results found")
    processing_time: float = Field(
        ..., description="Time taken to process the search in seconds"
    )


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    search_request: SearchRequest,
    current_user: User = Depends(get_current_active_user),
):
    """
    Search through indexed documents using RAG pipeline

    Supports semantic search, keyword search, and hybrid search modes.
    Results are ranked by relevance score.
    """

    try:
        from src.shared.services import get_rag_client
        import time

        start_time = time.time()

        logger.info(
            "Document search initiated",
            user_id=current_user.id,
            query=search_request.query[:100],  # Log first 100 chars
            max_results=search_request.max_results,
            search_type=search_request.search_type,
        )

        # Get RAG client
        rag_client = await get_rag_client()

        # Add user_id to filters to ensure users only search their own documents
        filters = search_request.filters or {}
        filters["user_id"] = current_user.id

        # Perform search through RAG pipeline
        search_result = await rag_client.search(
            query=search_request.query,
            filters=filters,
            max_results=search_request.max_results,
            search_type=search_request.search_type,
        )

        processing_time = time.time() - start_time

        # Transform RAG results to API response format
        results = []
        for item in search_result.get("results", []):
            results.append(
                SearchResult(
                    content=item.get("content", ""),
                    score=item.get("score", 0.0),
                    chunk_id=item.get("chunk_id", ""),
                    document_id=item.get("document_id", ""),
                    metadata=item.get("metadata"),
                )
            )

        logger.info(
            "Document search completed",
            user_id=current_user.id,
            results_count=len(results),
            processing_time=processing_time,
        )

        return SearchResponse(
            query=search_request.query,
            results=results,
            total_results=len(results),
            processing_time=processing_time,
        )

    except Exception as e:
        logger.error(
            "Document search failed",
            user_id=current_user.id,
            query=search_request.query[:100],
            error=str(e),
            error_type=type(e).__name__,
        )
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/download/{document_id}")
async def download_document(
    document_id: str, current_user: User = Depends(get_current_active_user)
):
    """Download a generated document"""
    # TODO: Implement document download for generated documents

    from src.shared.database import DocumentOperations

    document = DocumentOperations.get_document_by_id(document_id)

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if document.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Document access denied")

    file_path = Path(document.file_path)

    if not file_path.exists():
        logger.error(
            "Document file not found on disk",
            document_id=document_id,
            file_path=str(file_path),
        )
        raise HTTPException(status_code=404, detail="Document file not found")

    filename = document.original_filename

    ext = file_path.suffix.lower()
    media_type_map = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".ppt": "application/vnd.ms-powerpoint",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".html": "text/html",
    }
    media_type = media_type_map.get(ext, "application/octet-stream")

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename,
    )
