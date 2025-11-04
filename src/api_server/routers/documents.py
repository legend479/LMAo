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
from pydantic import BaseModel, Field
from enum import Enum

from .auth import get_current_active_user, User
from src.shared.logging import get_logger
from src.shared.config import get_settings

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
async def generate_document(request: DocumentGenerationRequest):
    """Generate a document in the specified format"""
    # TODO: Integrate with document generation tool
    return DocumentResponse(
        document_id="doc_123",
        download_url="/api/v1/documents/download/doc_123",
        format=request.format,
        status="pending",
    )


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

        # TODO: Remove from RAG pipeline/vector store

        # Delete from database
        # Note: This would need to be implemented in DocumentOperations

        logger.info(f"Document {document_id} deleted", user_id=current_user.id)

        return {"message": "Document deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")


@router.get("/download/{document_id}")
async def download_document(document_id: str):
    """Download a generated document"""
    # TODO: Implement document download for generated documents
    raise HTTPException(status_code=404, detail="Document not found")
