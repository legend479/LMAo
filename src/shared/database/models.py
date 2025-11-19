"""
SQLAlchemy Database Models
Based on existing Pydantic models in shared/models.py
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    Boolean,
    ForeignKey,
    JSON,
    Float,
)
import uuid
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, Any, Optional

Base = declarative_base()


class User(Base):
    """User model for authentication and session management"""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    last_login = Column(DateTime(timezone=True), nullable=True)

    # User preferences and settings
    preferences = Column(JSON, default=dict)

    # Relationships
    sessions = relationship(
        "Session", back_populates="user", cascade="all, delete-orphan"
    )
    documents = relationship(
        "Document", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return (
            f"<User(id='{self.id}', username='{self.username}', email='{self.email}')>"
        )


class Session(Base):
    """Session model for conversation and workflow tracking"""

    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    title = Column(String(255), nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    last_activity = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    is_active = Column(Boolean, default=True, nullable=False)

    # Session context and state
    context = Column(JSON, default=dict)
    workflow_state = Column(JSON, default=dict)

    # Session metadata
    session_type = Column(
        String(50), default="conversation", nullable=False
    )  # conversation, workflow, analysis
    session_metadata = Column(JSON, default=dict)

    # Relationships
    user = relationship("User", back_populates="sessions")
    tool_executions = relationship(
        "ToolExecution", back_populates="session", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return (
            f"<Session(id='{self.id}', user_id='{self.user_id}', title='{self.title}')>"
        )


class Document(Base):
    """Document model for RAG pipeline and knowledge management"""

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )

    # Document identification
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)

    # Document metadata
    file_size = Column(Integer, nullable=False)
    document_type = Column(String(50), nullable=False)  # pdf, docx, txt, etc.
    mime_type = Column(String(100), nullable=True)

    # Content information
    title = Column(String(500), nullable=True)
    author = Column(String(255), nullable=True)
    language = Column(String(10), nullable=True)
    page_count = Column(Integer, nullable=True)
    word_count = Column(Integer, nullable=True)

    # Processing information
    processing_status = Column(
        String(50), default="pending", nullable=False
    )  # pending, processing, completed, failed
    processing_started_at = Column(DateTime(timezone=True), nullable=True)
    processing_completed_at = Column(DateTime(timezone=True), nullable=True)
    processing_time = Column(Float, nullable=True)  # seconds

    # Chunking and embedding information
    total_chunks = Column(Integer, default=0, nullable=False)
    embedding_model = Column(String(100), nullable=True)
    indexed_at = Column(DateTime(timezone=True), nullable=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Additional metadata
    document_metadata = Column(JSON, default=dict)
    tags = Column(JSON, default=list)  # List of tags for categorization

    # Relationships
    user = relationship("User", back_populates="documents")

    def __repr__(self):
        return f"<Document(id='{self.id}', filename='{self.original_filename}', status='{self.processing_status}')>"


class ToolExecution(Base):
    """Tool execution model for tracking tool usage and performance"""

    __tablename__ = "tool_executions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False, index=True
    )

    # Tool identification
    tool_name = Column(String(100), nullable=False, index=True)
    tool_version = Column(String(20), nullable=True)
    tool_category = Column(String(50), nullable=True)

    # Execution details
    started_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)
    execution_time = Column(Float, nullable=True)  # seconds

    # Execution status and results
    status = Column(
        String(50), default="running", nullable=False
    )  # running, completed, failed, cancelled
    success = Column(Boolean, nullable=True)

    # Input and output
    input_data = Column(JSON, default=dict)
    output_data = Column(JSON, default=dict)
    error_message = Column(Text, nullable=True)
    error_code = Column(String(50), nullable=True)

    # Performance metrics
    memory_usage_mb = Column(Float, nullable=True)
    cpu_usage_percent = Column(Float, nullable=True)

    # Resource allocation
    allocated_resources = Column(JSON, default=dict)
    resource_pool = Column(String(50), nullable=True)

    # Metadata
    execution_metadata = Column(JSON, default=dict)
    request_id = Column(String(100), nullable=True, index=True)

    # Relationships
    session = relationship("Session", back_populates="tool_executions")

    def __repr__(self):
        return f"<ToolExecution(id='{self.id}', tool='{self.tool_name}', status='{self.status}')>"


class WorkflowExecution(Base):
    """Workflow execution model for LangGraph orchestrator tracking"""

    __tablename__ = "workflow_executions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False, index=True
    )

    # Workflow identification
    workflow_name = Column(String(100), nullable=False, index=True)
    workflow_version = Column(String(20), nullable=True)
    plan_id = Column(String(100), nullable=False, index=True)

    # Execution timeline
    started_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)
    paused_at = Column(DateTime(timezone=True), nullable=True)
    resumed_at = Column(DateTime(timezone=True), nullable=True)

    # Execution status
    status = Column(
        String(50), default="running", nullable=False
    )  # running, paused, completed, failed, cancelled
    current_step = Column(String(100), nullable=True)
    total_steps = Column(Integer, nullable=True)
    completed_steps = Column(Integer, default=0, nullable=False)

    # State management
    workflow_state = Column(JSON, default=dict)
    checkpoint_data = Column(JSON, default=dict)
    execution_path = Column(JSON, default=list)

    # Results and errors
    final_result = Column(JSON, default=dict)
    error_message = Column(Text, nullable=True)
    error_step = Column(String(100), nullable=True)

    # Performance metrics
    total_execution_time = Column(Float, nullable=True)  # seconds
    step_execution_times = Column(JSON, default=dict)

    # Metadata
    workflow_metadata = Column(JSON, default=dict)

    # Relationships
    session = relationship("Session", foreign_keys=[session_id])

    def __repr__(self):
        return f"<WorkflowExecution(id='{self.id}', workflow='{self.workflow_name}', status='{self.status}')>"


class SystemMetric(Base):
    """System metrics model for performance monitoring"""

    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Metric identification
    metric_name = Column(String(100), nullable=False, index=True)
    metric_type = Column(
        String(50), nullable=False
    )  # counter, gauge, histogram, summary

    # Metric value and metadata
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=True)
    labels = Column(JSON, default=dict)

    # Timestamp
    timestamp = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    # Source information
    source_service = Column(String(50), nullable=True)
    source_host = Column(String(100), nullable=True)

    def __repr__(self):
        return f"<SystemMetric(name='{self.metric_name}', value={self.value}, timestamp='{self.timestamp}')>"
