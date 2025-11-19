"""
Database Operations
Common database operations and utilities
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from datetime import datetime, timedelta
import uuid

from .models import (
    User,
    Session as DBSession,
    Document,
    ToolExecution,
    WorkflowExecution,
    SystemMetric,
)
from .connection import database_session_scope


class UserOperations:
    """User-related database operations"""

    @staticmethod
    def create_user(
        username: str,
        email: str,
        full_name: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> User:
        """Create a new user"""
        with database_session_scope() as session:
            user = User(
                id=str(uuid.uuid4()),
                username=username,
                email=email,
                full_name=full_name,
                preferences=preferences or {},
            )
            session.add(user)
            session.flush()
            session.refresh(user)
            # Expunge to make it detached so it can be accessed outside session
            session.expunge(user)
            return user

    @staticmethod
    def get_user_by_id(user_id: str) -> Optional[User]:
        """Get user by ID"""
        with database_session_scope() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                session.expunge(user)
            return user

    @staticmethod
    def get_user_by_username(username: str) -> Optional[User]:
        """Get user by username"""
        with database_session_scope() as session:
            user = session.query(User).filter(User.username == username).first()
            if user:
                session.expunge(user)
            return user

    @staticmethod
    def get_user_by_email(email: str) -> Optional[User]:
        """Get user by email"""
        with database_session_scope() as session:
            user = session.query(User).filter(User.email == email).first()
            if user:
                session.expunge(user)
            return user

    @staticmethod
    def update_user_last_login(user_id: str) -> bool:
        """Update user's last login timestamp"""
        with database_session_scope() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.last_login = datetime.utcnow()
                return True
            return False

    @staticmethod
    def update_user_preferences(user_id: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences"""
        with database_session_scope() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.preferences = preferences
                return True
            return False


class SessionOperations:
    """Session-related database operations"""

    @staticmethod
    def create_session(
        user_id: str,
        title: Optional[str] = None,
        session_type: str = "conversation",
        context: Optional[Dict[str, Any]] = None,
    ) -> DBSession:
        """Create a new session"""
        with database_session_scope() as session:
            db_session = DBSession(
                id=str(uuid.uuid4()),
                user_id=user_id,
                title=title,
                session_type=session_type,
                context=context or {},
            )
            session.add(db_session)
            session.flush()
            session.refresh(db_session)
            # Expunge to make it detached so it can be accessed outside session
            session.expunge(db_session)
            return db_session

    @staticmethod
    def get_session_by_id(session_id: str) -> Optional[DBSession]:
        """Get session by ID"""
        with database_session_scope() as session:
            db_session = (
                session.query(DBSession).filter(DBSession.id == session_id).first()
            )
            if db_session:
                session.expunge(db_session)
            return db_session

    @staticmethod
    def get_user_sessions(
        user_id: str, active_only: bool = True, limit: int = 50
    ) -> List[DBSession]:
        """Get user's sessions"""
        with database_session_scope() as session:
            query = session.query(DBSession).filter(DBSession.user_id == user_id)

            if active_only:
                query = query.filter(DBSession.is_active == True)

            sessions = query.order_by(desc(DBSession.last_activity)).limit(limit).all()
            # Expunge all sessions to make them detached
            for db_session in sessions:
                session.expunge(db_session)
            return sessions

    @staticmethod
    def update_session_activity(session_id: str) -> bool:
        """Update session's last activity timestamp"""
        with database_session_scope() as session:
            db_session = (
                session.query(DBSession).filter(DBSession.id == session_id).first()
            )
            if db_session:
                db_session.last_activity = datetime.utcnow()
                return True
            return False

    @staticmethod
    def update_session_context(session_id: str, context: Dict[str, Any]) -> bool:
        """Update session context"""
        with database_session_scope() as session:
            db_session = (
                session.query(DBSession).filter(DBSession.id == session_id).first()
            )
            if db_session:
                db_session.context = context
                return True
            return False


class DocumentOperations:
    """Document-related database operations"""

    @staticmethod
    def create_document(
        user_id: str,
        original_filename: str,
        file_path: str,
        content_hash: str,
        file_size: int,
        document_type: str,
        **kwargs,
    ) -> Document:
        """Create a new document record"""
        with database_session_scope() as session:
            document = Document(
                id=str(uuid.uuid4()),
                user_id=user_id,
                original_filename=original_filename,
                file_path=file_path,
                content_hash=content_hash,
                file_size=file_size,
                document_type=document_type,
                **kwargs,
            )
            session.add(document)
            session.flush()
            session.refresh(document)
            # Expunge to make it detached so it can be accessed outside session
            session.expunge(document)
            return document

    @staticmethod
    def get_document_by_id(document_id: str) -> Optional[Document]:
        """Get document by ID"""
        with database_session_scope() as session:
            document = (
                session.query(Document).filter(Document.id == document_id).first()
            )
            if document:
                session.expunge(document)
            return document

    @staticmethod
    def get_user_documents(
        user_id: str, status: Optional[str] = None, limit: int = 100
    ) -> List[Document]:
        """Get user's documents"""
        with database_session_scope() as session:
            query = session.query(Document).filter(Document.user_id == user_id)

            if status:
                query = query.filter(Document.processing_status == status)

            documents = query.order_by(desc(Document.created_at)).limit(limit).all()
            # Expunge all documents to make them detached
            for document in documents:
                session.expunge(document)
            return documents

    @staticmethod
    def update_document_processing_status(
        document_id: str, status: str, processing_time: Optional[float] = None
    ) -> bool:
        """Update document processing status"""
        with database_session_scope() as session:
            document = (
                session.query(Document).filter(Document.id == document_id).first()
            )
            if document:
                document.processing_status = status
                if status == "processing":
                    document.processing_started_at = datetime.utcnow()
                elif status in ["completed", "failed"]:
                    document.processing_completed_at = datetime.utcnow()
                    if processing_time:
                        document.processing_time = processing_time
                return True
            return False

    @staticmethod
    def delete_document(document_id: str) -> bool:
        """Delete a document record by ID"""
        with database_session_scope() as session:
            document = (
                session.query(Document).filter(Document.id == document_id).first()
            )
            if document:
                session.delete(document)
                return True
            return False


class ToolExecutionOperations:
    """Tool execution-related database operations"""

    @staticmethod
    def create_tool_execution(
        session_id: str, tool_name: str, input_data: Dict[str, Any], **kwargs
    ) -> ToolExecution:
        """Create a new tool execution record"""
        with database_session_scope() as session:
            execution = ToolExecution(
                id=str(uuid.uuid4()),
                session_id=session_id,
                tool_name=tool_name,
                input_data=input_data,
                **kwargs,
            )
            session.add(execution)
            session.flush()
            session.refresh(execution)
            # Expunge to make it detached so it can be accessed outside session
            session.expunge(execution)
            return execution

    @staticmethod
    def update_tool_execution_result(
        execution_id: str,
        status: str,
        success: bool,
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        execution_time: Optional[float] = None,
    ) -> bool:
        """Update tool execution result"""
        with database_session_scope() as session:
            execution = (
                session.query(ToolExecution)
                .filter(ToolExecution.id == execution_id)
                .first()
            )
            if execution:
                execution.status = status
                execution.success = success
                execution.completed_at = datetime.utcnow()

                if output_data:
                    execution.output_data = output_data
                if error_message:
                    execution.error_message = error_message
                if execution_time:
                    execution.execution_time = execution_time

                return True
            return False

    @staticmethod
    def get_session_tool_executions(session_id: str) -> List[ToolExecution]:
        """Get tool executions for a session"""
        with database_session_scope() as session:
            executions = (
                session.query(ToolExecution)
                .filter(ToolExecution.session_id == session_id)
                .order_by(desc(ToolExecution.started_at))
                .all()
            )
            # Expunge all executions to make them detached
            for execution in executions:
                session.expunge(execution)
            return executions


class MetricsOperations:
    """System metrics-related database operations"""

    @staticmethod
    def record_metric(
        metric_name: str,
        value: float,
        metric_type: str = "gauge",
        labels: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None,
    ) -> SystemMetric:
        """Record a system metric"""
        with database_session_scope() as session:
            metric = SystemMetric(
                metric_name=metric_name,
                value=value,
                metric_type=metric_type,
                labels=labels or {},
                unit=unit,
            )
            session.add(metric)
            session.flush()
            session.refresh(metric)
            # Expunge to make it detached so it can be accessed outside session
            session.expunge(metric)
            return metric

    @staticmethod
    def get_metrics(
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[SystemMetric]:
        """Get metrics by name and time range"""
        with database_session_scope() as session:
            query = session.query(SystemMetric).filter(
                SystemMetric.metric_name == metric_name
            )

            if start_time:
                query = query.filter(SystemMetric.timestamp >= start_time)
            if end_time:
                query = query.filter(SystemMetric.timestamp <= end_time)

            metrics = query.order_by(desc(SystemMetric.timestamp)).limit(limit).all()
            # Expunge all metrics to make them detached
            for metric in metrics:
                session.expunge(metric)
            return metrics

    @staticmethod
    def cleanup_old_metrics(days_to_keep: int = 30) -> int:
        """Clean up old metrics data"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

        with database_session_scope() as session:
            deleted_count = (
                session.query(SystemMetric)
                .filter(SystemMetric.timestamp < cutoff_date)
                .delete()
            )
            return deleted_count
