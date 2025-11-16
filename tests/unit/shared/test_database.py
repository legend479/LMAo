"""
Unit tests for database models and operations
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.shared.database.models import (
    Base,
    User,
    Session as DBSession,
    Document,
    ToolExecution,
    SystemMetric,
)
from src.shared.database.connection import DatabaseManager
from src.shared.database.operations import (
    UserOperations,
    SessionOperations,
    DocumentOperations,
    ToolExecutionOperations,
    MetricsOperations,
)


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing"""
    # Create temporary file
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)

    # Create engine and tables
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    yield engine, db_path

    # Cleanup
    os.unlink(db_path)


@pytest.fixture
def db_session(temp_db):
    """Create a database session for testing"""
    engine, db_path = temp_db
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    yield session

    session.close()


class TestDatabaseModels:
    """Test database model creation and relationships"""

    def test_user_model_creation(self, db_session):
        """Test User model creation"""
        user = User(
            id="test-user-1",
            username="testuser",
            email="test@example.com",
            full_name="Test User",
        )

        db_session.add(user)
        db_session.commit()

        # Verify user was created
        retrieved_user = db_session.query(User).filter(User.id == "test-user-1").first()
        assert retrieved_user is not None
        assert retrieved_user.username == "testuser"
        assert retrieved_user.email == "test@example.com"
        assert retrieved_user.full_name == "Test User"
        assert retrieved_user.is_active is True

    def test_session_model_creation(self, db_session):
        """Test Session model creation"""
        # Create user first
        user = User(id="test-user-1", username="testuser", email="test@example.com")
        db_session.add(user)
        db_session.commit()

        # Create session
        session = DBSession(
            id="test-session-1",
            user_id="test-user-1",
            title="Test Session",
            session_type="conversation",
        )

        db_session.add(session)
        db_session.commit()

        # Verify session was created
        retrieved_session = (
            db_session.query(DBSession).filter(DBSession.id == "test-session-1").first()
        )
        assert retrieved_session is not None
        assert retrieved_session.user_id == "test-user-1"
        assert retrieved_session.title == "Test Session"
        assert retrieved_session.session_type == "conversation"

    def test_document_model_creation(self, db_session):
        """Test Document model creation"""
        # Create user first
        user = User(id="test-user-1", username="testuser", email="test@example.com")
        db_session.add(user)
        db_session.commit()

        # Create document
        document = Document(
            id="test-doc-1",
            user_id="test-user-1",
            original_filename="test.pdf",
            file_path="/uploads/test.pdf",
            content_hash="abc123",
            file_size=1024,
            document_type="pdf",
        )

        db_session.add(document)
        db_session.commit()

        # Verify document was created
        retrieved_doc = (
            db_session.query(Document).filter(Document.id == "test-doc-1").first()
        )
        assert retrieved_doc is not None
        assert retrieved_doc.user_id == "test-user-1"
        assert retrieved_doc.original_filename == "test.pdf"
        assert retrieved_doc.file_size == 1024
        assert retrieved_doc.document_type == "pdf"

    def test_tool_execution_model_creation(self, db_session):
        """Test ToolExecution model creation"""
        # Create user and session first
        user = User(id="test-user-1", username="testuser", email="test@example.com")
        session = DBSession(
            id="test-session-1", user_id="test-user-1", title="Test Session"
        )
        db_session.add(user)
        db_session.add(session)
        db_session.commit()

        # Create tool execution
        execution = ToolExecution(
            id="test-exec-1",
            session_id="test-session-1",
            tool_name="test_tool",
            input_data={"param": "value"},
        )

        db_session.add(execution)
        db_session.commit()

        # Verify execution was created
        retrieved_exec = (
            db_session.query(ToolExecution)
            .filter(ToolExecution.id == "test-exec-1")
            .first()
        )
        assert retrieved_exec is not None
        assert retrieved_exec.session_id == "test-session-1"
        assert retrieved_exec.tool_name == "test_tool"
        assert retrieved_exec.input_data == {"param": "value"}

    def test_user_session_relationship(self, db_session):
        """Test User-Session relationship"""
        # Create user
        user = User(id="test-user-1", username="testuser", email="test@example.com")
        db_session.add(user)
        db_session.commit()

        # Create sessions
        session1 = DBSession(
            id="test-session-1", user_id="test-user-1", title="Session 1"
        )
        session2 = DBSession(
            id="test-session-2", user_id="test-user-1", title="Session 2"
        )

        db_session.add(session1)
        db_session.add(session2)
        db_session.commit()

        # Test relationship
        retrieved_user = db_session.query(User).filter(User.id == "test-user-1").first()
        assert len(retrieved_user.sessions) == 2
        assert retrieved_user.sessions[0].title in ["Session 1", "Session 2"]
        assert retrieved_user.sessions[1].title in ["Session 1", "Session 2"]


class TestDatabaseOperations:
    """Test database operations"""

    @pytest.fixture(autouse=True)
    def setup_operations(self, temp_db):
        """Setup database operations with test database"""
        engine, db_path = temp_db

        # Mock the database session scope to use our test database
        from src.shared.database import connection

        original_get_manager = connection.get_database_manager

        class TestDatabaseManager:
            def __init__(self):
                self.engine = engine
                self.session_factory = sessionmaker(bind=engine)

            def session_scope(self):
                from contextlib import contextmanager

                @contextmanager
                def scope():
                    session = self.session_factory()
                    try:
                        yield session
                        session.commit()
                    except Exception:
                        session.rollback()
                        raise
                    finally:
                        session.close()

                return scope()

        test_manager = TestDatabaseManager()
        connection.get_database_manager = lambda: test_manager
        connection.database_session_scope = test_manager.session_scope

        yield

        # Restore original function
        connection.get_database_manager = original_get_manager

    def test_user_operations_create_user(self):
        """Test user creation operation"""
        user = UserOperations.create_user(
            username="testuser", email="test@example.com", full_name="Test User"
        )

        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.id is not None

    def test_user_operations_get_user(self):
        """Test user retrieval operations"""
        # Create user
        user = UserOperations.create_user(username="testuser", email="test@example.com")

        # Test get by ID
        retrieved_user = UserOperations.get_user_by_id(user.id)
        assert retrieved_user is not None
        assert retrieved_user.username == "testuser"

        # Test get by username
        retrieved_user = UserOperations.get_user_by_username("testuser")
        assert retrieved_user is not None
        assert retrieved_user.email == "test@example.com"

        # Test get by email
        retrieved_user = UserOperations.get_user_by_email("test@example.com")
        assert retrieved_user is not None
        assert retrieved_user.username == "testuser"

    def test_session_operations_create_session(self):
        """Test session creation operation"""
        # Create user first
        user = UserOperations.create_user(username="testuser", email="test@example.com")

        # Create session
        session = SessionOperations.create_session(
            user_id=user.id, title="Test Session", session_type="conversation"
        )

        assert session.user_id == user.id
        assert session.title == "Test Session"
        assert session.session_type == "conversation"
        assert session.id is not None

    def test_document_operations_create_document(self):
        """Test document creation operation"""
        # Create user first
        user = UserOperations.create_user(username="testuser", email="test@example.com")

        # Create document
        document = DocumentOperations.create_document(
            user_id=user.id,
            original_filename="test.pdf",
            file_path="/uploads/test.pdf",
            content_hash="abc123",
            file_size=1024,
            document_type="pdf",
        )

        assert document.user_id == user.id
        assert document.original_filename == "test.pdf"
        assert document.file_size == 1024
        assert document.document_type == "pdf"
        assert document.id is not None

    def test_tool_execution_operations(self):
        """Test tool execution operations"""
        # Create user and session first
        user = UserOperations.create_user(username="testuser", email="test@example.com")
        session = SessionOperations.create_session(
            user_id=user.id, title="Test Session"
        )

        # Create tool execution
        execution = ToolExecutionOperations.create_tool_execution(
            session_id=session.id, tool_name="test_tool", input_data={"param": "value"}
        )

        assert execution.session_id == session.id
        assert execution.tool_name == "test_tool"
        assert execution.input_data == {"param": "value"}

        # Update execution result
        success = ToolExecutionOperations.update_tool_execution_result(
            execution_id=execution.id,
            status="completed",
            success=True,
            output_data={"result": "success"},
            execution_time=1.5,
        )

        assert success is True

    def test_metrics_operations(self):
        """Test metrics operations"""
        # Record a metric
        metric = MetricsOperations.record_metric(
            metric_name="test_metric", value=42.0, metric_type="gauge", unit="count"
        )

        assert metric.metric_name == "test_metric"
        assert metric.value == 42.0
        assert metric.metric_type == "gauge"
        assert metric.unit == "count"

        # Get metrics
        metrics = MetricsOperations.get_metrics("test_metric")
        assert len(metrics) == 1
        assert metrics[0].value == 42.0


class TestDatabaseManager:
    """Test DatabaseManager functionality"""

    def test_database_manager_initialization(self):
        """Test DatabaseManager initialization"""
        # Create a test database manager with SQLite
        import tempfile
        import os

        db_fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(db_fd)

        try:
            # Mock settings for test
            from unittest.mock import patch, MagicMock

            mock_settings = MagicMock()
            mock_settings.debug = True

            mock_db_config = {"url": f"sqlite:///{db_path}", "echo": True}

            with (
                patch(
                    "src.shared.database.connection.get_settings",
                    return_value=mock_settings,
                ),
                patch(
                    "src.shared.database.connection.get_database_config",
                    return_value=mock_db_config,
                ),
            ):
                manager = DatabaseManager()

                # Test engine creation
                engine = manager.engine
                assert engine is not None

                # Test session creation
                session = manager.get_session()
                assert session is not None
                session.close()

        finally:
            os.unlink(db_path)

    def test_database_health_check(self):
        """Test database health check"""
        import tempfile
        import os

        db_fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(db_fd)

        try:
            from unittest.mock import patch, MagicMock

            mock_settings = MagicMock()
            mock_settings.debug = False

            mock_db_config = {"url": f"sqlite:///{db_path}", "echo": False}

            with (
                patch(
                    "src.shared.database.connection.get_settings",
                    return_value=mock_settings,
                ),
                patch(
                    "src.shared.database.connection.get_database_config",
                    return_value=mock_db_config,
                ),
            ):
                manager = DatabaseManager()

                # Initialize database
                import asyncio

                asyncio.run(manager.initialize())

                # Test health check
                health = asyncio.run(manager.health_check())
                assert health["status"] == "healthy"
                assert "database_url" in health

        finally:
            os.unlink(db_path)
