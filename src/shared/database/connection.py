"""
Database Connection Management
Handles SQLAlchemy engine, sessions, and database initialization
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator, Optional
import logging
from functools import lru_cache

from ..config import get_settings, get_database_config
from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database connection and session management"""

    def __init__(self):
        self.settings = get_settings()
        self.db_config = get_database_config()
        self._engine = None
        self._session_factory = None
        self._initialized = False

    @property
    def engine(self):
        """Get or create database engine"""
        if self._engine is None:
            self._create_engine()
        return self._engine

    @property
    def session_factory(self):
        """Get or create session factory"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory

    def _create_engine(self):
        """Create SQLAlchemy engine with configuration"""
        engine_kwargs = {
            "echo": self.db_config.get("echo", False),
            "pool_size": self.db_config.get("pool_size", 10),
            "max_overflow": self.db_config.get("max_overflow", 20),
            "pool_timeout": self.db_config.get("pool_timeout", 30),
            "pool_recycle": self.db_config.get("pool_recycle", 3600),
        }

        # Handle SQLite special case
        if self.db_config["url"].startswith("sqlite"):
            engine_kwargs.update(
                {"poolclass": StaticPool, "connect_args": {"check_same_thread": False}}
            )
            # Remove PostgreSQL-specific settings for SQLite
            engine_kwargs.pop("pool_size", None)
            engine_kwargs.pop("max_overflow", None)
            engine_kwargs.pop("pool_timeout", None)
            engine_kwargs.pop("pool_recycle", None)

        self._engine = create_engine(self.db_config["url"], **engine_kwargs)

        # Add connection event listeners
        self._setup_event_listeners()

        logger.info(
            f"Database engine created for: {self._mask_url(self.db_config['url'])}"
        )

    def _setup_event_listeners(self):
        """Setup database event listeners for monitoring and optimization"""

        @event.listens_for(self._engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas for better performance"""
            if self.db_config["url"].startswith("sqlite"):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=1000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.close()

        @event.listens_for(self._engine, "before_cursor_execute")
        def receive_before_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            """Log slow queries in debug mode"""
            if self.settings.debug:
                context._query_start_time = (
                    logger.time() if hasattr(logger, "time") else None
                )

        @event.listens_for(self._engine, "after_cursor_execute")
        def receive_after_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            """Log query execution time in debug mode"""
            if (
                self.settings.debug
                and hasattr(context, "_query_start_time")
                and context._query_start_time
            ):
                total = (
                    logger.time() - context._query_start_time
                    if hasattr(logger, "time")
                    else 0
                )
                if total > 0.1:  # Log queries taking more than 100ms
                    logger.debug(f"Slow query ({total:.3f}s): {statement[:100]}...")

    async def initialize(self):
        """Initialize database - create tables and setup"""
        if self._initialized:
            logger.info("Database already initialized")
            return

        try:
            logger.info("Initializing database...")

            # Create all tables
            Base.metadata.create_all(bind=self.engine)

            # Verify connection
            from sqlalchemy import text

            with self.get_session() as session:
                session.execute(text("SELECT 1"))
                session.commit()

            self._initialized = True
            logger.info("Database initialization completed successfully")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def get_session(self) -> Session:
        """Get a new database session"""
        return self.session_factory()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    async def health_check(self) -> dict:
        """Perform database health check"""
        try:
            from sqlalchemy import text

            with self.session_scope() as session:
                result = session.execute(text("SELECT 1")).scalar()

            return {
                "status": "healthy",
                "database_url": self._mask_url(self.db_config["url"]),
                "connection_test": "passed" if result == 1 else "failed",
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "database_url": self._mask_url(self.db_config["url"]),
                "error": str(e),
            }

    def close(self):
        """Close database connections"""
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections closed")

    def _mask_url(self, url: str) -> str:
        """Mask sensitive information in database URL"""
        if "://" in url:
            scheme, rest = url.split("://", 1)
            if "@" in rest:
                credentials, host_part = rest.split("@", 1)
                return f"{scheme}://***:***@{host_part}"
        return url


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


@lru_cache()
def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance (cached)"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_database_session() -> Session:
    """Get a new database session - convenience function"""
    return get_database_manager().get_session()


@contextmanager
def database_session_scope() -> Generator[Session, None, None]:
    """Get a database session with automatic transaction management"""
    with get_database_manager().session_scope() as session:
        yield session


async def initialize_database():
    """Initialize the database - convenience function"""
    await get_database_manager().initialize()


async def database_health_check() -> dict:
    """Perform database health check - convenience function"""
    return await get_database_manager().health_check()


def close_database_connections():
    """Close all database connections - convenience function"""
    global _db_manager
    if _db_manager:
        _db_manager.close()
        _db_manager = None
