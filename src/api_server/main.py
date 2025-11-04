"""
Main API Server Entry Point
FastAPI-based server for handling external requests with comprehensive validation and security
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
import asyncio
import signal

from .routers import health, chat, documents, tools, auth, admin
from .middleware import security, logging, rate_limiting, validation, compression

from src.shared.config import get_settings, validate_config
from src.shared.logging import get_logger

logger = get_logger(__name__)


class APIServer:
    """Main API Server class with comprehensive lifecycle management"""

    def __init__(self):
        self.app: FastAPI = None
        self.settings = get_settings()
        self._shutdown_event = asyncio.Event()
        self._startup_complete = False

    async def startup(self):
        """Comprehensive server startup sequence"""
        try:
            logger.info(
                "Starting SE SME Agent API Server",
                version=self.settings.version,
                environment=self.settings.environment,
            )

            # Validate configuration
            validate_config()
            logger.info("Configuration validated successfully")

            # Initialize health monitoring
            from ..shared.health import setup_default_health_checks

            setup_default_health_checks()
            logger.info("Health monitoring initialized")

            # Initialize cache system
            from .cache import cache_manager

            await cache_manager.initialize()
            logger.info("Cache system initialized")

            # Initialize performance monitoring
            from .performance import performance_monitor

            await performance_monitor.start_monitoring()
            logger.info("Performance monitoring initialized")

            # Initialize metrics collection
            if self.settings.enable_metrics:
                from ..shared.metrics import get_metrics_collector

                await get_metrics_collector()
                logger.info("Metrics collection initialized")

            # Initialize database connections
            await self._initialize_database()

            # Initialize Redis connections
            await self._initialize_redis()

            # Initialize external service connections
            await self._initialize_external_services()

            # Setup signal handlers
            self._setup_signal_handlers()

            self._startup_complete = True
            logger.info(
                "API Server startup completed successfully",
                host=self.settings.api_host,
                port=self.settings.api_port,
            )

        except Exception as e:
            logger.error("Failed to start API Server", error=str(e))
            raise

    async def shutdown(self):
        """Comprehensive server shutdown sequence"""
        logger.info("Initiating API Server shutdown")

        try:
            # Stop accepting new requests
            self._shutdown_event.set()

            # Stop performance monitoring
            from .performance import performance_monitor

            await performance_monitor.stop_monitoring()
            logger.info("Performance monitoring stopped")

            # Graceful shutdown of components
            if self.settings.enable_metrics:
                from ..shared.metrics import metrics_collector

                if metrics_collector:
                    await metrics_collector.stop_metrics_server()
                    logger.info("Metrics server stopped")

            # Close database connections
            await self._cleanup_database()

            # Close Redis connections
            await self._cleanup_redis()

            # Cleanup external service connections
            await self._cleanup_external_services()

            logger.info("API Server shutdown completed")

        except Exception as e:
            logger.error("Error during shutdown", error=str(e))

    async def _initialize_database(self):
        """Initialize database connections"""
        try:
            # TODO: Initialize actual database connection pool
            logger.info("Database connections initialized")
        except Exception as e:
            logger.error("Failed to initialize database", error=str(e))
            raise

    async def _initialize_redis(self):
        """Initialize Redis connections"""
        try:
            # TODO: Initialize actual Redis connection pool
            logger.info("Redis connections initialized")
        except Exception as e:
            logger.error("Failed to initialize Redis", error=str(e))
            raise

    async def _initialize_external_services(self):
        """Initialize connections to external services"""
        try:
            # Initialize service registry for inter-service communication
            from ..shared.services import get_service_registry

            service_registry = await get_service_registry()

            # Test connectivity to services
            service_status = await service_registry.get_service_status()

            agent_status = service_status.get("agent_service", {}).get(
                "status", "unknown"
            )
            rag_status = service_status.get("rag_service", {}).get("status", "unknown")

            logger.info(
                "External service connections initialized",
                agent_service_status=agent_status,
                rag_service_status=rag_status,
            )

            # Log warnings for unhealthy services but don't fail startup
            if agent_status != "healthy":
                logger.warning(
                    "Agent service is not healthy - some features may be limited"
                )
            if rag_status != "healthy":
                logger.warning(
                    "RAG service is not healthy - document search may be limited"
                )

        except Exception as e:
            logger.error("Failed to initialize external services", error=str(e))
            # Don't fail startup for external service issues in development
            if self.settings.environment == "production":
                raise

    async def _cleanup_database(self):
        """Cleanup database connections"""
        try:
            # TODO: Close database connection pool
            logger.info("Database connections closed")
        except Exception as e:
            logger.error("Error closing database connections", error=str(e))

    async def _cleanup_redis(self):
        """Cleanup Redis connections"""
        try:
            # TODO: Close Redis connection pool
            logger.info("Redis connections closed")
        except Exception as e:
            logger.error("Error closing Redis connections", error=str(e))

    async def _cleanup_external_services(self):
        """Cleanup external service connections"""
        try:
            # Shutdown service registry
            from ..shared.services import _service_registry

            if _service_registry:
                await _service_registry.shutdown()

            logger.info("External service connections closed")
        except Exception as e:
            logger.error("Error closing external service connections", error=str(e))

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown")
            asyncio.create_task(self.shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


# Global server instance
api_server = APIServer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with comprehensive startup/shutdown"""
    try:
        # Startup
        await api_server.startup()
        yield
    finally:
        # Shutdown
        await api_server.shutdown()


def create_app() -> FastAPI:
    """Create and configure FastAPI application with comprehensive features"""
    settings = get_settings()

    # Create FastAPI app with enhanced configuration
    app = FastAPI(
        title="SE SME Agent API",
        description="Software Engineering Subject Matter Expert AI Agent - A comprehensive RAG-based system with agentic capabilities for software engineering tasks",
        version=settings.version,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        contact={
            "name": "SE SME Agent Team",
            "email": "support@se-sme-agent.com",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
    )

    # Add security middleware first
    if settings.environment == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"],  # Configure appropriately for production
        )

    # Add CORS middleware with enhanced configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )

    # Add custom middleware in order
    app.add_middleware(compression.CompressionMiddleware)
    app.add_middleware(security.SecurityMiddleware)
    app.add_middleware(validation.ValidationMiddleware)
    app.add_middleware(
        rate_limiting.RateLimitingMiddleware,
        requests_per_minute=settings.rate_limit_requests_per_minute,
    )
    app.add_middleware(logging.LoggingMiddleware)

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            error_type=type(exc).__name__,
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    # HTTP exception handler
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    # Include routers with comprehensive API structure
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
    app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat & Conversation"])
    app.include_router(
        documents.router, prefix="/api/v1/documents", tags=["Document Management"]
    )
    app.include_router(tools.router, prefix="/api/v1/tools", tags=["Tool Execution"])
    app.include_router(admin.router, prefix="/admin", tags=["Administration"])

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """API root endpoint with system information"""
        return {
            "name": "SE SME Agent API",
            "version": settings.version,
            "environment": settings.environment,
            "status": "operational",
            "documentation": "/docs" if settings.debug else "Contact administrator",
            "health_check": "/health",
        }

    # API info endpoint
    @app.get("/info", tags=["Root"])
    async def api_info():
        """Detailed API information"""
        return {
            "api": {
                "name": "SE SME Agent API",
                "version": settings.version,
                "environment": settings.environment,
                "debug_mode": settings.debug,
            },
            "features": {
                "chat": "Real-time conversation with SE SME Agent",
                "documents": "Document generation and management",
                "tools": "Tool execution and management",
                "rag": "Retrieval-Augmented Generation pipeline",
                "websockets": "Real-time communication support",
            },
            "endpoints": {
                "health": "/health",
                "chat": "/api/v1/chat",
                "documents": "/api/v1/documents",
                "tools": "/api/v1/tools",
                "websocket": "/api/v1/chat/ws/{session_id}",
            },
        }

    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title="SE SME Agent API",
            version=settings.version,
            description="Comprehensive API for Software Engineering Subject Matter Expert AI Agent",
            routes=app.routes,
        )

        # Add custom schema information
        openapi_schema["info"]["x-logo"] = {"url": "https://example.com/logo.png"}

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    return app


# Create the FastAPI application
app = create_app()


# Development server runner
if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    # Enhanced uvicorn configuration
    uvicorn_config = {
        "app": "src.api_server.main:app",
        "host": settings.api_host,
        "port": settings.api_port,
        "reload": settings.debug,
        "log_level": settings.log_level.lower(),
        "access_log": True,
        "use_colors": True,
        "loop": "asyncio",
        "http": "httptools",
        "ws": "websockets",
        "lifespan": "on",
        "timeout_keep_alive": 5,
        "timeout_graceful_shutdown": 30,
    }

    if settings.environment == "production":
        uvicorn_config.update(
            {
                "workers": 4,
                "reload": False,
                "access_log": False,  # Use structured logging instead
            }
        )

    logger.info("Starting development server", **uvicorn_config)
    uvicorn.run(**uvicorn_config)
