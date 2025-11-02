"""
Application Startup System
Centralized initialization and validation for all services
"""

import asyncio
import sys
from datetime import datetime

from .config import get_settings, validate_config
from .logging import configure_logging, get_logger
from .validation import validate_configuration, print_validation_results
from .health import setup_default_health_checks, health_monitor
from .metrics import get_metrics_collector

# Configure logging first
configure_logging()
logger = get_logger(__name__)


class StartupManager:
    """Manages application startup sequence"""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.settings = get_settings()
        self.startup_time = datetime.utcnow()
        self._initialized = False

    async def initialize(self, skip_validation: bool = False) -> bool:
        """Initialize the application with comprehensive checks"""

        if self._initialized:
            logger.warning("Application already initialized")
            return True

        logger.info(
            "Starting application initialization",
            service=self.service_name,
            environment=self.settings.environment,
        )

        try:
            # Step 1: Validate configuration
            if not skip_validation:
                if not await self._validate_configuration():
                    return False

            # Step 2: Initialize core systems
            await self._initialize_core_systems()

            # Step 3: Initialize service-specific components
            await self._initialize_service_components()

            # Step 4: Perform health checks
            await self._perform_initial_health_checks()

            # Step 5: Start monitoring
            await self._start_monitoring()

            self._initialized = True

            startup_duration = (datetime.utcnow() - self.startup_time).total_seconds()

            logger.info(
                "Application initialization completed successfully",
                service=self.service_name,
                startup_duration=startup_duration,
            )

            return True

        except Exception as e:
            logger.error(
                "Application initialization failed",
                service=self.service_name,
                error=str(e),
            )
            return False

    async def _validate_configuration(self) -> bool:
        """Validate application configuration"""

        logger.info("Validating configuration")

        try:
            # Basic config validation
            validate_config()

            # Comprehensive validation
            result = validate_configuration()
            print_validation_results(result)

            if not result.is_valid:
                logger.error(
                    "Configuration validation failed - cannot start application"
                )
                return False

            if result.warnings:
                logger.warning("Configuration has warnings but will continue startup")

            return True

        except Exception as e:
            logger.error("Configuration validation error", error=str(e))
            return False

    async def _initialize_core_systems(self):
        """Initialize core application systems"""

        logger.info("Initializing core systems")

        # Initialize database
        try:
            from .database import initialize_database, database_health_check

            await initialize_database()
            logger.info("Database initialized")

            # Register database health checker
            from .health import ServiceHealthChecker

            async def db_health_check():
                return await database_health_check()

            db_checker = ServiceHealthChecker("database", db_health_check)
            health_monitor.register_checker(db_checker)

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            # Don't fail startup for database issues in development
            if self.settings.environment == "production":
                raise

        # Initialize health monitoring
        setup_default_health_checks()
        logger.info("Health monitoring initialized")

        # Initialize metrics collection
        await get_metrics_collector()
        logger.info("Metrics collection initialized")

        # Add service-specific health checker
        from .health import ServiceHealthChecker

        service_checker = ServiceHealthChecker(
            self.service_name, check_function=self._service_health_check
        )
        health_monitor.register_checker(service_checker)

        logger.info("Core systems initialized")

    async def _initialize_service_components(self):
        """Initialize service-specific components - override in subclasses"""

        logger.info("Initializing service components", service=self.service_name)

        # This method should be overridden by specific service startup managers
        # For now, just log that it's a placeholder
        logger.info("No service-specific components to initialize")

    async def _perform_initial_health_checks(self):
        """Perform initial health checks"""

        logger.info("Performing initial health checks")

        system_health = await health_monitor.check_all()

        if system_health.overall_status.value == "unhealthy":
            logger.error("Initial health check failed - some components are unhealthy")
            # Don't fail startup for health check issues, just log them

        logger.info(
            "Initial health checks completed",
            overall_status=system_health.overall_status.value,
            component_count=len(system_health.components),
        )

    async def _start_monitoring(self):
        """Start monitoring and metrics collection"""

        logger.info("Starting monitoring systems")

        # Metrics server is already started by get_metrics_collector()
        # Additional monitoring setup can be added here

        logger.info("Monitoring systems started")

    async def _service_health_check(self) -> dict:
        """Service-specific health check - override in subclasses"""

        return {
            "status": "healthy" if self._initialized else "unhealthy",
            "message": f"{self.service_name} service is {'running' if self._initialized else 'not initialized'}",
            "details": {
                "service": self.service_name,
                "initialized": self._initialized,
                "startup_time": self.startup_time.isoformat(),
            },
        }

    async def shutdown(self):
        """Shutdown the application gracefully"""

        logger.info("Starting application shutdown", service=self.service_name)

        try:
            # Close database connections
            try:
                from .database import close_database_connections

                close_database_connections()
                logger.info("Database connections closed")
            except Exception as e:
                logger.error(f"Error closing database connections: {e}")

            # Stop monitoring
            from .metrics import metrics_collector

            await metrics_collector.stop_metrics_server()

            # Additional cleanup can be added here

            logger.info("Application shutdown completed", service=self.service_name)

        except Exception as e:
            logger.error(
                "Error during shutdown", service=self.service_name, error=str(e)
            )

    def is_initialized(self) -> bool:
        """Check if application is initialized"""
        return self._initialized


class APIServerStartup(StartupManager):
    """Startup manager for API Server"""

    def __init__(self):
        super().__init__("api_server")

    async def _initialize_service_components(self):
        """Initialize API server specific components"""

        logger.info("Initializing API server components")

        # API server specific initialization
        # This could include database connections, external service clients, etc.

        logger.info("API server components initialized")


class AgentServerStartup(StartupManager):
    """Startup manager for Agent Server"""

    def __init__(self):
        super().__init__("agent_server")

    async def _initialize_service_components(self):
        """Initialize Agent server specific components"""

        logger.info("Initializing Agent server components")

        # Initialize agent server components
        from ..agent_server.main import get_agent_server

        agent_server = await get_agent_server()
        logger.info("Agent server initialized")

        # Register agent-specific health checker
        from .health import ServiceHealthChecker

        async def agent_health_check():
            return {
                "status": "healthy" if agent_server._initialized else "unhealthy",
                "message": "Agent server status",
                "details": {
                    "initialized": agent_server._initialized,
                    "active_executions": (
                        len(agent_server.orchestrator.active_executions)
                        if hasattr(agent_server, "orchestrator")
                        else 0
                    ),
                },
            }

        agent_checker = ServiceHealthChecker("agent_orchestrator", agent_health_check)
        health_monitor.register_checker(agent_checker)


class RAGPipelineStartup(StartupManager):
    """Startup manager for RAG Pipeline"""

    def __init__(self):
        super().__init__("rag_pipeline")

    async def _initialize_service_components(self):
        """Initialize RAG pipeline specific components"""

        logger.info("Initializing RAG pipeline components")

        # Initialize RAG pipeline components
        from ..rag_pipeline.main import get_rag_pipeline

        rag_pipeline = await get_rag_pipeline()
        logger.info("RAG pipeline initialized")

        # Register RAG-specific health checker
        from .health import ServiceHealthChecker

        async def rag_health_check():
            return await rag_pipeline.health_check()

        rag_checker = ServiceHealthChecker("rag_components", rag_health_check)
        health_monitor.register_checker(rag_checker)


def create_startup_manager(service_name: str) -> StartupManager:
    """Create appropriate startup manager for service"""

    if service_name == "api_server":
        return APIServerStartup()
    elif service_name == "agent_server":
        return AgentServerStartup()
    elif service_name == "rag_pipeline":
        return RAGPipelineStartup()
    else:
        return StartupManager(service_name)


async def initialize_application(
    service_name: str, skip_validation: bool = False
) -> bool:
    """Initialize application for the specified service"""

    startup_manager = create_startup_manager(service_name)
    return await startup_manager.initialize(skip_validation)


def run_with_startup(service_name: str, main_function, *args, **kwargs):
    """Run application with proper startup sequence"""

    async def startup_and_run():
        # Initialize application
        success = await initialize_application(service_name)

        if not success:
            logger.error("Failed to initialize application")
            sys.exit(1)

        # Run main function
        if asyncio.iscoroutinefunction(main_function):
            return await main_function(*args, **kwargs)
        else:
            return main_function(*args, **kwargs)

    try:
        return asyncio.run(startup_and_run())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error("Application failed", error=str(e))
        sys.exit(1)


# Export commonly used functions and classes
__all__ = [
    "StartupManager",
    "APIServerStartup",
    "AgentServerStartup",
    "RAGPipelineStartup",
    "create_startup_manager",
    "initialize_application",
    "run_with_startup",
]
