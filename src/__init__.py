# SE SME Agent - Software Engineering Subject Matter Expert AI Agent
# A comprehensive RAG-based system with agentic capabilities for software engineering tasks

__version__ = "1.0.0"
__author__ = "LMAo"
__description__ = "Software Engineering Subject Matter Expert AI Agent"

# Main module exports
from . import shared, rag_pipeline, api_server, agent_server

# Core functionality shortcuts
from .shared import (
    get_service_registry,
    get_agent_client,
    get_rag_client,
    get_session_manager,
    initialize_database,
    get_database_session,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__description__",
    # Submodules
    "shared",
    "rag_pipeline",
    "api_server",
    "agent_server",
    # Core functions
    "get_service_registry",
    "get_agent_client",
    "get_rag_client",
    "get_session_manager",
    "initialize_database",
    "get_database_session",
]
