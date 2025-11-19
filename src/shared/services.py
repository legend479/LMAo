"""
Service Integration Layer
HTTP clients and service discovery for inter-service communication
"""

import httpx
import asyncio
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime

from .config import get_settings
from .logging import get_logger

logger = get_logger(__name__)


class ServiceClient:
    """Base HTTP client for service communication"""

    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={
                    "User-Agent": "SE-SME-Agent-API/1.0",
                },
            )
        return self._client

    async def close(self):
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed for {self.base_url}: {e}")
            return {"status": "unhealthy", "error": str(e)}


class AgentServiceClient(ServiceClient):
    """HTTP client for Agent Server communication"""

    def __init__(self):
        settings = get_settings()
        base_url = f"http://{settings.agent_host}:{settings.agent_port}"
        super().__init__(base_url, timeout=180)

    async def process_message(
        self, message: str, session_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process message through agent server"""
        try:
            client = await self._get_client()

            payload = {"message": message, "session_id": session_id, "user_id": user_id}

            response = await client.post("/process", json=payload)
            response.raise_for_status()

            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Agent service HTTP error: {e.response.status_code} - {e.response.text}"
            )
            raise
        except Exception as e:
            logger.error(f"Agent service communication error: {e}")
            raise

    async def get_available_tools(self) -> Dict[str, Any]:
        """Get available tools from agent server"""
        try:
            client = await self._get_client()
            response = await client.get("/tools")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get tools: {e}")
            raise

    async def execute_tool(
        self, tool_name: str, parameters: Dict[str, Any], session_id: str
    ) -> Dict[str, Any]:
        """Execute tool through agent server"""
        try:
            client = await self._get_client()

            payload = {"parameters": parameters, "session_id": session_id}

            response = await client.post(f"/tools/{tool_name}/execute", json=payload)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            raise


class RAGServiceClient(ServiceClient):
    """HTTP client for RAG Pipeline communication"""

    def __init__(self):
        settings = get_settings()
        base_url = f"http://{settings.rag_host}:{settings.rag_port}"
        super().__init__(base_url, timeout=180)

    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = 10,
        search_type: str = "hybrid",
    ) -> Dict[str, Any]:
        """Search documents through RAG pipeline"""
        try:
            client = await self._get_client()

            payload = {
                "query": query,
                "filters": filters or {},
                "max_results": max_results,
                "search_type": search_type,
            }

            response = await client.post("/search", json=payload)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            raise

    async def ingest_document(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ingest document through RAG pipeline"""
        try:
            client = await self._get_client()

            # Read file contents for multipart upload
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found for ingestion: {file_path}")

            file_name = os.path.basename(file_path)

            with open(file_path, "rb") as f:
                file_bytes = f.read()

            # RAG /ingest expects: file: UploadFile, metadata: JSON string
            files = {"file": (file_name, file_bytes)}
            data = {"metadata": json.dumps(metadata or {})}

            response = await client.post("/ingest", files=files, data=data)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            raise

    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete document from RAG pipeline"""
        try:
            client = await self._get_client()

            response = await client.delete(f"/documents/{document_id}")
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"RAG document deletion failed: {e}")
            raise


class ServiceRegistry:
    """Service registry for managing service clients"""

    def __init__(self):
        self.agent_client = AgentServiceClient()
        self.rag_client = RAGServiceClient()
        self._health_check_interval = 60  # seconds
        self._health_check_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize service registry"""
        logger.info("Initializing service registry")

        # Start health check monitoring
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info("Service registry initialized")

    async def shutdown(self):
        """Shutdown service registry"""
        logger.info("Shutting down service registry")

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close service clients
        await self.agent_client.close()
        await self.rag_client.close()

        logger.info("Service registry shutdown complete")

    async def _health_check_loop(self):
        """Periodic health check of services"""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)

                # Check agent service health
                agent_health = await self.agent_client.health_check()
                logger.debug(
                    f"Agent service health: {agent_health.get('status', 'unknown')}"
                )

                # Check RAG service health
                rag_health = await self.rag_client.health_check()
                logger.debug(
                    f"RAG service health: {rag_health.get('status', 'unknown')}"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        agent_health = await self.agent_client.health_check()
        rag_health = await self.rag_client.health_check()

        return {
            "agent_service": agent_health,
            "rag_service": rag_health,
            "timestamp": datetime.utcnow().isoformat(),
        }


# Global service registry instance
_service_registry: Optional[ServiceRegistry] = None


async def get_service_registry() -> ServiceRegistry:
    """Get global service registry instance"""
    global _service_registry
    if _service_registry is None:
        _service_registry = ServiceRegistry()
        await _service_registry.initialize()
    return _service_registry


async def get_agent_client() -> AgentServiceClient:
    """Get agent service client"""
    registry = await get_service_registry()
    return registry.agent_client


async def get_rag_client() -> RAGServiceClient:
    """Get RAG service client"""
    registry = await get_service_registry()
    return registry.rag_client
