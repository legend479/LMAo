"""
Agent Server Main Module
Core orchestration engine managing conversations and tool execution
"""

from typing import Dict, Any, Optional
from datetime import datetime

from .orchestrator import LangGraphOrchestrator
from .planning import PlanningModule
from .memory import MemoryManager
from .tools.registry import ToolExecutionRegistry
from src.shared.config import get_settings
from src.shared.logging import get_logger

logger = get_logger(__name__)


class AgentServer:
    """Main agent server class for handling conversations and tool orchestration"""

    def __init__(self):
        self.settings = get_settings()
        self.orchestrator = LangGraphOrchestrator()
        self.planning_module = PlanningModule()
        self.memory_manager = MemoryManager()
        self.tool_registry = ToolExecutionRegistry()
        self._initialized = False

    async def initialize(self):
        """Initialize all agent components"""
        if self._initialized:
            return

        logger.info("Initializing Agent Server")

        # Initialize components
        await self.orchestrator.initialize()
        await self.planning_module.initialize()
        await self.memory_manager.initialize()
        await self.tool_registry.initialize()

        self._initialized = True
        logger.info("Agent Server initialized successfully")

    async def process_message(
        self, message: str, session_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a user message through the agent pipeline"""
        if not self._initialized:
            await self.initialize()

        logger.info("Processing message", session_id=session_id, user_id=user_id)

        try:
            # Retrieve conversation context
            context = await self.memory_manager.get_context(session_id)

            # Create execution plan
            plan = await self.planning_module.create_plan(message, context)

            # Execute plan through orchestrator
            result = await self.orchestrator.execute_plan(plan, session_id)

            # Store conversation state
            await self.memory_manager.store_interaction(
                session_id, message, result, user_id
            )

            return {
                "response": result.response,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": result.metadata,
            }

        except Exception as e:
            logger.error(
                "Error processing message", error=str(e), session_id=session_id
            )
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again.",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {"error": True, "error_type": type(e).__name__},
            }

    async def get_available_tools(self) -> Dict[str, Any]:
        """Get list of available tools"""
        return await self.tool_registry.list_tools()

    async def execute_tool(
        self, tool_name: str, parameters: Dict[str, Any], session_id: str
    ) -> Dict[str, Any]:
        """Execute a specific tool"""
        logger.info("Executing tool", tool_name=tool_name, session_id=session_id)

        try:
            # Import ExecutionContext here to avoid circular imports
            from .tools.registry import ExecutionContext, ExecutionPriority

            # Create execution context
            context = ExecutionContext(
                session_id=session_id,
                priority=ExecutionPriority.NORMAL,
                timeout=300,
                max_retries=3,
            )

            tool = await self.tool_registry.get_tool(tool_name)
            result = await tool.execute(parameters, context)

            return {
                "tool_name": tool_name,
                "result": result.data,
                "status": "success",
                "execution_time": result.execution_time,
                "metadata": result.metadata,
            }

        except Exception as e:
            logger.error("Tool execution failed", tool_name=tool_name, error=str(e))
            return {
                "tool_name": tool_name,
                "result": None,
                "status": "error",
                "error": str(e),
                "execution_time": 0.0,
            }

    async def shutdown(self):
        """Shutdown agent server and cleanup resources"""
        logger.info("Shutting down Agent Server")

        if hasattr(self, "orchestrator"):
            await self.orchestrator.shutdown()
        if hasattr(self, "memory_manager"):
            await self.memory_manager.shutdown()
        if hasattr(self, "tool_registry"):
            await self.tool_registry.shutdown()

        logger.info("Agent Server shutdown complete")


# Global agent server instance
agent_server = AgentServer()


async def get_agent_server() -> AgentServer:
    """Get the global agent server instance"""
    if not agent_server._initialized:
        await agent_server.initialize()
    return agent_server


# FastAPI application for HTTP endpoints
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    await agent_server.initialize()
    yield
    # Shutdown
    await agent_server.shutdown()


app = FastAPI(
    title="SE SME Agent - Agent Server",
    description="Agent orchestration and tool execution service",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "agent-server",
        "version": "1.0.0",
        "initialized": agent_server._initialized,
    }


@app.post("/process")
async def process_message(request: Dict[str, Any]):
    """Process a message through the agent"""
    try:
        message = request.get("message", "")
        session_id = request.get("session_id", "default")
        user_id = request.get("user_id")

        result = await agent_server.process_message(message, session_id, user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools")
async def get_available_tools():
    """Get available tools"""
    try:
        tools = await agent_server.get_available_tools()
        return tools
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/{tool_name}/execute")
async def execute_tool(tool_name: str, request: Dict[str, Any]):
    """Execute a specific tool"""
    try:
        parameters = request.get("parameters", {})
        session_id = request.get("session_id", "default")

        result = await agent_server.execute_tool(tool_name, parameters, session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    return {
        "service": "agent-server",
        "status": "healthy",
        "initialized": agent_server._initialized,
    }
