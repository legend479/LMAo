"""
Agent Server Main Module
Core orchestration engine managing conversations and tool execution
"""

from typing import Dict, Any, Optional
from datetime import datetime

from .orchestrator import LangGraphOrchestrator
from .planning import PlanningModule
from .memory import MemoryManager
from .tools.registry import ToolRegistry
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
        self.tool_registry = ToolRegistry()
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
