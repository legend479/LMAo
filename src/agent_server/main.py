"""
Agent Server Main Module
Core orchestration engine managing conversations and tool execution
"""

from typing import Dict, Any, Optional, List
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

        # Multi-step reasoning engine
        self.reasoning_engine = None

        # Feedback and learning components
        self.feedback_collector = None
        self.feedback_analyzer = None
        self.feedback_learning = None

        # Feature flags
        self.enable_feedback = True
        self.enable_learning = True
        self.enable_multi_step_reasoning = True

        self._initialized = False

    async def initialize(self):
        """Initialize all agent components in correct order"""
        if self._initialized:
            return

        logger.info("Initializing Agent Server")

        try:
            # Step 1: Initialize tool registry FIRST
            await self.tool_registry.initialize()
            logger.info("Tool registry initialized")

            # Step 2: Initialize orchestrator and connect to tool registry
            await self.orchestrator.initialize()
            self.orchestrator.set_tool_registry(self.tool_registry)
            logger.info("Orchestrator initialized and connected to tool registry")

            # Step 3: Auto-register default tools (orchestrator is now ready)
            from .tools.auto_register import register_default_tools

            registered_tool_ids = await register_default_tools(self.tool_registry)
            logger.info(f"Auto-registered {len(registered_tool_ids)} default tools")

            # Step 4: Initialize planning module
            await self.planning_module.initialize()
            logger.info("Planning module initialized")

            # Step 5: Initialize memory manager
            await self.memory_manager.initialize()
            logger.info("Memory manager initialized")

            # Step 6: Initialize multi-step reasoning engine
            if self.enable_multi_step_reasoning and self.orchestrator.adaptive_engine:
                from .multi_step_reasoning import MultiStepReasoningEngine

                self.reasoning_engine = MultiStepReasoningEngine(
                    self.orchestrator, self.orchestrator.adaptive_engine
                )
                logger.info("Multi-step reasoning engine initialized")

            # Step 7: Initialize feedback and learning systems
            if self.enable_feedback:
                from .feedback_system import FeedbackCollector, FeedbackAnalyzer

                self.feedback_collector = FeedbackCollector()
                self.feedback_analyzer = FeedbackAnalyzer()
                await self.feedback_collector.initialize()
                await self.feedback_analyzer.initialize()
                logger.info("Feedback system initialized")

            if self.enable_learning:
                from .feedback_learning import FeedbackLearningSystem

                self.feedback_learning = FeedbackLearningSystem()
                await self.feedback_learning.initialize()
                logger.info("Learning system initialized")

            self._initialized = True
            logger.info("Agent Server initialized successfully with all enhancements")

        except Exception as e:
            logger.error(f"Failed to initialize Agent Server: {e}")
            # Cleanup any partially initialized components
            await self.shutdown()
            raise

    async def process_message(
        self, message: str, session_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a user message through the agent pipeline with multi-step reasoning support"""
        if not self._initialized:
            await self.initialize()

        logger.info("Processing message", session_id=session_id, user_id=user_id)

        try:
            # Retrieve conversation context
            context = await self.memory_manager.get_context(session_id)

            # Create execution plan
            plan = await self.planning_module.create_plan(message, context)

            # Check if this requires multi-step reasoning
            requires_multi_step = self._requires_multi_step_reasoning(plan, message)

            if requires_multi_step and self.reasoning_engine:
                logger.info("Using multi-step reasoning for complex query")

                # Create reasoning chain
                reasoning_chain = await self.reasoning_engine.create_reasoning_chain(
                    goal=message,
                    initial_context={"plan": plan, "conversation_context": context},
                    session_id=session_id,
                )

                # Execute reasoning chain
                reasoning_result = await self.reasoning_engine.execute_reasoning_chain(
                    reasoning_chain, session_id
                )

                # Convert reasoning result to standard format
                result = self._convert_reasoning_result_to_execution_result(
                    reasoning_result, plan, session_id
                )
            else:
                # Standard execution through orchestrator
                result = await self.orchestrator.execute_plan(
                    plan, session_id, original_query=message
                )

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

    def _requires_multi_step_reasoning(self, plan, message: str) -> bool:
        """Determine if a query requires multi-step reasoning"""

        # Check for multi-step indicators in message
        multi_step_indicators = [
            "and then",
            "after that",
            "next",
            "also",
            "additionally",
            "first",
            "second",
            "finally",
            "step by step",
        ]

        message_lower = message.lower()
        has_indicators = any(
            indicator in message_lower for indicator in multi_step_indicators
        )

        # Check if plan has many tasks with complex dependencies
        has_complex_plan = len(plan.tasks) > 3

        # Check for analysis or synthesis tasks
        has_reasoning_tasks = any(
            task.get("type") in ["analysis", "planning", "synthesis"]
            for task in plan.tasks
        )

        return has_indicators or (has_complex_plan and has_reasoning_tasks)

    def _convert_reasoning_result_to_execution_result(
        self, reasoning_result: Dict[str, Any], plan, session_id: str
    ):
        """Convert reasoning chain result to ExecutionResult format"""

        from .orchestrator import ExecutionResult, ExecutionState

        # Extract results from reasoning chain
        results = reasoning_result.get("results", [])

        # Build tool results
        tool_results = []
        for i, result in enumerate(results):
            tool_results.append(
                {
                    "task_id": f"reasoning_step_{i+1}",
                    "result": result,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        # Generate response from final context
        final_context = reasoning_result.get("final_context", {})
        response = self._synthesize_reasoning_response(
            reasoning_result.get("goal", ""), results, final_context
        )

        # Determine state
        state = (
            ExecutionState.COMPLETED
            if reasoning_result.get("completed")
            else ExecutionState.FAILED
        )

        return ExecutionResult(
            response=response,
            metadata={
                "reasoning_chain_id": reasoning_result.get("chain_id"),
                "steps_completed": reasoning_result.get("steps_completed", 0),
                "total_steps": reasoning_result.get("total_steps", 0),
                "multi_step_reasoning": True,
                "plan_id": plan.plan_id,
            },
            execution_time=0.0,  # Would need to track this
            state=state,
            tool_results=tool_results,
            execution_path=[f"reasoning_step_{i+1}" for i in range(len(results))],
        )

    def _synthesize_reasoning_response(
        self, goal: str, results: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> str:
        """Synthesize a coherent response from reasoning chain results"""

        if not results:
            return f"I've analyzed your request: '{goal}', but couldn't complete all reasoning steps."

        # Extract meaningful content from results
        response_parts = []

        for i, result in enumerate(results):
            if isinstance(result, dict):
                result_content = result.get("result", "")
                if isinstance(result_content, str) and len(result_content) > 20:
                    response_parts.append(result_content)

        if response_parts:
            # Combine results coherently
            if len(response_parts) == 1:
                return response_parts[0]
            else:
                return "\n\n".join(response_parts)

        return f"I've completed the multi-step analysis for: '{goal}'"

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

            tool = await self.tool_registry.get_tool_by_name(tool_name)
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

    async def collect_feedback(
        self,
        session_id: str,
        query: str,
        response: str,
        rating: Optional[float] = None,
        text_feedback: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Collect user feedback"""
        if not self.enable_feedback or not self.feedback_collector:
            return {"success": False, "error": "Feedback system not enabled"}

        try:
            from .feedback_system import FeedbackType, FeedbackCategory

            feedback = await self.feedback_collector.collect_feedback(
                session_id=session_id,
                feedback_type=(
                    FeedbackType.RELEVANCE_RATING
                    if rating
                    else FeedbackType.DETAILED_FEEDBACK
                ),
                category=FeedbackCategory.OVERALL_SATISFACTION,
                query=query,
                response=response,
                rating=rating,
                text_feedback=text_feedback,
                user_id=user_id,
            )

            return {
                "success": True,
                "feedback_id": feedback.feedback_id,
                "timestamp": feedback.timestamp,
            }
        except Exception as e:
            logger.error(f"Failed to collect feedback: {e}")
            return {"success": False, "error": str(e)}

    async def analyze_feedback(self, days: int = 7) -> Dict[str, Any]:
        """Analyze recent feedback"""
        if not self.enable_feedback or not self.feedback_analyzer:
            return {"success": False, "error": "Feedback system not enabled"}

        try:
            recent_feedback = self.feedback_collector.get_recent_feedback(days=days)

            if not recent_feedback:
                return {
                    "success": True,
                    "message": "No feedback to analyze",
                    "feedback_count": 0,
                }

            analysis = await self.feedback_analyzer.analyze_feedback(
                recent_feedback, days
            )

            return {
                "success": True,
                "feedback_count": len(recent_feedback),
                "analysis": {
                    "total_feedback": analysis.total_feedback,
                    "positive_feedback": analysis.positive_feedback,
                    "negative_feedback": analysis.negative_feedback,
                    "average_rating": analysis.average_rating,
                    "common_issues": analysis.common_issues,
                    "suggested_improvements": analysis.suggested_improvements,
                },
            }
        except Exception as e:
            logger.error(f"Failed to analyze feedback: {e}")
            return {"success": False, "error": str(e)}

    async def learn_from_feedback(
        self, days: int = 7, auto_apply: bool = False
    ) -> Dict[str, Any]:
        """Learn from feedback and generate optimizations"""
        if not self.enable_learning or not self.feedback_learning:
            return {"success": False, "error": "Learning system not enabled"}

        try:
            recent_feedback = self.feedback_collector.get_recent_feedback(days=days)

            if len(recent_feedback) < 10:
                return {
                    "success": False,
                    "error": f"Insufficient feedback: {len(recent_feedback)} < 10",
                }

            analysis = await self.feedback_analyzer.analyze_feedback(
                recent_feedback, days
            )
            insights = await self.feedback_learning.learn_from_feedback(
                recent_feedback, analysis
            )

            application_result = None
            if auto_apply:
                application_result = await self.feedback_learning.apply_optimizations(
                    insights, auto_apply=True
                )

            return {
                "success": True,
                "feedback_analyzed": len(recent_feedback),
                "insights": {
                    "prompt_optimizations": len(insights.prompt_optimizations),
                    "strategy_adjustments": len(insights.strategy_adjustments),
                    "recommendations": insights.recommendations,
                },
                "optimizations_applied": application_result is not None,
                "application_result": application_result,
            }
        except Exception as e:
            logger.error(f"Failed to learn from feedback: {e}")
            return {"success": False, "error": str(e)}

    async def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        if not self.enable_feedback or not self.feedback_collector:
            return {"enabled": False}

        try:
            health = await self.feedback_collector.health_check()
            return {
                "enabled": True,
                "total_feedback": health.get("total_feedback", 0),
                "feedback_by_category": health.get("feedback_by_category", {}),
                "recent_feedback_7d": health.get("recent_feedback_7d", 0),
            }
        except Exception as e:
            return {"enabled": True, "error": str(e)}

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
