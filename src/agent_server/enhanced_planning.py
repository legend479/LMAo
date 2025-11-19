"""
Enhanced Planning Module
Improved task decomposition with tool-aware intent classification
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import uuid

from .orchestrator import ExecutionPlan
from .tool_intent_mapper import ToolIntentMapper, EnhancedIntentType
from .prompt_templates import PromptTemplates, PromptType
from .planning import (
    ConversationContext,
    Entity,
    QueryAnalysis,
    Goal,
    ComplexityLevel,
    EntityType,
)
from src.shared.logging import get_logger
from src.shared.llm.integration import get_llm_integration

logger = get_logger(__name__)


class EnhancedPlanningModule:
    """Enhanced planning with tool-aware intent classification"""

    def __init__(self):
        self._initialized = False
        self.llm_integration = None
        self.tool_intent_mapper = ToolIntentMapper()
        self.prompt_templates = PromptTemplates()

    async def initialize(self):
        """Initialize planning components"""
        if self._initialized:
            return

        logger.info("Initializing Enhanced Planning Module")

        # Initialize LLM integration
        self.llm_integration = await get_llm_integration()
        logger.info("LLM integration initialized for enhanced planning")

        self._initialized = True
        logger.info("Enhanced Planning Module initialized")

    async def create_plan(
        self, message: str, context: ConversationContext
    ) -> ExecutionPlan:
        """Create execution plan with enhanced intent classification"""

        logger.info("Creating enhanced execution plan", session_id=context.session_id)

        # Step 1: Enhanced intent classification
        intent_match = self.tool_intent_mapper.classify_intent(
            message, context={"history": context.message_history}
        )

        logger.info(
            "Intent classified",
            primary_intent=intent_match.primary_intent.value,
            confidence=intent_match.confidence,
            suggested_tools=intent_match.suggested_tools,
            reasoning=intent_match.reasoning,
        )

        # Step 2: Create tasks based on tool sequence
        tasks = []
        dependencies = {}

        for i, (tool_name, tool_params) in enumerate(intent_match.tool_sequence):
            task_id = f"task_{i+1}"

            # Extract additional parameters from message
            extracted_params = self.tool_intent_mapper.get_tool_parameters(
                tool_name, message, context
            )

            # Merge parameters
            parameters = {**tool_params, **extracted_params}

            # Add query/description to parameters
            if "query" not in parameters and tool_name in [
                "knowledge_retrieval",
                "rag_search",
            ]:
                parameters["query"] = message
            elif "description" not in parameters and tool_name == "code_generation":
                parameters["description"] = message
            elif "topic" not in parameters and tool_name == "content_generation":
                parameters["topic"] = message

            # Determine task type
            task_type = self._map_tool_to_task_type(tool_name)

            # Create task
            task = {
                "id": task_id,
                "name": f"{tool_name}_task",
                "type": task_type,
                "tool": tool_name if task_type == "tool_execution" else None,
                "parameters": parameters,
                "priority": 1,
                "estimated_duration": self._estimate_task_duration(tool_name),
                "intent": intent_match.primary_intent.value,
            }

            tasks.append(task)

            # Set up dependencies (sequential by default)
            if i > 0:
                dependencies[task_id] = [f"task_{i}"]

                # Handle parameter injection from previous task
                if tool_params.get("content_from_previous") or tool_params.get(
                    "code_from_previous"
                ):
                    # Use placeholder syntax for orchestrator to resolve
                    prev_task_id = f"task_{i}"
                    parameters["content"] = f"{{{{ {prev_task_id}.result }}}}"
            else:
                dependencies[task_id] = []

        # Calculate estimated duration
        estimated_duration = sum(task["estimated_duration"] for task in tasks)

        # Create execution plan
        plan = ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            tasks=tasks,
            dependencies=dependencies,
            estimated_duration=estimated_duration,
            priority=1,
            recovery_strategies=self._create_recovery_strategies(tasks),
            parallel_groups=self._identify_parallel_groups(tasks, dependencies),
        )

        logger.info(
            "Enhanced execution plan created",
            plan_id=plan.plan_id,
            task_count=len(tasks),
            estimated_duration=estimated_duration,
            intent=intent_match.primary_intent.value,
        )

        return plan

    def _map_tool_to_task_type(self, tool_name: str) -> str:
        """Map tool name to task type"""
        mapping = {
            "knowledge_retrieval": "tool_execution",
            "rag_search": "tool_execution",
            "code_generation": "code_generation",
            "compiler_runtime": "tool_execution",
            "content_generation": "content_generation",
            "document_generation": "tool_execution",
            "email_automation": "tool_execution",
            "general_processing": "general_processing",
        }
        return mapping.get(tool_name, "tool_execution")

    def _estimate_task_duration(self, tool_name: str) -> float:
        """Estimate task duration based on tool"""
        durations = {
            "knowledge_retrieval": 2.0,
            "rag_search": 2.0,
            "code_generation": 5.0,
            "compiler_runtime": 3.0,
            "content_generation": 4.0,
            "document_generation": 2.0,
            "email_automation": 1.5,
            "general_processing": 1.0,
        }
        return durations.get(tool_name, 2.0)

    def _create_recovery_strategies(
        self, tasks: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Create recovery strategies for tasks"""
        strategies = {}

        for task in tasks:
            task_id = task["id"]
            tool_name = task.get("tool", "")

            if tool_name in ["compiler_runtime", "code_generation"]:
                strategies[task_id] = {
                    "strategy": "retry",
                    "max_retries": 2,
                    "backoff_delay": 1.0,
                }
            elif tool_name in ["email_automation", "document_generation"]:
                strategies[task_id] = {
                    "strategy": "fallback",
                    "fallback_result": f"{tool_name} completed with fallback method",
                }
            else:
                strategies[task_id] = {
                    "strategy": "retry",
                    "max_retries": 3,
                    "backoff_delay": 0.5,
                }

        return strategies

    def _identify_parallel_groups(
        self, tasks: List[Dict[str, Any]], dependencies: Dict[str, List[str]]
    ) -> List[List[str]]:
        """Identify tasks that can run in parallel"""
        parallel_groups = []

        # Find tasks with no dependencies
        independent_tasks = [
            task["id"] for task in tasks if not dependencies.get(task["id"], [])
        ]

        if len(independent_tasks) > 1:
            parallel_groups.append(independent_tasks)

        return parallel_groups
