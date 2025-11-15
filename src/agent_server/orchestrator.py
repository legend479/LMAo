"""
LangGraph Orchestrator
Stateful workflow management using LangGraph
"""

from typing import Dict, Any, List, Optional, Callable, Annotated
import asyncio
import redis.asyncio as redis
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, UTC
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.runnables import RunnableConfig

# Note: Using MemorySaver for in-memory state persistence
# MemorySaver provides a complete implementation of all async checkpoint methods
# For production with persistence across restarts, consider implementing a custom
# checkpointer with Redis or SQLite backend

from src.shared.logging import get_logger
from src.shared.config import get_settings
from src.shared.llm.integration import get_llm_integration

logger = get_logger(__name__)


class ExecutionState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class NodeType(Enum):
    TOOL_EXECUTION = "tool_execution"
    CONTENT_GENERATION = "content_generation"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    GENERAL_PROCESSING = "general_processing"
    PLANNING = "planning"
    DECISION = "decision"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    ERROR_HANDLER = "error_handler"
    RECOVERY = "recovery"


@dataclass
class ExecutionResult:
    response: str
    metadata: Dict[str, Any]
    execution_time: float
    state: ExecutionState
    tool_results: List[Dict[str, Any]]
    execution_path: List[str] = None
    checkpoints: List[Dict[str, Any]] = None


@dataclass
class ExecutionPlan:
    plan_id: str
    tasks: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]
    estimated_duration: float
    priority: int = 1
    recovery_strategies: Dict[str, Dict[str, Any]] = None
    parallel_groups: List[List[str]] = None


@dataclass
class WorkflowState:
    """State object for LangGraph workflows"""

    session_id: str
    plan_id: str
    current_task: Optional[str] = None
    # Use Annotated with operator.add for lists that can be updated by parallel nodes
    completed_tasks: Annotated[List[str], operator.add] = None
    failed_tasks: Annotated[List[str], operator.add] = None
    # Use custom reducer for dict merging
    task_results: Annotated[Dict[str, Any], lambda x, y: {**x, **y}] = None
    context: Annotated[Dict[str, Any], lambda x, y: {**x, **y}] = None
    error_count: int = 0
    last_error: Optional[str] = None
    execution_path: Annotated[List[str], operator.add] = None

    def __post_init__(self):
        if self.completed_tasks is None:
            self.completed_tasks = []
        if self.failed_tasks is None:
            self.failed_tasks = []
        if self.task_results is None:
            self.task_results = {}
        if self.context is None:
            self.context = {}
        if self.execution_path is None:
            self.execution_path = []


class WorkflowNode:
    """Represents a node in the workflow graph"""

    def __init__(
        self,
        node_id: str,
        node_type: NodeType,
        config: Dict[str, Any],
        executor: Callable,
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.config = config
        self.executor = executor
        self.retry_count = config.get("retry_count", 3)
        self.timeout = config.get("timeout", 300)  # 5 minutes default


class LangGraphOrchestrator:
    """LangGraph-based orchestration for stateful workflow management"""

    def __init__(self):
        self.settings = get_settings()
        self.redis_client: Optional[redis.Redis] = None
        self.checkpointer: Optional[BaseCheckpointSaver] = None
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.execution_history: Dict[str, List[ExecutionResult]] = {}
        self.workflow_graphs: Dict[str, StateGraph] = {}
        self.workflow_nodes: Dict[str, Dict[str, WorkflowNode]] = {}
        self.llm_integration = None
        self.tool_registry = None  # Will be set by AgentServer
        self._initialized = False

    def set_tool_registry(self, tool_registry):
        """Set the tool registry for generic tool execution"""
        self.tool_registry = tool_registry
        logger.info("Tool registry configured for orchestrator")

    async def initialize(self):
        """Initialize the orchestrator with Redis and LangGraph components"""
        if self._initialized:
            return

        logger.info("Initializing LangGraph Orchestrator")

        try:
            # Initialize Redis connection for state persistence
            redis_url = getattr(self.settings, "REDIS_URL", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)

            # Test Redis connection
            await self.redis_client.ping()
            logger.info("Redis connection established")

            # Initialize LangGraph checkpointer
            # Using MemorySaver for in-memory state persistence
            # For production persistence, implement custom checkpointer with Redis/SQLite
            try:
                self.checkpointer = MemorySaver()

                # Validate checkpointer has required async methods
                test_config = RunnableConfig(
                    configurable={
                        "thread_id": "test_init",
                        "checkpoint_ns": "test_namespace",
                    }
                )

                # Test if aget_tuple is implemented (this is what causes the error)
                try:
                    await self.checkpointer.aget_tuple(test_config)
                    logger.info(
                        "Checkpointer validation successful - aget_tuple is implemented"
                    )
                except NotImplementedError:
                    logger.error(
                        "Checkpointer aget_tuple not implemented, disabling checkpointer"
                    )
                    self.checkpointer = None
                except Exception:
                    # Other exceptions are fine (e.g., checkpoint not found)
                    pass

                if self.checkpointer:
                    logger.info(
                        "Using MemorySaver for checkpointing (in-memory state persistence)"
                    )
                else:
                    logger.warning(
                        "Checkpointer disabled - workflows will run in stateless mode"
                    )

            except Exception as e:
                logger.error(
                    "Failed to initialize checkpointer, running in stateless mode",
                    error=str(e),
                )
                self.checkpointer = None

            # Initialize LLM integration
            self.llm_integration = await get_llm_integration()
            logger.info("LLM integration initialized")

            # Initialize workflow monitoring
            await self._setup_workflow_monitoring()

            self._initialized = True
            logger.info("LangGraph Orchestrator initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize LangGraph Orchestrator", error=str(e))
            raise

    async def create_workflow_graph(self, plan: ExecutionPlan) -> StateGraph:
        """Create a LangGraph workflow from an execution plan"""

        # Create state graph
        workflow = StateGraph(WorkflowState)

        # Create nodes for each task
        nodes = {}
        for task in plan.tasks:
            task_id = task["id"]
            node_type = NodeType(task.get("type", "tool_execution"))

            # Create workflow node
            node = WorkflowNode(
                node_id=task_id,
                node_type=node_type,
                config=task,
                executor=self._create_task_executor(task),
            )
            nodes[task_id] = node

            # Add node to graph
            workflow.add_node(task_id, self._create_node_function(node))

        # Store nodes for this plan
        self.workflow_nodes[plan.plan_id] = nodes

        # Add edges based on dependencies
        self._add_workflow_edges(workflow, plan)

        # Add error handling and recovery nodes
        await self._add_error_handling_nodes(workflow, plan)

        # Set entry point
        entry_tasks = self._find_entry_tasks(plan)
        if len(entry_tasks) == 1:
            workflow.set_entry_point(entry_tasks[0])
        else:
            # Multiple entry points - create a start node
            workflow.add_node("start", self._create_start_node())
            workflow.set_entry_point("start")
            for task_id in entry_tasks:
                workflow.add_edge("start", task_id)

        # Compile workflow with checkpointer
        # Only use checkpointer if it's properly initialized
        if self.checkpointer is not None:
            compiled_workflow = workflow.compile(checkpointer=self.checkpointer)
            logger.debug(
                "Workflow compiled with checkpointer",
                plan_id=plan.plan_id,
                checkpointer_type=type(self.checkpointer).__name__,
            )
        else:
            # Compile without checkpointer as fallback
            compiled_workflow = workflow.compile()
            logger.warning(
                "Workflow compiled without checkpointer (stateless mode)",
                plan_id=plan.plan_id,
            )

        # Store compiled workflow
        self.workflow_graphs[plan.plan_id] = compiled_workflow

        return compiled_workflow

    async def execute_plan(
        self, plan: ExecutionPlan, session_id: str
    ) -> ExecutionResult:
        """Execute a plan using LangGraph workflow management"""
        logger.info(
            "Executing plan with LangGraph", plan_id=plan.plan_id, session_id=session_id
        )

        execution_id = f"{session_id}_{plan.plan_id}"
        start_time = asyncio.get_event_loop().time()

        # Track execution
        self.active_executions[execution_id] = {
            "plan": plan,
            "session_id": session_id,
            "start_time": start_time,
            "state": ExecutionState.RUNNING,
            "completed_tasks": [],
            "failed_tasks": [],
            "checkpoints": [],
        }

        try:
            # Create or get workflow graph
            if plan.plan_id not in self.workflow_graphs:
                await self.create_workflow_graph(plan)

            workflow = self.workflow_graphs[plan.plan_id]

            # Create initial state
            initial_state = WorkflowState(
                session_id=session_id,
                plan_id=plan.plan_id,
                context={"execution_id": execution_id, "start_time": start_time},
            )

            # Configure execution
            config = RunnableConfig(
                configurable={
                    "thread_id": execution_id,
                    "checkpoint_ns": f"execution_{execution_id}",
                }
            )

            # Execute workflow
            final_state = None
            accumulated_state = {
                "execution_path": [],
                "task_results": {},
                "completed_tasks": [],
                "failed_tasks": [],
            }

            try:
                async for state_update in workflow.astream(
                    asdict(initial_state), config
                ):
                    # LangGraph astream returns dict with node names as keys
                    # e.g., {"task_1": {"execution_path": [...], "task_results": {...}}}

                    if isinstance(state_update, dict):
                        # Iterate through each node's state update
                        for node_name, node_state in state_update.items():
                            if isinstance(node_state, dict):
                                # Accumulate execution path
                                if "execution_path" in node_state:
                                    accumulated_state["execution_path"].extend(
                                        node_state["execution_path"]
                                    )

                                # Accumulate task results
                                if "task_results" in node_state:
                                    accumulated_state["task_results"].update(
                                        node_state["task_results"]
                                    )

                                # Accumulate completed tasks
                                if "completed_tasks" in node_state:
                                    accumulated_state["completed_tasks"].extend(
                                        node_state["completed_tasks"]
                                    )

                                # Accumulate failed tasks
                                if "failed_tasks" in node_state:
                                    accumulated_state["failed_tasks"].extend(
                                        node_state["failed_tasks"]
                                    )

                    final_state = state_update

                    # Store checkpoint
                    checkpoint = {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "state": state_update,
                        "execution_path": accumulated_state["execution_path"].copy(),
                    }
                    self.active_executions[execution_id]["checkpoints"].append(
                        checkpoint
                    )

                # Build tool_results from accumulated task_results
                tool_results = []
                for task_id, result in accumulated_state["task_results"].items():
                    if result and isinstance(result, dict):
                        tool_results.append(
                            {
                                "task_id": task_id,
                                "result": result,
                                "timestamp": datetime.now(UTC).isoformat(),
                            }
                        )

                execution_path = accumulated_state["execution_path"]

            except NotImplementedError as nie:
                # Checkpointer method not implemented - recreate workflow without checkpointer
                logger.warning(
                    "Checkpointer method not implemented, retrying without checkpointer",
                    plan_id=plan.plan_id,
                    error=str(nie),
                )

                # Recreate workflow without checkpointer
                workflow_no_checkpoint = await self.create_workflow_graph(plan)

                # Temporarily disable checkpointer for this execution
                original_checkpointer = self.checkpointer
                self.checkpointer = None

                try:
                    # Recreate the workflow graph without checkpointer
                    workflow_no_checkpoint = await self.create_workflow_graph(plan)

                    # Execute without checkpointer (stateless mode)
                    async for state_update in workflow_no_checkpoint.astream(
                        asdict(initial_state), config
                    ):
                        # LangGraph astream returns dict with node names as keys
                        if isinstance(state_update, dict):
                            # Iterate through each node's state update
                            for node_name, node_state in state_update.items():
                                if isinstance(node_state, dict):
                                    # Accumulate execution path
                                    if "execution_path" in node_state:
                                        accumulated_state["execution_path"].extend(
                                            node_state["execution_path"]
                                        )

                                    # Accumulate task results
                                    if "task_results" in node_state:
                                        accumulated_state["task_results"].update(
                                            node_state["task_results"]
                                        )

                                    # Accumulate completed tasks
                                    if "completed_tasks" in node_state:
                                        accumulated_state["completed_tasks"].extend(
                                            node_state["completed_tasks"]
                                        )

                                    # Accumulate failed tasks
                                    if "failed_tasks" in node_state:
                                        accumulated_state["failed_tasks"].extend(
                                            node_state["failed_tasks"]
                                        )

                        final_state = state_update

                        # Store checkpoint in memory (not using checkpointer)
                        checkpoint = {
                            "timestamp": datetime.now(UTC).isoformat(),
                            "state": state_update,
                            "execution_path": accumulated_state[
                                "execution_path"
                            ].copy(),
                        }
                        self.active_executions[execution_id]["checkpoints"].append(
                            checkpoint
                        )

                    # Build tool_results from accumulated task_results
                    tool_results = []
                    for task_id, result in accumulated_state["task_results"].items():
                        if result and isinstance(result, dict):
                            tool_results.append(
                                {
                                    "task_id": task_id,
                                    "result": result,
                                    "timestamp": datetime.now(UTC).isoformat(),
                                }
                            )

                    execution_path = accumulated_state["execution_path"]

                    logger.info(
                        "Workflow executed successfully in stateless mode",
                        plan_id=plan.plan_id,
                    )
                finally:
                    # Restore original checkpointer
                    self.checkpointer = original_checkpointer

            execution_time = asyncio.get_event_loop().time() - start_time

            # Determine final response
            response = self._generate_final_response(final_state, tool_results)

            # Calculate tasks completed from tool_results (more reliable than final_state)
            # Count unique task_ids in tool_results that don't have errors
            completed_task_ids = set()
            failed_task_ids = set()

            for tool_result in tool_results:
                task_id = tool_result.get("task_id")
                result_data = tool_result.get("result", {})

                if task_id:
                    # Check if task failed
                    if isinstance(result_data, dict) and result_data.get("error"):
                        failed_task_ids.add(task_id)
                    else:
                        completed_task_ids.add(task_id)

            # Remove failed tasks from completed set
            completed_task_ids -= failed_task_ids

            # Create execution result
            result = ExecutionResult(
                response=response,
                metadata={
                    "plan_id": plan.plan_id,
                    "execution_id": execution_id,
                    "tasks_planned": len(plan.tasks),
                    "tasks_completed": len(completed_task_ids),
                    "tasks_failed": len(failed_task_ids),
                    "execution_path": execution_path,
                    "checkpoints_count": len(
                        self.active_executions[execution_id]["checkpoints"]
                    ),
                },
                execution_time=execution_time,
                state=ExecutionState.COMPLETED,
                tool_results=tool_results,
                execution_path=execution_path,
                checkpoints=self.active_executions[execution_id]["checkpoints"],
            )

            # Store execution history
            if session_id not in self.execution_history:
                self.execution_history[session_id] = []
            self.execution_history[session_id].append(result)

            # Clean up active execution
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]

            logger.info(
                "Plan execution completed",
                plan_id=plan.plan_id,
                execution_time=execution_time,
                tasks_completed=result.metadata["tasks_completed"],
            )

            return result

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time

            # Log detailed error information
            import traceback

            error_traceback = traceback.format_exc()
            logger.error(
                "Plan execution failed",
                plan_id=plan.plan_id,
                error=str(e),
                error_type=type(e).__name__,
                traceback=error_traceback,
            )

            # Mark execution as failed
            if execution_id in self.active_executions:
                self.active_executions[execution_id]["state"] = ExecutionState.FAILED

            return ExecutionResult(
                response=f"I encountered an error while processing your request: {str(e) if str(e) else type(e).__name__}",
                metadata={
                    "error": True,
                    "plan_id": plan.plan_id,
                    "execution_id": execution_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "error_traceback": (
                        error_traceback if str(e) else "No error message available"
                    ),
                },
                execution_time=execution_time,
                state=ExecutionState.FAILED,
                tool_results=[],
                execution_path=[],
                checkpoints=self.active_executions.get(execution_id, {}).get(
                    "checkpoints", []
                ),
            )

    def _create_task_executor(self, task: Dict[str, Any]) -> Callable:
        """Create an executor function for a task"""

        async def executor(state: WorkflowState) -> Dict[str, Any]:
            task_id = task["id"]
            task_type = task.get("type", "tool_execution")

            logger.info("Executing task", task_id=task_id, task_type=task_type)

            try:
                # Add to execution path
                state.execution_path.append(task_id)
                # Note: Not setting current_task to avoid conflicts in parallel execution

                # Execute based on task type
                if task_type == "tool_execution":
                    result = await self._execute_tool_task(task, state)
                elif task_type == "content_generation":
                    result = await self._execute_content_generation_task(task, state)
                elif task_type == "code_generation":
                    result = await self._execute_code_generation_task(task, state)
                elif task_type == "analysis":
                    result = await self._execute_analysis_task(task, state)
                else:
                    result = await self._execute_general_task(task, state)

                # Store result
                state.task_results[task_id] = result
                state.completed_tasks.append(task_id)

                logger.info("Task completed successfully", task_id=task_id)

                # Only return the fields that were modified (not session_id, plan_id, current_task)
                # Note: current_task is not returned to avoid conflicts in parallel execution
                return {
                    "completed_tasks": [task_id],
                    "task_results": {task_id: result},
                    "execution_path": [task_id],
                }

            except Exception as e:
                logger.error("Task execution failed", task_id=task_id, error=str(e))

                state.failed_tasks.append(task_id)
                # Note: Not modifying error_count or last_error to avoid conflicts in parallel execution
                # Error info is preserved in task_results

                # Store error result
                error_result = {
                    "error": True,
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                }
                state.task_results[task_id] = error_result

                # Only return the fields that were modified
                # Note: error_count and last_error not returned to avoid conflicts in parallel execution
                return {
                    "failed_tasks": [task_id],
                    "task_results": {task_id: error_result},
                    "execution_path": [task_id],
                }

        return executor

    def _state_to_workflow_state(self, state: Any) -> WorkflowState:
        """Convert state (dict or WorkflowState) to WorkflowState object"""
        if isinstance(state, WorkflowState):
            return state
        elif isinstance(state, dict):
            return WorkflowState(**state)
        else:
            # Fallback: try to convert to dict first
            try:
                state_dict = (
                    asdict(state)
                    if hasattr(state, "__dataclass_fields__")
                    else dict(state)
                )
                return WorkflowState(**state_dict)
            except Exception as e:
                logger.error(
                    "Failed to convert state to WorkflowState",
                    state_type=type(state).__name__,
                    error=str(e),
                )
                raise

    def _create_node_function(self, node: WorkflowNode) -> Callable:
        """Create a node function for LangGraph"""

        async def node_function(state: Dict[str, Any]) -> Dict[str, Any]:
            # Convert state to WorkflowState
            workflow_state = self._state_to_workflow_state(state)

            # Execute the node
            result = await node.executor(workflow_state)

            return result

        return node_function

    def _create_start_node(self) -> Callable:
        """Create a start node for workflows with multiple entry points"""

        async def start_node(state: Dict[str, Any]) -> Dict[str, Any]:
            # Only return the modified field
            return {"execution_path": ["start"]}

        return start_node

    def _add_workflow_edges(self, workflow: StateGraph, plan: ExecutionPlan):
        """Add edges to workflow based on task dependencies"""

        for task_id, dependencies in plan.dependencies.items():
            if not dependencies:
                continue

            for dep_id in dependencies:
                workflow.add_edge(dep_id, task_id)

        # Add edges to END for tasks with no dependents
        all_dependencies = set()
        for deps in plan.dependencies.values():
            all_dependencies.update(deps)

        for task in plan.tasks:
            task_id = task["id"]
            if task_id not in all_dependencies:
                workflow.add_edge(task_id, END)

    async def _add_error_handling_nodes(
        self, workflow: StateGraph, plan: ExecutionPlan
    ):
        """Add error handling and recovery nodes to workflow"""

        # Add global error handler
        workflow.add_node("error_handler", self._create_error_handler())

        # Add recovery strategies if defined
        if plan.recovery_strategies:
            for task_id, recovery_config in plan.recovery_strategies.items():
                recovery_node_id = f"recovery_{task_id}"
                workflow.add_node(
                    recovery_node_id, self._create_recovery_node(recovery_config)
                )

                # Create conditional edge function for this specific task
                def create_recovery_condition(task_id):
                    def should_recover_for_task(state: Dict[str, Any]) -> str:
                        """Determine recovery path for specific task"""
                        try:
                            workflow_state = self._state_to_workflow_state(state)

                            # Check if this task failed
                            if task_id in workflow_state.failed_tasks:
                                # Use the enhanced recovery logic
                                if self._should_recover(state):
                                    return f"recovery_{task_id}"
                                else:
                                    return "error_handler"

                            # No error, continue normal flow
                            return "continue"

                        except Exception as e:
                            logger.error("Error in recovery condition", error=str(e))
                            return "error_handler"

                    return should_recover_for_task

                # Add conditional edge from task with multiple paths
                workflow.add_conditional_edges(
                    task_id,
                    create_recovery_condition(task_id),
                    {
                        f"recovery_{task_id}": recovery_node_id,
                        "error_handler": "error_handler",
                        "continue": END,
                    },
                )

                # Add edge from recovery node back to task for retry or to error handler
                def create_post_recovery_condition(task_id, recovery_config):
                    def post_recovery_decision(state: Dict[str, Any]) -> str:
                        """Decide what to do after recovery attempt"""
                        try:
                            workflow_state = self._state_to_workflow_state(state)
                            recovery_strategy = recovery_config.get("strategy", "retry")

                            if recovery_strategy == "retry":
                                # Check if we should retry the task
                                retry_count = workflow_state.context.get(
                                    f"{task_id}_retry_count", 0
                                )
                                max_retries = recovery_config.get("max_retries", 2)

                                if retry_count < max_retries:
                                    return task_id  # Retry the task
                                else:
                                    return "error_handler"  # Too many retries
                            else:
                                return "continue"  # Fallback strategy completed

                        except Exception as e:
                            logger.error(
                                "Error in post-recovery decision", error=str(e)
                            )
                            return "error_handler"

                    return post_recovery_decision

                workflow.add_conditional_edges(
                    recovery_node_id,
                    create_post_recovery_condition(task_id, recovery_config),
                    {
                        task_id: task_id,  # Retry the original task
                        "error_handler": "error_handler",
                        "continue": END,
                    },
                )

        # Add fallback error handling for tasks without specific recovery strategies
        for task in plan.tasks:
            task_id = task["id"]
            if not plan.recovery_strategies or task_id not in plan.recovery_strategies:
                # Add simple conditional edge to error handler for tasks without recovery
                def create_simple_error_condition(task_id):
                    def simple_error_check(state: Dict[str, Any]) -> str:
                        try:
                            workflow_state = self._state_to_workflow_state(state)
                            if task_id in workflow_state.failed_tasks:
                                return "error_handler"
                            return "continue"
                        except Exception:
                            return "error_handler"

                    return simple_error_check

                workflow.add_conditional_edges(
                    task_id,
                    create_simple_error_condition(task_id),
                    {"error_handler": "error_handler", "continue": END},
                )

    def _create_error_handler(self) -> Callable:
        """Create global error handler node"""

        async def error_handler(state: Dict[str, Any]) -> Dict[str, Any]:
            workflow_state = self._state_to_workflow_state(state)

            logger.info(
                "Handling workflow error",
                session_id=workflow_state.session_id,
                error_count=workflow_state.error_count,
                last_error=workflow_state.last_error,
            )

            # Implement error handling logic
            context_update = {}
            if workflow_state.error_count > 3:
                # Too many errors, fail the workflow
                context_update["workflow_failed"] = True
            else:
                # Try to recover
                context_update["recovery_attempted"] = True

            # Only return modified fields
            return {"context": context_update}

        return error_handler

    def _create_recovery_node(self, recovery_config: Dict[str, Any]) -> Callable:
        """Create recovery node for failed tasks"""

        async def recovery_node(state: Dict[str, Any]) -> Dict[str, Any]:
            workflow_state = self._state_to_workflow_state(state)

            strategy = recovery_config.get("strategy", "retry")
            logger.info(
                "Attempting task recovery",
                recovery_strategy=strategy,
                current_task=workflow_state.current_task,
                error_count=workflow_state.error_count,
            )

            try:
                if strategy == "retry":
                    # Implement retry logic with backoff
                    current_task = workflow_state.current_task
                    if current_task:
                        # Track retry attempts
                        retry_key = f"{current_task}_retry_count"
                        retry_count = workflow_state.context.get(retry_key, 0)
                        max_retries = recovery_config.get("max_retries", 2)

                        if retry_count < max_retries:
                            # Increment retry count
                            workflow_state.context[retry_key] = retry_count + 1

                            # Remove task from failed list to allow retry
                            if current_task in workflow_state.failed_tasks:
                                workflow_state.failed_tasks.remove(current_task)

                            # Add backoff delay information
                            backoff_delay = recovery_config.get(
                                "backoff_delay", 1.0
                            ) * (2**retry_count)
                            workflow_state.context[f"{current_task}_backoff_delay"] = (
                                backoff_delay
                            )

                            # Reset error for this task
                            workflow_state.last_error = None

                            logger.info(
                                "Task prepared for retry",
                                task=current_task,
                                retry_attempt=retry_count + 1,
                                backoff_delay=backoff_delay,
                            )
                        else:
                            logger.warning(
                                "Maximum retries exceeded for task",
                                task=current_task,
                                max_retries=max_retries,
                            )
                            # Mark as permanently failed
                            workflow_state.context[
                                f"{current_task}_permanently_failed"
                            ] = True

                elif strategy == "fallback":
                    # Use fallback approach
                    current_task = workflow_state.current_task
                    fallback_result = recovery_config.get(
                        "fallback_result", "Recovery completed using fallback method"
                    )

                    if current_task:
                        # Remove from failed tasks and add fallback result
                        if current_task in workflow_state.failed_tasks:
                            workflow_state.failed_tasks.remove(current_task)

                        # Add to completed tasks with fallback result
                        if current_task not in workflow_state.completed_tasks:
                            workflow_state.completed_tasks.append(current_task)

                        workflow_state.task_results[current_task] = {
                            "recovered": True,
                            "recovery_strategy": "fallback",
                            "result": fallback_result,
                            "original_error": workflow_state.last_error,
                        }

                        logger.info(
                            "Task recovered using fallback strategy",
                            task=current_task,
                            fallback_result=fallback_result,
                        )

                elif strategy == "skip":
                    # Skip the failed task and continue
                    current_task = workflow_state.current_task
                    if current_task:
                        # Remove from failed tasks
                        if current_task in workflow_state.failed_tasks:
                            workflow_state.failed_tasks.remove(current_task)

                        # Mark as skipped
                        workflow_state.task_results[current_task] = {
                            "skipped": True,
                            "recovery_strategy": "skip",
                            "reason": "Task skipped due to recovery strategy",
                            "original_error": workflow_state.last_error,
                        }

                        logger.info(
                            "Task skipped due to recovery strategy", task=current_task
                        )

                # Update recovery metadata
                workflow_state.context["last_recovery_attempt"] = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "strategy": strategy,
                    "task": workflow_state.current_task,
                    "config": recovery_config,
                }

                # Build return dict with only modified fields
                result = {
                    "context": workflow_state.context,
                    "execution_path": [f"recovery_{strategy}"],
                }

                # Add other modified fields based on strategy
                # Note: Not returning last_error or error_count to avoid conflicts

                if strategy in ["fallback", "skip"]:
                    # These strategies modify completed_tasks and task_results
                    current_task = workflow_state.current_task
                    if current_task:
                        if (
                            strategy == "fallback"
                            and current_task not in workflow_state.completed_tasks
                        ):
                            result["completed_tasks"] = [current_task]
                        result["task_results"] = {
                            current_task: workflow_state.task_results.get(
                                current_task, {}
                            )
                        }

                return result

            except Exception as e:
                logger.error("Recovery node execution failed", error=str(e))
                # Only return context with error info to avoid conflicts
                return {
                    "context": {
                        "recovery_error": f"Recovery failed: {str(e)}",
                        "recovery_failed": True,
                    }
                }

        return recovery_node

    def _should_recover(self, state: Dict[str, Any]) -> bool:
        """Determine if recovery should be attempted based on error type and count"""
        try:
            workflow_state = self._state_to_workflow_state(state)

            # Don't attempt recovery if no errors
            if workflow_state.error_count == 0:
                return False

            # Don't attempt recovery if too many errors
            if workflow_state.error_count > 3:
                logger.warning(
                    "Too many errors, skipping recovery",
                    error_count=workflow_state.error_count,
                )
                return False

            # Check if the last error is recoverable
            last_error = workflow_state.last_error
            if last_error:
                # Classify error types that are recoverable
                recoverable_errors = [
                    "timeout",
                    "connection",
                    "rate_limit",
                    "temporary",
                    "network",
                    "service_unavailable",
                    "throttled",
                ]

                error_lower = last_error.lower()
                is_recoverable = any(
                    err_type in error_lower for err_type in recoverable_errors
                )

                if not is_recoverable:
                    logger.info("Error type not recoverable", error=last_error)
                    return False

            # Check if current task has failed multiple times
            current_task = workflow_state.current_task
            if current_task and workflow_state.failed_tasks.count(current_task) >= 2:
                logger.warning(
                    "Task has failed multiple times, skipping recovery",
                    task=current_task,
                )
                return False

            logger.info(
                "Recovery conditions met",
                error_count=workflow_state.error_count,
                current_task=current_task,
            )
            return True

        except Exception as e:
            logger.error("Error in recovery decision logic", error=str(e))
            return False

    def _find_entry_tasks(self, plan: ExecutionPlan) -> List[str]:
        """Find tasks that have no dependencies (entry points)"""

        entry_tasks = []
        all_dependencies = set()

        # Collect all tasks that are dependencies
        for deps in plan.dependencies.values():
            all_dependencies.update(deps)

        # Find tasks that are not dependencies of any other task
        for task in plan.tasks:
            task_id = task["id"]
            if task_id not in all_dependencies:
                entry_tasks.append(task_id)

        return entry_tasks

    async def _execute_tool_task(
        self, task: Dict[str, Any], state: WorkflowState
    ) -> Dict[str, Any]:
        """Execute a tool-based task"""

        start_time = asyncio.get_event_loop().time()
        tool_name = task.get("tool", "unknown")
        parameters = task.get("parameters", {})

        try:
            # Handle RAG retrieval task
            if tool_name == "rag_search" or tool_name == "knowledge_retrieval":
                return await self._execute_rag_search(task, state)

            # Handle document analysis task
            elif tool_name == "document_analysis":
                return await self._execute_document_analysis(task, state)

            # Handle code analysis task
            elif tool_name == "code_analysis":
                return await self._execute_code_analysis(task, state)

            # Generic tool execution via tool registry
            else:
                try:
                    # Try to execute via tool registry if available
                    if hasattr(self, "tool_registry") and self.tool_registry:
                        from .tools.registry import ExecutionContext, ExecutionPriority

                        # Create execution context
                        context = ExecutionContext(
                            session_id=state.session_id,
                            priority=ExecutionPriority.NORMAL,
                            timeout=task.get("timeout", 300),
                            max_retries=task.get("max_retries", 3),
                        )

                        # Get and execute tool
                        tool = await self.tool_registry.get_tool(tool_name)
                        result = await tool.execute(parameters, context)

                        execution_time = asyncio.get_event_loop().time() - start_time

                        return {
                            "tool": tool_name,
                            "parameters": parameters,
                            "result": result.data,
                            "execution_time": execution_time,
                            "metadata": {
                                "tool_type": "registry",
                                "success": result.success,
                                "tool_metadata": result.metadata,
                            },
                        }
                    else:
                        # Fallback: basic execution without registry
                        execution_time = asyncio.get_event_loop().time() - start_time
                        logger.warning(
                            f"Tool registry not available, using basic execution for {tool_name}"
                        )
                        return {
                            "tool": tool_name,
                            "parameters": parameters,
                            "result": f"Tool {tool_name} executed (basic mode - tool registry not initialized)",
                            "execution_time": execution_time,
                            "metadata": {
                                "tool_type": "basic",
                                "parameter_count": len(parameters),
                                "warning": "Tool registry not available",
                            },
                        }
                except Exception as tool_error:
                    # If tool registry fails, log and return error
                    logger.warning(
                        f"Tool registry execution failed for {tool_name}, using fallback",
                        error=str(tool_error),
                    )
                    execution_time = asyncio.get_event_loop().time() - start_time
                    return {
                        "tool": tool_name,
                        "parameters": parameters,
                        "result": f"Tool {tool_name} execution attempted (registry unavailable: {str(tool_error)})",
                        "execution_time": execution_time,
                        "metadata": {
                            "tool_type": "fallback",
                            "parameter_count": len(parameters),
                            "registry_error": str(tool_error),
                        },
                    }

        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name}, error: {str(e)}")
            execution_time = asyncio.get_event_loop().time() - start_time

            return {
                "tool": tool_name,
                "parameters": parameters,
                "result": f"Tool execution failed: {str(e)}",
                "execution_time": execution_time,
                "error": True,
                "error_message": str(e),
            }

    async def _execute_rag_search(
        self, task: Dict[str, Any], state: WorkflowState
    ) -> Dict[str, Any]:
        """Execute RAG-based knowledge retrieval"""

        start_time = asyncio.get_event_loop().time()

        try:
            query = task.get("query", task.get("parameters", {}).get("query", ""))
            max_results = task.get("max_results", 5)

            # Import RAG pipeline here to avoid circular imports
            from src.rag_pipeline.main import RAGPipeline

            # Initialize RAG pipeline if not already done
            if not hasattr(self, "rag_pipeline"):
                self.rag_pipeline = RAGPipeline()
                await self.rag_pipeline.initialize()

            # Perform search
            search_results = await self.rag_pipeline.search(
                query=query, max_results=max_results
            )

            # Format results for LLM consumption
            context_chunks = []
            for result in search_results.get("results", []):
                # Extract source from metadata (use document_title or document_id as source)
                metadata = result.get("metadata", {})
                source = (
                    metadata.get("document_title")
                    or result.get("document_id")
                    or "unknown"
                )

                context_chunks.append(
                    {
                        "content": result.get("content", ""),
                        "source": source,
                        "score": result.get("score", 0.0),
                        "chunk_type": result.get("chunk_type", "text"),
                    }
                )

            # Generate response using retrieved context
            if context_chunks:
                context_text = "\n\n".join(
                    [
                        f"Source: {chunk['source']}\nContent: {chunk['content']}"
                        for chunk in context_chunks[:3]  # Use top 3 results
                    ]
                )

                system_prompt = """You are a knowledgeable software engineering assistant with access to relevant documentation and resources.
                Use the provided context to answer questions accurately and comprehensively.
                
                Guidelines:
                - Base your answer primarily on the provided context
                - If the context doesn't contain enough information, acknowledge this
                - Cite sources when possible
                - Provide practical, actionable information
                - Maintain technical accuracy
                """

                prompt = f"""Context from knowledge base:
{context_text}

Query: {query}

Please provide a comprehensive answer based on the context above."""

                response = await self.llm_integration.generate_response(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.4,
                    max_tokens=1500,
                )

                result_text = response
            else:
                result_text = (
                    f"No relevant information found in the knowledge base for: {query}"
                )

            execution_time = asyncio.get_event_loop().time() - start_time

            return {
                "tool": "rag_search",
                "result": result_text,
                "execution_time": execution_time,
                "metadata": {
                    "query": query,
                    "results_found": len(context_chunks),
                    "sources": [chunk["source"] for chunk in context_chunks],
                    "top_score": context_chunks[0]["score"] if context_chunks else 0.0,
                },
            }

        except Exception as e:
            logger.error(f"RAG search failed: {str(e)}")
            execution_time = asyncio.get_event_loop().time() - start_time

            return {
                "tool": "rag_search",
                "result": f"Knowledge retrieval failed: {str(e)}",
                "execution_time": execution_time,
                "error": True,
                "error_message": str(e),
            }

    async def _execute_document_analysis(
        self, task: Dict[str, Any], state: WorkflowState
    ) -> Dict[str, Any]:
        """Execute document analysis task"""

        start_time = asyncio.get_event_loop().time()

        try:
            document_content = task.get("content", "")
            analysis_type = task.get("analysis_type", "general")

            system_prompt = """You are an expert document analyst specializing in software engineering documentation.
            Analyze documents thoroughly and provide structured insights.
            
            Guidelines:
            - Identify key concepts, patterns, and structures
            - Highlight important technical details
            - Note any issues, gaps, or inconsistencies
            - Provide actionable recommendations
            - Structure your analysis clearly
            """

            prompt = f"""Please analyze the following document with focus on {analysis_type} analysis:

Document Content:
{document_content}

Provide a structured analysis including:
1. Summary of key points
2. Technical details and concepts
3. Quality assessment
4. Recommendations for improvement
5. Potential issues or gaps"""

            analysis = await self.llm_integration.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=2000,
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            return {
                "tool": "document_analysis",
                "result": analysis,
                "execution_time": execution_time,
                "metadata": {
                    "analysis_type": analysis_type,
                    "document_length": len(document_content),
                    "analysis_length": len(analysis),
                },
            }

        except Exception as e:
            logger.error(f"Document analysis failed: {str(e)}")
            execution_time = asyncio.get_event_loop().time() - start_time

            return {
                "tool": "document_analysis",
                "result": f"Document analysis failed: {str(e)}",
                "execution_time": execution_time,
                "error": True,
                "error_message": str(e),
            }

    async def _execute_code_analysis(
        self, task: Dict[str, Any], state: WorkflowState
    ) -> Dict[str, Any]:
        """Execute code analysis task"""

        start_time = asyncio.get_event_loop().time()

        try:
            code_content = task.get("code", "")
            language = task.get("language", "python")
            analysis_focus = task.get("focus", ["quality", "security", "performance"])

            system_prompt = f"""You are an expert code reviewer specializing in {language} development.
            Analyze code thoroughly for quality, security, performance, and best practices.
            
            Guidelines:
            - Identify potential bugs, security vulnerabilities, and performance issues
            - Check adherence to coding standards and best practices
            - Suggest specific improvements with examples
            - Consider maintainability and readability
            - Provide actionable recommendations
            """

            focus_text = (
                ", ".join(analysis_focus)
                if isinstance(analysis_focus, list)
                else analysis_focus
            )

            prompt = f"""Please analyze the following {language} code with focus on {focus_text}:

Code:
```{language}
{code_content}
```

Provide a comprehensive analysis including:
1. Code quality assessment
2. Security considerations
3. Performance implications
4. Best practices compliance
5. Specific improvement suggestions
6. Overall recommendations"""

            analysis = await self.llm_integration.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,  # Low temperature for consistent analysis
                max_tokens=2500,
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            return {
                "tool": "code_analysis",
                "result": analysis,
                "execution_time": execution_time,
                "metadata": {
                    "language": language,
                    "analysis_focus": analysis_focus,
                    "code_lines": len(code_content.split("\n")),
                    "analysis_length": len(analysis),
                },
            }

        except Exception as e:
            logger.error(f"Code analysis failed: {str(e)}")
            execution_time = asyncio.get_event_loop().time() - start_time

            return {
                "tool": "code_analysis",
                "result": f"Code analysis failed: {str(e)}",
                "execution_time": execution_time,
                "error": True,
                "error_message": str(e),
            }

    async def _execute_content_generation_task(
        self, task: Dict[str, Any], state: WorkflowState
    ) -> Dict[str, Any]:
        """Execute content generation task"""

        start_time = asyncio.get_event_loop().time()

        try:
            # Extract task parameters
            prompt = task.get("prompt", "Generate content based on the given context")
            topic = task.get("topic", "software engineering")
            audience = task.get("audience", "intermediate")
            content_type = task.get("content_type", "explanation")

            # Create system prompt for content generation
            system_prompt = f"""You are an expert software engineering content creator. 
            Generate high-quality {content_type} content about {topic} for {audience} level audience.
            
            Guidelines:
            - Be accurate and technically precise
            - Use appropriate examples and analogies
            - Structure content clearly with headings and bullet points
            - Include practical applications where relevant
            - Maintain appropriate technical depth for the audience level
            """

            # Generate content using LLM
            content = await self.llm_integration.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=2000,
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            return {
                "type": "content_generation",
                "result": content,
                "execution_time": execution_time,
                "metadata": {
                    "topic": topic,
                    "audience": audience,
                    "content_type": content_type,
                    "tokens_used": len(content.split()) * 1.3,  # Rough estimate
                },
            }

        except Exception as e:
            logger.error(f"Content generation task failed: {str(e)}")
            execution_time = asyncio.get_event_loop().time() - start_time

            return {
                "type": "content_generation",
                "result": f"I apologize, but I encountered an error while generating content: {str(e)}",
                "execution_time": execution_time,
                "error": True,
                "error_message": str(e),
            }

    async def _execute_code_generation_task(
        self, task: Dict[str, Any], state: WorkflowState
    ) -> Dict[str, Any]:
        """Execute code generation task"""

        start_time = asyncio.get_event_loop().time()

        try:
            # Extract task parameters
            description = task.get("description", "Write code based on requirements")
            language = task.get("language", "python")
            requirements = task.get("requirements", [])
            style_guide = task.get("style_guide", "Follow best practices")

            # Create system prompt for code generation
            system_prompt = f"""You are an expert {language} developer. Generate clean, efficient, and well-documented code.

            Guidelines:
            - Follow {language} best practices and conventions
            - Include comprehensive docstrings and comments
            - Handle edge cases and errors appropriately
            - Write maintainable and readable code
            - Include type hints where applicable
            - Follow the specified style guide: {style_guide}
            """

            # Build the prompt with requirements
            prompt_parts = [description]

            if requirements:
                prompt_parts.append("\nRequirements:")
                for req in requirements:
                    prompt_parts.append(f"- {req}")

            prompt_parts.append(
                f"\nGenerate {language} code that fulfills these requirements."
            )
            prompt = "\n".join(prompt_parts)

            # Generate code using LLM
            code = await self.llm_integration.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,  # Lower temperature for more consistent code
                max_tokens=2500,
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            return {
                "type": "code_generation",
                "result": code,
                "execution_time": execution_time,
                "metadata": {
                    "language": language,
                    "requirements_count": len(requirements),
                    "style_guide": style_guide,
                    "code_lines": len(code.split("\n")),
                },
            }

        except Exception as e:
            logger.error(f"Code generation task failed: {str(e)}")
            execution_time = asyncio.get_event_loop().time() - start_time

            return {
                "type": "code_generation",
                "result": f"# Error generating code: {str(e)}\n# Please try again with more specific requirements",
                "execution_time": execution_time,
                "error": True,
                "error_message": str(e),
            }

    async def _execute_analysis_task(
        self, task: Dict[str, Any], state: WorkflowState
    ) -> Dict[str, Any]:
        """Execute analysis task"""

        start_time = asyncio.get_event_loop().time()

        try:
            # Extract task parameters
            content = task.get("content", "")
            analysis_type = task.get("analysis_type", "general")
            focus_areas = task.get("focus_areas", [])

            # Create system prompt for analysis
            system_prompt = f"""You are an expert software engineering analyst. Perform thorough {analysis_type} analysis.

            Guidelines:
            - Provide detailed, structured analysis
            - Identify key patterns, issues, and opportunities
            - Give actionable recommendations
            - Support findings with specific examples
            - Consider best practices and industry standards
            - Be objective and comprehensive
            """

            # Build analysis prompt
            prompt_parts = [
                f"Please perform a {analysis_type} analysis of the following:"
            ]
            prompt_parts.append(f"\nContent to analyze:\n{content}")

            if focus_areas:
                prompt_parts.append("\nFocus particularly on:")
                for area in focus_areas:
                    prompt_parts.append(f"- {area}")

            prompt_parts.append(
                "\nProvide a structured analysis with findings and recommendations."
            )
            prompt = "\n".join(prompt_parts)

            # Generate analysis using LLM
            analysis = await self.llm_integration.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.4,  # Balanced temperature for analytical thinking
                max_tokens=2000,
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            return {
                "type": "analysis",
                "result": analysis,
                "execution_time": execution_time,
                "metadata": {
                    "analysis_type": analysis_type,
                    "focus_areas": focus_areas,
                    "content_length": len(content),
                    "analysis_length": len(analysis),
                },
            }

        except Exception as e:
            logger.error(f"Analysis task failed: {str(e)}")
            execution_time = asyncio.get_event_loop().time() - start_time

            return {
                "type": "analysis",
                "result": f"Analysis could not be completed due to an error: {str(e)}",
                "execution_time": execution_time,
                "error": True,
                "error_message": str(e),
            }

    async def _execute_general_task(
        self, task: Dict[str, Any], state: WorkflowState
    ) -> Dict[str, Any]:
        """Execute general task"""

        start_time = asyncio.get_event_loop().time()

        try:
            # Extract task parameters
            query = task.get("query", task.get("prompt", ""))
            context = task.get("context", {})
            task_type = task.get("task_type", "general_query")

            # Create system prompt based on task type
            if task_type == "question_answering":
                system_prompt = """You are a knowledgeable software engineering expert. Answer questions accurately and comprehensively.
                
                Guidelines:
                - Provide clear, accurate answers
                - Include relevant examples when helpful
                - Explain complex concepts in understandable terms
                - Cite best practices and industry standards
                - If uncertain, acknowledge limitations
                """
            elif task_type == "explanation":
                system_prompt = """You are an expert technical educator. Explain concepts clearly and thoroughly.
                
                Guidelines:
                - Break down complex topics into digestible parts
                - Use analogies and examples to clarify concepts
                - Structure explanations logically
                - Consider the audience's technical level
                - Provide practical context and applications
                """
            elif task_type == "recommendation":
                system_prompt = """You are a software engineering consultant. Provide thoughtful recommendations.
                
                Guidelines:
                - Consider multiple approaches and trade-offs
                - Provide specific, actionable recommendations
                - Explain the reasoning behind suggestions
                - Consider context and constraints
                - Highlight potential risks and benefits
                """
            else:
                system_prompt = """You are a helpful software engineering assistant. Provide accurate and useful responses.
                
                Guidelines:
                - Be helpful and informative
                - Provide accurate technical information
                - Structure responses clearly
                - Include relevant examples
                - Be concise but comprehensive
                """

            # Add context to the prompt if available
            if context:
                context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
                full_prompt = f"Context:\n{context_str}\n\nQuery: {query}"
            else:
                full_prompt = query

            # Generate response using LLM
            response = await self.llm_integration.generate_response(
                prompt=full_prompt,
                system_prompt=system_prompt,
                temperature=0.6,
                max_tokens=1500,
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            return {
                "type": "general",
                "result": response,
                "execution_time": execution_time,
                "metadata": {
                    "task_type": task_type,
                    "query_length": len(query),
                    "context_provided": bool(context),
                    "response_length": len(response),
                },
            }

        except Exception as e:
            logger.error(f"General task failed: {str(e)}")
            execution_time = asyncio.get_event_loop().time() - start_time

            return {
                "type": "general",
                "result": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "execution_time": execution_time,
                "error": True,
                "error_message": str(e),
            }

    def _generate_final_response(
        self, final_state: Dict[str, Any], tool_results: List[Dict[str, Any]]
    ) -> str:
        """Generate final response from workflow execution"""

        # Extract the actual response from tool results
        # Tool results contain the actual LLM-generated responses
        if tool_results:
            # Get the most recent tool result (usually the last task executed)
            for tool_result in reversed(tool_results):
                result_data = tool_result.get("result", {})

                # Check if this is a RAG search result with an LLM response
                if isinstance(result_data, dict):
                    # RAG search returns result as a string in the "result" field
                    if "result" in result_data and isinstance(
                        result_data["result"], str
                    ):
                        response_text = result_data["result"]
                        if (
                            response_text
                            and not response_text.startswith("Tool")
                            and not response_text.startswith(
                                "Knowledge retrieval failed"
                            )
                        ):
                            return response_text
                    # Check for other result formats
                    elif "response" in result_data:
                        return result_data["response"]
                    elif "content" in result_data:
                        return result_data["content"]
                elif isinstance(result_data, str) and result_data:
                    # Direct string result
                    if not result_data.startswith(
                        "Tool"
                    ) and not result_data.startswith("Knowledge retrieval failed"):
                        return result_data

        # Fallback to generic message if no response found
        if not isinstance(final_state, dict):
            return "Workflow completed successfully."

        completed_tasks = final_state.get("completed_tasks", [])
        failed_tasks = final_state.get("failed_tasks", [])

        if failed_tasks:
            return f"Workflow completed with {len(completed_tasks)} successful tasks and {len(failed_tasks)} failed tasks. Some operations may need to be retried."
        else:
            return f"Workflow completed successfully. All {len(completed_tasks)} tasks were executed successfully."

    async def _setup_workflow_monitoring(self):
        """Setup workflow monitoring and metrics collection"""

        try:
            # Initialize metrics storage in Redis
            if self.redis_client:
                # Create metrics keys with TTL
                await self.redis_client.setex(
                    "orchestrator:metrics:initialized",
                    86400,  # 24 hours
                    datetime.now(UTC).isoformat(),
                )

                # Initialize counters
                await self.redis_client.set("orchestrator:metrics:total_executions", 0)
                await self.redis_client.set(
                    "orchestrator:metrics:successful_executions", 0
                )
                await self.redis_client.set("orchestrator:metrics:failed_executions", 0)

                logger.info(
                    "Workflow monitoring initialized with Redis metrics storage"
                )
            else:
                logger.warning(
                    "Redis not available, metrics will be stored in memory only"
                )

            # Set up periodic metrics collection (would be done by a background task in production)
            logger.info("Workflow monitoring setup completed")

        except Exception as e:
            logger.error("Failed to setup workflow monitoring", error=str(e))
            # Non-critical, continue without monitoring

    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active execution"""
        return self.active_executions.get(execution_id)

    async def pause_execution(self, execution_id: str) -> bool:
        """Pause an active execution"""
        if execution_id not in self.active_executions:
            return False

        try:
            execution_info = self.active_executions[execution_id]
            plan = execution_info["plan"]

            # Get the workflow graph for this execution
            if plan.plan_id not in self.workflow_graphs:
                logger.error(
                    "Workflow graph not found for execution", execution_id=execution_id
                )
                return False

            workflow = self.workflow_graphs[plan.plan_id]

            # Update execution state
            self.active_executions[execution_id]["state"] = ExecutionState.PAUSED
            self.active_executions[execution_id]["paused_at"] = datetime.now(
                UTC
            ).isoformat()

            # Store pause checkpoint in Redis
            config = RunnableConfig(
                configurable={
                    "thread_id": execution_id,
                    "checkpoint_ns": f"execution_{execution_id}",
                }
            )

            # Get current state from checkpointer
            try:
                current_checkpoint = await self.checkpointer.aget(config)
                if current_checkpoint:
                    # Mark checkpoint as paused
                    pause_metadata = {
                        "paused": True,
                        "pause_timestamp": datetime.now(UTC).isoformat(),
                        "execution_id": execution_id,
                    }

                    # Store pause state in Redis
                    await self.redis_client.hset(
                        f"paused_execution:{execution_id}",
                        mapping={
                            "checkpoint_id": current_checkpoint.id,
                            "metadata": str(pause_metadata),
                            "state": ExecutionState.PAUSED.value,
                        },
                    )
                else:
                    # No checkpoint available, store minimal pause state
                    logger.warning(
                        "No checkpoint available for paused execution",
                        execution_id=execution_id,
                    )
                    await self.redis_client.hset(
                        f"paused_execution:{execution_id}",
                        mapping={
                            "checkpoint_id": "none",
                            "metadata": str(
                                {
                                    "paused": True,
                                    "pause_timestamp": datetime.now(UTC).isoformat(),
                                    "execution_id": execution_id,
                                }
                            ),
                            "state": ExecutionState.PAUSED.value,
                        },
                    )
            except NotImplementedError as nie:
                # Checkpointer method not implemented, store basic pause state
                logger.warning(
                    "Checkpointer.aget not implemented, using fallback pause mechanism",
                    execution_id=execution_id,
                    error=str(nie),
                )
                await self.redis_client.hset(
                    f"paused_execution:{execution_id}",
                    mapping={
                        "checkpoint_id": "fallback",
                        "metadata": str(
                            {
                                "paused": True,
                                "pause_timestamp": datetime.now(UTC).isoformat(),
                                "execution_id": execution_id,
                                "fallback": True,
                            }
                        ),
                        "state": ExecutionState.PAUSED.value,
                    },
                )

            logger.info("Execution paused successfully", execution_id=execution_id)
            return True

        except Exception as e:
            logger.error(
                "Failed to pause execution", execution_id=execution_id, error=str(e)
            )
            return False

    async def resume_execution(self, execution_id: str) -> bool:
        """Resume a paused execution"""
        if execution_id not in self.active_executions:
            return False

        try:
            execution_info = self.active_executions[execution_id]

            if execution_info["state"] != ExecutionState.PAUSED:
                logger.warning(
                    "Execution is not in paused state",
                    execution_id=execution_id,
                    current_state=execution_info["state"],
                )
                return False

            plan = execution_info["plan"]

            # Get the workflow graph for this execution
            if plan.plan_id not in self.workflow_graphs:
                logger.error(
                    "Workflow graph not found for execution", execution_id=execution_id
                )
                return False

            workflow = self.workflow_graphs[plan.plan_id]

            # Check if pause state exists in Redis
            pause_data = await self.redis_client.hgetall(
                f"paused_execution:{execution_id}"
            )
            if not pause_data:
                logger.error(
                    "No pause state found for execution", execution_id=execution_id
                )
                return False

            # Configure execution for resume
            config = RunnableConfig(
                configurable={
                    "thread_id": execution_id,
                    "checkpoint_ns": f"execution_{execution_id}",
                }
            )

            # Update execution state
            self.active_executions[execution_id]["state"] = ExecutionState.RUNNING
            self.active_executions[execution_id]["resumed_at"] = datetime.now(
                UTC
            ).isoformat()

            # Remove pause state from Redis
            await self.redis_client.delete(f"paused_execution:{execution_id}")

            # Resume workflow execution from last checkpoint
            # Note: LangGraph will automatically resume from the last checkpoint
            # when we call astream with the same thread_id
            logger.info("Execution resumed successfully", execution_id=execution_id)

            # The actual resumption will happen when the workflow is called again
            # with the same config - LangGraph handles checkpoint restoration automatically

            return True

        except Exception as e:
            logger.error(
                "Failed to resume execution", execution_id=execution_id, error=str(e)
            )
            return False

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution"""
        if execution_id not in self.active_executions:
            return False

        try:
            execution_info = self.active_executions[execution_id]
            plan = execution_info["plan"]

            # Update execution state to cancelled
            self.active_executions[execution_id]["state"] = ExecutionState.CANCELLED
            self.active_executions[execution_id]["cancelled_at"] = datetime.now(
                UTC
            ).isoformat()

            # Clean up checkpoints and state in Redis
            config = RunnableConfig(
                configurable={
                    "thread_id": execution_id,
                    "checkpoint_ns": f"execution_{execution_id}",
                }
            )

            # Store cancellation metadata
            cancellation_metadata = {
                "cancelled": True,
                "cancellation_timestamp": datetime.now(UTC).isoformat(),
                "execution_id": execution_id,
                "reason": "user_requested",
            }

            # Mark execution as cancelled in Redis
            await self.redis_client.hset(
                f"cancelled_execution:{execution_id}",
                mapping={
                    "metadata": str(cancellation_metadata),
                    "state": ExecutionState.CANCELLED.value,
                    "cancelled_at": datetime.now(UTC).isoformat(),
                },
            )

            # Clean up pause state if it exists
            await self.redis_client.delete(f"paused_execution:{execution_id}")

            # Remove from active executions
            del self.active_executions[execution_id]

            # Note: LangGraph doesn't have explicit cancellation, but we can prevent
            # further execution by removing the execution from active tracking
            # and marking it as cancelled in our state management

            logger.info("Execution cancelled successfully", execution_id=execution_id)
            return True

        except Exception as e:
            logger.error(
                "Failed to cancel execution", execution_id=execution_id, error=str(e)
            )
            return False

    async def get_execution_history(self, session_id: str) -> List[ExecutionResult]:
        """Get execution history for a session"""
        return self.execution_history.get(session_id, [])

    async def get_workflow_checkpoints(self, execution_id: str) -> List[Dict[str, Any]]:
        """Get checkpoints for a workflow execution"""
        if execution_id in self.active_executions:
            return self.active_executions[execution_id].get("checkpoints", [])
        return []

    async def restore_from_checkpoint(
        self, execution_id: str, checkpoint_index: int
    ) -> bool:
        """Restore workflow execution from a specific checkpoint"""

        if execution_id not in self.active_executions:
            logger.error("Execution not found", execution_id=execution_id)
            return False

        try:
            execution_info = self.active_executions[execution_id]
            checkpoints = execution_info.get("checkpoints", [])

            if checkpoint_index >= len(checkpoints) or checkpoint_index < 0:
                logger.error(
                    "Invalid checkpoint index",
                    execution_id=execution_id,
                    checkpoint_index=checkpoint_index,
                    available_checkpoints=len(checkpoints),
                )
                return False

            plan = execution_info["plan"]

            # Get the workflow graph for this execution
            if plan.plan_id not in self.workflow_graphs:
                logger.error(
                    "Workflow graph not found for execution", execution_id=execution_id
                )
                return False

            # Get the target checkpoint
            target_checkpoint = checkpoints[checkpoint_index]
            checkpoint_state = target_checkpoint.get("state", {})

            # Configure execution for checkpoint restoration
            config = RunnableConfig(
                configurable={
                    "thread_id": execution_id,
                    "checkpoint_ns": f"execution_{execution_id}",
                }
            )

            # Create a new WorkflowState from the checkpoint
            if isinstance(checkpoint_state, dict):
                restored_state = WorkflowState(
                    session_id=checkpoint_state.get(
                        "session_id", execution_info["session_id"]
                    ),
                    plan_id=checkpoint_state.get("plan_id", plan.plan_id),
                    current_task=checkpoint_state.get("current_task"),
                    completed_tasks=checkpoint_state.get("completed_tasks", []),
                    failed_tasks=checkpoint_state.get("failed_tasks", []),
                    task_results=checkpoint_state.get("task_results", {}),
                    context=checkpoint_state.get("context", {}),
                    error_count=checkpoint_state.get("error_count", 0),
                    last_error=checkpoint_state.get("last_error"),
                    execution_path=checkpoint_state.get("execution_path", []),
                )
            else:
                logger.error(
                    "Invalid checkpoint state format", execution_id=execution_id
                )
                return False

            # Update execution info with restored state
            self.active_executions[execution_id].update(
                {
                    "state": ExecutionState.RUNNING,
                    "restored_from_checkpoint": checkpoint_index,
                    "restored_at": datetime.now(UTC).isoformat(),
                    "completed_tasks": restored_state.completed_tasks,
                    "failed_tasks": restored_state.failed_tasks,
                }
            )

            # Store restoration metadata in Redis
            restoration_metadata = {
                "restored": True,
                "restoration_timestamp": datetime.now(UTC).isoformat(),
                "checkpoint_index": checkpoint_index,
                "execution_id": execution_id,
            }

            await self.redis_client.hset(
                f"restored_execution:{execution_id}",
                mapping={
                    "metadata": str(restoration_metadata),
                    "checkpoint_index": str(checkpoint_index),
                    "restored_at": datetime.now(UTC).isoformat(),
                },
            )

            logger.info(
                "Checkpoint restoration completed successfully",
                execution_id=execution_id,
                checkpoint_index=checkpoint_index,
                restored_tasks=len(restored_state.completed_tasks),
                failed_tasks=len(restored_state.failed_tasks),
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to restore from checkpoint",
                execution_id=execution_id,
                checkpoint_index=checkpoint_index,
                error=str(e),
            )
            return False

    async def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get workflow execution metrics"""

        total_executions = sum(
            len(history) for history in self.execution_history.values()
        )
        active_count = len(self.active_executions)

        # Calculate success rate
        successful_executions = 0
        total_execution_time = 0.0

        for history in self.execution_history.values():
            for result in history:
                if result.state == ExecutionState.COMPLETED:
                    successful_executions += 1
                total_execution_time += result.execution_time

        success_rate = (
            successful_executions / total_executions if total_executions > 0 else 0
        )
        avg_execution_time = (
            total_execution_time / total_executions if total_executions > 0 else 0
        )

        return {
            "total_executions": total_executions,
            "active_executions": active_count,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "workflow_graphs_cached": len(self.workflow_graphs),
        }

    async def shutdown(self):
        """Shutdown orchestrator and cleanup resources"""
        logger.info("Shutting down LangGraph Orchestrator")

        # Cancel all active executions
        for execution_id in list(self.active_executions.keys()):
            await self.cancel_execution(execution_id)

        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()

        logger.info("LangGraph Orchestrator shutdown complete")
