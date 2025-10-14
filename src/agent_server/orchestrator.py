"""
LangGraph Orchestrator
Stateful workflow management using LangGraph
"""

from typing import Dict, Any, List, Optional, Callable
import asyncio
import redis.asyncio as redis
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver
from langchain_core.runnables import RunnableConfig

from src.shared.logging import get_logger
from src.shared.config import get_settings

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
    completed_tasks: List[str] = None
    failed_tasks: List[str] = None
    task_results: Dict[str, Any] = None
    context: Dict[str, Any] = None
    error_count: int = 0
    last_error: Optional[str] = None
    execution_path: List[str] = None

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
        self.checkpointer: Optional[RedisSaver] = None
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.execution_history: Dict[str, List[ExecutionResult]] = {}
        self.workflow_graphs: Dict[str, StateGraph] = {}
        self.workflow_nodes: Dict[str, Dict[str, WorkflowNode]] = {}
        self._initialized = False

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
            self.checkpointer = RedisSaver(redis_url)

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
        compiled_workflow = workflow.compile(checkpointer=self.checkpointer)

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
            execution_path = []
            tool_results = []

            async for state in workflow.astream(asdict(initial_state), config):
                # Update execution tracking
                if isinstance(state, dict) and "execution_path" in state:
                    execution_path.extend(state["execution_path"])

                if isinstance(state, dict) and "task_results" in state:
                    for task_id, result in state["task_results"].items():
                        if result and isinstance(result, dict):
                            tool_results.append(
                                {
                                    "task_id": task_id,
                                    "result": result,
                                    "timestamp": datetime.utcnow().isoformat(),
                                }
                            )

                final_state = state

                # Store checkpoint
                checkpoint = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "state": state,
                    "execution_path": execution_path.copy(),
                }
                self.active_executions[execution_id]["checkpoints"].append(checkpoint)

            execution_time = asyncio.get_event_loop().time() - start_time

            # Determine final response
            response = self._generate_final_response(final_state, tool_results)

            # Create execution result
            result = ExecutionResult(
                response=response,
                metadata={
                    "plan_id": plan.plan_id,
                    "execution_id": execution_id,
                    "tasks_planned": len(plan.tasks),
                    "tasks_completed": (
                        len(final_state.get("completed_tasks", []))
                        if isinstance(final_state, dict)
                        else 0
                    ),
                    "tasks_failed": (
                        len(final_state.get("failed_tasks", []))
                        if isinstance(final_state, dict)
                        else 0
                    ),
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
            logger.error("Plan execution failed", plan_id=plan.plan_id, error=str(e))

            # Mark execution as failed
            if execution_id in self.active_executions:
                self.active_executions[execution_id]["state"] = ExecutionState.FAILED

            return ExecutionResult(
                response=f"I encountered an error while processing your request: {str(e)}",
                metadata={
                    "error": True,
                    "plan_id": plan.plan_id,
                    "execution_id": execution_id,
                    "error_type": type(e).__name__,
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
                state.current_task = task_id

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

                return asdict(state)

            except Exception as e:
                logger.error("Task execution failed", task_id=task_id, error=str(e))

                state.failed_tasks.append(task_id)
                state.error_count += 1
                state.last_error = str(e)

                # Store error result
                state.task_results[task_id] = {
                    "error": True,
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                }

                return asdict(state)

        return executor

    def _create_node_function(self, node: WorkflowNode) -> Callable:
        """Create a node function for LangGraph"""

        async def node_function(state: Dict[str, Any]) -> Dict[str, Any]:
            # Convert dict back to WorkflowState
            workflow_state = WorkflowState(**state)

            # Execute the node
            result = await node.executor(workflow_state)

            return result

        return node_function

    def _create_start_node(self) -> Callable:
        """Create a start node for workflows with multiple entry points"""

        async def start_node(state: Dict[str, Any]) -> Dict[str, Any]:
            workflow_state = WorkflowState(**state)
            workflow_state.execution_path.append("start")
            return asdict(workflow_state)

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

                # Add conditional edge from task to recovery
                workflow.add_conditional_edges(
                    task_id, self._should_recover, {True: recovery_node_id, False: END}
                )

    def _create_error_handler(self) -> Callable:
        """Create global error handler node"""

        async def error_handler(state: Dict[str, Any]) -> Dict[str, Any]:
            workflow_state = WorkflowState(**state)

            logger.info(
                "Handling workflow error",
                session_id=workflow_state.session_id,
                error_count=workflow_state.error_count,
                last_error=workflow_state.last_error,
            )

            # Implement error handling logic
            if workflow_state.error_count > 3:
                # Too many errors, fail the workflow
                workflow_state.context["workflow_failed"] = True
            else:
                # Try to recover
                workflow_state.context["recovery_attempted"] = True

            return asdict(workflow_state)

        return error_handler

    def _create_recovery_node(self, recovery_config: Dict[str, Any]) -> Callable:
        """Create recovery node for failed tasks"""

        async def recovery_node(state: Dict[str, Any]) -> Dict[str, Any]:
            workflow_state = WorkflowState(**state)

            logger.info(
                "Attempting task recovery",
                recovery_strategy=recovery_config.get("strategy", "retry"),
            )

            # Implement recovery logic based on strategy
            strategy = recovery_config.get("strategy", "retry")

            if strategy == "retry":
                # Reset failed task for retry
                failed_task = workflow_state.last_error
                if failed_task in workflow_state.failed_tasks:
                    workflow_state.failed_tasks.remove(failed_task)
            elif strategy == "fallback":
                # Use fallback approach
                fallback_result = recovery_config.get(
                    "fallback_result", "Recovery completed"
                )
                workflow_state.task_results[workflow_state.current_task] = {
                    "recovered": True,
                    "result": fallback_result,
                }

            return asdict(workflow_state)

        return recovery_node

    def _should_recover(self, state: Dict[str, Any]) -> bool:
        """Determine if recovery should be attempted"""
        workflow_state = WorkflowState(**state)
        return workflow_state.error_count > 0 and workflow_state.error_count <= 3

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

        tool_name = task.get("tool", "unknown")
        parameters = task.get("parameters", {})

        # TODO: Integrate with tool registry
        # For now, return placeholder result

        return {
            "tool": tool_name,
            "parameters": parameters,
            "result": f"Tool {tool_name} executed successfully (placeholder)",
            "execution_time": 0.1,
        }

    async def _execute_content_generation_task(
        self, task: Dict[str, Any], state: WorkflowState
    ) -> Dict[str, Any]:
        """Execute content generation task"""

        return {
            "type": "content_generation",
            "result": "Content generated successfully (placeholder)",
            "execution_time": 0.2,
        }

    async def _execute_code_generation_task(
        self, task: Dict[str, Any], state: WorkflowState
    ) -> Dict[str, Any]:
        """Execute code generation task"""

        return {
            "type": "code_generation",
            "result": "Code generated successfully (placeholder)",
            "execution_time": 0.3,
        }

    async def _execute_analysis_task(
        self, task: Dict[str, Any], state: WorkflowState
    ) -> Dict[str, Any]:
        """Execute analysis task"""

        return {
            "type": "analysis",
            "result": "Analysis completed successfully (placeholder)",
            "execution_time": 0.1,
        }

    async def _execute_general_task(
        self, task: Dict[str, Any], state: WorkflowState
    ) -> Dict[str, Any]:
        """Execute general task"""

        return {
            "type": "general",
            "result": "General task completed successfully (placeholder)",
            "execution_time": 0.1,
        }

    def _generate_final_response(
        self, final_state: Dict[str, Any], tool_results: List[Dict[str, Any]]
    ) -> str:
        """Generate final response from workflow execution"""

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

        # TODO: Implement comprehensive monitoring
        logger.info("Workflow monitoring setup completed")

    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active execution"""
        return self.active_executions.get(execution_id)

    async def pause_execution(self, execution_id: str) -> bool:
        """Pause an active execution"""
        if execution_id in self.active_executions:
            self.active_executions[execution_id]["state"] = ExecutionState.PAUSED
            # TODO: Implement actual workflow pausing in LangGraph
            return True
        return False

    async def resume_execution(self, execution_id: str) -> bool:
        """Resume a paused execution"""
        if execution_id in self.active_executions:
            if self.active_executions[execution_id]["state"] == ExecutionState.PAUSED:
                self.active_executions[execution_id]["state"] = ExecutionState.RUNNING
                # TODO: Implement actual workflow resuming in LangGraph
                return True
        return False

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution"""
        if execution_id in self.active_executions:
            self.active_executions[execution_id]["state"] = ExecutionState.CANCELLED
            del self.active_executions[execution_id]
            # TODO: Implement actual workflow cancellation in LangGraph
            return True
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
            return False

        checkpoints = self.active_executions[execution_id].get("checkpoints", [])

        if checkpoint_index >= len(checkpoints):
            return False

        # TODO: Implement checkpoint restoration with LangGraph
        logger.info(
            "Checkpoint restoration requested",
            execution_id=execution_id,
            checkpoint_index=checkpoint_index,
        )

        return True

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
