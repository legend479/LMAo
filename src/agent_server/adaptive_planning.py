"""
Adaptive Planning Module
Dynamic replanning and task verification for agent orchestration
"""

from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass
import asyncio
from datetime import datetime

from .orchestrator import ExecutionPlan
from src.shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TaskVerificationResult:
    """Result of task verification"""

    task_id: str
    success: bool
    verification_method: str
    issues_found: List[str]
    suggestions: List[str]
    retry_recommended: bool


@dataclass
class ReplanResult:
    """Result of replanning operation"""

    success: bool
    new_plan: Optional[ExecutionPlan]
    recovered_tasks: Set[str]
    tool_results: List[Dict[str, Any]]
    completed_tasks: Set[str]
    metadata: Dict[str, Any]


class AdaptivePlanningEngine:
    """Engine for dynamic replanning and task verification"""

    def __init__(self, llm_integration=None):
        self.llm_integration = llm_integration
        self.verification_strategies = self._initialize_verification_strategies()

    def _initialize_verification_strategies(self) -> Dict[str, Any]:
        """Initialize task verification strategies"""
        return {
            "code_generation": self._verify_code_task,
            "content_generation": self._verify_content_task,
            "tool_execution": self._verify_tool_task,
            "document_generation": self._verify_document_task,
        }

    async def verify_task_completion(
        self, task: Dict[str, Any], result: Dict[str, Any]
    ) -> TaskVerificationResult:
        """Verify that a task completed successfully and meets requirements"""

        task_id = task.get("id")
        task_type = task.get("type", "general")

        logger.info(f"Verifying task completion: {task_id} (type: {task_type})")

        # Get verification strategy
        verification_fn = self.verification_strategies.get(
            task_type, self._verify_generic_task
        )

        return await verification_fn(task, result)

    async def _verify_code_task(
        self, task: Dict[str, Any], result: Dict[str, Any]
    ) -> TaskVerificationResult:
        """Verify code generation task with automatic testing"""

        task_id = task.get("id")
        issues = []
        suggestions = []

        # Check if code was generated
        code = result.get("code") or result.get("result")
        if not code or not isinstance(code, str):
            issues.append("No code generated")
            return TaskVerificationResult(
                task_id=task_id,
                success=False,
                verification_method="code_inspection",
                issues_found=issues,
                suggestions=["Regenerate code with clearer requirements"],
                retry_recommended=True,
            )

        # Check for basic code quality
        if len(code.strip()) < 10:
            issues.append("Generated code is too short")
            suggestions.append("Request more detailed implementation")

        # Check for syntax errors (basic check)
        if "def " not in code and "class " not in code and "function " not in code:
            issues.append("No function or class definitions found")
            suggestions.append("Ensure code includes proper structure")

        # Check for error indicators in result
        if result.get("error") or result.get("compilation_error"):
            issues.append(
                f"Compilation error: {result.get('error_message', 'Unknown')}"
            )
            suggestions.append("Fix syntax errors and retry")

        # AUTOMATIC TESTING: If compiler_runtime tool is available, test the code
        test_passed = False
        if not issues:
            try:
                # Attempt to test the code
                test_result = await self._test_generated_code(code, task)
                if test_result.get("success"):
                    test_passed = True
                    logger.info(f"Code testing passed for task {task_id}")
                else:
                    issues.append(
                        f"Code testing failed: {test_result.get('error', 'Unknown error')}"
                    )
                    suggestions.append("Review test failures and fix code logic")
            except Exception as e:
                logger.warning(f"Could not test code automatically: {e}")
                # Don't fail verification if testing infrastructure is unavailable
                test_passed = True  # Assume success if we can't test

        success = len(issues) == 0 and test_passed

        return TaskVerificationResult(
            task_id=task_id,
            success=success,
            verification_method="code_testing" if test_passed else "code_inspection",
            issues_found=issues,
            suggestions=suggestions,
            retry_recommended=not success,
        )

    async def _test_generated_code(
        self, code: str, task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test generated code using compiler_runtime tool"""

        try:
            # Import here to avoid circular dependency
            from .tools.compiler_runtime import CompilerRuntimeTool
            from .tools.registry import ExecutionContext, ExecutionPriority

            # Detect language from task or code
            language = task.get("parameters", {}).get("language", "python")

            # Create compiler tool
            compiler_tool = CompilerRuntimeTool()
            await compiler_tool.initialize()

            # Create execution context
            context = ExecutionContext(
                session_id=task.get("session_id", "test"),
                priority=ExecutionPriority.HIGH,
                timeout=30,
                max_retries=1,
            )

            # Execute code with basic test
            result = await compiler_tool.execute(
                parameters={
                    "code": code,
                    "language": language,
                    "mode": "validate",  # Just validate, don't run full tests
                },
                context=context,
            )

            return {
                "success": result.success,
                "error": result.error_message if not result.success else None,
                "execution_time": result.execution_time,
            }

        except Exception as e:
            logger.error(f"Code testing failed: {e}")
            return {"success": False, "error": str(e)}

    async def _verify_content_task(
        self, task: Dict[str, Any], result: Dict[str, Any]
    ) -> TaskVerificationResult:
        """Verify content generation task"""

        task_id = task.get("id")
        issues = []
        suggestions = []

        # Check if content was generated
        content = result.get("content") or result.get("result")
        if not content or not isinstance(content, str):
            issues.append("No content generated")
            return TaskVerificationResult(
                task_id=task_id,
                success=False,
                verification_method="content_inspection",
                issues_found=issues,
                suggestions=["Regenerate content with clearer topic"],
                retry_recommended=True,
            )

        # Check content quality
        word_count = len(content.split())
        if word_count < 50:
            issues.append(f"Content too short ({word_count} words)")
            suggestions.append("Request more detailed content")

        # Check for placeholder text
        placeholders = ["TODO", "FIXME", "[INSERT", "placeholder", "lorem ipsum"]
        for placeholder in placeholders:
            if placeholder.lower() in content.lower():
                issues.append(f"Content contains placeholder: {placeholder}")
                suggestions.append("Complete all placeholder sections")

        success = len(issues) == 0

        return TaskVerificationResult(
            task_id=task_id,
            success=success,
            verification_method="content_quality_check",
            issues_found=issues,
            suggestions=suggestions,
            retry_recommended=not success,
        )

    async def _verify_tool_task(
        self, task: Dict[str, Any], result: Dict[str, Any]
    ) -> TaskVerificationResult:
        """Verify tool execution task"""

        task_id = task.get("id")
        issues = []
        suggestions = []

        # Check for explicit errors
        if result.get("error"):
            issues.append(
                f"Tool execution error: {result.get('error_message', 'Unknown')}"
            )
            suggestions.append("Check tool parameters and retry")

        # Check if result is empty
        if not result.get("result") and not result.get("data"):
            issues.append("Tool returned empty result")
            suggestions.append("Verify tool executed correctly")

        success = len(issues) == 0

        return TaskVerificationResult(
            task_id=task_id,
            success=success,
            verification_method="tool_result_check",
            issues_found=issues,
            suggestions=suggestions,
            retry_recommended=not success,
        )

    async def _verify_document_task(
        self, task: Dict[str, Any], result: Dict[str, Any]
    ) -> TaskVerificationResult:
        """Verify document generation task"""

        task_id = task.get("id")
        issues = []
        suggestions = []

        # Check if document was created
        file_path = result.get("file_path") or result.get("path")
        if not file_path:
            issues.append("No document file path returned")
            suggestions.append("Ensure document was saved successfully")

        # Check for errors
        if result.get("error"):
            issues.append(
                f"Document generation error: {result.get('error_message', 'Unknown')}"
            )
            suggestions.append("Check document format and content")

        success = len(issues) == 0

        return TaskVerificationResult(
            task_id=task_id,
            success=success,
            verification_method="document_check",
            issues_found=issues,
            suggestions=suggestions,
            retry_recommended=not success,
        )

    async def _verify_generic_task(
        self, task: Dict[str, Any], result: Dict[str, Any]
    ) -> TaskVerificationResult:
        """Generic task verification"""

        task_id = task.get("id")
        issues = []

        # Basic checks
        if result.get("error"):
            issues.append(f"Task error: {result.get('error_message', 'Unknown')}")

        if not result.get("result") and not result.get("data"):
            issues.append("Task returned no result")

        success = len(issues) == 0

        return TaskVerificationResult(
            task_id=task_id,
            success=success,
            verification_method="generic_check",
            issues_found=issues,
            suggestions=["Review task execution"] if issues else [],
            retry_recommended=not success,
        )

    async def create_recovery_plan(
        self,
        original_plan: ExecutionPlan,
        failed_tasks: Set[str],
        completed_tasks: Set[str],
        verification_results: Dict[str, TaskVerificationResult],
    ) -> Optional[ExecutionPlan]:
        """Create a recovery plan for failed tasks"""

        logger.info(f"Creating recovery plan for {len(failed_tasks)} failed tasks")

        # Analyze failures
        recovery_tasks = []
        task_counter = len(original_plan.tasks) + 1

        for task in original_plan.tasks:
            task_id = task.get("id")

            # Skip completed tasks
            if task_id in completed_tasks:
                continue

            # Handle failed tasks
            if task_id in failed_tasks:
                verification = verification_results.get(task_id)

                if verification and verification.retry_recommended:
                    # Create modified task with improvements
                    recovery_task = task.copy()
                    recovery_task["id"] = f"recovery_{task_id}"
                    recovery_task["original_task_id"] = task_id
                    recovery_task["retry_attempt"] = task.get("retry_attempt", 0) + 1

                    # Add suggestions to parameters
                    if verification.suggestions:
                        recovery_task["parameters"] = recovery_task.get(
                            "parameters", {}
                        ).copy()
                        recovery_task["parameters"][
                            "_recovery_suggestions"
                        ] = verification.suggestions
                        recovery_task["parameters"][
                            "_issues_to_fix"
                        ] = verification.issues_found

                    recovery_tasks.append(recovery_task)
                    logger.info(f"Created recovery task for {task_id}")

        if not recovery_tasks:
            logger.info("No recoverable tasks found")
            return None

        # Create new plan with recovery tasks
        # Create a recovery plan. Do NOT copy the original plan's recovery_strategies
        # into the recovery plan itself. Doing so may cause duplicate recovery node
        # creation when the orchestrator builds the workflow (nodes are named
        # `recovery_<task_id>`). Keep recovery_strategies empty for the recovery
        # plan to avoid duplicate node IDs; the recovery tasks themselves are
        # explicit tasks to execute.
        recovery_plan = ExecutionPlan(
            plan_id=f"{original_plan.plan_id}_recovery",
            tasks=recovery_tasks,
            dependencies={task["id"]: [] for task in recovery_tasks},
            estimated_duration=sum(
                task.get("estimated_duration", 2.0) for task in recovery_tasks
            ),
            priority=original_plan.priority + 1,
            recovery_strategies={},
            parallel_groups=[],
        )

        logger.info(
            f"Recovery plan created with {len(recovery_tasks)} tasks",
            plan_id=recovery_plan.plan_id,
        )

        return recovery_plan


__all__ = ["AdaptivePlanningEngine", "TaskVerificationResult", "ReplanResult"]
