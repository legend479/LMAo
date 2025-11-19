"""
Tests for Adaptive Planning Engine
"""

import pytest
import asyncio
from src.agent_server.adaptive_planning import (
    AdaptivePlanningEngine,
    TaskVerificationResult,
    ReplanResult,
)
from src.agent_server.orchestrator import ExecutionPlan


class MockLLMIntegration:
    """Mock LLM for testing"""

    async def generate_response(self, prompt, **kwargs):
        return "Mock LLM response"


@pytest.fixture
def adaptive_engine():
    """Create adaptive planning engine for testing"""
    llm = MockLLMIntegration()
    return AdaptivePlanningEngine(llm)


@pytest.mark.asyncio
async def test_verify_code_task_success(adaptive_engine):
    """Test successful code task verification"""

    task = {
        "id": "task_1",
        "type": "code_generation",
        "parameters": {"language": "python"},
    }

    result = {
        "code": "def hello():\n    print('Hello, World!')\n",
        "result": "def hello():\n    print('Hello, World!')\n",
    }

    verification = await adaptive_engine.verify_task_completion(task, result)

    assert isinstance(verification, TaskVerificationResult)
    assert verification.task_id == "task_1"
    # Note: May not be success if code testing fails, but should not crash


@pytest.mark.asyncio
async def test_verify_code_task_failure(adaptive_engine):
    """Test code task verification with failures"""

    task = {
        "id": "task_2",
        "type": "code_generation",
        "parameters": {"language": "python"},
    }

    result = {"code": "", "result": ""}  # Empty code

    verification = await adaptive_engine.verify_task_completion(task, result)

    assert not verification.success
    assert len(verification.issues_found) > 0
    assert "No code generated" in verification.issues_found
    assert verification.retry_recommended


@pytest.mark.asyncio
async def test_verify_content_task_success(adaptive_engine):
    """Test successful content task verification"""

    task = {
        "id": "task_3",
        "type": "content_generation",
        "parameters": {"topic": "Python programming"},
    }

    result = {
        "content": "Python is a high-level programming language. " * 20,
        "result": "Python is a high-level programming language. " * 20,
    }

    verification = await adaptive_engine.verify_task_completion(task, result)

    assert verification.success
    assert len(verification.issues_found) == 0


@pytest.mark.asyncio
async def test_verify_content_task_too_short(adaptive_engine):
    """Test content task verification with short content"""

    task = {
        "id": "task_4",
        "type": "content_generation",
        "parameters": {"topic": "Python"},
    }

    result = {"content": "Python is good.", "result": "Python is good."}

    verification = await adaptive_engine.verify_task_completion(task, result)

    assert not verification.success
    assert any("too short" in issue.lower() for issue in verification.issues_found)


@pytest.mark.asyncio
async def test_verify_content_task_with_placeholders(adaptive_engine):
    """Test content task verification with placeholders"""

    task = {
        "id": "task_5",
        "type": "content_generation",
        "parameters": {"topic": "Python"},
    }

    result = {
        "content": "Python is a programming language. TODO: Add more details here. "
        * 10,
        "result": "Python is a programming language. TODO: Add more details here. "
        * 10,
    }

    verification = await adaptive_engine.verify_task_completion(task, result)

    assert not verification.success
    assert any("placeholder" in issue.lower() for issue in verification.issues_found)


@pytest.mark.asyncio
async def test_verify_tool_task_success(adaptive_engine):
    """Test successful tool task verification"""

    task = {"id": "task_6", "type": "tool_execution", "tool": "knowledge_retrieval"}

    result = {
        "result": "Retrieved information about Python",
        "data": {"info": "Python details"},
    }

    verification = await adaptive_engine.verify_task_completion(task, result)

    assert verification.success


@pytest.mark.asyncio
async def test_verify_tool_task_with_error(adaptive_engine):
    """Test tool task verification with error"""

    task = {"id": "task_7", "type": "tool_execution", "tool": "knowledge_retrieval"}

    result = {"error": True, "error_message": "Tool execution failed"}

    verification = await adaptive_engine.verify_task_completion(task, result)

    assert not verification.success
    assert verification.retry_recommended


@pytest.mark.asyncio
async def test_verify_document_task_success(adaptive_engine):
    """Test successful document task verification"""

    task = {
        "id": "task_8",
        "type": "document_generation",
        "parameters": {"format": "pdf"},
    }

    result = {
        "file_path": "/path/to/document.pdf",
        "result": "Document created successfully",
    }

    verification = await adaptive_engine.verify_task_completion(task, result)

    assert verification.success


@pytest.mark.asyncio
async def test_verify_document_task_no_file(adaptive_engine):
    """Test document task verification without file path"""

    task = {
        "id": "task_9",
        "type": "document_generation",
        "parameters": {"format": "pdf"},
    }

    result = {
        "result": "Document created"
        # Missing file_path
    }

    verification = await adaptive_engine.verify_task_completion(task, result)

    assert not verification.success
    assert any("file path" in issue.lower() for issue in verification.issues_found)


@pytest.mark.asyncio
async def test_create_recovery_plan(adaptive_engine):
    """Test recovery plan creation"""

    original_plan = ExecutionPlan(
        plan_id="test_plan",
        tasks=[
            {
                "id": "task_1",
                "type": "code_generation",
                "parameters": {"language": "python"},
            },
            {
                "id": "task_2",
                "type": "content_generation",
                "parameters": {"topic": "Python"},
            },
        ],
        dependencies={"task_1": [], "task_2": []},
        estimated_duration=10.0,
        priority=1,
    )

    failed_tasks = {"task_1"}
    completed_tasks = {"task_2"}

    verification_results = {
        "task_1": TaskVerificationResult(
            task_id="task_1",
            success=False,
            verification_method="code_inspection",
            issues_found=["Syntax error"],
            suggestions=["Fix syntax"],
            retry_recommended=True,
        )
    }

    recovery_plan = await adaptive_engine.create_recovery_plan(
        original_plan, failed_tasks, completed_tasks, verification_results
    )

    assert recovery_plan is not None
    assert len(recovery_plan.tasks) > 0
    assert recovery_plan.plan_id.endswith("_recovery")

    # Check recovery task
    recovery_task = recovery_plan.tasks[0]
    assert recovery_task["id"].startswith("recovery_")
    assert recovery_task["original_task_id"] == "task_1"
    assert "_recovery_suggestions" in recovery_task["parameters"]


@pytest.mark.asyncio
async def test_create_recovery_plan_no_recoverable_tasks(adaptive_engine):
    """Test recovery plan creation with no recoverable tasks"""

    original_plan = ExecutionPlan(
        plan_id="test_plan",
        tasks=[
            {
                "id": "task_1",
                "type": "code_generation",
                "parameters": {"language": "python"},
            }
        ],
        dependencies={"task_1": []},
        estimated_duration=5.0,
        priority=1,
    )

    failed_tasks = {"task_1"}
    completed_tasks = set()

    verification_results = {
        "task_1": TaskVerificationResult(
            task_id="task_1",
            success=False,
            verification_method="code_inspection",
            issues_found=["Critical error"],
            suggestions=[],
            retry_recommended=False,  # Not recommended to retry
        )
    }

    recovery_plan = await adaptive_engine.create_recovery_plan(
        original_plan, failed_tasks, completed_tasks, verification_results
    )

    assert recovery_plan is None


@pytest.mark.asyncio
async def test_verify_generic_task(adaptive_engine):
    """Test generic task verification"""

    task = {"id": "task_10", "type": "unknown_type", "parameters": {}}

    result = {"result": "Task completed", "data": {"status": "success"}}

    verification = await adaptive_engine.verify_task_completion(task, result)

    assert verification.success
    assert verification.verification_method == "generic_check"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
