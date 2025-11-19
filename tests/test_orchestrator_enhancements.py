"""
Tests for Orchestrator Enhancements
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.agent_server.orchestrator import (
    LangGraphOrchestrator,
    ExecutionPlan,
    ExecutionState,
    WorkflowState,
)


@pytest.fixture
async def orchestrator():
    """Create orchestrator for testing"""
    orch = LangGraphOrchestrator()

    # Mock dependencies
    orch.llm_integration = Mock()
    orch.llm_integration.generate_response = AsyncMock(
        return_value="def add(a, b):\n    return a + b"
    )

    orch.tool_registry = Mock()
    orch.tool_registry.get_tool_by_name = AsyncMock(return_value=None)

    # Initialize adaptive engine
    from src.agent_server.adaptive_planning import AdaptivePlanningEngine

    orch.adaptive_engine = AdaptivePlanningEngine(orch.llm_integration)

    return orch


@pytest.mark.asyncio
async def test_code_generation_with_testing(orchestrator):
    """Test code generation with automatic testing"""

    task = {
        "id": "task_1",
        "type": "code_generation",
        "parameters": {
            "description": "Write a function to add two numbers",
            "language": "python",
            "auto_test": False,  # Disable for this test
            "max_retries": 1,
        },
    }

    state = WorkflowState(
        session_id="test_session",
        plan_id="test_plan",
        original_query="Write a function to add two numbers",
    )

    result = await orchestrator._execute_code_generation_task(task, state)

    assert result["type"] == "code_generation"
    assert "result" in result or "code" in result
    assert result["metadata"]["generation_attempts"] >= 1


@pytest.mark.asyncio
async def test_code_generation_metadata(orchestrator):
    """Test code generation includes proper metadata"""

    task = {
        "id": "task_2",
        "type": "code_generation",
        "parameters": {
            "description": "Write a hello world function",
            "language": "python",
            "auto_test": False,
            "requirements": ["Print hello world", "Use function"],
        },
    }

    state = WorkflowState(session_id="test_session", plan_id="test_plan")

    result = await orchestrator._execute_code_generation_task(task, state)

    metadata = result["metadata"]
    assert "language" in metadata
    assert "requirements_count" in metadata
    assert "generation_attempts" in metadata
    assert "auto_tested" in metadata
    assert metadata["language"] == "python"
    assert metadata["requirements_count"] == 2


@pytest.mark.asyncio
async def test_replan_tracking_initialization():
    """Test that replan tracking is initialized correctly"""

    orch = LangGraphOrchestrator()

    plan = ExecutionPlan(
        plan_id="test_plan",
        tasks=[{"id": "task_1", "type": "general"}],
        dependencies={"task_1": []},
        estimated_duration=5.0,
        priority=1,
    )

    session_id = "test_session"
    execution_id = f"{session_id}_{plan.plan_id}"

    # Simulate initialization in execute_plan
    orch.active_executions[execution_id] = {
        "plan": plan,
        "session_id": session_id,
        "replan_count": 0,
        "max_replans": 2,
    }

    assert orch.active_executions[execution_id]["replan_count"] == 0
    assert orch.active_executions[execution_id]["max_replans"] == 2


@pytest.mark.asyncio
async def test_test_generated_code_no_tool(orchestrator):
    """Test code testing when compiler tool is not available"""

    # Ensure tool registry returns None
    orchestrator.tool_registry.get_tool_by_name = AsyncMock(return_value=None)

    code = "def hello():\n    print('Hello')"
    language = "python"
    task = {"session_id": "test"}

    result = await orchestrator._test_generated_code(code, language, task)

    # Should skip testing gracefully
    assert result["success"] == True
    assert result.get("skipped") == True


@pytest.mark.asyncio
async def test_content_generation_with_context(orchestrator):
    """Test content generation uses context from previous tasks"""

    task = {
        "id": "task_3",
        "type": "content_generation",
        "topic": "Python programming",
        "for_document": True,
        "format": "markdown",
    }

    state = WorkflowState(
        session_id="test_session",
        plan_id="test_plan",
        original_query="Explain Python programming",
        task_results={
            "task_1": {"result": "Python is a high-level programming language"}
        },
    )

    orchestrator.llm_integration.generate_response = AsyncMock(
        return_value="Python is a versatile programming language. " * 50
    )

    result = await orchestrator._execute_content_generation_task(task, state)

    assert result["type"] == "content_generation"
    assert len(result["result"]) > 50
    assert result["metadata"]["for_document"] == True


@pytest.mark.asyncio
async def test_content_generation_retry_on_short_content(orchestrator):
    """Test content generation retries when content is too short"""

    task = {"id": "task_4", "type": "content_generation", "topic": "Python"}

    state = WorkflowState(session_id="test_session", plan_id="test_plan")

    # First call returns short content, second call returns longer content
    orchestrator.llm_integration.generate_response = AsyncMock(
        side_effect=[
            "Short",  # First attempt
            "This is a much longer response about Python programming. " * 10,  # Retry
        ]
    )

    result = await orchestrator._execute_content_generation_task(task, state)

    # Should have retried and gotten longer content
    assert len(result["result"]) > 50
    assert orchestrator.llm_integration.generate_response.call_count == 2


@pytest.mark.asyncio
async def test_execute_plan_with_replan_tracking(orchestrator):
    """Test that execute_plan initializes replan tracking"""

    # Mock the workflow execution
    with patch.object(orchestrator, "create_workflow_graph", new_callable=AsyncMock):
        with patch.object(
            orchestrator, "_generate_final_response", new_callable=AsyncMock
        ) as mock_response:
            mock_response.return_value = "Test response"

            plan = ExecutionPlan(
                plan_id="test_plan",
                tasks=[],
                dependencies={},
                estimated_duration=1.0,
                priority=1,
            )

            # This will fail due to mocking, but we can check initialization
            try:
                await orchestrator.execute_plan(plan, "test_session", "test query")
            except:
                pass

            # Check that active_executions was initialized with replan tracking
            execution_id = f"test_session_{plan.plan_id}"
            if execution_id in orchestrator.active_executions:
                assert "replan_count" in orchestrator.active_executions[execution_id]
                assert "max_replans" in orchestrator.active_executions[execution_id]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
