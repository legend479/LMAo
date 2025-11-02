"""
Enhanced tests for LangGraph Orchestrator functionality
Tests the completed workflow control methods and error recovery mechanisms
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

from src.agent_server.orchestrator import (
    LangGraphOrchestrator,
    ExecutionState,
    ExecutionPlan,
    WorkflowState,
    ExecutionResult,
    NodeType,
)

# Configure pytest for async tests
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture
def orchestrator():
    """Create orchestrator instance for testing"""
    orchestrator = LangGraphOrchestrator()

    # Mock dependencies
    orchestrator.redis_client = AsyncMock()
    orchestrator.checkpointer = AsyncMock()
    orchestrator.llm_integration = AsyncMock()
    orchestrator._initialized = True

    return orchestrator


@pytest.fixture
def sample_execution_plan():
    """Create sample execution plan for testing"""
    return ExecutionPlan(
        plan_id="test_plan_123",
        tasks=[
            {
                "id": "task_1",
                "type": "tool_execution",
                "tool": "test_tool",
                "parameters": {"param1": "value1"},
            },
            {
                "id": "task_2",
                "type": "content_generation",
                "prompt": "Generate test content",
            },
        ],
        dependencies={"task_2": ["task_1"]},
        estimated_duration=60.0,
        recovery_strategies={
            "task_1": {"strategy": "retry", "max_retries": 2, "backoff_delay": 1.0}
        },
    )


@pytest.fixture
def sample_workflow_state():
    """Create sample workflow state for testing"""
    return WorkflowState(
        session_id="test_session_123",
        plan_id="test_plan_123",
        current_task="task_1",
        completed_tasks=["task_0"],
        failed_tasks=[],
        task_results={"task_0": {"result": "success"}},
        context={"test_key": "test_value"},
        error_count=0,
        execution_path=["task_0"],
    )


class TestWorkflowControlMethods:
    """Test enhanced workflow control methods"""

    @pytest.mark.asyncio
    async def test_pause_execution_success(self, orchestrator, sample_execution_plan):
        """Test successful execution pausing"""
        execution_id = "test_session_123_test_plan_123"

        # Setup active execution
        orchestrator.active_executions[execution_id] = {
            "plan": sample_execution_plan,
            "session_id": "test_session_123",
            "state": ExecutionState.RUNNING,
            "checkpoints": [],
        }

        # Setup workflow graph
        orchestrator.workflow_graphs[sample_execution_plan.plan_id] = Mock()

        # Mock checkpointer response
        mock_checkpoint = Mock()
        mock_checkpoint.id = "checkpoint_123"
        orchestrator.checkpointer.aget.return_value = mock_checkpoint

        # Test pause execution
        result = await orchestrator.pause_execution(execution_id)

        assert result is True
        assert (
            orchestrator.active_executions[execution_id]["state"]
            == ExecutionState.PAUSED
        )
        assert "paused_at" in orchestrator.active_executions[execution_id]

        # Verify Redis operations
        orchestrator.redis_client.hset.assert_called_once()
        call_args = orchestrator.redis_client.hset.call_args
        assert call_args[0][0] == f"paused_execution:{execution_id}"

    @pytest.mark.asyncio
    async def test_pause_execution_not_found(self, orchestrator):
        """Test pausing non-existent execution"""
        result = await orchestrator.pause_execution("non_existent_id")
        assert result is False

    @pytest.mark.asyncio
    async def test_resume_execution_success(self, orchestrator, sample_execution_plan):
        """Test successful execution resuming"""
        execution_id = "test_session_123_test_plan_123"

        # Setup paused execution
        orchestrator.active_executions[execution_id] = {
            "plan": sample_execution_plan,
            "session_id": "test_session_123",
            "state": ExecutionState.PAUSED,
            "checkpoints": [],
        }

        # Setup workflow graph
        orchestrator.workflow_graphs[sample_execution_plan.plan_id] = Mock()

        # Mock Redis pause data
        orchestrator.redis_client.hgetall.return_value = {
            "checkpoint_id": "checkpoint_123",
            "metadata": "pause_metadata",
            "state": ExecutionState.PAUSED.value,
        }

        # Test resume execution
        result = await orchestrator.resume_execution(execution_id)

        assert result is True
        assert (
            orchestrator.active_executions[execution_id]["state"]
            == ExecutionState.RUNNING
        )
        assert "resumed_at" in orchestrator.active_executions[execution_id]

        # Verify Redis cleanup
        orchestrator.redis_client.delete.assert_called_once_with(
            f"paused_execution:{execution_id}"
        )

    @pytest.mark.asyncio
    async def test_resume_execution_not_paused(
        self, orchestrator, sample_execution_plan
    ):
        """Test resuming execution that is not paused"""
        execution_id = "test_session_123_test_plan_123"

        # Setup running execution
        orchestrator.active_executions[execution_id] = {
            "plan": sample_execution_plan,
            "session_id": "test_session_123",
            "state": ExecutionState.RUNNING,
            "checkpoints": [],
        }

        result = await orchestrator.resume_execution(execution_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_execution_success(self, orchestrator, sample_execution_plan):
        """Test successful execution cancellation"""
        execution_id = "test_session_123_test_plan_123"

        # Setup active execution
        orchestrator.active_executions[execution_id] = {
            "plan": sample_execution_plan,
            "session_id": "test_session_123",
            "state": ExecutionState.RUNNING,
            "checkpoints": [],
        }

        # Test cancel execution
        result = await orchestrator.cancel_execution(execution_id)

        assert result is True
        assert execution_id not in orchestrator.active_executions

        # Verify Redis operations
        orchestrator.redis_client.hset.assert_called()
        orchestrator.redis_client.delete.assert_called_once_with(
            f"paused_execution:{execution_id}"
        )

    @pytest.mark.asyncio
    async def test_cancel_execution_not_found(self, orchestrator):
        """Test cancelling non-existent execution"""
        result = await orchestrator.cancel_execution("non_existent_id")
        assert result is False


class TestCheckpointRestoration:
    """Test checkpoint restoration functionality"""

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint_success(
        self, orchestrator, sample_execution_plan
    ):
        """Test successful checkpoint restoration"""
        execution_id = "test_session_123_test_plan_123"
        checkpoint_index = 1

        # Setup execution with checkpoints
        sample_checkpoint = {
            "timestamp": "2024-01-01T12:00:00",
            "state": {
                "session_id": "test_session_123",
                "plan_id": "test_plan_123",
                "current_task": "task_1",
                "completed_tasks": ["task_0"],
                "failed_tasks": [],
                "task_results": {"task_0": {"result": "success"}},
                "context": {"restored": True},
                "error_count": 0,
                "execution_path": ["task_0"],
            },
            "execution_path": ["task_0"],
        }

        orchestrator.active_executions[execution_id] = {
            "plan": sample_execution_plan,
            "session_id": "test_session_123",
            "state": ExecutionState.RUNNING,
            "checkpoints": [
                {"timestamp": "2024-01-01T11:00:00", "state": {}},
                sample_checkpoint,
                {"timestamp": "2024-01-01T13:00:00", "state": {}},
            ],
        }

        # Setup workflow graph
        orchestrator.workflow_graphs[sample_execution_plan.plan_id] = Mock()

        # Test checkpoint restoration
        result = await orchestrator.restore_from_checkpoint(
            execution_id, checkpoint_index
        )

        assert result is True
        assert (
            orchestrator.active_executions[execution_id]["restored_from_checkpoint"]
            == checkpoint_index
        )
        assert "restored_at" in orchestrator.active_executions[execution_id]

        # Verify Redis operations
        orchestrator.redis_client.hset.assert_called_once()

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint_invalid_index(
        self, orchestrator, sample_execution_plan
    ):
        """Test checkpoint restoration with invalid index"""
        execution_id = "test_session_123_test_plan_123"

        orchestrator.active_executions[execution_id] = {
            "plan": sample_execution_plan,
            "session_id": "test_session_123",
            "checkpoints": [{"state": {}}],  # Only one checkpoint
        }

        # Test with invalid index
        result = await orchestrator.restore_from_checkpoint(execution_id, 5)
        assert result is False

        # Test with negative index
        result = await orchestrator.restore_from_checkpoint(execution_id, -1)
        assert result is False

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint_execution_not_found(self, orchestrator):
        """Test checkpoint restoration for non-existent execution"""
        result = await orchestrator.restore_from_checkpoint("non_existent_id", 0)
        assert result is False


class TestErrorRecoveryMechanisms:
    """Test enhanced error recovery mechanisms"""

    def test_should_recover_success_conditions(self, orchestrator):
        """Test recovery decision logic for recoverable conditions"""
        # Test recoverable error
        state = {
            "session_id": "test_session",
            "plan_id": "test_plan",
            "error_count": 1,
            "last_error": "Connection timeout occurred",
            "current_task": "task_1",
            "failed_tasks": ["task_1"],
            "completed_tasks": [],
            "task_results": {},
            "context": {},
            "execution_path": [],
        }

        result = orchestrator._should_recover(state)
        assert result is True

    def test_should_recover_too_many_errors(self, orchestrator):
        """Test recovery decision when too many errors occurred"""
        state = {
            "session_id": "test_session",
            "plan_id": "test_plan",
            "error_count": 5,  # Too many errors
            "last_error": "Connection timeout",
            "current_task": "task_1",
            "failed_tasks": ["task_1"],
            "completed_tasks": [],
            "task_results": {},
            "context": {},
            "execution_path": [],
        }

        result = orchestrator._should_recover(state)
        assert result is False

    def test_should_recover_non_recoverable_error(self, orchestrator):
        """Test recovery decision for non-recoverable errors"""
        state = {
            "session_id": "test_session",
            "plan_id": "test_plan",
            "error_count": 1,
            "last_error": "Invalid syntax error",  # Not recoverable
            "current_task": "task_1",
            "failed_tasks": ["task_1"],
            "completed_tasks": [],
            "task_results": {},
            "context": {},
            "execution_path": [],
        }

        result = orchestrator._should_recover(state)
        assert result is False

    def test_should_recover_task_failed_multiple_times(self, orchestrator):
        """Test recovery decision when task has failed multiple times"""
        state = {
            "session_id": "test_session",
            "plan_id": "test_plan",
            "error_count": 2,
            "last_error": "Connection timeout",
            "current_task": "task_1",
            "failed_tasks": ["task_1", "task_1"],  # Task failed twice
            "completed_tasks": [],
            "task_results": {},
            "context": {},
            "execution_path": [],
        }

        result = orchestrator._should_recover(state)
        assert result is False

    @pytest.mark.asyncio
    async def test_recovery_node_retry_strategy(self, orchestrator):
        """Test recovery node with retry strategy"""
        recovery_config = {"strategy": "retry", "max_retries": 2, "backoff_delay": 1.0}

        recovery_node = orchestrator._create_recovery_node(recovery_config)

        state = {
            "session_id": "test_session",
            "plan_id": "test_plan",
            "current_task": "task_1",
            "failed_tasks": ["task_1"],
            "completed_tasks": [],
            "task_results": {},
            "context": {},
            "error_count": 1,
            "last_error": "Connection timeout",
            "execution_path": [],
        }

        result = await recovery_node(state)

        # Verify retry logic
        assert result["context"]["task_1_retry_count"] == 1
        assert "task_1_backoff_delay" in result["context"]
        assert "task_1" not in result["failed_tasks"]
        assert result["last_error"] is None

    @pytest.mark.asyncio
    async def test_recovery_node_fallback_strategy(self, orchestrator):
        """Test recovery node with fallback strategy"""
        recovery_config = {
            "strategy": "fallback",
            "fallback_result": "Fallback result used",
        }

        recovery_node = orchestrator._create_recovery_node(recovery_config)

        state = {
            "session_id": "test_session",
            "plan_id": "test_plan",
            "current_task": "task_1",
            "failed_tasks": ["task_1"],
            "completed_tasks": [],
            "task_results": {},
            "context": {},
            "error_count": 1,
            "last_error": "Connection timeout",
            "execution_path": [],
        }

        result = await recovery_node(state)

        # Verify fallback logic
        assert "task_1" not in result["failed_tasks"]
        assert "task_1" in result["completed_tasks"]
        assert result["task_results"]["task_1"]["recovered"] is True
        assert result["task_results"]["task_1"]["recovery_strategy"] == "fallback"

    @pytest.mark.asyncio
    async def test_recovery_node_skip_strategy(self, orchestrator):
        """Test recovery node with skip strategy"""
        recovery_config = {"strategy": "skip"}

        recovery_node = orchestrator._create_recovery_node(recovery_config)

        state = {
            "session_id": "test_session",
            "plan_id": "test_plan",
            "current_task": "task_1",
            "failed_tasks": ["task_1"],
            "completed_tasks": [],
            "task_results": {},
            "context": {},
            "error_count": 1,
            "last_error": "Connection timeout",
            "execution_path": [],
        }

        result = await recovery_node(state)

        # Verify skip logic
        assert "task_1" not in result["failed_tasks"]
        assert result["task_results"]["task_1"]["skipped"] is True
        assert result["task_results"]["task_1"]["recovery_strategy"] == "skip"


class TestWorkflowStateManagement:
    """Test workflow state management enhancements"""

    def test_workflow_state_initialization(self):
        """Test WorkflowState initialization with defaults"""
        state = WorkflowState(session_id="test_session", plan_id="test_plan")

        assert state.session_id == "test_session"
        assert state.plan_id == "test_plan"
        assert state.completed_tasks == []
        assert state.failed_tasks == []
        assert state.task_results == {}
        assert state.context == {}
        assert state.execution_path == []
        assert state.error_count == 0

    def test_workflow_state_with_data(self):
        """Test WorkflowState initialization with provided data"""
        state = WorkflowState(
            session_id="test_session",
            plan_id="test_plan",
            current_task="task_1",
            completed_tasks=["task_0"],
            failed_tasks=["task_2"],
            task_results={"task_0": {"result": "success"}},
            context={"key": "value"},
            error_count=1,
            last_error="Test error",
            execution_path=["task_0", "task_1"],
        )

        assert state.current_task == "task_1"
        assert state.completed_tasks == ["task_0"]
        assert state.failed_tasks == ["task_2"]
        assert state.task_results == {"task_0": {"result": "success"}}
        assert state.context == {"key": "value"}
        assert state.error_count == 1
        assert state.last_error == "Test error"
        assert state.execution_path == ["task_0", "task_1"]


class TestIntegrationScenarios:
    """Test integration scenarios for orchestrator enhancements"""

    @pytest.mark.asyncio
    async def test_pause_resume_workflow_cycle(
        self, orchestrator, sample_execution_plan
    ):
        """Test complete pause-resume cycle"""
        execution_id = "test_session_123_test_plan_123"

        # Setup active execution
        orchestrator.active_executions[execution_id] = {
            "plan": sample_execution_plan,
            "session_id": "test_session_123",
            "state": ExecutionState.RUNNING,
            "checkpoints": [],
        }
        orchestrator.workflow_graphs[sample_execution_plan.plan_id] = Mock()

        # Mock checkpointer and Redis
        mock_checkpoint = Mock()
        mock_checkpoint.id = "checkpoint_123"
        orchestrator.checkpointer.aget.return_value = mock_checkpoint
        orchestrator.redis_client.hgetall.return_value = {
            "checkpoint_id": "checkpoint_123"
        }

        # Test pause
        pause_result = await orchestrator.pause_execution(execution_id)
        assert pause_result is True
        assert (
            orchestrator.active_executions[execution_id]["state"]
            == ExecutionState.PAUSED
        )

        # Test resume
        resume_result = await orchestrator.resume_execution(execution_id)
        assert resume_result is True
        assert (
            orchestrator.active_executions[execution_id]["state"]
            == ExecutionState.RUNNING
        )

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, orchestrator):
        """Test error recovery workflow with retry logic"""
        recovery_config = {"strategy": "retry", "max_retries": 2, "backoff_delay": 0.5}

        # Test multiple recovery attempts
        recovery_node = orchestrator._create_recovery_node(recovery_config)

        # First recovery attempt
        state = {
            "session_id": "test_session",
            "plan_id": "test_plan",
            "current_task": "task_1",
            "failed_tasks": ["task_1"],
            "completed_tasks": [],
            "task_results": {},
            "context": {},
            "error_count": 1,
            "last_error": "Temporary failure",
            "execution_path": [],
        }

        result1 = await recovery_node(state)
        assert result1["context"]["task_1_retry_count"] == 1

        # Second recovery attempt
        state = result1.copy()
        state["failed_tasks"] = ["task_1"]  # Task failed again
        state["error_count"] = 2

        result2 = await recovery_node(state)
        assert result2["context"]["task_1_retry_count"] == 2

        # Third attempt should not increment beyond max_retries
        state = result2.copy()
        state["failed_tasks"] = ["task_1"]
        state["error_count"] = 3

        result3 = await recovery_node(state)
        # Should still be 2 as we don't increment beyond max_retries in this implementation
        assert result3["context"]["task_1_retry_count"] == 2
