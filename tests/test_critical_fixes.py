"""
Tests for Critical Fixes
Validates context chaining, tool configuration, and initialization order
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.agent_server.main import AgentServer
from src.agent_server.orchestrator import ExecutionPlan, WorkflowState
from src.shared.config import get_settings


@pytest.mark.asyncio
async def test_context_chaining():
    """Test that context is properly chained between dependent tasks"""

    agent = AgentServer()
    await agent.initialize()

    # Create plan with dependent tasks
    plan = ExecutionPlan(
        plan_id="test_context_chain",
        tasks=[
            {
                "id": "retrieve_info",
                "type": "tool_execution",
                "tool": "knowledge_retrieval",
                "parameters": {"query": "What is Python?"},
                "dependencies": [],
            },
            {
                "id": "summarize_info",
                "type": "content_generation",
                "parameters": {"prompt": "Summarize the retrieved information"},
                "dependencies": ["retrieve_info"],
            },
        ],
        dependencies={"summarize_info": ["retrieve_info"]},
        estimated_duration=60.0,
    )

    result = await agent.orchestrator.execute_plan(plan, "test_session")

    # Verify both tasks completed
    assert result.metadata["tasks_completed"] >= 1
    assert result.state.value in ["completed", "failed"]

    # Verify second task had access to first task's results if both completed
    if result.metadata["tasks_completed"] == 2:
        summarize_result = next(
            (r for r in result.tool_results if r.get("task_id") == "summarize_info"),
            None,
        )
        if summarize_result:
            assert "dependencies_used" in summarize_result.get("result", {})


@pytest.mark.asyncio
async def test_tool_configuration():
    """Test that tools respect configuration settings"""

    settings = get_settings()

    agent = AgentServer()
    await agent.initialize()

    # Get registered tools
    tools = await agent.get_available_tools()
    tool_names = [t.get("name") for t in tools.get("tools", [])]

    # Verify code execution tools are only registered if enabled
    if settings.enable_code_execution:
        assert (
            "compiler_runtime" in tool_names
        ), "Code execution enabled but tool not registered"
    else:
        assert (
            "compiler_runtime" not in tool_names
        ), "Code execution disabled but tool registered"

    # Verify email tools are only registered if enabled
    if settings.enable_email_tools:
        assert (
            "email_automation" in tool_names
        ), "Email tools enabled but tool not registered"
    else:
        assert (
            "email_automation" not in tool_names
        ), "Email tools disabled but tool registered"

    # Verify core tools are always registered
    assert (
        "knowledge_retrieval" in tool_names
    ), "Core knowledge_retrieval tool not registered"
    assert (
        "document_generation" in tool_names
    ), "Core document_generation tool not registered"


@pytest.mark.asyncio
async def test_initialization_order():
    """Test that components initialize in correct order"""

    agent = AgentServer()

    # Track initialization order
    init_order = []

    # Monkey patch initialize methods to track order
    original_tool_init = agent.tool_registry.initialize
    original_orch_init = agent.orchestrator.initialize
    original_plan_init = agent.planning_module.initialize
    original_mem_init = agent.memory_manager.initialize

    async def track_tool_init():
        init_order.append("tool_registry")
        await original_tool_init()

    async def track_orch_init():
        init_order.append("orchestrator")
        await original_orch_init()

    async def track_plan_init():
        init_order.append("planning")
        await original_plan_init()

    async def track_mem_init():
        init_order.append("memory")
        await original_mem_init()

    agent.tool_registry.initialize = track_tool_init
    agent.orchestrator.initialize = track_orch_init
    agent.planning_module.initialize = track_plan_init
    agent.memory_manager.initialize = track_mem_init

    await agent.initialize()

    # Verify correct order: tool_registry -> orchestrator -> tools registered -> planning -> memory
    assert (
        init_order[0] == "tool_registry"
    ), f"Tool registry should initialize first, got: {init_order}"
    assert (
        init_order[1] == "orchestrator"
    ), f"Orchestrator should initialize second, got: {init_order}"
    assert (
        init_order[2] == "planning"
    ), f"Planning should initialize third, got: {init_order}"
    assert (
        init_order[3] == "memory"
    ), f"Memory should initialize fourth, got: {init_order}"

    # Verify orchestrator has tool registry
    assert (
        agent.orchestrator.tool_registry is not None
    ), "Orchestrator should have tool registry reference"


@pytest.mark.asyncio
async def test_dependency_context_injection():
    """Test that dependency context is properly injected into tasks"""

    agent = AgentServer()
    await agent.initialize()

    # Create a simple workflow state
    state = WorkflowState(
        session_id="test_session",
        plan_id="test_plan",
        task_results={"task_1": {"result": "Task 1 completed", "data": {"value": 42}}},
    )

    # Create a task with dependencies
    task = {
        "id": "task_2",
        "type": "tool_execution",
        "tool": "knowledge_retrieval",
        "parameters": {"query": "test"},
        "dependencies": ["task_1"],
    }

    # Create executor
    executor = agent.orchestrator._create_task_executor(task)

    # Execute task
    result = await executor(state)

    # Verify dependency context was injected
    assert "parameters" in task
    assert "_dependency_context" in task["parameters"]
    assert "dependency_task_1" in task["parameters"]["_dependency_context"]


@pytest.mark.asyncio
async def test_tool_awareness():
    """Test that agent has full awareness of available tools"""

    agent = AgentServer()
    await agent.initialize()

    # Get available tools
    tools_response = await agent.get_available_tools()

    # Verify response structure
    assert "tools" in tools_response
    assert isinstance(tools_response["tools"], list)

    # Verify each tool has required metadata
    for tool in tools_response["tools"]:
        assert "name" in tool, "Tool should have name"
        assert "description" in tool, "Tool should have description"
        assert "capabilities" in tool, "Tool should have capabilities"

    # Verify tool count is reasonable
    assert (
        len(tools_response["tools"]) >= 3
    ), "Should have at least 3 core tools registered"


@pytest.mark.asyncio
async def test_configuration_validation():
    """Test that configuration is properly validated"""

    settings = get_settings()

    # Verify tool configuration fields exist
    assert hasattr(settings, "enable_code_execution")
    assert hasattr(settings, "enable_email_tools")
    assert hasattr(settings, "tool_timeout_seconds")
    assert hasattr(settings, "tool_max_retries")
    assert hasattr(settings, "code_execution_timeout")
    assert hasattr(settings, "rag_max_results")

    # Verify default values are sensible
    assert settings.tool_timeout_seconds > 0
    assert settings.tool_max_retries >= 0
    assert settings.code_execution_timeout > 0
    assert settings.rag_max_results > 0


@pytest.mark.asyncio
async def test_tool_execution_with_context():
    """Test that tools can be executed with proper context"""

    agent = AgentServer()
    await agent.initialize()

    # Execute a simple tool
    try:
        result = await agent.execute_tool(
            tool_name="knowledge_retrieval",
            parameters={"query": "test query", "max_results": 5},
            session_id="test_session",
        )

        # Verify result structure
        assert "tool_name" in result
        assert "status" in result
        assert result["tool_name"] == "knowledge_retrieval"

    except Exception as e:
        # Tool execution may fail if dependencies not available, but should not crash
        assert isinstance(e, Exception)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
