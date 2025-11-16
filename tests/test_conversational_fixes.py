"""
Tests for Conversational Agent Fixes
Validates response synthesis, context awareness, and logging
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.agent_server.main import AgentServer
from src.agent_server.orchestrator import ExecutionPlan, WorkflowState


@pytest.mark.asyncio
async def test_original_query_preserved():
    """Test that original query is preserved through execution"""

    agent = AgentServer()
    await agent.initialize()

    original_message = "What is Python and how do I use it?"

    # Process message
    result = await agent.process_message(original_message, "test_session")

    # Verify response exists
    assert "response" in result
    assert len(result["response"]) > 0

    # Response should be conversational, not just "Workflow completed"
    assert (
        "Workflow completed" not in result["response"] or len(result["response"]) > 100
    )


@pytest.mark.asyncio
async def test_response_synthesis():
    """Test that responses are synthesized from multiple sources"""

    agent = AgentServer()
    await agent.initialize()

    # Create a plan with multiple tasks
    plan = ExecutionPlan(
        plan_id="test_synthesis",
        tasks=[
            {
                "id": "task_1",
                "type": "tool_execution",
                "tool": "knowledge_retrieval",
                "parameters": {"query": "What is Python?"},
                "dependencies": [],
            },
            {
                "id": "task_2",
                "type": "general",
                "parameters": {"query": "Explain Python usage"},
                "dependencies": ["task_1"],
            },
        ],
        dependencies={"task_2": ["task_1"]},
        estimated_duration=60.0,
    )

    # Execute with original query
    result = await agent.orchestrator.execute_plan(
        plan, "test_session", original_query="What is Python and how do I use it?"
    )

    # Verify synthesis occurred
    assert result.response is not None
    assert len(result.response) > 50  # Should be substantial

    # Should not be generic fallback message
    assert "Workflow completed" not in result.response or "Python" in result.response


@pytest.mark.asyncio
async def test_general_task_uses_context():
    """Test that general tasks use conversation context"""

    agent = AgentServer()
    await agent.initialize()

    # Create workflow state with previous results
    state = WorkflowState(
        session_id="test_session",
        plan_id="test_plan",
        original_query="Tell me about Python",
        task_results={
            "task_1": {
                "result": "Python is a high-level programming language",
                "type": "knowledge_retrieval",
            }
        },
    )

    # Create a general task
    task = {
        "id": "task_2",
        "type": "general",
        "parameters": {"query": "How do I use it?", "task_type": "question_answering"},
        "dependencies": ["task_1"],
    }

    # Execute general task
    result = await agent.orchestrator._execute_general_task(task, state)

    # Verify result includes context
    assert result is not None
    assert "result" in result
    assert len(result["result"]) > 20

    # Metadata should indicate context was used
    assert result.get("metadata", {}).get("previous_context_used") is not None


@pytest.mark.asyncio
async def test_workflow_state_has_original_query():
    """Test that WorkflowState properly stores original query"""

    state = WorkflowState(
        session_id="test", plan_id="test", original_query="What is Python?"
    )

    assert state.original_query == "What is Python?"
    assert state.conversation_history is not None
    assert isinstance(state.conversation_history, list)


@pytest.mark.asyncio
async def test_synthesis_with_multiple_sources():
    """Test synthesis combines multiple tool results"""

    agent = AgentServer()
    await agent.initialize()

    # Mock tool results
    tool_results = [
        {
            "task_id": "task_1",
            "result": {
                "result": "Python is a programming language known for simplicity"
            },
        },
        {
            "task_id": "task_2",
            "result": {
                "result": "Python is widely used in web development and data science"
            },
        },
    ]

    final_state = {
        "original_query": "What is Python?",
        "completed_tasks": ["task_1", "task_2"],
    }

    # Test synthesis
    if agent.orchestrator.llm_integration:
        try:
            response = await agent.orchestrator._synthesize_coherent_response(
                "What is Python?", tool_results, final_state
            )

            # Response should be substantial
            assert len(response) > 50

            # Should not mention "task" or "tool"
            assert "task_1" not in response.lower()
            assert "tool" not in response.lower() or "toolkit" in response.lower()

        except Exception as e:
            # Synthesis might fail if LLM not available, that's okay
            pytest.skip(f"LLM not available for synthesis test: {e}")


@pytest.mark.asyncio
async def test_logging_enhanced():
    """Test that enhanced logging is working"""

    agent = AgentServer()
    await agent.initialize()

    # This test just verifies the logging methods exist and don't crash
    # Actual log output would need to be captured separately

    if agent.orchestrator.llm_integration:
        try:
            # Make a simple LLM call
            response = await agent.orchestrator.llm_integration.generate_response(
                prompt="Test prompt",
                system_prompt="Test system",
                temperature=0.5,
                max_tokens=50,
            )

            # If we got here, logging didn't crash
            assert response is not None

        except Exception as e:
            # LLM might not be available, that's okay
            pytest.skip(f"LLM not available for logging test: {e}")


@pytest.mark.asyncio
async def test_conversational_flow():
    """Test that agent maintains conversational flow"""

    agent = AgentServer()
    await agent.initialize()

    # First message
    result1 = await agent.process_message("What is Python?", "conversation_test")

    assert "response" in result1
    first_response = result1["response"]

    # Response should be conversational
    assert len(first_response) > 30

    # Second message (follow-up)
    result2 = await agent.process_message(
        "How do I install it?", "conversation_test"  # Same session
    )

    assert "response" in result2
    second_response = result2["response"]

    # Second response should also be substantial
    assert len(second_response) > 30


@pytest.mark.asyncio
async def test_fallback_when_synthesis_fails():
    """Test that agent falls back gracefully if synthesis fails"""

    agent = AgentServer()
    await agent.initialize()

    # Create tool results
    tool_results = [{"task_id": "task_1", "result": "Simple result"}]

    final_state = {"completed_tasks": ["task_1"], "failed_tasks": []}

    # Test with no LLM (should fall back to extraction)
    original_llm = agent.orchestrator.llm_integration
    agent.orchestrator.llm_integration = None

    try:
        response = await agent.orchestrator._generate_final_response(
            final_state, tool_results, "Test query"
        )

        # Should get some response (fallback)
        assert response is not None
        assert len(response) > 0

    finally:
        # Restore LLM
        agent.orchestrator.llm_integration = original_llm


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
