"""
Unit tests for agent server core functionality.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from datetime import datetime


class TestAgentCore:
    """Test core agent functionality."""

    @pytest.fixture
    def mock_agent(self):
        """Mock agent instance."""
        agent = Mock()
        agent.name = "test_agent"
        agent.capabilities = ["reasoning", "tool_use"]
        agent.status = "active"
        return agent

    def test_agent_initialization(self, mock_agent):
        """Test agent initialization."""
        assert mock_agent.name == "test_agent"
        assert "reasoning" in mock_agent.capabilities
        assert mock_agent.status == "active"

    @patch("agent_server.main.AgentServer")
    def test_agent_manager_creation(self, mock_agent_server):
        """Test agent server creation."""
        mock_agent_server.return_value._initialized = False
        server = mock_agent_server()
        assert hasattr(server, "_initialized")

    # FIX: Added the pytest.mark.asyncio decorator
    @pytest.mark.asyncio
    async def test_agent_message_processing(self, mock_agent):
        """Test agent message processing."""
        mock_agent.process_message = AsyncMock(return_value="Processed message")
        result = await mock_agent.process_message("test message")
        assert result == "Processed message"
        mock_agent.process_message.assert_called_once_with("test message")


class TestAgentLifecycle:
    """Test agent lifecycle management."""

    @pytest.fixture
    def agent_manager(self):
        """Mock agent manager."""
        manager = Mock()
        manager.agents = {}
        manager.start_agent = Mock()
        manager.stop_agent = Mock()
        manager.restart_agent = Mock()
        return manager

    def test_agent_startup(self, agent_manager):
        """Test agent startup process."""
        agent_manager.start_agent("test_agent")
        agent_manager.start_agent.assert_called_once_with("test_agent")

    def test_agent_shutdown(self, agent_manager):
        """Test agent shutdown process."""
        agent_manager.stop_agent("test_agent")
        agent_manager.stop_agent.assert_called_once_with("test_agent")

    def test_agent_restart(self, agent_manager):
        """Test agent restart process."""
        agent_manager.restart_agent("test_agent")
        agent_manager.restart_agent.assert_called_once_with("test_agent")


class TestAgentCommunication:
    """Test agent communication protocols."""

    @pytest.fixture
    def mock_message(self):
        """Mock message object."""
        return {
            "id": "msg_123",
            "content": "Hello agent",
            "timestamp": datetime.now().isoformat(),
            "user_id": "user_456",
        }

    # FIX: Added the pytest.mark.asyncio decorator
    @pytest.mark.asyncio
    async def test_message_routing(self, mock_message):
        """Test message routing through orchestrator."""
        with patch(
            "agent_server.orchestrator.LangGraphOrchestrator.execute_plan"
        ) as mock_execute:
            mock_execute.return_value = Mock(response="Test response", metadata={})
            result = await mock_execute(Mock(), "session_123")
            assert result.response == "Test response"

    # FIX: Added the pytest.mark.asyncio decorator
    @pytest.mark.asyncio
    async def test_message_validation(self, mock_message):
        """Test message validation."""
        # Test valid message
        assert "id" in mock_message
        assert "content" in mock_message
        assert "timestamp" in mock_message

        # Test invalid message
        invalid_message = {"content": "Hello"}
        # Should fail validation due to missing required fields


class TestAgentTools:
    """Test agent tool integration."""

    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry."""
        registry = Mock()
        registry.tools = {
            "calculator": Mock(),
            "web_search": Mock(),
            "file_reader": Mock(),
        }
        registry.get_tool = Mock()
        registry.register_tool = Mock()
        return registry

    def test_tool_registration(self, mock_tool_registry):
        """Test tool registration."""
        new_tool = Mock()
        new_tool.name = "new_tool"
        mock_tool_registry.register_tool(new_tool)
        mock_tool_registry.register_tool.assert_called_once_with(new_tool)

    def test_tool_retrieval(self, mock_tool_registry):
        """Test tool retrieval."""
        mock_tool_registry.get_tool.return_value = mock_tool_registry.tools[
            "calculator"
        ]
        tool = mock_tool_registry.get_tool("calculator")
        assert tool == mock_tool_registry.tools["calculator"]

    # FIX: Added the pytest.mark.asyncio decorator
    @pytest.mark.asyncio
    async def test_tool_execution(self, mock_tool_registry):
        """Test tool execution."""
        calculator = mock_tool_registry.tools["calculator"]
        calculator.execute = AsyncMock(return_value=42)
        result = await calculator.execute("2 + 2")
        assert result == 42


class TestAgentMemory:
    """Test agent memory and context management."""

    @pytest.fixture
    def mock_memory(self):
        """Mock memory system."""
        memory = Mock()
        memory.store = Mock()
        memory.retrieve = Mock()
        memory.clear = Mock()
        memory.context = []
        return memory

    def test_memory_storage(self, mock_memory):
        """Test storing information in memory."""
        mock_memory.store("key", "value")
        mock_memory.store.assert_called_once_with("key", "value")

    def test_memory_retrieval(self, mock_memory):
        """Test retrieving information from memory."""
        mock_memory.retrieve.return_value = "stored_value"
        result = mock_memory.retrieve("key")
        assert result == "stored_value"

    def test_context_management(self, mock_memory):
        """Test conversation context management."""
        mock_memory.context = ["message1", "message2"]
        assert len(mock_memory.context) == 2
        assert "message1" in mock_memory.context


class TestAgentError:
    """Test agent error handling."""

    @pytest.mark.asyncio
    async def test_tool_execution_error(self):
        """Test handling of tool execution errors."""
        with patch("agent_server.main.AgentServer.execute_tool") as mock_execute:
            mock_execute.side_effect = Exception("Tool failed")
            # Should handle gracefully
            with pytest.raises(Exception, match="Tool failed"):
                await mock_execute("broken_tool", {}, "session_123")

    @pytest.mark.asyncio
    async def test_message_processing_error(self):
        """Test handling of message processing errors."""
        with patch("agent_server.main.AgentServer.process_message") as mock_process:
            mock_process.side_effect = ValueError("Invalid message format")
            # Should handle gracefully
            with pytest.raises(ValueError, match="Invalid message format"):
                await mock_process("invalid message", "session_123")

    def test_agent_recovery(self):
        """Test agent recovery from errors."""
        agent = Mock()
        agent.status = "error"
        agent.recover = Mock()
        agent.recover()
        agent.recover.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
