#!/usr/bin/env python3
"""
Quick test script for Agent Server initialization
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_agent_server():
    """Test agent server initialization"""
    print("Testing Agent Server initialization...")

    try:
        from src.agent_server.main import AgentServer

        print("✓ Imported AgentServer")

        agent_server = AgentServer()
        print("✓ Created AgentServer instance")

        await agent_server.initialize()
        print("✓ Initialized AgentServer")

        # Test health check
        print("\nTesting components:")
        print(f"  - Orchestrator initialized: {agent_server.orchestrator._initialized}")
        print(
            f"  - Planning module initialized: {agent_server.planning_module._initialized}"
        )
        print(
            f"  - Memory manager initialized: {agent_server.memory_manager._initialized}"
        )
        print(
            f"  - Tool registry initialized: {agent_server.tool_registry._initialized}"
        )

        # Test list tools
        print("\nTesting tool listing...")
        tools = await agent_server.get_available_tools()
        print(f"✓ Got tools: {tools.get('total_count', 0)} total")

        # Cleanup
        await agent_server.shutdown()
        print("\n✓ Agent Server test completed successfully!")

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_agent_server())
    sys.exit(0 if success else 1)
