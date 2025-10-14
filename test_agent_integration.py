#!/usr/bin/env python3
"""
Test script to verify the Core Agent System integration
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agent_server.main import AgentServer


async def test_agent_integration():
    """Test the core agent system integration"""

    print("🚀 Testing Core Agent System Integration...")

    try:
        # Initialize agent server
        agent = AgentServer()
        await agent.initialize()

        print("✅ Agent server initialized successfully")

        # Test message processing
        test_message = "Explain what Python is and how to write a simple function"
        session_id = "test_session_001"

        print(f"📝 Processing test message: {test_message}")

        result = await agent.process_message(test_message, session_id)

        print("✅ Message processed successfully")
        print(f"📋 Response: {result['response'][:100]}...")
        print(f"🔧 Metadata: {result['metadata']}")

        # Test tool listing
        tools = await agent.get_available_tools()
        print(f"🛠️  Available tools: {list(tools.keys())}")

        # Test tool execution
        if "knowledge_retrieval" in tools:
            tool_result = await agent.execute_tool(
                "knowledge_retrieval",
                {"query": "Python programming basics"},
                session_id,
            )
            print(f"🔍 Tool execution result: {tool_result['status']}")

        # Shutdown
        await agent.shutdown()
        print("✅ Agent server shutdown successfully")

        print("\n🎉 All tests passed! Core Agent System is working correctly.")

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(test_agent_integration())
    sys.exit(0 if success else 1)
