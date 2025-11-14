#!/usr/bin/env python3
"""
Test tool registration
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_tool_registration():
    """Test tool registration"""
    print("Testing Tool Registration...")

    try:
        from src.agent_server.main import AgentServer

        print("✓ Imported AgentServer")

        agent_server = AgentServer()
        print("✓ Created AgentServer instance")

        print("\nInitializing Agent Server (this will auto-register tools)...")
        await agent_server.initialize()
        print("✓ Initialized AgentServer")

        # Test list tools
        print("\nListing registered tools...")
        tools = await agent_server.get_available_tools()

        total_tools = tools.get("total_count", 0)
        active_tools = tools.get("active_count", 0)

        print(f"\n📊 Tool Registry Status:")
        print(f"  Total tools: {total_tools}")
        print(f"  Active tools: {active_tools}")

        if total_tools > 0:
            print(f"\n✅ Successfully registered {total_tools} tools!")
            print("\n📋 Registered Tools:")
            for tool in tools.get("tools", []):
                print(f"  • {tool['name']}")
                print(f"    - Category: {tool['category']}")
                print(f"    - Status: {tool['status']}")
                print(f"    - Description: {tool['description'][:60]}...")
                print()
        else:
            print("\n⚠️  No tools were registered")
            print("This might be because:")
            print("  1. Tool classes don't have required methods")
            print("  2. Tool initialization failed")
            print("  3. RAG pipeline dependency is missing")

        # Cleanup
        await agent_server.shutdown()
        print("✓ Agent Server shutdown complete")

        return total_tools > 0

    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_tool_registration())
    sys.exit(0 if success else 1)
