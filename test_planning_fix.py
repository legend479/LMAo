#!/usr/bin/env python3
"""
Test planning feature fix - verify asdict() error is resolved
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_planning_fix():
    """Test that planning feature works without asdict() error"""
    print("=" * 70)
    print("PLANNING FEATURE FIX TEST")
    print("=" * 70)
    print("\nTesting that planning feature works without asdict() error...")
    print()

    try:
        from src.agent_server.main import AgentServer

        print("✓ Imported AgentServer")

        agent_server = AgentServer()
        print("✓ Created AgentServer instance")

        print("\nInitializing Agent Server...")
        await agent_server.initialize()
        print("✓ Agent Server initialized")

        # Test planning with a sample query
        test_query = "i want to code a game from scratch"

        print("\n" + "=" * 70)
        print("TEST: PLANNING FEATURE")
        print("=" * 70)
        print(f"\nQuery: '{test_query}'")
        print("\nCreating execution plan...")

        try:
            # Get memory context (this was failing before)
            context = await agent_server.memory_manager.get_context("test_session")
            print("✓ Memory context retrieved successfully")

            # Create plan (this uses the context)
            plan = await agent_server.planning_module.create_plan(test_query, context)
            print("✓ Execution plan created successfully")

            # Display plan details
            print("\n" + "=" * 70)
            print("PLAN DETAILS")
            print("=" * 70)
            print(f"\nPlan ID: {plan.plan_id}")
            print(f"Total Tasks: {len(plan.tasks)}")
            print(f"Estimated Duration: {plan.estimated_duration:.2f}s")
            print(f"Priority: {plan.priority}")

            if plan.tasks:
                print(f"\nFirst few tasks:")
                for i, task in enumerate(plan.tasks[:3], 1):
                    print(f"  {i}. {task.get('id', 'N/A')} - {task.get('type', 'N/A')}")

            print("\n" + "=" * 70)
            print("TEST RESULTS")
            print("=" * 70)
            print("\n✅ PASSED: Planning feature works!")
            print("   • No asdict() error")
            print("   • Memory context retrieved")
            print("   • Plan created successfully")
            print(f"   • {len(plan.tasks)} tasks generated")

            success = True

        except TypeError as e:
            if "asdict()" in str(e):
                print("\n❌ FAILED: asdict() error still present!")
                print(f"   Error: {str(e)}")
                success = False
            else:
                raise

        # Cleanup
        await agent_server.shutdown()
        print("\n✓ Agent Server shutdown complete")

        return success

    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_planning_fix())

    print("\n" + "=" * 70)
    if success:
        print("✅ PLANNING FIX VERIFIED!")
        print("=" * 70)
        print("\nThe planning feature now works correctly!")
        print("You can use option 6 in the CLI without errors.")
    else:
        print("❌ TEST FAILED")
        print("=" * 70)
        print("\nCheck the output above for details.")

    sys.exit(0 if success else 1)
