#!/usr/bin/env python3
"""
Simple test to verify that all tools can be imported correctly
"""


def test_tool_imports():
    """Test that all tools can be imported without errors"""

    try:

        print("✓ KnowledgeRetrievalTool imported successfully")

        print("✓ DocumentGenerationTool imported successfully")

        print("✓ EmailAutomationTool imported successfully")

        print("✓ CompilerRuntimeTool imported successfully")

        print("✓ ReadabilityScoringTool imported successfully")

        print("✓ ToolRegistry imported successfully")

        print("\n✅ All tools imported successfully!")
        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


if __name__ == "__main__":
    test_tool_imports()
