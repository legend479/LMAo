#!/usr/bin/env python3
"""
Simple test for code generation fix
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agent_server.tools.code_tool_generator import CodeToolGenerator


async def test_single_generation():
    """Test a single code generation"""

    print("Testing Single Code Generation")
    print("=" * 40)

    code_generator = CodeToolGenerator()

    description = "Create a tool that processes CSV files and calculates statistics"

    try:
        result = await code_generator.execute(
            parameters={"description": description}, context=None
        )

        if result.success:
            tool_data = result.data
            print("✅ Tool Generated Successfully")
            print(f"   Tool Name: {tool_data['tool_name']}")
            print(f"   Tool Type: {tool_data['tool_type']}")
            print(f"   Validation: {tool_data['validation_result']['valid']}")
            print(
                f"   Quality Score: {tool_data['validation_result']['quality_score']:.2f}"
            )

            if tool_data["validation_result"]["errors"]:
                print(f"   Errors: {tool_data['validation_result']['errors']}")

            # Show a snippet of the generated code
            code_lines = tool_data["generated_code"].split("\n")
            print("\n   Generated Code Preview (first 10 lines):")
            for i, line in enumerate(code_lines[:10]):
                print(f"   {i+1:2d}: {line}")

        else:
            print(f"❌ Generation Failed: {result.error_message}")

    except Exception as e:
        print(f"❌ Test Failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_single_generation())
