#!/usr/bin/env python3
"""
Comprehensive Integration Test for the Complete Agentic System
Tests the full pipeline: LLM integration, RAG, tools, planning, and chat interface
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.shared.llm.integration import get_llm_integration
from src.agent_server.main import get_agent_server
from src.agent_server.planning import ConversationContext
from src.shared.logging import get_logger

logger = get_logger(__name__)


class IntegratedSystemTester:
    """Comprehensive tester for the integrated agentic system"""

    def __init__(self):
        self.agent_server = None
        self.llm_integration = None
        self.test_results = {}

    async def initialize(self):
        """Initialize all system components"""
        print("🚀 Initializing Integrated Agentic System...")

        try:
            # Initialize LLM integration
            self.llm_integration = await get_llm_integration()
            print("✓ LLM Integration initialized")

            # Initialize agent server
            self.agent_server = await get_agent_server()
            print("✓ Agent Server initialized")

            print("✅ System initialization complete!\n")
            return True

        except Exception as e:
            print(f"❌ System initialization failed: {str(e)}")
            return False

    async def test_llm_providers(self):
        """Test LLM provider functionality"""
        print("🧠 Testing LLM Providers...")

        test_prompt = "What is the difference between REST and GraphQL APIs?"

        try:
            # Test basic generation
            response = await self.llm_integration.generate_response(
                prompt=test_prompt,
                system_prompt="You are a helpful software engineering assistant.",
                temperature=0.5,
                max_tokens=200,
            )

            print(f"✓ LLM Response: {response[:100]}...")

            # Test provider validation
            validation = await self.llm_integration.validate_providers()
            valid_providers = [p for p, v in validation.items() if v]

            print(f"✓ Valid providers: {valid_providers}")

            # Test metrics
            metrics = self.llm_integration.get_metrics()
            print(
                f"✓ LLM Metrics: {metrics['total_requests']} requests, {metrics['providers_count']} providers"
            )

            self.test_results["llm_providers"] = {
                "status": "success",
                "valid_providers": valid_providers,
                "response_length": len(response),
            }

            return True

        except Exception as e:
            print(f"❌ LLM Provider test failed: {str(e)}")
            self.test_results["llm_providers"] = {"status": "failed", "error": str(e)}
            return False

    async def test_planning_module(self):
        """Test the planning module with LLM integration"""
        print("📋 Testing Planning Module...")

        try:
            # Create test context
            context = ConversationContext(
                session_id="test_session",
                user_id="test_user",
                message_history=[],
                user_preferences={},
            )

            # Test different types of queries
            test_queries = [
                "Explain how microservices architecture works",
                "Write a Python function to implement binary search",
                "Analyze this code for security vulnerabilities",
                "Generate documentation for a REST API",
            ]

            for query in test_queries:
                plan = await self.agent_server.planning_module.create_plan(
                    query, context
                )

                print(f"✓ Plan created for: '{query[:30]}...'")
                print(f"  - Tasks: {len(plan.tasks)}")
                print(f"  - Dependencies: {len(plan.dependencies)}")
                print(f"  - Estimated duration: {plan.estimated_duration:.2f}s")

            self.test_results["planning"] = {
                "status": "success",
                "queries_tested": len(test_queries),
            }

            return True

        except Exception as e:
            print(f"❌ Planning module test failed: {str(e)}")
            self.test_results["planning"] = {"status": "failed", "error": str(e)}
            return False

    async def test_agent_orchestration(self):
        """Test the complete agent orchestration"""
        print("🎭 Testing Agent Orchestration...")

        test_messages = [
            {
                "message": "What are the SOLID principles in software engineering?",
                "expected_type": "knowledge_retrieval",
            },
            {
                "message": "Write a Python class for a binary search tree",
                "expected_type": "code_generation",
            },
            {
                "message": "Explain the benefits of using Docker containers",
                "expected_type": "content_generation",
            },
            {
                "message": "Analyze this algorithm for time complexity",
                "expected_type": "analysis",
            },
        ]

        successful_tests = 0

        for i, test_case in enumerate(test_messages):
            try:
                print(f"\n  Test {i+1}: {test_case['message'][:50]}...")

                # Process message through agent
                result = await self.agent_server.process_message(
                    message=test_case["message"],
                    session_id=f"test_session_{i}",
                    user_id="test_user",
                )

                response = result.get("response", "")
                metadata = result.get("metadata", {})

                print(f"    ✓ Response length: {len(response)} characters")
                print(f"    ✓ Processing metadata: {list(metadata.keys())}")
                print(f"    ✓ Response preview: {response[:100]}...")

                successful_tests += 1

            except Exception as e:
                print(f"    ❌ Test {i+1} failed: {str(e)}")

        success_rate = successful_tests / len(test_messages)
        print(
            f"\n✓ Agent orchestration success rate: {success_rate:.1%} ({successful_tests}/{len(test_messages)})"
        )

        self.test_results["orchestration"] = {
            "status": "success" if success_rate > 0.5 else "partial",
            "success_rate": success_rate,
            "successful_tests": successful_tests,
            "total_tests": len(test_messages),
        }

        return success_rate > 0.5

    async def test_tool_integration(self):
        """Test tool integration and execution"""
        print("🔧 Testing Tool Integration...")

        try:
            # Test available tools
            tools = await self.agent_server.get_available_tools()
            print(
                f"✓ Available tools: {list(tools.keys()) if tools else 'None configured'}"
            )

            # Test tool execution (if any tools are available)
            if tools:
                tool_name = list(tools.keys())[0]
                result = await self.agent_server.execute_tool(
                    tool_name=tool_name,
                    parameters={"test": True},
                    session_id="test_session",
                )
                print(
                    f"✓ Tool '{tool_name}' executed: {result.get('status', 'unknown')}"
                )

            self.test_results["tools"] = {
                "status": "success",
                "available_tools": len(tools) if tools else 0,
            }

            return True

        except Exception as e:
            print(f"❌ Tool integration test failed: {str(e)}")
            self.test_results["tools"] = {"status": "failed", "error": str(e)}
            return False

    async def test_conversation_flow(self):
        """Test multi-turn conversation flow"""
        print("💬 Testing Conversation Flow...")

        conversation = [
            "Hello, I'm working on a web application project",
            "What database should I use for a high-traffic application?",
            "How do I implement caching for better performance?",
            "Can you show me a Python example of Redis caching?",
            "What are the security considerations I should keep in mind?",
        ]

        session_id = "conversation_test"
        successful_turns = 0

        for i, message in enumerate(conversation):
            try:
                print(f"\n  Turn {i+1}: {message}")

                result = await self.agent_server.process_message(
                    message=message, session_id=session_id, user_id="test_user"
                )

                response = result.get("response", "")
                print(f"    ✓ Agent: {response[:150]}...")

                successful_turns += 1

                # Small delay to simulate real conversation
                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"    ❌ Turn {i+1} failed: {str(e)}")

        success_rate = successful_turns / len(conversation)
        print(
            f"\n✓ Conversation flow success rate: {success_rate:.1%} ({successful_turns}/{len(conversation)})"
        )

        self.test_results["conversation"] = {
            "status": "success" if success_rate > 0.7 else "partial",
            "success_rate": success_rate,
            "successful_turns": successful_turns,
            "total_turns": len(conversation),
        }

        return success_rate > 0.7

    async def test_streaming_responses(self):
        """Test streaming response functionality"""
        print("🌊 Testing Streaming Responses...")

        try:
            prompt = (
                "Explain the Model-View-Controller (MVC) architecture pattern in detail"
            )

            print("  Starting stream...")
            chunks_received = 0
            total_content = ""

            async for chunk in self.llm_integration.stream_response(
                prompt=prompt,
                system_prompt="You are a software architecture expert.",
                temperature=0.6,
            ):
                total_content += chunk
                chunks_received += 1

                # Print first few chunks
                if chunks_received <= 3:
                    print(f"    Chunk {chunks_received}: '{chunk[:20]}...'")

            print(
                f"✓ Streaming complete: {chunks_received} chunks, {len(total_content)} characters"
            )

            self.test_results["streaming"] = {
                "status": "success",
                "chunks_received": chunks_received,
                "total_length": len(total_content),
            }

            return True

        except Exception as e:
            print(f"❌ Streaming test failed: {str(e)}")
            self.test_results["streaming"] = {"status": "failed", "error": str(e)}
            return False

    async def test_error_handling(self):
        """Test error handling and recovery"""
        print("🛡️ Testing Error Handling...")

        error_scenarios = [
            {"message": "", "description": "Empty message"},
            {"message": "x" * 10000, "description": "Very long message"},
            {
                "message": "Generate code in InvalidLanguage",
                "description": "Invalid parameters",
            },
        ]

        handled_errors = 0

        for scenario in error_scenarios:
            try:
                print(f"  Testing: {scenario['description']}")

                result = await self.agent_server.process_message(
                    message=scenario["message"],
                    session_id="error_test",
                    user_id="test_user",
                )

                # Should get a response even for error cases
                response = result.get("response", "")
                if response:
                    print(f"    ✓ Graceful handling: {response[:50]}...")
                    handled_errors += 1
                else:
                    print(f"    ⚠️ No response received")

            except Exception as e:
                print(f"    ❌ Unhandled error: {str(e)}")

        success_rate = handled_errors / len(error_scenarios)
        print(
            f"\n✓ Error handling success rate: {success_rate:.1%} ({handled_errors}/{len(error_scenarios)})"
        )

        self.test_results["error_handling"] = {
            "status": "success" if success_rate > 0.8 else "partial",
            "success_rate": success_rate,
            "handled_errors": handled_errors,
            "total_scenarios": len(error_scenarios),
        }

        return success_rate > 0.8

    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📊 INTEGRATED SYSTEM TEST REPORT")
        print("=" * 60)

        total_tests = len(self.test_results)
        successful_tests = sum(
            1 for result in self.test_results.values() if result["status"] == "success"
        )
        partial_tests = sum(
            1 for result in self.test_results.values() if result["status"] == "partial"
        )

        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Partial: {partial_tests}")
        print(f"Failed: {total_tests - successful_tests - partial_tests}")
        print(f"Success Rate: {successful_tests/total_tests:.1%}")

        print("\nDetailed Results:")
        print("-" * 40)

        for test_name, result in self.test_results.items():
            status_icon = (
                "✅"
                if result["status"] == "success"
                else "⚠️" if result["status"] == "partial" else "❌"
            )
            print(
                f"{status_icon} {test_name.replace('_', ' ').title()}: {result['status']}"
            )

            # Show additional details
            if "success_rate" in result:
                print(f"    Success Rate: {result['success_rate']:.1%}")
            if "error" in result:
                print(f"    Error: {result['error']}")

        print("\n" + "=" * 60)

        # Overall assessment
        if successful_tests == total_tests:
            print("🎉 ALL TESTS PASSED! The integrated system is fully functional.")
        elif successful_tests + partial_tests >= total_tests * 0.8:
            print("✅ SYSTEM MOSTLY FUNCTIONAL with some minor issues.")
        elif successful_tests >= total_tests * 0.5:
            print("⚠️ SYSTEM PARTIALLY FUNCTIONAL. Some components need attention.")
        else:
            print("❌ SYSTEM HAS SIGNIFICANT ISSUES. Major components are not working.")

        print("\nRecommendations:")
        if successful_tests < total_tests:
            print("- Check configuration files (.env)")
            print("- Verify API keys are valid")
            print("- Ensure all dependencies are installed")
            print("- Check logs for detailed error information")

        if (
            "llm_providers" in self.test_results
            and self.test_results["llm_providers"]["status"] != "success"
        ):
            print(
                "- Configure at least one LLM provider (OpenAI, Google, Anthropic, or Ollama)"
            )

        print("\n📝 For detailed logs, check the application logs.")
        print("🔧 For troubleshooting, see docs/LLM_INTEGRATION.md")


async def main():
    """Run comprehensive integration tests"""
    print("🧪 Starting Comprehensive Integration Tests")
    print("=" * 60)

    tester = IntegratedSystemTester()

    # Initialize system
    if not await tester.initialize():
        print("❌ Cannot proceed with tests due to initialization failure")
        return

    # Run all tests
    tests = [
        ("LLM Providers", tester.test_llm_providers),
        ("Planning Module", tester.test_planning_module),
        ("Agent Orchestration", tester.test_agent_orchestration),
        ("Tool Integration", tester.test_tool_integration),
        ("Conversation Flow", tester.test_conversation_flow),
        ("Streaming Responses", tester.test_streaming_responses),
        ("Error Handling", tester.test_error_handling),
    ]

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            await test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {str(e)}")
            tester.test_results[test_name.lower().replace(" ", "_")] = {
                "status": "failed",
                "error": f"Test crashed: {str(e)}",
            }

    # Generate final report
    tester.generate_report()

    # Cleanup
    if tester.agent_server:
        await tester.agent_server.shutdown()
    if tester.llm_integration:
        await tester.llm_integration.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
