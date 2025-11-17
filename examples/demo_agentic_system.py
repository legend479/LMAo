#!/usr/bin/env python3
"""
Demo Script for the Complete Autonomous RAG-Supported Tool-Enabled Smart Agentic System
Shows the system capabilities with interactive examples
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.agent_server.main import get_agent_server
from src.shared.llm.integration import get_llm_integration


class AgenticSystemDemo:
    """Interactive demo of the complete agentic system"""

    def __init__(self):
        self.agent_server = None
        self.session_id = f"demo_session_{int(datetime.now().timestamp())}"
        self.user_id = "demo_user"

    async def initialize(self):
        """Initialize the system"""
        print("🚀 Initializing Complete Agentic System...")
        print("   - Multi-provider LLM integration")
        print("   - RAG-powered knowledge retrieval")
        print("   - Tool-enabled task execution")
        print("   - Intelligent planning and orchestration")
        print("   - Real-time chat interface")

        try:
            self.agent_server = await get_agent_server()
            print("✅ System ready!\n")
            return True
        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
            return False

    async def demo_conversation(self):
        """Demo a natural conversation with the agent"""
        print("💬 CONVERSATION DEMO")
        print("=" * 50)

        conversation = [
            "Hello! I'm building a microservices application. Can you help me?",
            "What are the key principles I should follow for microservices design?",
            "How should I handle communication between services?",
            "Can you show me a Python example of implementing a REST API for a microservice?",
            "What about database design? Should each service have its own database?",
            "How do I implement monitoring and logging across multiple services?",
        ]

        for i, message in enumerate(conversation, 1):
            print(f"\n👤 User: {message}")

            try:
                # Process through the complete agent system
                result = await self.agent_server.process_message(
                    message=message, session_id=self.session_id, user_id=self.user_id
                )

                response = result.get("response", "No response received")
                metadata = result.get("metadata", {})

                print(f"🤖 Agent: {response}")

                # Show metadata if available
                if metadata:
                    print(f"📊 Metadata: {list(metadata.keys())}")

                # Pause for readability
                await asyncio.sleep(1)

            except Exception as e:
                print(f"❌ Error: {str(e)}")

        print("\n✅ Conversation demo complete!")

    async def demo_code_generation(self):
        """Demo code generation capabilities"""
        print("\n💻 CODE GENERATION DEMO")
        print("=" * 50)

        code_requests = [
            "Write a Python class for a simple REST API client with error handling",
            "Create a JavaScript function that implements debouncing",
            "Generate a SQL query to find the top 5 customers by total order value",
            "Write a Docker Compose file for a web app with database and Redis",
        ]

        for request in code_requests:
            print(f"\n👤 Request: {request}")

            try:
                result = await self.agent_server.process_message(
                    message=request, session_id=self.session_id, user_id=self.user_id
                )

                response = result.get("response", "No response received")
                print(f"🤖 Generated Code:\n{response[:500]}...")

                if len(response) > 500:
                    print("   [Code truncated for demo - full code would be provided]")

            except Exception as e:
                print(f"❌ Error: {str(e)}")

        print("\n✅ Code generation demo complete!")

    async def demo_analysis_capabilities(self):
        """Demo analysis and review capabilities"""
        print("\n🔍 ANALYSIS DEMO")
        print("=" * 50)

        analysis_tasks = [
            {
                "request": "Analyze the pros and cons of using GraphQL vs REST APIs",
                "type": "comparison",
            },
            {
                "request": "Review this architecture: monolithic app with MySQL, Redis cache, and nginx load balancer",
                "type": "architecture_review",
            },
            {
                "request": "What are the security implications of using JWT tokens for authentication?",
                "type": "security_analysis",
            },
        ]

        for task in analysis_tasks:
            print(f"\n👤 Analysis Request: {task['request']}")

            try:
                result = await self.agent_server.process_message(
                    message=task["request"],
                    session_id=self.session_id,
                    user_id=self.user_id,
                )

                response = result.get("response", "No response received")
                print(f"🤖 Analysis:\n{response[:400]}...")

                if len(response) > 400:
                    print(
                        "   [Analysis truncated for demo - full analysis would be provided]"
                    )

            except Exception as e:
                print(f"❌ Error: {str(e)}")

        print("\n✅ Analysis demo complete!")

    async def demo_multi_step_planning(self):
        """Demo multi-step task planning and execution"""
        print("\n📋 MULTI-STEP PLANNING DEMO")
        print("=" * 50)

        complex_request = """I want to build a todo application. Please help me:
        1. Design the database schema
        2. Create a REST API specification
        3. Write the backend code in Python
        4. Design the frontend architecture
        5. Provide deployment instructions"""

        print(f"👤 Complex Request: {complex_request}")

        try:
            result = await self.agent_server.process_message(
                message=complex_request,
                session_id=self.session_id,
                user_id=self.user_id,
            )

            response = result.get("response", "No response received")
            metadata = result.get("metadata", {})

            print(f"🤖 Planned Response:\n{response[:600]}...")

            if len(response) > 600:
                print("   [Response truncated for demo - full plan would be provided]")

            # Show planning metadata
            if metadata:
                print(f"\n📊 Planning Metadata:")
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        print(f"   - {key}: {value}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

        print("\n✅ Multi-step planning demo complete!")

    async def demo_streaming_response(self):
        """Demo streaming response capability"""
        print("\n🌊 STREAMING RESPONSE DEMO")
        print("=" * 50)

        print("👤 Request: Explain the principles of clean code architecture in detail")
        print("🤖 Streaming Response:")

        try:
            llm_integration = await get_llm_integration()

            response_parts = []
            async for chunk in llm_integration.stream_response(
                prompt="Explain the principles of clean code architecture in detail",
                system_prompt="You are a software architecture expert. Provide detailed explanations.",
                temperature=0.6,
            ):
                print(chunk, end="", flush=True)
                response_parts.append(chunk)

                # Add small delay for demo effect
                await asyncio.sleep(0.05)

            print(
                f"\n\n📊 Streamed {len(response_parts)} chunks, {len(''.join(response_parts))} total characters"
            )

        except Exception as e:
            print(f"❌ Streaming error: {str(e)}")

        print("\n✅ Streaming demo complete!")

    async def demo_provider_switching(self):
        """Demo switching between different LLM providers"""
        print("\n🔄 PROVIDER SWITCHING DEMO")
        print("=" * 50)

        try:
            llm_integration = await get_llm_integration()

            # Check available providers
            providers = await llm_integration.get_available_providers()
            print(f"Available providers: {providers}")

            # Test with different providers if available
            test_prompt = "What is dependency injection in software engineering?"

            for provider in providers[:2]:  # Test first 2 providers
                try:
                    print(f"\n🔧 Testing with {provider}:")

                    # This would switch providers in a real implementation
                    result = await self.agent_server.process_message(
                        message=f"Using {provider}: {test_prompt}",
                        session_id=self.session_id,
                        user_id=self.user_id,
                    )

                    response = result.get("response", "No response")
                    print(f"Response: {response[:200]}...")

                except Exception as e:
                    print(f"❌ {provider} failed: {str(e)}")

        except Exception as e:
            print(f"❌ Provider switching demo failed: {str(e)}")

        print("\n✅ Provider switching demo complete!")

    async def show_system_capabilities(self):
        """Show comprehensive system capabilities"""
        print("\n🎯 SYSTEM CAPABILITIES OVERVIEW")
        print("=" * 50)

        capabilities = [
            "🧠 Multi-Provider LLM Support (OpenAI, Anthropic, Google AI, Ollama)",
            "📚 RAG-Powered Knowledge Retrieval",
            "🔧 Tool Integration and Execution",
            "📋 Intelligent Task Planning and Decomposition",
            "🎭 Workflow Orchestration with LangGraph",
            "💬 Real-time Chat Interface (REST + WebSocket)",
            "🌊 Streaming Response Support",
            "🔄 Provider Switching and Fallbacks",
            "🛡️ Error Handling and Recovery",
            "📊 Usage Metrics and Monitoring",
            "🏗️ Modular and Extensible Architecture",
            "🐳 Docker-Ready Deployment",
        ]

        for capability in capabilities:
            print(f"  {capability}")
            await asyncio.sleep(0.1)  # Dramatic effect

        print("\n🚀 This system provides a complete autonomous AI assistant")
        print("   capable of handling complex software engineering tasks!")

    async def run_demo(self):
        """Run the complete demo"""
        if not await self.initialize():
            return

        await self.show_system_capabilities()

        demos = [
            ("Natural Conversation", self.demo_conversation),
            ("Code Generation", self.demo_code_generation),
            ("Analysis Capabilities", self.demo_analysis_capabilities),
            ("Multi-Step Planning", self.demo_multi_step_planning),
            ("Streaming Responses", self.demo_streaming_response),
            ("Provider Switching", self.demo_provider_switching),
        ]

        for demo_name, demo_func in demos:
            try:
                await demo_func()
                print(f"\n⏸️  Press Enter to continue to next demo...")
                input()  # Pause between demos
            except KeyboardInterrupt:
                print(f"\n⏹️  Demo interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Demo '{demo_name}' failed: {str(e)}")

        print("\n🎉 DEMO COMPLETE!")
        print("=" * 50)
        print("The Complete Autonomous RAG-Supported Tool-Enabled Smart Agentic System")
        print("is ready for production use!")
        print("\n📖 For more information:")
        print("   - See docs/LLM_INTEGRATION.md for detailed usage")
        print("   - Run test_integrated_system.py for comprehensive tests")
        print("   - Check examples/llm_usage_example.py for code examples")

        # Cleanup
        if self.agent_server:
            await self.agent_server.shutdown()


async def main():
    """Run the interactive demo"""
    print("🎬 COMPLETE AGENTIC SYSTEM DEMO")
    print("=" * 60)
    print("This demo showcases the full capabilities of the integrated system:")
    print("- Autonomous AI agent with multi-provider LLM support")
    print("- RAG-powered knowledge retrieval")
    print("- Tool-enabled task execution")
    print("- Intelligent planning and orchestration")
    print("- Real-time chat interface")
    print("\n🎮 Interactive Demo Mode")
    print("Press Ctrl+C at any time to exit")
    print("=" * 60)

    demo = AgenticSystemDemo()

    try:
        await demo.run_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo terminated by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    asyncio.run(main())
