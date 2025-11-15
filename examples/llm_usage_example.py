"""
LLM Integration Usage Examples
Demonstrates how to use the LLM integration in the SE SME Agent system
"""

import asyncio
from typing import Dict, Any

from src.shared.llm import (
    create_llm_client,
    create_openai_client,
    create_google_client,
    create_ollama_client,
    LLMProvider,
    ChatMessage,
    MessageRole,
)
from src.shared.llm.integration import (
    get_llm_integration,
    generate_text,
    stream_text,
    chat_with_llm,
)


async def basic_usage_example():
    """Basic usage with the integration helper"""
    print("=== Basic Usage Example ===")

    # Simple text generation
    response = await generate_text(
        prompt="Explain the concept of microservices architecture",
        system_prompt="You are a software engineering expert. Provide clear, concise explanations.",
        temperature=0.7,
    )
    print(f"Response: {response[:200]}...")

    # Streaming response
    print("\n=== Streaming Response ===")
    async for chunk in stream_text(
        prompt="What are the benefits of using Docker containers?",
        system_prompt="You are a DevOps expert.",
        temperature=0.5,
    ):
        print(chunk, end="", flush=True)
    print("\n")


async def multi_provider_example():
    """Example using multiple providers"""
    print("=== Multi-Provider Example ===")

    # Create client with multiple providers
    providers = {
        "openai": {"api_key": "your-openai-key", "model": "gpt-3.5-turbo"},
        "google": {"api_key": "your-google-key", "model": "gemini-2.5-flash "},
        "ollama": {"base_url": "http://localhost:11434", "model": "llama2"},
    }

    client = await create_llm_client(providers)

    # Test each provider
    prompt = "What is the difference between REST and GraphQL APIs?"

    for provider in [LLMProvider.OPENAI, LLMProvider.GOOGLE, LLMProvider.OLLAMA]:
        try:
            response = await client.simple_chat(
                message=prompt, provider=provider, temperature=0.5
            )
            print(f"\n{provider.value.upper()} Response:")
            print(f"{response[:150]}...")
        except Exception as e:
            print(f"\n{provider.value.upper()} Error: {str(e)}")

    await client.cleanup()


async def google_specific_example():
    """Example specifically for Google AI (Gemini)"""
    print("=== Google AI (Gemini) Example ===")

    # Create Google-only client
    client = await create_google_client(
        api_key="your-google-api-key", model="gemini-2.5-flash "
    )

    # Use Google-specific features
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="You are a helpful AI assistant specializing in software engineering.",
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="Explain the SOLID principles in software design.",
        ),
    ]

    try:
        response = await client.chat(
            messages=messages,
            model="gemini-2.5-flash ",
            temperature=0.7,
            # Google-specific parameters
            extra_params={
                "top_k": 40,
                "safety_settings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                    }
                ],
            },
        )

        print(f"Google AI Response: {response.content[:200]}...")
        print(f"Usage: {response.usage}")
        print(f"Response time: {response.response_time:.2f}s")

    except Exception as e:
        print(f"Google AI Error: {str(e)}")

    await client.cleanup()


async def ollama_local_example():
    """Example for local Ollama deployment"""
    print("=== Ollama Local Deployment Example ===")

    # Create Ollama client
    client = await create_ollama_client(
        model="llama2", base_url="http://localhost:11434"
    )

    # Check available models
    try:
        models = await client.get_available_models()
        print(f"Available Ollama models: {models}")

        # Use Ollama-specific features
        response = await client.simple_chat(
            message="Write a Python function to calculate fibonacci numbers",
            model="llama2",
            temperature=0.3,
            # Ollama-specific parameters
            extra_params={
                "top_k": 40,
                "repeat_penalty": 1.1,
                "num_ctx": 4096,  # Context window size
            },
        )

        print(f"Ollama Response: {response[:300]}...")

    except Exception as e:
        print(f"Ollama Error: {str(e)}")
        print("Make sure Ollama is running: ollama serve")
        print("And the model is available: ollama pull llama2")

    await client.cleanup()


async def advanced_integration_example():
    """Advanced integration with the SE SME Agent system"""
    print("=== Advanced Integration Example ===")

    integration = await get_llm_integration()

    # Check available providers
    providers = await integration.get_available_providers()
    print(f"Available providers: {providers}")

    # Validate providers
    validation = await integration.validate_providers()
    print(f"Provider validation: {validation}")

    # Get available models
    models = await integration.get_available_models()
    for provider, model_list in models.items():
        print(f"{provider}: {model_list[:3]}...")  # Show first 3 models

    # Use different providers for different tasks
    tasks = [
        {
            "task": "code_generation",
            "prompt": "Write a Python class for a binary search tree",
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.2,
        },
        {
            "task": "explanation",
            "prompt": "Explain how machine learning works in simple terms",
            "provider": "google",
            "model": "gemini-2.5-flash ",
            "temperature": 0.7,
        },
        {
            "task": "analysis",
            "prompt": "Analyze the pros and cons of microservices vs monolithic architecture",
            "provider": "anthropic",
            "model": "claude-3-haiku-20240307",
            "temperature": 0.5,
        },
    ]

    for task in tasks:
        try:
            print(f"\n--- {task['task'].upper()} TASK ---")

            # Use specific provider for this task
            async with integration.use_provider(LLMProvider(task["provider"])):
                response = await integration.generate_response(
                    prompt=task["prompt"],
                    model=task["model"],
                    temperature=task["temperature"],
                )
                print(f"Response: {response[:150]}...")

        except Exception as e:
            print(f"Task {task['task']} failed: {str(e)}")

    # Get usage metrics
    metrics = integration.get_metrics()
    print(f"\nUsage metrics: {metrics}")


async def chat_conversation_example():
    """Example of a multi-turn conversation"""
    print("=== Chat Conversation Example ===")

    conversation = [
        {"role": "system", "content": "You are a helpful software engineering mentor."},
        {
            "role": "user",
            "content": "I'm learning about design patterns. Can you explain the Observer pattern?",
        },
    ]

    # First response
    response = await chat_with_llm(conversation, temperature=0.7)
    print(f"Assistant: {response['content'][:200]}...")

    # Add to conversation
    conversation.append({"role": "assistant", "content": response["content"]})
    conversation.append(
        {"role": "user", "content": "Can you show me a Python implementation?"}
    )

    # Second response
    response = await chat_with_llm(conversation, temperature=0.3)
    print(f"\nAssistant: {response['content'][:300]}...")

    print(f"\nUsage: {response['usage']}")
    print(f"Provider: {response['provider']}")
    print(f"Response time: {response['response_time']:.2f}s")


async def error_handling_example():
    """Example of error handling"""
    print("=== Error Handling Example ===")

    integration = await get_llm_integration()

    # Test with invalid provider
    try:
        await integration.switch_provider(LLMProvider.GOOGLE)
        response = await integration.generate_response(
            prompt="Test prompt", model="invalid-model"
        )
    except Exception as e:
        print(f"Expected error with invalid model: {type(e).__name__}: {str(e)}")

    # Test with unavailable provider
    try:
        async with integration.use_provider(LLMProvider.OLLAMA):
            response = await integration.generate_response(
                prompt="Test prompt", model="nonexistent-model"
            )
    except Exception as e:
        print(f"Expected error with unavailable provider: {type(e).__name__}: {str(e)}")


async def main():
    """Run all examples"""
    examples = [
        basic_usage_example,
        # multi_provider_example,  # Uncomment when you have API keys
        # google_specific_example,  # Uncomment when you have Google API key
        # ollama_local_example,     # Uncomment when Ollama is running
        advanced_integration_example,
        chat_conversation_example,
        error_handling_example,
    ]

    for example in examples:
        try:
            await example()
            print("\n" + "=" * 50 + "\n")
        except Exception as e:
            print(f"Example {example.__name__} failed: {str(e)}")
            print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
