#!/usr/bin/env python3
"""
Test script for LLM integration
Run this to verify that the LLM integration is working correctly
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.shared.llm import (
    create_llm_client,
    LLMProvider,
    ChatMessage,
    MessageRole,
    LLMError,
)
from src.shared.llm.integration import generate_text


async def test_configuration():
    """Test LLM configuration loading"""
    print("=== Testing Configuration ===")

    try:
        from src.shared.config import get_settings

        settings = get_settings()

        print(f"LLM Provider: {settings.llm_provider}")
        print(f"LLM Model: {settings.llm_model}")
        print(f"LLM API Key: {'***' if settings.llm_api_key else 'Not set'}")
        print(f"LLM Base URL: {settings.llm_base_url or 'Default'}")

        # Check provider-specific settings
        if hasattr(settings, "openai_api_key"):
            print(f"OpenAI API Key: {'***' if settings.openai_api_key else 'Not set'}")
        if hasattr(settings, "google_api_key"):
            print(f"Google API Key: {'***' if settings.google_api_key else 'Not set'}")
        if hasattr(settings, "anthropic_api_key"):
            print(
                f"Anthropic API Key: {'***' if settings.anthropic_api_key else 'Not set'}"
            )
        if hasattr(settings, "ollama_base_url"):
            print(f"Ollama Base URL: {settings.ollama_base_url}")

        print("✓ Configuration loaded successfully")
        return True

    except Exception as e:
        print(f"✗ Configuration error: {str(e)}")
        return False


async def test_client_creation():
    """Test LLM client creation"""
    print("\n=== Testing Client Creation ===")

    try:
        # Test with minimal configuration
        providers = {
            "ollama": {"base_url": "http://localhost:11434", "model": "llama2"}
        }

        client = await create_llm_client(providers)
        print("✓ LLM client created successfully")

        # Test metrics
        metrics = client.get_metrics()
        print(f"Available providers: {metrics['available_providers']}")
        print(f"Default provider: {metrics['default_provider']}")

        await client.cleanup()
        return True

    except Exception as e:
        print(f"✗ Client creation error: {str(e)}")
        return False


async def test_provider_validation():
    """Test provider validation"""
    print("\n=== Testing Provider Validation ===")

    # Test different provider configurations
    test_configs = [
        {
            "name": "OpenAI",
            "config": {
                "openai": {
                    "api_key": os.getenv("OPENAI_API_KEY", "test-key"),
                    "model": "gpt-3.5-turbo",
                }
            },
        },
        {
            "name": "Google AI",
            "config": {
                "google": {
                    "api_key": os.getenv("GOOGLE_API_KEY", "test-key"),
                    "model": "gemini-pro",
                }
            },
        },
        {
            "name": "Anthropic",
            "config": {
                "anthropic": {
                    "api_key": os.getenv("ANTHROPIC_API_KEY", "test-key"),
                    "model": "claude-3-haiku-20240307",
                }
            },
        },
        {
            "name": "Ollama",
            "config": {
                "ollama": {"base_url": "http://localhost:11434", "model": "llama2"}
            },
        },
    ]

    results = {}

    for test_config in test_configs:
        try:
            client = await create_llm_client(test_config["config"])
            validation = await client.validate_providers()

            for provider, is_valid in validation.items():
                status = "✓" if is_valid else "✗"
                print(
                    f"{status} {test_config['name']}: {'Valid' if is_valid else 'Invalid'}"
                )
                results[test_config["name"]] = is_valid

            await client.cleanup()

        except Exception as e:
            print(f"✗ {test_config['name']}: Error - {str(e)}")
            results[test_config["name"]] = False

    return results


async def test_simple_generation():
    """Test simple text generation"""
    print("\n=== Testing Simple Generation ===")

    try:
        # Try with environment configuration first
        response = await generate_text(
            prompt="What is 2+2?",
            system_prompt="You are a helpful assistant. Give brief answers.",
            temperature=0.1,
        )

        print(f"✓ Generated response: {response[:100]}...")
        return True

    except Exception as e:
        print(f"✗ Generation error: {str(e)}")

        # Try with Ollama as fallback
        try:
            print("Trying with Ollama fallback...")

            from src.shared.llm.integration import LLMIntegration

            integration = LLMIntegration()

            providers = {
                "ollama": {"base_url": "http://localhost:11434", "model": "llama2"}
            }

            await integration.initialize(providers)

            response = await integration.generate_response(
                prompt="What is 2+2?",
                system_prompt="You are a helpful assistant. Give brief answers.",
                temperature=0.1,
            )

            print(f"✓ Generated response with Ollama: {response[:100]}...")
            await integration.cleanup()
            return True

        except Exception as e2:
            print(f"✗ Ollama fallback also failed: {str(e2)}")
            return False


async def test_model_listing():
    """Test model listing functionality"""
    print("\n=== Testing Model Listing ===")

    try:
        # Test with Ollama (most likely to work locally)
        from src.shared.llm.providers import OllamaProvider

        provider = OllamaProvider({"base_url": "http://localhost:11434"})

        await provider.initialize()
        models = await provider.get_available_models()

        if models:
            print(f"✓ Available Ollama models: {models}")
        else:
            print("✓ Ollama connected but no models found")
            print("  Run 'ollama pull llama2' to download a model")

        await provider.cleanup()
        return True

    except Exception as e:
        print(f"✗ Model listing error: {str(e)}")
        print("  Make sure Ollama is running: 'ollama serve'")
        return False


async def main():
    """Run all tests"""
    print("LLM Integration Test Suite")
    print("=" * 50)

    tests = [
        ("Configuration", test_configuration),
        ("Client Creation", test_client_creation),
        ("Provider Validation", test_provider_validation),
        ("Simple Generation", test_simple_generation),
        ("Model Listing", test_model_listing),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {str(e)}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed!")
    elif passed > 0:
        print("⚠️  Some tests passed. Check configuration for failed tests.")
    else:
        print("❌ All tests failed. Check your configuration and dependencies.")

    # Recommendations
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS")
    print("=" * 50)

    if not results.get("Configuration", False):
        print("• Check your .env file and ensure it follows .env.example")

    if not results.get("Provider Validation", False):
        print("• Verify your API keys are correct")
        print("• For Ollama: ensure it's running with 'ollama serve'")
        print("• For cloud providers: check your account quotas")

    if not results.get("Simple Generation", False):
        print("• Start with Ollama for local testing")
        print("• Verify at least one provider is properly configured")

    print("\nFor more help, see examples/llm_usage_example.py")


if __name__ == "__main__":
    asyncio.run(main())
