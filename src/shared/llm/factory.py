"""
LLM Client Factory
Creates and configures LLM clients based on configuration
"""

from typing import Dict, Any, Optional
import os

from .client import LLMClient
from .providers import (
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    OllamaProvider,
)
from .models import LLMProvider, LLMError
from ..config import get_settings
from ..logging import get_logger

logger = get_logger(__name__)


async def create_llm_client(
    providers: Optional[Dict[str, Dict[str, Any]]] = None
) -> LLMClient:
    """
    Create and configure LLM client with specified providers

    Args:
        providers: Dict of provider configurations. If None, uses settings from config.
                  Format: {
                      "openai": {"api_key": "...", "model": "gpt-4"},
                      "google": {"api_key": "...", "model": "gemini-pro"},
                      "ollama": {"base_url": "http://localhost:11434", "model": "llama2"}
                  }

    Returns:
        Configured LLMClient instance
    """

    client = LLMClient()
    settings = get_settings()

    # Use provided providers or get from settings
    if providers is None:
        providers = _get_providers_from_settings(settings)

    # Initialize each provider
    for provider_name, config in providers.items():
        try:
            provider_enum = LLMProvider(provider_name.lower())
            provider_instance = _create_provider(provider_enum, config)

            if provider_instance:
                await client.add_provider(provider_instance)
                logger.info(f"Successfully added {provider_name} provider")

        except ValueError:
            logger.warning(f"Unknown provider: {provider_name}")
        except Exception as e:
            logger.error(f"Failed to initialize {provider_name} provider: {str(e)}")

    # Set default provider based on settings
    default_provider = settings.llm_provider.lower()
    if default_provider in [p.value for p in LLMProvider]:
        try:
            await client.set_default_provider(LLMProvider(default_provider))
        except Exception as e:
            logger.warning(
                f"Could not set default provider to {default_provider}: {str(e)}"
            )

    return client


def _get_providers_from_settings(settings) -> Dict[str, Dict[str, Any]]:
    """Extract provider configurations from settings"""

    providers = {}

    # Get the primary provider from settings
    primary_provider = settings.llm_provider.lower()

    if primary_provider == "openai":
        if settings.llm_api_key:
            providers["openai"] = {
                "api_key": settings.llm_api_key,
                "model": settings.llm_model,
                "base_url": getattr(settings, "openai_base_url", None),
                "organization": getattr(settings, "openai_organization", None),
            }

    elif primary_provider == "anthropic":
        if settings.llm_api_key:
            providers["anthropic"] = {
                "api_key": settings.llm_api_key,
                "model": settings.llm_model,
                "base_url": getattr(settings, "anthropic_base_url", None),
            }

    elif primary_provider == "google":
        if settings.llm_api_key:
            providers["google"] = {
                "api_key": settings.llm_api_key,
                "model": settings.llm_model,
                "base_url": getattr(settings, "google_base_url", None),
            }

    elif primary_provider in ["local", "ollama"]:
        providers["ollama"] = {
            "base_url": settings.llm_base_url or "http://localhost:11434",
            "model": settings.llm_model,
        }

    # Check for additional provider configurations from environment
    _add_env_providers(providers)

    return providers


def _add_env_providers(providers: Dict[str, Dict[str, Any]]) -> None:
    """Add additional providers from environment variables"""

    # OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and "openai" not in providers:
        providers["openai"] = {
            "api_key": openai_key,
            "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            "base_url": os.getenv("OPENAI_BASE_URL"),
            "organization": os.getenv("OPENAI_ORGANIZATION"),
        }

    # Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key and "anthropic" not in providers:
        providers["anthropic"] = {
            "api_key": anthropic_key,
            "model": os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
            "base_url": os.getenv("ANTHROPIC_BASE_URL"),
        }

    # Google AI
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if google_key and "google" not in providers:
        providers["google"] = {
            "api_key": google_key,
            "model": os.getenv("GOOGLE_MODEL", "gemini-pro"),
            "base_url": os.getenv("GOOGLE_BASE_URL"),
        }

    # Ollama - Only add if explicitly configured
    ollama_url = os.getenv("OLLAMA_BASE_URL")
    if ollama_url and "ollama" not in providers:
        providers["ollama"] = {
            "base_url": ollama_url,
            "model": os.getenv("OLLAMA_MODEL", "llama2"),
        }
    # Note: Removed automatic Ollama initialization
    # Ollama will only be added if OLLAMA_BASE_URL is set in environment


def _create_provider(provider_type: LLMProvider, config: Dict[str, Any]):
    """Create provider instance based on type"""

    # Remove None values from config
    config = {k: v for k, v in config.items() if v is not None}

    if provider_type == LLMProvider.OPENAI:
        return OpenAIProvider(config)

    elif provider_type == LLMProvider.ANTHROPIC:
        return AnthropicProvider(config)

    elif provider_type == LLMProvider.GOOGLE:
        return GoogleProvider(config)

    elif provider_type in [LLMProvider.OLLAMA, LLMProvider.LOCAL]:
        return OllamaProvider(config)

    else:
        raise LLMError(f"Unsupported provider type: {provider_type}")


async def create_simple_client(
    provider: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """
    Create a simple LLM client with a single provider

    Args:
        provider: Provider name (openai, anthropic, google, ollama)
        api_key: API key (not needed for Ollama)
        model: Model name
        base_url: Base URL for API
        **kwargs: Additional provider-specific configuration

    Returns:
        Configured LLMClient instance
    """

    config = {"api_key": api_key, "model": model, "base_url": base_url, **kwargs}

    providers = {provider: config}
    return await create_llm_client(providers)


# Convenience functions for specific providers
async def create_openai_client(
    api_key: str,
    model: str = "gpt-3.5-turbo",
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
) -> LLMClient:
    """Create OpenAI-only client"""
    return await create_simple_client(
        "openai",
        api_key=api_key,
        model=model,
        base_url=base_url,
        organization=organization,
    )


async def create_anthropic_client(
    api_key: str, model: str = "claude-3-haiku-20240307", base_url: Optional[str] = None
) -> LLMClient:
    """Create Anthropic-only client"""
    return await create_simple_client(
        "anthropic", api_key=api_key, model=model, base_url=base_url
    )


async def create_google_client(
    api_key: str, model: str = "gemini-pro", base_url: Optional[str] = None
) -> LLMClient:
    """Create Google AI-only client"""
    return await create_simple_client(
        "google", api_key=api_key, model=model, base_url=base_url
    )


async def create_ollama_client(
    model: str = "llama2", base_url: str = "http://localhost:11434"
) -> LLMClient:
    """Create Ollama-only client"""
    return await create_simple_client("ollama", model=model, base_url=base_url)


# Global client instance (singleton pattern)
_global_client: Optional[LLMClient] = None


async def get_global_client() -> LLMClient:
    """Get or create global LLM client instance"""
    global _global_client

    if _global_client is None:
        _global_client = await create_llm_client()

    return _global_client


async def reset_global_client() -> None:
    """Reset global client (useful for testing or reconfiguration)"""
    global _global_client

    if _global_client:
        await _global_client.cleanup()
        _global_client = None
