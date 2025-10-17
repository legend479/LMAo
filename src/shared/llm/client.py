"""
LLM Client Interface
Unified interface for all LLM providers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator
import asyncio
import time
from contextlib import asynccontextmanager

from .models import (
    LLMRequest,
    LLMResponse,
    LLMProvider,
    LLMError,
    ChatMessage,
    MessageRole,
)
from ..logging import get_logger

logger = get_logger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = self.get_provider_name()
        self._initialized = False

    @abstractmethod
    def get_provider_name(self) -> LLMProvider:
        """Get the provider name"""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider"""
        pass

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from LLM"""
        pass

    @abstractmethod
    async def stream_generate(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream generate response from LLM"""
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate connection to the provider"""
        pass

    @abstractmethod
    async def get_available_models(self) -> List[str]:
        """Get list of available models"""
        pass

    async def cleanup(self) -> None:
        """Cleanup resources"""
        pass


class LLMClient:
    """Unified LLM client that manages multiple providers"""

    def __init__(self):
        self.providers: Dict[LLMProvider, BaseLLMProvider] = {}
        self.default_provider: Optional[LLMProvider] = None
        self._request_count = 0
        self._total_tokens = 0
        self._total_cost = 0.0

    async def add_provider(self, provider: BaseLLMProvider) -> None:
        """Add a provider to the client"""
        try:
            await provider.initialize()
            provider_name = provider.get_provider_name()
            self.providers[provider_name] = provider

            # Set as default if it's the first provider
            if self.default_provider is None:
                self.default_provider = provider_name

            logger.info(f"Added LLM provider: {provider_name}")

        except Exception as e:
            logger.error(
                f"Failed to add provider {provider.get_provider_name()}: {str(e)}"
            )
            raise LLMError(f"Failed to initialize provider: {str(e)}")

    async def generate(
        self, request: LLMRequest, provider: Optional[LLMProvider] = None
    ) -> LLMResponse:
        """Generate response using specified or default provider"""

        # Use default provider if none specified
        if provider is None:
            provider = self.default_provider

        if provider is None:
            raise LLMError("No provider specified and no default provider set")

        if provider not in self.providers:
            raise LLMError(f"Provider {provider} not available")

        start_time = time.time()

        try:
            # Log request
            self._request_count += 1
            logger.info(
                f"Generating response with {provider}",
                model=request.model,
                messages_count=len(request.messages),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            # Generate response
            response = await self.providers[provider].generate(request)

            # Update metrics
            response.response_time = time.time() - start_time
            response.provider = provider

            if response.usage:
                self._total_tokens += response.usage.get("total_tokens", 0)

            logger.info(
                f"Response generated successfully",
                provider=provider,
                model=response.model,
                response_time=response.response_time,
                tokens_used=(
                    response.usage.get("total_tokens", 0) if response.usage else 0
                ),
            )

            return response

        except Exception as e:
            logger.error(
                f"Failed to generate response with {provider}",
                error=str(e),
                model=request.model,
            )
            raise

    async def stream_generate(
        self, request: LLMRequest, provider: Optional[LLMProvider] = None
    ) -> AsyncGenerator[str, None]:
        """Stream generate response using specified or default provider"""

        # Use default provider if none specified
        if provider is None:
            provider = self.default_provider

        if provider is None:
            raise LLMError("No provider specified and no default provider set")

        if provider not in self.providers:
            raise LLMError(f"Provider {provider} not available")

        try:
            logger.info(
                f"Streaming response with {provider}",
                model=request.model,
                messages_count=len(request.messages),
            )

            async for chunk in self.providers[provider].stream_generate(request):
                yield chunk

        except Exception as e:
            logger.error(
                f"Failed to stream response with {provider}",
                error=str(e),
                model=request.model,
            )
            raise

    async def chat(
        self,
        messages: List[ChatMessage],
        model: str,
        provider: Optional[LLMProvider] = None,
        **kwargs,
    ) -> LLMResponse:
        """Convenience method for chat completion"""

        request = LLMRequest(messages=messages, model=model, **kwargs)

        return await self.generate(request, provider)

    async def simple_chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        provider: Optional[LLMProvider] = None,
        **kwargs,
    ) -> str:
        """Simple chat interface that returns just the content"""

        messages = []

        if system_prompt:
            messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))

        messages.append(ChatMessage(role=MessageRole.USER, content=message))

        response = await self.chat(messages, model, provider, **kwargs)
        return response.content

    async def validate_providers(self) -> Dict[LLMProvider, bool]:
        """Validate all registered providers"""
        results = {}

        for provider_name, provider in self.providers.items():
            try:
                is_valid = await provider.validate_connection()
                results[provider_name] = is_valid
                logger.info(
                    f"Provider {provider_name} validation: {'✓' if is_valid else '✗'}"
                )
            except Exception as e:
                results[provider_name] = False
                logger.error(f"Provider {provider_name} validation failed: {str(e)}")

        return results

    async def get_available_models(
        self, provider: Optional[LLMProvider] = None
    ) -> Dict[LLMProvider, List[str]]:
        """Get available models for all or specific provider"""
        results = {}

        providers_to_check = [provider] if provider else list(self.providers.keys())

        for provider_name in providers_to_check:
            if provider_name in self.providers:
                try:
                    models = await self.providers[provider_name].get_available_models()
                    results[provider_name] = models
                except Exception as e:
                    logger.error(f"Failed to get models for {provider_name}: {str(e)}")
                    results[provider_name] = []

        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics"""
        return {
            "total_requests": self._request_count,
            "total_tokens": self._total_tokens,
            "total_cost": self._total_cost,
            "providers_count": len(self.providers),
            "available_providers": list(self.providers.keys()),
            "default_provider": self.default_provider,
        }

    async def set_default_provider(self, provider: LLMProvider) -> None:
        """Set default provider"""
        if provider not in self.providers:
            raise LLMError(f"Provider {provider} not available")

        self.default_provider = provider
        logger.info(f"Default provider set to: {provider}")

    @asynccontextmanager
    async def provider_context(self, provider: LLMProvider):
        """Context manager for temporarily switching providers"""
        original_provider = self.default_provider
        await self.set_default_provider(provider)
        try:
            yield self
        finally:
            if original_provider:
                await self.set_default_provider(original_provider)

    async def cleanup(self) -> None:
        """Cleanup all providers"""
        logger.info("Cleaning up LLM client")

        for provider_name, provider in self.providers.items():
            try:
                await provider.cleanup()
                logger.info(f"Cleaned up provider: {provider_name}")
            except Exception as e:
                logger.error(f"Failed to cleanup provider {provider_name}: {str(e)}")

        self.providers.clear()
        self.default_provider = None
