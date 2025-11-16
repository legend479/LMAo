"""
LLM Integration Helper
Provides easy integration with the existing system
"""

from typing import Dict, Any, List, Optional, AsyncGenerator
import asyncio
from contextlib import asynccontextmanager

from .factory import get_global_client, create_llm_client
from .models import (
    LLMRequest,
    LLMResponse,
    LLMProvider,
    ChatMessage,
    MessageRole,
    LLMError,
)
from ..logging import get_logger

logger = get_logger(__name__)


class LLMIntegration:
    """High-level LLM integration for the SE SME Agent system"""

    def __init__(self):
        self._client = None
        self._initialized = False

    async def initialize(self, providers: Optional[Dict[str, Dict[str, Any]]] = None):
        """Initialize LLM integration"""
        try:
            if providers:
                self._client = await create_llm_client(providers)
            else:
                self._client = await get_global_client()

            self._initialized = True
            logger.info("LLM integration initialized successfully")

            # Log available providers
            metrics = self._client.get_metrics()
            logger.info(f"Available providers: {metrics['available_providers']}")
            logger.info(f"Default provider: {metrics['default_provider']}")

        except Exception as e:
            logger.error(f"Failed to initialize LLM integration: {str(e)}")
            raise

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Generate a response using the LLM

        Args:
            prompt: User prompt/question
            system_prompt: System instructions
            model: Specific model to use
            provider: Specific provider to use
            temperature: Response creativity (0.0-2.0)
            max_tokens: Maximum response length
            **kwargs: Additional parameters

        Returns:
            Generated response text
        """
        if not self._initialized:
            await self.initialize()

        try:
            messages = []

            if system_prompt:
                messages.append(
                    ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
                )

            messages.append(ChatMessage(role=MessageRole.USER, content=prompt))

            # Use default model if not specified
            if not model:
                metrics = self._client.get_metrics()
                default_provider = metrics.get("default_provider")
                if default_provider == LLMProvider.OPENAI:
                    model = "gpt-3.5-turbo"
                elif default_provider == LLMProvider.ANTHROPIC:
                    model = "claude-3-haiku-20240307"
                elif default_provider == LLMProvider.GOOGLE:
                    model = "gemini-2.5-flash"
                elif default_provider == LLMProvider.OLLAMA:
                    model = "llama2"
                else:
                    model = "gpt-3.5-turbo"  # Fallback

            # ENHANCED LOGGING: Log LLM request details
            logger.debug(
                "LLM Request",
                model=model,
                provider=str(provider) if provider else "default",
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt_length=len(system_prompt) if system_prompt else 0,
                prompt_length=len(prompt),
            )

            # Log first 200 chars of prompt for debugging
            logger.debug(f"LLM Prompt Preview: {prompt[:200]}...")
            if system_prompt:
                logger.debug(f"LLM System Prompt Preview: {system_prompt[:200]}...")

            request = LLMRequest(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            response = await self._client.generate(request, provider)

            # ENHANCED LOGGING: Log LLM response details
            logger.debug(
                "LLM Response",
                response_length=len(response.content),
                model_used=response.model if hasattr(response, "model") else model,
            )
            logger.debug(f"LLM Response Preview: {response.content[:200]}...")

            return response.content

        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise

    async def stream_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response using the LLM

        Args:
            prompt: User prompt/question
            system_prompt: System instructions
            model: Specific model to use
            provider: Specific provider to use
            temperature: Response creativity (0.0-2.0)
            max_tokens: Maximum response length
            **kwargs: Additional parameters

        Yields:
            Response chunks as they are generated
        """
        if not self._initialized:
            await self.initialize()

        try:
            messages = []

            if system_prompt:
                messages.append(
                    ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
                )

            messages.append(ChatMessage(role=MessageRole.USER, content=prompt))

            # Use default model if not specified
            if not model:
                metrics = self._client.get_metrics()
                default_provider = metrics.get("default_provider")
                if default_provider == LLMProvider.OPENAI:
                    model = "gpt-3.5-turbo"
                elif default_provider == LLMProvider.ANTHROPIC:
                    model = "claude-3-haiku-20240307"
                elif default_provider == LLMProvider.GOOGLE:
                    model = "gemini-2.5-flash"
                elif default_provider == LLMProvider.OLLAMA:
                    model = "llama2"
                else:
                    model = "gpt-3.5-turbo"  # Fallback

            request = LLMRequest(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            async for chunk in self._client.stream_generate(request, provider):
                yield chunk

        except Exception as e:
            logger.error(f"Failed to stream response: {str(e)}")
            raise

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Complete a chat conversation

        Args:
            messages: List of messages in format [{"role": "user", "content": "..."}]
            model: Specific model to use
            provider: Specific provider to use
            **kwargs: Additional parameters

        Returns:
            Complete LLM response object
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Convert dict messages to ChatMessage objects
            chat_messages = []
            for msg in messages:
                role = MessageRole(msg["role"])
                content = msg["content"]
                chat_messages.append(ChatMessage(role=role, content=content))

            # Use default model if not specified
            if not model:
                metrics = self._client.get_metrics()
                default_provider = metrics.get("default_provider")
                if default_provider == LLMProvider.OPENAI:
                    model = "gpt-3.5-turbo"
                elif default_provider == LLMProvider.ANTHROPIC:
                    model = "claude-3-haiku-20240307"
                elif default_provider == LLMProvider.GOOGLE:
                    model = "gemini-2.5-flash"
                elif default_provider == LLMProvider.OLLAMA:
                    model = "llama2"
                else:
                    model = "gpt-3.5-turbo"  # Fallback

            request = LLMRequest(messages=chat_messages, model=model, **kwargs)

            return await self._client.generate(request, provider)

        except Exception as e:
            logger.error(f"Failed to complete chat: {str(e)}")
            raise

    async def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        if not self._initialized:
            await self.initialize()

        metrics = self._client.get_metrics()
        return metrics.get("available_providers", [])

    async def get_available_models(
        self, provider: Optional[LLMProvider] = None
    ) -> Dict[str, List[str]]:
        """Get available models for providers"""
        if not self._initialized:
            await self.initialize()

        return await self._client.get_available_models(provider)

    async def validate_providers(self) -> Dict[str, bool]:
        """Validate all providers"""
        if not self._initialized:
            await self.initialize()

        return await self._client.validate_providers()

    async def switch_provider(self, provider: LLMProvider) -> None:
        """Switch default provider"""
        if not self._initialized:
            await self.initialize()

        await self._client.set_default_provider(provider)
        logger.info(f"Switched to provider: {provider}")

    @asynccontextmanager
    async def use_provider(self, provider: LLMProvider):
        """Context manager for temporarily using a specific provider"""
        if not self._initialized:
            await self.initialize()

        async with self._client.provider_context(provider):
            yield self

    def get_metrics(self) -> Dict[str, Any]:
        """Get usage metrics"""
        if not self._initialized:
            return {}

        return self._client.get_metrics()

    async def cleanup(self):
        """Cleanup resources"""
        if self._client:
            await self._client.cleanup()
            self._client = None
        self._initialized = False


# Global integration instance
_global_integration: Optional[LLMIntegration] = None


async def get_llm_integration() -> LLMIntegration:
    """Get or create global LLM integration instance"""
    global _global_integration

    if _global_integration is None:
        _global_integration = LLMIntegration()
        await _global_integration.initialize()

    return _global_integration


# Convenience functions for common operations
async def generate_text(
    prompt: str,
    system_prompt: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> str:
    """Generate text using the default LLM integration"""
    integration = await get_llm_integration()

    provider_enum = None
    if provider:
        provider_enum = LLMProvider(provider.lower())

    return await integration.generate_response(
        prompt=prompt,
        system_prompt=system_prompt,
        provider=provider_enum,
        model=model,
        **kwargs,
    )


async def stream_text(
    prompt: str,
    system_prompt: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> AsyncGenerator[str, None]:
    """Stream text using the default LLM integration"""
    integration = await get_llm_integration()

    provider_enum = None
    if provider:
        provider_enum = LLMProvider(provider.lower())

    async for chunk in integration.stream_response(
        prompt=prompt,
        system_prompt=system_prompt,
        provider=provider_enum,
        model=model,
        **kwargs,
    ):
        yield chunk


async def chat_with_llm(
    messages: List[Dict[str, str]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Chat with LLM and return response data"""
    integration = await get_llm_integration()

    provider_enum = None
    if provider:
        provider_enum = LLMProvider(provider.lower())

    response = await integration.chat_completion(
        messages=messages, provider=provider_enum, model=model, **kwargs
    )

    return {
        "content": response.content,
        "model": response.model,
        "provider": response.provider.value,
        "usage": response.usage,
        "response_time": response.response_time,
    }
