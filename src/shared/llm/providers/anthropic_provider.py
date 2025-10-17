"""
Anthropic Provider Implementation
"""

import asyncio
import httpx
from typing import Dict, Any, List, Optional, AsyncGenerator
import json
import time

from ..client import BaseLLMProvider
from ..models import (
    LLMRequest,
    LLMResponse,
    LLMProvider,
    LLMError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMQuotaExceededError,
    LLMConnectionError,
    MessageRole,
)
from ...logging import get_logger

logger = get_logger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic (Claude) API provider implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.anthropic.com")
        self.timeout = config.get("timeout", 60)
        self.max_retries = config.get("max_retries", 3)
        self.client: Optional[httpx.AsyncClient] = None

    def get_provider_name(self) -> LLMProvider:
        return LLMProvider.ANTHROPIC

    async def initialize(self) -> None:
        """Initialize Anthropic client"""
        if not self.api_key:
            raise LLMError(
                "Anthropic API key is required", provider=LLMProvider.ANTHROPIC
            )

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        self.client = httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=self.timeout
        )

        # Test connection (Anthropic doesn't have a simple health check endpoint)
        self._initialized = True
        logger.info("Anthropic provider initialized successfully")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Anthropic API"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Convert request to Anthropic format
        anthropic_request = self._convert_request(request)

        try:
            response = await self._make_request("/v1/messages", anthropic_request)

            # Parse response
            content = ""
            if "content" in response and response["content"]:
                # Anthropic returns content as a list of content blocks
                for block in response["content"]:
                    if block.get("type") == "text":
                        content += block.get("text", "")

            # Extract usage information
            usage = {}
            if "usage" in response:
                usage_data = response["usage"]
                usage = {
                    "prompt_tokens": usage_data.get("input_tokens", 0),
                    "completion_tokens": usage_data.get("output_tokens", 0),
                    "total_tokens": usage_data.get("input_tokens", 0)
                    + usage_data.get("output_tokens", 0),
                }

            return LLMResponse(
                content=content,
                model=response.get("model", request.model),
                provider=LLMProvider.ANTHROPIC,
                usage=usage,
                finish_reason=response.get("stop_reason", "").lower(),
                response_time=time.time() - start_time,
                request_id=response.get("id"),
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Anthropic generation failed: {str(e)}")
            raise self._handle_error(e)

    async def stream_generate(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream generate response using Anthropic API"""
        if not self._initialized:
            await self.initialize()

        # Convert request to Anthropic format with streaming
        anthropic_request = self._convert_request(request)
        anthropic_request["stream"] = True

        try:
            async with self.client.stream(
                "POST", "/v1/messages", json=anthropic_request
            ) as response:

                if response.status_code != 200:
                    error_text = await response.aread()
                    raise LLMError(f"Anthropic API error: {error_text}")

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix

                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)

                            if chunk.get("type") == "content_block_delta":
                                delta = chunk.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text = delta.get("text", "")
                                    if text:
                                        yield text

                        except (json.JSONDecodeError, KeyError):
                            continue

        except Exception as e:
            logger.error(f"Anthropic streaming failed: {str(e)}")
            raise self._handle_error(e)

    async def validate_connection(self) -> bool:
        """Validate connection to Anthropic API"""
        try:
            # Make a minimal request to test the connection
            test_request = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "Hi"}],
            }
            await self._make_request("/v1/messages", test_request)
            return True
        except Exception as e:
            logger.error(f"Anthropic connection validation failed: {str(e)}")
            return False

    async def get_available_models(self) -> List[str]:
        """Get available Anthropic models"""
        # Anthropic doesn't provide a models endpoint, return known models
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ]

    def _convert_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Convert LLMRequest to Anthropic API format"""

        # Separate system messages from other messages
        system_content = ""
        messages = []

        for msg in request.messages:
            if msg.role == MessageRole.SYSTEM:
                # Anthropic uses a separate system parameter
                system_content += msg.content + "\n"
            else:
                # Map roles (assistant stays as assistant, user stays as user)
                role = "user" if msg.role == MessageRole.USER else "assistant"
                messages.append({"role": role, "content": msg.content})

        anthropic_request = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens
            or 4096,  # Required parameter for Anthropic
            "temperature": request.temperature,
        }

        # Add system content if present
        if system_content.strip():
            anthropic_request["system"] = system_content.strip()

        # Add optional parameters
        if request.top_p is not None:
            anthropic_request["top_p"] = request.top_p
        if request.stop is not None:
            if isinstance(request.stop, str):
                anthropic_request["stop_sequences"] = [request.stop]
            else:
                anthropic_request["stop_sequences"] = request.stop

        # Handle Anthropic-specific parameters
        if "top_k" in request.extra_params:
            anthropic_request["top_k"] = request.extra_params["top_k"]
        if "metadata" in request.extra_params:
            anthropic_request["metadata"] = request.extra_params["metadata"]

        return anthropic_request

    async def _make_request(
        self, endpoint: str, data: Dict[str, Any], method: str = "POST"
    ) -> Dict[str, Any]:
        """Make HTTP request to Anthropic API with retry logic"""

        for attempt in range(self.max_retries + 1):
            try:
                if method == "GET":
                    response = await self.client.get(endpoint)
                else:
                    response = await self.client.post(endpoint, json=data)

                if response.status_code == 200:
                    return response.json()

                # Handle specific error codes
                if response.status_code == 429:
                    if attempt < self.max_retries:
                        wait_time = 2**attempt
                        logger.warning(
                            f"Rate limited, waiting {wait_time}s before retry"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    raise LLMRateLimitError(
                        "Rate limit exceeded", provider=LLMProvider.ANTHROPIC
                    )

                elif response.status_code == 401:
                    raise LLMAuthenticationError(
                        "Invalid API key", provider=LLMProvider.ANTHROPIC
                    )

                elif response.status_code == 403:
                    raise LLMQuotaExceededError(
                        "Quota exceeded", provider=LLMProvider.ANTHROPIC
                    )

                else:
                    error_text = response.text
                    raise LLMError(
                        f"Anthropic API error: {response.status_code} - {error_text}",
                        provider=LLMProvider.ANTHROPIC,
                    )

            except httpx.RequestError as e:
                if attempt < self.max_retries:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Connection error, retrying in {wait_time}s: {str(e)}"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise LLMConnectionError(
                    f"Connection failed: {str(e)}", provider=LLMProvider.ANTHROPIC
                )

        raise LLMError("Max retries exceeded", provider=LLMProvider.ANTHROPIC)

    def _handle_error(self, error: Exception) -> LLMError:
        """Convert exceptions to appropriate LLM errors"""
        if isinstance(error, LLMError):
            return error

        error_str = str(error)

        if "rate limit" in error_str.lower():
            return LLMRateLimitError(error_str, provider=LLMProvider.ANTHROPIC)
        elif (
            "unauthorized" in error_str.lower()
            or "invalid api key" in error_str.lower()
        ):
            return LLMAuthenticationError(error_str, provider=LLMProvider.ANTHROPIC)
        elif "quota" in error_str.lower():
            return LLMQuotaExceededError(error_str, provider=LLMProvider.ANTHROPIC)
        elif "connection" in error_str.lower():
            return LLMConnectionError(error_str, provider=LLMProvider.ANTHROPIC)
        else:
            return LLMError(error_str, provider=LLMProvider.ANTHROPIC)

    async def cleanup(self) -> None:
        """Cleanup Anthropic client"""
        if self.client:
            await self.client.aclose()
            self.client = None
        self._initialized = False
