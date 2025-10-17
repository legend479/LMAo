"""
OpenAI Provider Implementation
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


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.organization = config.get("organization")
        self.timeout = config.get("timeout", 60)
        self.max_retries = config.get("max_retries", 3)
        self.client: Optional[httpx.AsyncClient] = None

    def get_provider_name(self) -> LLMProvider:
        return LLMProvider.OPENAI

    async def initialize(self) -> None:
        """Initialize OpenAI client"""
        if not self.api_key:
            raise LLMError("OpenAI API key is required", provider=LLMProvider.OPENAI)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        self.client = httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=self.timeout
        )

        # Test connection
        await self.validate_connection()
        self._initialized = True
        logger.info("OpenAI provider initialized successfully")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenAI API"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Convert request to OpenAI format
        openai_request = self._convert_request(request)

        try:
            response = await self._make_request("/chat/completions", openai_request)

            # Parse response
            choice = response["choices"][0]
            content = choice["message"]["content"] or ""

            return LLMResponse(
                content=content,
                model=response["model"],
                provider=LLMProvider.OPENAI,
                usage=response.get("usage", {}),
                finish_reason=choice.get("finish_reason"),
                tool_calls=choice["message"].get("tool_calls"),
                function_call=choice["message"].get("function_call"),
                response_time=time.time() - start_time,
                request_id=response.get("id"),
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"OpenAI generation failed: {str(e)}")
            raise self._handle_error(e)

    async def stream_generate(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream generate response using OpenAI API"""
        if not self._initialized:
            await self.initialize()

        # Convert request to OpenAI format with streaming
        openai_request = self._convert_request(request)
        openai_request["stream"] = True

        try:
            async with self.client.stream(
                "POST", "/chat/completions", json=openai_request
            ) as response:

                if response.status_code != 200:
                    error_text = await response.aread()
                    raise LLMError(f"OpenAI API error: {error_text}")

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix

                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"]

                            if "content" in delta and delta["content"]:
                                yield delta["content"]

                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

        except Exception as e:
            logger.error(f"OpenAI streaming failed: {str(e)}")
            raise self._handle_error(e)

    async def validate_connection(self) -> bool:
        """Validate connection to OpenAI API"""
        try:
            response = await self._make_request("/models", {}, method="GET")
            return "data" in response
        except Exception as e:
            logger.error(f"OpenAI connection validation failed: {str(e)}")
            return False

    async def get_available_models(self) -> List[str]:
        """Get available OpenAI models"""
        try:
            response = await self._make_request("/models", {}, method="GET")
            models = [model["id"] for model in response.get("data", [])]
            # Filter to chat models only
            chat_models = [
                m
                for m in models
                if any(prefix in m for prefix in ["gpt-", "text-davinci"])
            ]
            return sorted(chat_models)
        except Exception as e:
            logger.error(f"Failed to get OpenAI models: {str(e)}")
            return []

    def _convert_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Convert LLMRequest to OpenAI API format"""
        openai_request = {
            "model": request.model,
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    **({"name": msg.name} if msg.name else {}),
                    **(
                        {"function_call": msg.function_call}
                        if msg.function_call
                        else {}
                    ),
                    **({"tool_calls": msg.tool_calls} if msg.tool_calls else {}),
                }
                for msg in request.messages
            ],
            "temperature": request.temperature,
        }

        # Add optional parameters
        if request.max_tokens is not None:
            openai_request["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            openai_request["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            openai_request["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            openai_request["presence_penalty"] = request.presence_penalty
        if request.stop is not None:
            openai_request["stop"] = request.stop
        if request.tools is not None:
            openai_request["tools"] = request.tools
        if request.tool_choice is not None:
            openai_request["tool_choice"] = request.tool_choice
        if request.response_format is not None:
            openai_request["response_format"] = request.response_format
        if request.seed is not None:
            openai_request["seed"] = request.seed
        if request.user is not None:
            openai_request["user"] = request.user

        # Add extra parameters
        openai_request.update(request.extra_params)

        return openai_request

    async def _make_request(
        self, endpoint: str, data: Dict[str, Any], method: str = "POST"
    ) -> Dict[str, Any]:
        """Make HTTP request to OpenAI API with retry logic"""

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
                        "Rate limit exceeded", provider=LLMProvider.OPENAI
                    )

                elif response.status_code == 401:
                    raise LLMAuthenticationError(
                        "Invalid API key", provider=LLMProvider.OPENAI
                    )

                elif response.status_code == 403:
                    raise LLMQuotaExceededError(
                        "Quota exceeded", provider=LLMProvider.OPENAI
                    )

                else:
                    error_text = response.text
                    raise LLMError(
                        f"OpenAI API error: {response.status_code} - {error_text}",
                        provider=LLMProvider.OPENAI,
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
                    f"Connection failed: {str(e)}", provider=LLMProvider.OPENAI
                )

        raise LLMError("Max retries exceeded", provider=LLMProvider.OPENAI)

    def _handle_error(self, error: Exception) -> LLMError:
        """Convert exceptions to appropriate LLM errors"""
        if isinstance(error, LLMError):
            return error

        error_str = str(error)

        if "rate limit" in error_str.lower():
            return LLMRateLimitError(error_str, provider=LLMProvider.OPENAI)
        elif (
            "unauthorized" in error_str.lower()
            or "invalid api key" in error_str.lower()
        ):
            return LLMAuthenticationError(error_str, provider=LLMProvider.OPENAI)
        elif "quota" in error_str.lower():
            return LLMQuotaExceededError(error_str, provider=LLMProvider.OPENAI)
        elif "connection" in error_str.lower():
            return LLMConnectionError(error_str, provider=LLMProvider.OPENAI)
        else:
            return LLMError(error_str, provider=LLMProvider.OPENAI)

    async def cleanup(self) -> None:
        """Cleanup OpenAI client"""
        if self.client:
            await self.client.aclose()
            self.client = None
        self._initialized = False
