"""
Google AI (Gemini) Provider Implementation
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


class GoogleProvider(BaseLLMProvider):
    """Google AI (Gemini) API provider implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.base_url = config.get(
            "base_url", "https://generativelanguage.googleapis.com/v1beta"
        )
        self.timeout = config.get("timeout", 60)
        self.max_retries = config.get("max_retries", 3)
        self.client: Optional[httpx.AsyncClient] = None

    def get_provider_name(self) -> LLMProvider:
        return LLMProvider.GOOGLE

    async def initialize(self) -> None:
        """Initialize Google AI client"""
        if not self.api_key:
            raise LLMError("Google AI API key is required", provider=LLMProvider.GOOGLE)

        self.client = httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout, params={"key": self.api_key}
        )

        # Test connection
        await self.validate_connection()
        self._initialized = True
        logger.info("Google AI provider initialized successfully")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Google AI API"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Convert request to Google AI format
        google_request = self._convert_request(request)

        try:
            endpoint = f"/models/{request.model}:generateContent"
            response = await self._make_request(endpoint, google_request)

            # Parse response
            if "candidates" not in response or not response["candidates"]:
                raise LLMError("No candidates in response", provider=LLMProvider.GOOGLE)

            candidate = response["candidates"][0]
            content = ""

            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                content = "".join(part.get("text", "") for part in parts)

            # Extract usage information
            usage = {}
            if "usageMetadata" in response:
                usage_meta = response["usageMetadata"]
                usage = {
                    "prompt_tokens": usage_meta.get("promptTokenCount", 0),
                    "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
                    "total_tokens": usage_meta.get("totalTokenCount", 0),
                }

            return LLMResponse(
                content=content,
                model=request.model,
                provider=LLMProvider.GOOGLE,
                usage=usage,
                finish_reason=candidate.get("finishReason", "").lower(),
                response_time=time.time() - start_time,
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Google AI generation failed: {str(e)}")
            raise self._handle_error(e)

    async def stream_generate(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream generate response using Google AI API"""
        if not self._initialized:
            await self.initialize()

        # Convert request to Google AI format
        google_request = self._convert_request(request)

        try:
            endpoint = f"/models/{request.model}:streamGenerateContent"

            async with self.client.stream(
                "POST", endpoint, json=google_request
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise LLMError(f"Google AI API error: {error_text}")

                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            # Google AI streaming returns JSON objects
                            chunk = json.loads(line)

                            if "candidates" in chunk and chunk["candidates"]:
                                candidate = chunk["candidates"][0]

                                if (
                                    "content" in candidate
                                    and "parts" in candidate["content"]
                                ):
                                    parts = candidate["content"]["parts"]
                                    for part in parts:
                                        if "text" in part:
                                            yield part["text"]

                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

        except Exception as e:
            logger.error(f"Google AI streaming failed: {str(e)}")
            raise self._handle_error(e)

    async def validate_connection(self) -> bool:
        """Validate connection to Google AI API"""
        try:
            response = await self._make_request("/models", {}, method="GET")
            return "models" in response
        except Exception as e:
            logger.error(f"Google AI connection validation failed: {str(e)}")
            return False

    async def get_available_models(self) -> List[str]:
        """Get available Google AI models"""
        try:
            response = await self._make_request("/models", {}, method="GET")
            models = []

            for model in response.get("models", []):
                model_name = model.get("name", "")
                if model_name.startswith("models/"):
                    model_id = model_name[7:]  # Remove "models/" prefix
                    # Filter to generative models
                    if "gemini" in model_id.lower() or "generate" in model.get(
                        "supportedGenerationMethods", []
                    ):
                        models.append(model_id)

            return sorted(models)
        except Exception as e:
            logger.error(f"Failed to get Google AI models: {str(e)}")
            return [
                "gemini-2.5-flash ",
                "gemini-2.5-flash -vision",
            ]  # Fallback to known models

    def _convert_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Convert LLMRequest to Google AI API format"""

        # Convert messages to Google AI format
        contents = []
        system_instruction = None

        for msg in request.messages:
            if msg.role == MessageRole.SYSTEM:
                # Google AI uses systemInstruction separately
                system_instruction = {"parts": [{"text": msg.content}]}
            else:
                # Map roles
                role = "user" if msg.role == MessageRole.USER else "model"
                contents.append({"role": role, "parts": [{"text": msg.content}]})

        google_request = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
            },
        }

        # Add system instruction if present
        if system_instruction:
            google_request["systemInstruction"] = system_instruction

        # Add optional parameters to generationConfig
        generation_config = google_request["generationConfig"]

        if request.max_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_tokens
        if request.top_p is not None:
            generation_config["topP"] = request.top_p
        if request.stop is not None:
            if isinstance(request.stop, str):
                generation_config["stopSequences"] = [request.stop]
            else:
                generation_config["stopSequences"] = request.stop

        # Handle Google AI specific parameters
        if "top_k" in request.extra_params:
            generation_config["topK"] = request.extra_params["top_k"]
        if "candidate_count" in request.extra_params:
            generation_config["candidateCount"] = request.extra_params[
                "candidate_count"
            ]

        # Safety settings (Google AI specific)
        if "safety_settings" in request.extra_params:
            google_request["safetySettings"] = request.extra_params["safety_settings"]

        return google_request

    async def _make_request(
        self, endpoint: str, data: Dict[str, Any], method: str = "POST"
    ) -> Dict[str, Any]:
        """Make HTTP request to Google AI API with retry logic"""

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
                        "Rate limit exceeded", provider=LLMProvider.GOOGLE
                    )

                elif response.status_code == 401:
                    raise LLMAuthenticationError(
                        "Invalid API key", provider=LLMProvider.GOOGLE
                    )

                elif response.status_code == 403:
                    raise LLMQuotaExceededError(
                        "Quota exceeded", provider=LLMProvider.GOOGLE
                    )

                else:
                    error_text = response.text
                    raise LLMError(
                        f"Google AI API error: {response.status_code} - {error_text}",
                        provider=LLMProvider.GOOGLE,
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
                    f"Connection failed: {str(e)}", provider=LLMProvider.GOOGLE
                )

        raise LLMError("Max retries exceeded", provider=LLMProvider.GOOGLE)

    def _handle_error(self, error: Exception) -> LLMError:
        """Convert exceptions to appropriate LLM errors"""
        if isinstance(error, LLMError):
            return error

        error_str = str(error)

        if "rate limit" in error_str.lower() or "quota" in error_str.lower():
            return LLMRateLimitError(error_str, provider=LLMProvider.GOOGLE)
        elif (
            "unauthorized" in error_str.lower()
            or "invalid api key" in error_str.lower()
        ):
            return LLMAuthenticationError(error_str, provider=LLMProvider.GOOGLE)
        elif "connection" in error_str.lower():
            return LLMConnectionError(error_str, provider=LLMProvider.GOOGLE)
        else:
            return LLMError(error_str, provider=LLMProvider.GOOGLE)

    async def cleanup(self) -> None:
        """Cleanup Google AI client"""
        if self.client:
            await self.client.aclose()
            self.client = None
        self._initialized = False
