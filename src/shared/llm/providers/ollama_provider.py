"""
Ollama Provider Implementation
Local LLM deployment using Ollama API format
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
    LLMConnectionError,
    MessageRole,
)
from ...logging import get_logger

logger = get_logger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama API provider implementation for local LLM deployment"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.timeout = config.get("timeout", 300)  # Longer timeout for local models
        self.max_retries = config.get("max_retries", 2)
        self.client: Optional[httpx.AsyncClient] = None

    def get_provider_name(self) -> LLMProvider:
        return LLMProvider.OLLAMA

    async def initialize(self) -> None:
        """Initialize Ollama client"""
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

        # Test connection
        await self.validate_connection()
        self._initialized = True
        logger.info("Ollama provider initialized successfully")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Ollama API"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Convert request to Ollama format
        ollama_request = self._convert_request(request)

        try:
            response = await self._make_request("/api/chat", ollama_request)

            # Parse Ollama response
            content = response.get("message", {}).get("content", "")

            # Ollama doesn't provide detailed usage stats by default
            usage = {}
            if "prompt_eval_count" in response:
                usage["prompt_tokens"] = response["prompt_eval_count"]
            if "eval_count" in response:
                usage["completion_tokens"] = response["eval_count"]
                usage["total_tokens"] = (
                    usage.get("prompt_tokens", 0) + response["eval_count"]
                )

            return LLMResponse(
                content=content,
                model=response.get("model", request.model),
                provider=LLMProvider.OLLAMA,
                usage=usage,
                finish_reason="stop" if response.get("done", False) else "length",
                response_time=time.time() - start_time,
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Ollama generation failed: {str(e)}")
            raise self._handle_error(e)

    async def stream_generate(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream generate response using Ollama API"""
        if not self._initialized:
            await self.initialize()

        # Convert request to Ollama format with streaming
        ollama_request = self._convert_request(request)
        ollama_request["stream"] = True

        try:
            async with self.client.stream(
                "POST", "/api/chat", json=ollama_request
            ) as response:

                if response.status_code != 200:
                    error_text = await response.aread()
                    raise LLMError(f"Ollama API error: {error_text}")

                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)

                            if "message" in chunk and "content" in chunk["message"]:
                                content = chunk["message"]["content"]
                                if content:
                                    yield content

                            # Check if done
                            if chunk.get("done", False):
                                break

                        except (json.JSONDecodeError, KeyError):
                            continue

        except Exception as e:
            logger.error(f"Ollama streaming failed: {str(e)}")
            raise self._handle_error(e)

    async def validate_connection(self) -> bool:
        """Validate connection to Ollama API"""
        try:
            response = await self._make_request("/api/tags", {}, method="GET")
            return "models" in response
        except Exception as e:
            logger.error(f"Ollama connection validation failed: {str(e)}")
            return False

    async def get_available_models(self) -> List[str]:
        """Get available Ollama models"""
        try:
            response = await self._make_request("/api/tags", {}, method="GET")
            models = []

            for model in response.get("models", []):
                model_name = model.get("name", "")
                if model_name:
                    # Remove tag if present (e.g., "llama2:latest" -> "llama2")
                    base_name = model_name.split(":")[0]
                    if base_name not in models:
                        models.append(model_name)  # Keep full name with tag

            return sorted(models)
        except Exception as e:
            logger.error(f"Failed to get Ollama models: {str(e)}")
            return []

    async def pull_model(self, model_name: str) -> bool:
        """Pull a model to Ollama (download if not available)"""
        try:
            logger.info(f"Pulling Ollama model: {model_name}")

            pull_request = {"name": model_name}

            # Use streaming to monitor pull progress
            async with self.client.stream(
                "POST", "/api/pull", json=pull_request
            ) as response:

                if response.status_code != 200:
                    error_text = await response.aread()
                    raise LLMError(f"Failed to pull model: {error_text}")

                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            status = chunk.get("status", "")

                            if "error" in chunk:
                                raise LLMError(f"Model pull error: {chunk['error']}")

                            # Log progress
                            if status:
                                logger.info(f"Pull progress: {status}")

                            # Check if completed
                            if (
                                chunk.get("status") == "success"
                                or "successfully" in status.lower()
                            ):
                                logger.info(f"Model {model_name} pulled successfully")
                                return True

                        except (json.JSONDecodeError, KeyError):
                            continue

            return True

        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {str(e)}")
            return False

    async def delete_model(self, model_name: str) -> bool:
        """Delete a model from Ollama"""
        try:
            delete_request = {"name": model_name}
            await self._make_request("/api/delete", delete_request)
            logger.info(f"Model {model_name} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {str(e)}")
            return False

    def _convert_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Convert LLMRequest to Ollama API format"""

        # Convert messages to Ollama format
        messages = []
        for msg in request.messages:
            ollama_msg = {"role": msg.role.value, "content": msg.content}
            messages.append(ollama_msg)

        ollama_request = {
            "model": request.model,
            "messages": messages,
            "stream": False,  # Will be overridden for streaming
            "options": {
                "temperature": request.temperature,
            },
        }

        # Add optional parameters to options
        options = ollama_request["options"]

        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.stop is not None:
            if isinstance(request.stop, str):
                options["stop"] = [request.stop]
            else:
                options["stop"] = request.stop

        # Handle Ollama-specific parameters
        if "top_k" in request.extra_params:
            options["top_k"] = request.extra_params["top_k"]
        if "repeat_penalty" in request.extra_params:
            options["repeat_penalty"] = request.extra_params["repeat_penalty"]
        if "seed" in request.extra_params:
            options["seed"] = request.extra_params["seed"]
        if "num_ctx" in request.extra_params:
            options["num_ctx"] = request.extra_params["num_ctx"]
        if "num_gpu" in request.extra_params:
            options["num_gpu"] = request.extra_params["num_gpu"]
        if "num_thread" in request.extra_params:
            options["num_thread"] = request.extra_params["num_thread"]

        return ollama_request

    async def _make_request(
        self, endpoint: str, data: Dict[str, Any], method: str = "POST"
    ) -> Dict[str, Any]:
        """Make HTTP request to Ollama API with retry logic"""

        for attempt in range(self.max_retries + 1):
            try:
                if method == "GET":
                    response = await self.client.get(endpoint)
                else:
                    response = await self.client.post(endpoint, json=data)

                if response.status_code == 200:
                    return response.json()

                # Handle error responses
                error_text = response.text

                if response.status_code == 404:
                    if "model" in data and endpoint == "/api/chat":
                        model_name = data["model"]
                        raise LLMError(
                            f"Model '{model_name}' not found. Use pull_model() to download it first.",
                            provider=LLMProvider.OLLAMA,
                        )
                    else:
                        raise LLMError(
                            f"Endpoint not found: {endpoint}",
                            provider=LLMProvider.OLLAMA,
                        )

                raise LLMError(
                    f"Ollama API error: {response.status_code} - {error_text}",
                    provider=LLMProvider.OLLAMA,
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
                    f"Connection failed: {str(e)}", provider=LLMProvider.OLLAMA
                )

        raise LLMError("Max retries exceeded", provider=LLMProvider.OLLAMA)

    def _handle_error(self, error: Exception) -> LLMError:
        """Convert exceptions to appropriate LLM errors"""
        if isinstance(error, LLMError):
            return error

        error_str = str(error)

        if "connection" in error_str.lower() or "refused" in error_str.lower():
            return LLMConnectionError(
                f"Cannot connect to Ollama server at {self.base_url}. "
                f"Make sure Ollama is running: {error_str}",
                provider=LLMProvider.OLLAMA,
            )
        else:
            return LLMError(error_str, provider=LLMProvider.OLLAMA)

    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model"""
        try:
            show_request = {"name": model_name}
            response = await self._make_request("/api/show", show_request)
            return response
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {str(e)}")
            return None

    async def cleanup(self) -> None:
        """Cleanup Ollama client"""
        if self.client:
            await self.client.aclose()
            self.client = None
        self._initialized = False
