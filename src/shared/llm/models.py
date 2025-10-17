"""
LLM Models and Data Structures
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


class LLMProvider(str, Enum):
    """Supported LLM providers"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    LOCAL = "local"  # Alias for Ollama


class MessageRole(str, Enum):
    """Message roles in chat conversations"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """Individual chat message"""

    role: MessageRole
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class LLMRequest(BaseModel):
    """Request to LLM provider"""

    messages: List[ChatMessage]
    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    user: Optional[str] = None

    # Provider-specific parameters
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Response from LLM provider"""

    content: str
    model: str
    provider: LLMProvider
    usage: Dict[str, int] = Field(default_factory=dict)
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Dict[str, Any]] = None

    # Metadata
    response_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None

    # Raw response for debugging
    raw_response: Optional[Dict[str, Any]] = None


class LLMError(Exception):
    """Base exception for LLM operations"""

    def __init__(
        self,
        message: str,
        provider: Optional[LLMProvider] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.error_code = error_code
        self.details = details or {}


class LLMRateLimitError(LLMError):
    """Rate limit exceeded error"""

    pass


class LLMAuthenticationError(LLMError):
    """Authentication error"""

    pass


class LLMQuotaExceededError(LLMError):
    """Quota exceeded error"""

    pass


class LLMModelNotFoundError(LLMError):
    """Model not found error"""

    pass


class LLMConnectionError(LLMError):
    """Connection error"""

    pass


class LLMValidationError(LLMError):
    """Request validation error"""

    pass


# Provider-specific model configurations
PROVIDER_MODELS = {
    LLMProvider.OPENAI: {
        "gpt-4": {
            "max_tokens": 8192,
            "supports_functions": True,
            "supports_vision": False,
        },
        "gpt-4-turbo": {
            "max_tokens": 128000,
            "supports_functions": True,
            "supports_vision": True,
        },
        "gpt-4o": {
            "max_tokens": 128000,
            "supports_functions": True,
            "supports_vision": True,
        },
        "gpt-3.5-turbo": {
            "max_tokens": 4096,
            "supports_functions": True,
            "supports_vision": False,
        },
        "gpt-3.5-turbo-16k": {
            "max_tokens": 16384,
            "supports_functions": True,
            "supports_vision": False,
        },
    },
    LLMProvider.ANTHROPIC: {
        "claude-3-opus-20240229": {
            "max_tokens": 200000,
            "supports_functions": True,
            "supports_vision": True,
        },
        "claude-3-sonnet-20240229": {
            "max_tokens": 200000,
            "supports_functions": True,
            "supports_vision": True,
        },
        "claude-3-haiku-20240307": {
            "max_tokens": 200000,
            "supports_functions": True,
            "supports_vision": True,
        },
        "claude-2.1": {
            "max_tokens": 200000,
            "supports_functions": False,
            "supports_vision": False,
        },
        "claude-2.0": {
            "max_tokens": 100000,
            "supports_functions": False,
            "supports_vision": False,
        },
    },
    LLMProvider.GOOGLE: {
        "gemini-pro": {
            "max_tokens": 32768,
            "supports_functions": True,
            "supports_vision": False,
        },
        "gemini-pro-vision": {
            "max_tokens": 16384,
            "supports_functions": True,
            "supports_vision": True,
        },
        "gemini-1.5-pro": {
            "max_tokens": 1048576,
            "supports_functions": True,
            "supports_vision": True,
        },
        "gemini-1.5-flash": {
            "max_tokens": 1048576,
            "supports_functions": True,
            "supports_vision": True,
        },
    },
    LLMProvider.OLLAMA: {
        # Common Ollama models - actual availability depends on local installation
        "llama2": {
            "max_tokens": 4096,
            "supports_functions": False,
            "supports_vision": False,
        },
        "llama2:13b": {
            "max_tokens": 4096,
            "supports_functions": False,
            "supports_vision": False,
        },
        "llama2:70b": {
            "max_tokens": 4096,
            "supports_functions": False,
            "supports_vision": False,
        },
        "codellama": {
            "max_tokens": 16384,
            "supports_functions": False,
            "supports_vision": False,
        },
        "mistral": {
            "max_tokens": 8192,
            "supports_functions": False,
            "supports_vision": False,
        },
        "mixtral": {
            "max_tokens": 32768,
            "supports_functions": False,
            "supports_vision": False,
        },
        "phi": {
            "max_tokens": 2048,
            "supports_functions": False,
            "supports_vision": False,
        },
        "neural-chat": {
            "max_tokens": 4096,
            "supports_functions": False,
            "supports_vision": False,
        },
        "starcode": {
            "max_tokens": 8192,
            "supports_functions": False,
            "supports_vision": False,
        },
    },
}


def get_model_info(provider: LLMProvider, model: str) -> Dict[str, Any]:
    """Get model information for a specific provider and model"""
    provider_models = PROVIDER_MODELS.get(provider, {})
    return provider_models.get(
        model,
        {"max_tokens": 4096, "supports_functions": False, "supports_vision": False},
    )


def validate_model_support(provider: LLMProvider, model: str) -> bool:
    """Check if a model is supported by a provider"""
    return model in PROVIDER_MODELS.get(provider, {})
