"""
Configuration Management
Centralized configuration using environment variables and config files
"""

from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from typing import List, Optional
import os
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Application
    app_name: str = Field(default="SE SME Agent", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)",
    )

    # API Server
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000"], description="CORS allowed origins"
    )

    # Agent Server
    agent_host: str = Field(default="localhost", description="Agent server host")
    agent_port: int = Field(default=8001, description="Agent server port")

    # RAG Pipeline
    rag_host: str = Field(default="localhost", description="RAG pipeline host")
    rag_port: int = Field(default=8002, description="RAG pipeline port")

    # Database
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/se_sme_agent",
        description="Database URL",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")

    # Elasticsearch
    elasticsearch_host: str = Field(
        default="localhost", description="Elasticsearch host"
    )
    elasticsearch_port: int = Field(default=9200, description="Elasticsearch port")
    elasticsearch_index: str = Field(
        default="se_sme_documents", description="Elasticsearch index name"
    )

    # Security
    secret_key: str = Field(
        default="your-secret-key-change-in-production", description="Secret key for JWT"
    )
    access_token_expire_minutes: int = Field(
        default=30, description="Access token expiration in minutes"
    )

    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(
        default=60, description="Rate limit requests per minute"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="text", description="Log format (json, text)")

    # File Storage
    upload_dir: str = Field(default="./uploads", description="Upload directory")
    max_file_size: int = Field(
        default=10 * 1024 * 1024, description="Maximum file size in bytes (10MB)"
    )

    # Embedding Models
    general_embedding_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="General purpose embedding model",
    )
    domain_embedding_model: str = Field(
        default="microsoft/graphcodebert-base",
        description="Domain-specific embedding model",
    )

    # LLM Configuration
    llm_provider: str = Field(
        default="openai",
        description="LLM provider (openai, anthropic, google, ollama, local)",
    )
    llm_model: str = Field(default="gpt-3.5-turbo", description="LLM model name")
    llm_api_key: Optional[str] = Field(default=None, description="LLM API key")
    llm_base_url: Optional[str] = Field(
        default=None, description="LLM base URL for local/custom models"
    )

    # Provider-specific configurations
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_organization: Optional[str] = Field(
        default=None, description="OpenAI organization ID"
    )
    openai_base_url: Optional[str] = Field(default=None, description="OpenAI base URL")

    anthropic_api_key: Optional[str] = Field(
        default=None, description="Anthropic API key"
    )
    anthropic_base_url: Optional[str] = Field(
        default=None, description="Anthropic base URL"
    )

    google_api_key: Optional[str] = Field(default=None, description="Google AI API key")
    google_base_url: Optional[str] = Field(
        default=None, description="Google AI base URL"
    )

    ollama_base_url: Optional[str] = Field(
        default="http://localhost:11434", description="Ollama server URL"
    )
    ollama_model: Optional[str] = Field(
        default="llama2", description="Default Ollama model"
    )

    # Tool Configuration
    enable_code_execution: bool = Field(
        default=False, description="Enable code execution tools"
    )
    enable_email_tools: bool = Field(
        default=False, description="Enable email automation tools"
    )

    # Performance
    max_concurrent_requests: int = Field(
        default=100, description="Maximum concurrent requests"
    )
    request_timeout: int = Field(default=300, description="Request timeout in seconds")

    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics server port")

    model_config = ConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )


class DevelopmentSettings(Settings):
    """Development environment settings"""

    debug: bool = True
    log_level: str = "DEBUG"
    environment: str = "development"


class ProductionSettings(Settings):
    """Production environment settings"""

    debug: bool = False
    log_level: str = "INFO"
    environment: str = "production"

    # Override defaults for production
    secret_key: str = Field(
        ..., description="Secret key for JWT (required in production)"
    )
    database_url: str = Field(..., description="Database URL (required in production)")


class TestingSettings(Settings):
    """Testing environment settings"""

    debug: bool = True
    log_level: str = "DEBUG"
    environment: str = "testing"

    # Use in-memory databases for testing
    database_url: str = "sqlite:///:memory:"
    redis_url: str = "redis://localhost:6379/1"  # Different Redis DB for testing


@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)"""

    environment = os.getenv("ENVIRONMENT", "development").lower()

    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Configuration validation
def validate_config():
    """Validate configuration settings"""
    settings = get_settings()

    errors = []

    # Check required production settings
    if settings.environment == "production":
        if settings.secret_key == "your-secret-key-change-in-production":
            errors.append("SECRET_KEY must be changed in production")

        if "localhost" in settings.database_url:
            errors.append("DATABASE_URL should not use localhost in production")

    # Check file paths
    if not os.path.exists(settings.upload_dir):
        try:
            os.makedirs(settings.upload_dir, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create upload directory: {e}")

    # Check port availability (basic check)
    if settings.api_port == settings.agent_port:
        errors.append("API and Agent servers cannot use the same port")

    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

    return True


# Environment-specific configurations
def get_database_config() -> dict:
    """Get database configuration"""
    settings = get_settings()

    return {
        "url": settings.database_url,
        "echo": settings.debug,
        "pool_size": 10,
        "max_overflow": 20,
        "pool_timeout": 30,
        "pool_recycle": 3600,
    }


def get_redis_config() -> dict:
    """Get Redis configuration"""
    settings = get_settings()

    return {
        "url": settings.redis_url,
        "decode_responses": True,
        "socket_timeout": 5,
        "socket_connect_timeout": 5,
        "retry_on_timeout": True,
    }


def get_elasticsearch_config() -> dict:
    """Get Elasticsearch configuration"""
    settings = get_settings()

    return {
        "hosts": [f"{settings.elasticsearch_host}:{settings.elasticsearch_port}"],
        "timeout": 30,
        "max_retries": 3,
        "retry_on_timeout": True,
    }


def get_cors_config() -> dict:
    """Get CORS configuration"""
    settings = get_settings()

    return {
        "allow_origins": settings.allowed_origins,
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }


# Export commonly used settings
__all__ = [
    "Settings",
    "get_settings",
    "validate_config",
    "get_database_config",
    "get_redis_config",
    "get_elasticsearch_config",
    "get_cors_config",
]
