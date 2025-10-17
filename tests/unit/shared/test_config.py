"""
Unit tests for shared configuration module
"""

import pytest
import os
from unittest.mock import patch, MagicMock

from src.shared.config import (
    Settings,
    DevelopmentSettings,
    ProductionSettings,
    TestingSettings,
    get_settings,
    validate_config,
    get_database_config,
    get_redis_config,
    get_elasticsearch_config,
    get_cors_config,
)


@pytest.mark.unit
class TestSettings:
    """Test Settings class and its variants."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()

        assert settings.app_name == "SE SME Agent"
        assert settings.version == "1.0.0"
        assert settings.debug is False
        assert settings.environment == "development"
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000

    def test_development_settings(self):
        """Test development-specific settings."""
        settings = DevelopmentSettings()

        assert settings.debug is True
        assert settings.log_level == "DEBUG"
        assert settings.environment == "development"

    def test_production_settings(self):
        """Test production-specific settings."""
        with pytest.raises(ValueError):
            # Should fail without required production settings
            ProductionSettings()

    def test_testing_settings(self):
        """Test testing-specific settings."""
        settings = TestingSettings()

        assert settings.debug is True
        assert settings.log_level == "DEBUG"
        assert settings.environment == "testing"
        assert "sqlite:///:memory:" in settings.database_url

    @patch.dict(os.environ, {"ENVIRONMENT": "development"})
    def test_get_settings_development(self):
        """Test get_settings returns development settings."""
        settings = get_settings()
        assert isinstance(settings, DevelopmentSettings)

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def test_get_settings_testing(self):
        """Test get_settings returns testing settings."""
        settings = get_settings()
        assert isinstance(settings, TestingSettings)


@pytest.mark.unit
class TestConfigValidation:
    """Test configuration validation."""

    @patch("src.shared.config.get_settings")
    def test_validate_config_success(self, mock_get_settings):
        """Test successful configuration validation."""
        mock_settings = MagicMock()
        mock_settings.environment = "development"
        mock_settings.secret_key = "test-secret-key"
        mock_settings.database_url = "postgresql://user:pass@localhost/db"
        mock_settings.upload_dir = "/tmp/test"
        mock_settings.api_port = 8000
        mock_settings.agent_port = 8001
        mock_get_settings.return_value = mock_settings

        with patch("os.path.exists", return_value=True):
            with patch("os.makedirs"):
                result = validate_config()
                assert result is True

    @patch("src.shared.config.get_settings")
    def test_validate_config_production_errors(self, mock_get_settings):
        """Test configuration validation errors in production."""
        mock_settings = MagicMock()
        mock_settings.environment = "production"
        mock_settings.secret_key = "your-secret-key-change-in-production"
        mock_settings.database_url = "postgresql://user:pass@localhost/db"
        mock_settings.upload_dir = "/tmp/test"
        mock_settings.api_port = 8000
        mock_settings.agent_port = 8000  # Same port - should cause error
        mock_get_settings.return_value = mock_settings

        with pytest.raises(ValueError) as exc_info:
            validate_config()
        assert "SECRET_KEY must be changed in production" in str(exc_info.value)
        assert "cannot use the same port" in str(exc_info.value)


@pytest.mark.unit
class TestConfigHelpers:
    """Test configuration helper functions."""

    @patch("src.shared.config.get_settings")
    def test_get_database_config(self, mock_get_settings):
        """Test database configuration generation."""
        mock_settings = MagicMock()
        mock_settings.database_url = "postgresql://user:pass@localhost/db"
        mock_settings.debug = True
        mock_get_settings.return_value = mock_settings

        config = get_database_config()

        assert config["url"] == "postgresql://user:pass@localhost/db"
        assert config["echo"] is True
        assert config["pool_size"] == 10

    @patch("src.shared.config.get_settings")
    def test_get_redis_config(self, mock_get_settings):
        """Test Redis configuration generation."""
        mock_settings = MagicMock()
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_get_settings.return_value = mock_settings

        config = get_redis_config()

        assert config["url"] == "redis://localhost:6379/0"
        assert config["decode_responses"] is True
        assert config["socket_timeout"] == 5

    @patch("src.shared.config.get_settings")
    def test_get_elasticsearch_config(self, mock_get_settings):
        """Test Elasticsearch configuration generation."""
        mock_settings = MagicMock()
        mock_settings.elasticsearch_host = "localhost"
        mock_settings.elasticsearch_port = 9200
        mock_get_settings.return_value = mock_settings

        config = get_elasticsearch_config()

        assert config["hosts"] == ["localhost:9200"]
        assert config["timeout"] == 30
        assert config["max_retries"] == 3

    @patch("src.shared.config.get_settings")
    def test_get_cors_config(self, mock_get_settings):
        """Test CORS configuration generation."""
        mock_settings = MagicMock()
        mock_settings.allowed_origins = ["http://localhost:3000"]
        mock_get_settings.return_value = mock_settings

        config = get_cors_config()

        assert config["allow_origins"] == ["http://localhost:3000"]
        assert config["allow_credentials"] is True
        assert config["allow_methods"] == ["*"]


@pytest.mark.unit
class TestEnvironmentVariables:
    """Test environment variable handling."""

    @patch.dict(
        os.environ,
        {
            "APP_NAME": "Test App",
            "VERSION": "2.0.0",
            "DEBUG": "true",
            "API_PORT": "9000",
        },
    )
    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        settings = Settings()

        assert settings.app_name == "Test App"
        assert settings.version == "2.0.0"
        assert settings.debug is True
        assert settings.api_port == 9000

    @patch.dict(os.environ, {"INVALID_BOOL": "not_a_boolean"})
    def test_invalid_boolean_environment_variable(self):
        """Test handling of invalid boolean environment variables."""
        # This should not raise an error, pydantic should handle it
        settings = Settings()
        # The invalid boolean should default to False
        assert hasattr(settings, "debug")

    @patch.dict(os.environ, {"API_PORT": "not_a_number"})
    def test_invalid_integer_environment_variable(self):
        """Test handling of invalid integer environment variables."""
        with pytest.raises(ValueError):
            Settings()


@pytest.mark.unit
class TestConfigCaching:
    """Test configuration caching behavior."""

    def test_get_settings_caching(self):
        """Test that get_settings caches the result."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            settings1 = get_settings()
            settings2 = get_settings()

            # Should return the same instance due to caching
            assert settings1 is settings2

    def test_settings_immutability(self):
        """Test that settings objects are effectively immutable."""
        settings = Settings()

        # Pydantic models are immutable by default
        with pytest.raises(ValueError):
            settings.app_name = "Modified Name"
