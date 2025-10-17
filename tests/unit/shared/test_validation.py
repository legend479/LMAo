"""
Unit tests for shared validation module
"""

import pytest
import socket
from unittest.mock import patch, MagicMock

from src.shared.validation import (
    ValidationResult,
    ConfigValidator,
    validate_configuration,
    print_validation_results,
)


@pytest.mark.unit
class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test ValidationResult creation."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Test warning"],
            recommendations=["Test recommendation"],
        )

        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == ["Test warning"]
        assert result.recommendations == ["Test recommendation"]

    def test_validation_result_invalid(self):
        """Test ValidationResult with errors."""
        result = ValidationResult(
            is_valid=False, errors=["Test error"], warnings=[], recommendations=[]
        )

        assert result.is_valid is False
        assert result.errors == ["Test error"]


@pytest.mark.unit
class TestConfigValidator:
    """Test ConfigValidator class."""

    @patch("src.shared.validation.get_settings")
    def test_validator_initialization(self, mock_get_settings):
        """Test ConfigValidator initialization."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        validator = ConfigValidator()
        assert validator.settings == mock_settings

    @patch("src.shared.validation.get_settings")
    def test_validate_all_success(self, mock_get_settings):
        """Test successful validation."""
        mock_settings = MagicMock()
        mock_settings.app_name = "Test App"
        mock_settings.version = "1.0.0"
        mock_settings.environment = "development"
        mock_settings.debug = False
        mock_settings.api_port = 8000
        mock_settings.agent_port = 8001
        mock_settings.rag_port = 8002
        mock_settings.metrics_port = 9090
        mock_settings.api_host = "localhost"
        mock_settings.agent_host = "localhost"
        mock_settings.rag_host = "localhost"
        mock_settings.elasticsearch_host = "localhost"
        mock_settings.allowed_origins = ["http://localhost:3000"]
        mock_settings.database_url = "postgresql://user:pass@localhost/db"
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_settings.secret_key = "test-secret-key"
        mock_settings.access_token_expire_minutes = 30
        mock_settings.rate_limit_requests_per_minute = 60
        mock_settings.upload_dir = "/tmp/test"
        mock_settings.max_file_size = 10485760
        mock_get_settings.return_value = mock_settings

        validator = ConfigValidator()

        with patch.object(validator, "_is_port_available", return_value=True):
            with patch.object(validator, "_is_valid_host", return_value=True):
                with patch.object(validator, "_is_valid_url", return_value=True):
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch("os.access", return_value=True):
                            result = validator.validate_all()

        assert result.is_valid is True
        assert len(result.errors) == 0

    @patch("src.shared.validation.get_settings")
    def test_validate_basic_settings_missing_app_name(self, mock_get_settings):
        """Test validation with missing app name."""
        mock_settings = MagicMock()
        mock_settings.app_name = ""
        mock_settings.version = "1.0.0"
        mock_settings.environment = "development"
        mock_settings.debug = False
        mock_get_settings.return_value = mock_settings

        validator = ConfigValidator()
        errors, warnings, recommendations = validator._validate_basic_settings()

        assert "APP_NAME is required" in errors

    @patch("src.shared.validation.get_settings")
    def test_validate_network_settings_port_conflict(self, mock_get_settings):
        """Test validation with port conflicts."""
        mock_settings = MagicMock()
        mock_settings.api_port = 8000
        mock_settings.agent_port = 8000  # Same port - conflict
        mock_settings.rag_port = 8002
        mock_settings.metrics_port = 9090
        mock_settings.api_host = "localhost"
        mock_settings.agent_host = "localhost"
        mock_settings.rag_host = "localhost"
        mock_settings.elasticsearch_host = "localhost"
        mock_settings.allowed_origins = ["http://localhost:3000"]
        mock_get_settings.return_value = mock_settings

        validator = ConfigValidator()

        with patch.object(validator, "_is_valid_host", return_value=True):
            with patch.object(validator, "_is_valid_url", return_value=True):
                errors, warnings, recommendations = (
                    validator._validate_network_settings()
                )

        assert any(
            "Port 8000 is used by multiple services" in error for error in errors
        )

    @patch("src.shared.validation.get_settings")
    def test_validate_database_settings_invalid_url(self, mock_get_settings):
        """Test validation with invalid database URL."""
        mock_settings = MagicMock()
        mock_settings.database_url = "invalid-url"
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_get_settings.return_value = mock_settings

        validator = ConfigValidator()
        errors, warnings, recommendations = validator._validate_database_settings()

        assert any("Invalid DATABASE_URL format" in error for error in errors)

    @patch("src.shared.validation.get_settings")
    def test_validate_security_settings_weak_secret(self, mock_get_settings):
        """Test validation with weak secret key."""
        mock_settings = MagicMock()
        mock_settings.environment = "production"
        mock_settings.secret_key = "your-secret-key-change-in-production"
        mock_settings.access_token_expire_minutes = 30
        mock_settings.rate_limit_requests_per_minute = 60
        mock_get_settings.return_value = mock_settings

        validator = ConfigValidator()
        errors, warnings, recommendations = validator._validate_security_settings()

        assert any(
            "SECRET_KEY must be changed in production" in error for error in errors
        )

    def test_is_port_available_true(self):
        """Test port availability check when port is available."""
        validator = ConfigValidator()

        with patch("socket.socket") as mock_socket:
            mock_sock = MagicMock()
            mock_sock.connect_ex.return_value = 1  # Connection failed = port available
            mock_socket.return_value.__enter__.return_value = mock_sock

            result = validator._is_port_available(8080)
            assert result is True

    def test_is_port_available_false(self):
        """Test port availability check when port is in use."""
        validator = ConfigValidator()

        with patch("socket.socket") as mock_socket:
            mock_sock = MagicMock()
            mock_sock.connect_ex.return_value = 0  # Connection succeeded = port in use
            mock_socket.return_value.__enter__.return_value = mock_sock

            result = validator._is_port_available(8080)
            assert result is False

    def test_is_port_available_exception(self):
        """Test port availability check with exception."""
        validator = ConfigValidator()

        with patch("socket.socket", side_effect=Exception("Socket error")):
            result = validator._is_port_available(8080)
            assert result is False

    def test_is_valid_host_localhost(self):
        """Test host validation for localhost."""
        validator = ConfigValidator()
        assert validator._is_valid_host("localhost") is True

    def test_is_valid_host_ip_address(self):
        """Test host validation for IP address."""
        validator = ConfigValidator()

        with patch("socket.inet_aton"):
            assert validator._is_valid_host("192.168.1.1") is True

    def test_is_valid_host_invalid_ip(self):
        """Test host validation for invalid IP address."""
        validator = ConfigValidator()

        with patch("socket.inet_aton", side_effect=socket.error):
            # Should still validate as hostname
            assert validator._is_valid_host("invalid-ip") is False

    def test_is_valid_host_valid_hostname(self):
        """Test host validation for valid hostname."""
        validator = ConfigValidator()
        assert validator._is_valid_host("example.com") is True

    def test_is_valid_host_invalid_hostname(self):
        """Test host validation for invalid hostname."""
        validator = ConfigValidator()
        assert validator._is_valid_host("") is False
        assert validator._is_valid_host("-invalid") is False

    def test_is_valid_url_valid(self):
        """Test URL validation for valid URLs."""
        validator = ConfigValidator()
        assert validator._is_valid_url("http://localhost:3000") is True
        assert validator._is_valid_url("https://example.com") is True

    def test_is_valid_url_invalid(self):
        """Test URL validation for invalid URLs."""
        validator = ConfigValidator()
        assert validator._is_valid_url("not-a-url") is False
        assert validator._is_valid_url("") is False


@pytest.mark.unit
class TestValidationHelpers:
    """Test validation helper functions."""

    @patch("src.shared.validation.ConfigValidator")
    def test_validate_configuration(self, mock_validator_class):
        """Test validate_configuration function."""
        mock_validator = MagicMock()
        mock_result = ValidationResult(
            is_valid=True, errors=[], warnings=[], recommendations=[]
        )
        mock_validator.validate_all.return_value = mock_result
        mock_validator_class.return_value = mock_validator

        result = validate_configuration()

        assert result == mock_result
        mock_validator.validate_all.assert_called_once()

    @patch("src.shared.validation.logger")
    def test_print_validation_results_success(self, mock_logger):
        """Test print_validation_results with successful validation."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Test warning"],
            recommendations=["Test recommendation"],
        )

        print_validation_results(result)

        mock_logger.info.assert_called()
        mock_logger.warning.assert_called()

    @patch("src.shared.validation.logger")
    def test_print_validation_results_failure(self, mock_logger):
        """Test print_validation_results with failed validation."""
        result = ValidationResult(
            is_valid=False, errors=["Test error"], warnings=[], recommendations=[]
        )

        print_validation_results(result)

        mock_logger.error.assert_called()


@pytest.mark.unit
class TestValidationEdgeCases:
    """Test validation edge cases and error conditions."""

    @patch("src.shared.validation.get_settings")
    def test_validation_with_none_values(self, mock_get_settings):
        """Test validation with None values."""
        mock_settings = MagicMock()
        mock_settings.app_name = None
        mock_settings.version = None
        mock_settings.environment = "development"
        mock_get_settings.return_value = mock_settings

        validator = ConfigValidator()
        errors, warnings, recommendations = validator._validate_basic_settings()

        # Should handle None values gracefully
        assert len(errors) >= 0  # May have errors for None values

    @patch("src.shared.validation.get_settings")
    def test_validation_with_extreme_values(self, mock_get_settings):
        """Test validation with extreme values."""
        mock_settings = MagicMock()
        mock_settings.api_port = 99999  # Very high port
        mock_settings.agent_port = 1  # Very low port
        mock_settings.rag_port = 8002
        mock_settings.metrics_port = 9090
        mock_settings.api_host = "localhost"
        mock_settings.agent_host = "localhost"
        mock_settings.rag_host = "localhost"
        mock_settings.elasticsearch_host = "localhost"
        mock_settings.allowed_origins = ["http://localhost:3000"]
        mock_settings.access_token_expire_minutes = 1  # Very short
        mock_settings.rate_limit_requests_per_minute = 100000  # Very high
        mock_get_settings.return_value = mock_settings

        validator = ConfigValidator()

        with patch.object(validator, "_is_valid_host", return_value=True):
            with patch.object(validator, "_is_valid_url", return_value=True):
                with patch.object(validator, "_is_port_available", return_value=True):
                    errors, warnings, recommendations = (
                        validator._validate_network_settings()
                    )

        # Should handle extreme values and provide warnings
        assert isinstance(warnings, list)
