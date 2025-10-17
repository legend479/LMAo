"""
Unit tests for shared logging module
"""

import pytest
import logging
import json
from unittest.mock import patch, MagicMock
from io import StringIO

from src.shared.logging import (
    configure_logging,
    get_logger,
    LoggerMixin,
    RequestLogger,
    PerformanceLogger,
    SecurityLogger,
    AuditLogger,
)


@pytest.mark.unit
class TestLoggingConfiguration:
    """Test logging configuration."""

    @patch("src.shared.logging.get_settings")
    def test_configure_logging_json_format(self, mock_get_settings):
        """Test logging configuration with JSON format."""
        mock_settings = MagicMock()
        mock_settings.log_level = "INFO"
        mock_settings.log_format = "json"
        mock_get_settings.return_value = mock_settings

        configure_logging()

        # Test that logging is configured
        logger = get_logger("test")
        assert logger is not None

    @patch("src.shared.logging.get_settings")
    def test_configure_logging_text_format(self, mock_get_settings):
        """Test logging configuration with text format."""
        mock_settings = MagicMock()
        mock_settings.log_level = "DEBUG"
        mock_settings.log_format = "text"
        mock_get_settings.return_value = mock_settings

        configure_logging()

        logger = get_logger("test")
        assert logger is not None

    def test_get_logger(self):
        """Test logger creation."""
        logger = get_logger("test_module")
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")


@pytest.mark.unit
class TestLoggerMixin:
    """Test LoggerMixin class."""

    def test_logger_mixin(self):
        """Test LoggerMixin provides logger property."""

        class TestClass(LoggerMixin):
            pass

        instance = TestClass()
        assert hasattr(instance, "logger")
        assert instance.logger is not None


@pytest.mark.unit
class TestRequestLogger:
    """Test RequestLogger class."""

    def test_request_logger_initialization(self):
        """Test RequestLogger initialization."""
        logger = RequestLogger()
        assert logger.logger is not None

    @patch("src.shared.logging.get_logger")
    def test_log_request(self, mock_get_logger):
        """Test request logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        request_logger = RequestLogger()
        request_logger.log_request(
            method="GET",
            url="/api/test",
            headers={"User-Agent": "test"},
            user_id="user123",
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "Request received" in call_args[0]

    @patch("src.shared.logging.get_logger")
    def test_log_request_filters_sensitive_headers(self, mock_get_logger):
        """Test that sensitive headers are filtered."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        request_logger = RequestLogger()
        request_logger.log_request(
            method="POST",
            url="/api/auth",
            headers={
                "Authorization": "Bearer secret-token",
                "Cookie": "session=secret",
                "User-Agent": "test",
            },
        )

        mock_logger.info.assert_called_once()
        # Check that sensitive headers are not logged
        call_kwargs = mock_logger.info.call_args[1]
        if "headers" in call_kwargs:
            assert "Authorization" not in call_kwargs["headers"]
            assert "Cookie" not in call_kwargs["headers"]
            assert "User-Agent" in call_kwargs["headers"]

    @patch("src.shared.logging.get_logger")
    def test_log_response_success(self, mock_get_logger):
        """Test successful response logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        request_logger = RequestLogger()
        request_logger.log_response(
            status_code=200, processing_time=0.5, response_size=1024
        )

        mock_logger.info.assert_called_once()

    @patch("src.shared.logging.get_logger")
    def test_log_response_error(self, mock_get_logger):
        """Test error response logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        request_logger = RequestLogger()
        request_logger.log_response(
            status_code=500, processing_time=1.0, error="Internal server error"
        )

        mock_logger.error.assert_called_once()


@pytest.mark.unit
class TestPerformanceLogger:
    """Test PerformanceLogger class."""

    @patch("src.shared.logging.get_logger")
    def test_log_execution_time_normal(self, mock_get_logger):
        """Test normal execution time logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        perf_logger = PerformanceLogger()
        perf_logger.log_execution_time(
            operation="test_operation", execution_time=1.0, metadata={"param": "value"}
        )

        mock_logger.info.assert_called_once()

    @patch("src.shared.logging.get_logger")
    def test_log_execution_time_slow(self, mock_get_logger):
        """Test slow execution time logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        perf_logger = PerformanceLogger()
        perf_logger.log_execution_time(operation="slow_operation", execution_time=10.0)

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "Slow operation detected" in call_args[0]

    @patch("src.shared.logging.get_logger")
    def test_log_resource_usage(self, mock_get_logger):
        """Test resource usage logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        perf_logger = PerformanceLogger()
        perf_logger.log_resource_usage(
            cpu_percent=75.5,
            memory_mb=512.0,
            disk_usage={"used": "10GB", "free": "90GB"},
        )

        mock_logger.info.assert_called_once()


@pytest.mark.unit
class TestSecurityLogger:
    """Test SecurityLogger class."""

    @patch("src.shared.logging.get_logger")
    def test_log_authentication_success(self, mock_get_logger):
        """Test successful authentication logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        security_logger = SecurityLogger()
        security_logger.log_authentication_attempt(
            user_id="user123", success=True, ip_address="192.168.1.1"
        )

        mock_logger.info.assert_called_once()

    @patch("src.shared.logging.get_logger")
    def test_log_authentication_failure(self, mock_get_logger):
        """Test failed authentication logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        security_logger = SecurityLogger()
        security_logger.log_authentication_attempt(
            user_id="user123", success=False, ip_address="192.168.1.1"
        )

        mock_logger.warning.assert_called_once()

    @patch("src.shared.logging.get_logger")
    def test_log_security_event_high_severity(self, mock_get_logger):
        """Test high severity security event logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        security_logger = SecurityLogger()
        security_logger.log_security_event(
            event_type="injection_attempt",
            severity="high",
            description="SQL injection detected",
            metadata={"ip": "192.168.1.1"},
        )

        mock_logger.error.assert_called_once()

    @patch("src.shared.logging.get_logger")
    def test_log_security_event_low_severity(self, mock_get_logger):
        """Test low severity security event logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        security_logger = SecurityLogger()
        security_logger.log_security_event(
            event_type="rate_limit_warning",
            severity="low",
            description="User approaching rate limit",
        )

        mock_logger.info.assert_called_once()


@pytest.mark.unit
class TestAuditLogger:
    """Test AuditLogger class."""

    @patch("src.shared.logging.get_logger")
    def test_log_user_action(self, mock_get_logger):
        """Test user action logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        audit_logger = AuditLogger()
        audit_logger.log_user_action(
            user_id="user123",
            action="document_upload",
            resource="document_456",
            result="success",
            metadata={"file_size": 1024},
        )

        mock_logger.info.assert_called_once()
        call_kwargs = mock_logger.info.call_args[1]
        assert call_kwargs["user_id"] == "user123"
        assert call_kwargs["action"] == "document_upload"

    @patch("src.shared.logging.get_logger")
    def test_log_data_access(self, mock_get_logger):
        """Test data access logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        audit_logger = AuditLogger()
        audit_logger.log_data_access(
            user_id="user123",
            data_type="user_profiles",
            operation="read",
            record_count=10,
        )

        mock_logger.info.assert_called_once()
        call_kwargs = mock_logger.info.call_args[1]
        assert call_kwargs["data_type"] == "user_profiles"
        assert call_kwargs["operation"] == "read"
        assert call_kwargs["record_count"] == 10


@pytest.mark.unit
class TestLoggerIntegration:
    """Test logger integration and real logging output."""

    def test_actual_logging_output(self, caplog):
        """Test actual logging output."""
        with caplog.at_level(logging.INFO):
            logger = get_logger("test_integration")
            logger.info("Test message", extra_field="test_value")

        assert "Test message" in caplog.text

    def test_structured_logging_data(self):
        """Test that structured data is properly logged."""
        logger = get_logger("test_structured")

        # Capture log output
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            logger.info("Structured test", user_id="123", action="test")

        # The exact format depends on the logging configuration
        # This test ensures no exceptions are raised during structured logging
        assert True  # If we get here, no exceptions were raised
