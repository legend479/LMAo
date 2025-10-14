"""
Configuration Validation System
Comprehensive validation for application configuration and environment setup
"""

from typing import List, Tuple
from dataclasses import dataclass
import os
import socket
import re
from pathlib import Path
from urllib.parse import urlparse

from .config import get_settings
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]


class ConfigValidator:
    """Validates application configuration"""

    def __init__(self):
        self.settings = get_settings()

    def validate_all(self) -> ValidationResult:
        """Perform comprehensive configuration validation"""

        errors = []
        warnings = []
        recommendations = []

        # Validate basic settings
        basic_errors, basic_warnings, basic_recs = self._validate_basic_settings()
        errors.extend(basic_errors)
        warnings.extend(basic_warnings)
        recommendations.extend(basic_recs)

        # Validate network settings
        net_errors, net_warnings, net_recs = self._validate_network_settings()
        errors.extend(net_errors)
        warnings.extend(net_warnings)
        recommendations.extend(net_recs)

        # Validate database settings
        db_errors, db_warnings, db_recs = self._validate_database_settings()
        errors.extend(db_errors)
        warnings.extend(db_warnings)
        recommendations.extend(db_recs)

        # Validate security settings
        sec_errors, sec_warnings, sec_recs = self._validate_security_settings()
        errors.extend(sec_errors)
        warnings.extend(sec_warnings)
        recommendations.extend(sec_recs)

        # Validate file system settings
        fs_errors, fs_warnings, fs_recs = self._validate_filesystem_settings()
        errors.extend(fs_errors)
        warnings.extend(fs_warnings)
        recommendations.extend(fs_recs)

        # Validate environment-specific settings
        env_errors, env_warnings, env_recs = self._validate_environment_settings()
        errors.extend(env_errors)
        warnings.extend(env_warnings)
        recommendations.extend(env_recs)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _validate_basic_settings(self) -> Tuple[List[str], List[str], List[str]]:
        """Validate basic application settings"""

        errors = []
        warnings = []
        recommendations = []

        # Check required fields
        if not self.settings.app_name:
            errors.append("APP_NAME is required")

        if not self.settings.version:
            errors.append("VERSION is required")

        # Check environment
        valid_environments = ["development", "staging", "production", "testing"]
        if self.settings.environment not in valid_environments:
            warnings.append(
                f"Environment '{self.settings.environment}' is not standard. Valid: {valid_environments}"
            )

        # Check debug mode in production
        if self.settings.environment == "production" and self.settings.debug:
            warnings.append("Debug mode is enabled in production environment")

        return errors, warnings, recommendations

    def _validate_network_settings(self) -> Tuple[List[str], List[str], List[str]]:
        """Validate network and port settings"""

        errors = []
        warnings = []
        recommendations = []

        # Validate ports
        ports = [
            ("API_PORT", self.settings.api_port),
            ("AGENT_PORT", self.settings.agent_port),
            ("RAG_PORT", self.settings.rag_port),
            ("METRICS_PORT", self.settings.metrics_port),
        ]

        used_ports = []
        for name, port in ports:
            # Check port range
            if not (1 <= port <= 65535):
                errors.append(f"{name} must be between 1 and 65535, got {port}")

            # Check for port conflicts
            if port in used_ports:
                errors.append(f"Port {port} is used by multiple services")
            used_ports.append(port)

            # Check if port is available
            if not self._is_port_available(port):
                warnings.append(f"Port {port} ({name}) may already be in use")

        # Validate hosts
        hosts = [
            ("API_HOST", self.settings.api_host),
            ("AGENT_HOST", self.settings.agent_host),
            ("RAG_HOST", self.settings.rag_host),
            ("ELASTICSEARCH_HOST", self.settings.elasticsearch_host),
        ]

        for name, host in hosts:
            if not self._is_valid_host(host):
                errors.append(f"{name} '{host}' is not a valid hostname or IP address")

        # Validate CORS origins
        for origin in self.settings.allowed_origins:
            if not self._is_valid_url(origin):
                warnings.append(f"CORS origin '{origin}' may not be a valid URL")

        return errors, warnings, recommendations

    def _validate_database_settings(self) -> Tuple[List[str], List[str], List[str]]:
        """Validate database connection settings"""

        errors = []
        warnings = []
        recommendations = []

        # Validate database URL
        if not self.settings.database_url:
            errors.append("DATABASE_URL is required")
        else:
            try:
                parsed = urlparse(self.settings.database_url)

                # Check scheme
                if parsed.scheme not in ["postgresql", "postgres", "sqlite"]:
                    warnings.append(
                        f"Database scheme '{parsed.scheme}' may not be supported"
                    )

                # Check for localhost in production
                if self.settings.environment == "production" and parsed.hostname in [
                    "localhost",
                    "127.0.0.1",
                ]:
                    warnings.append(
                        "Using localhost database in production environment"
                    )

                # Check for default credentials
                if parsed.username == "postgres" and parsed.password == "password":
                    warnings.append("Using default database credentials")

            except Exception as e:
                errors.append(f"Invalid DATABASE_URL format: {str(e)}")

        # Validate Redis URL
        if not self.settings.redis_url:
            errors.append("REDIS_URL is required")
        else:
            try:
                parsed = urlparse(self.settings.redis_url)

                if parsed.scheme != "redis":
                    warnings.append(
                        f"Redis scheme should be 'redis', got '{parsed.scheme}'"
                    )

            except Exception as e:
                errors.append(f"Invalid REDIS_URL format: {str(e)}")

        return errors, warnings, recommendations

    def _validate_security_settings(self) -> Tuple[List[str], List[str], List[str]]:
        """Validate security-related settings"""

        errors = []
        warnings = []
        recommendations = []

        # Check secret key
        if not self.settings.secret_key:
            errors.append("SECRET_KEY is required")
        elif self.settings.secret_key == "your-secret-key-change-in-production":
            if self.settings.environment == "production":
                errors.append("SECRET_KEY must be changed in production")
            else:
                warnings.append("Using default SECRET_KEY")
        elif len(self.settings.secret_key) < 32:
            warnings.append("SECRET_KEY should be at least 32 characters long")

        # Check token expiration
        if self.settings.access_token_expire_minutes < 5:
            warnings.append(
                "Very short token expiration time may cause usability issues"
            )
        elif self.settings.access_token_expire_minutes > 1440:  # 24 hours
            warnings.append("Long token expiration time may pose security risks")

        # Check rate limiting
        if self.settings.rate_limit_requests_per_minute < 1:
            errors.append("Rate limit must be at least 1 request per minute")
        elif self.settings.rate_limit_requests_per_minute > 10000:
            warnings.append("Very high rate limit may not provide effective protection")

        return errors, warnings, recommendations

    def _validate_filesystem_settings(self) -> Tuple[List[str], List[str], List[str]]:
        """Validate file system settings"""

        errors = []
        warnings = []
        recommendations = []

        # Check upload directory
        upload_path = Path(self.settings.upload_dir)

        if not upload_path.exists():
            try:
                upload_path.mkdir(parents=True, exist_ok=True)
                recommendations.append(f"Created upload directory: {upload_path}")
            except Exception as e:
                errors.append(
                    f"Cannot create upload directory '{upload_path}': {str(e)}"
                )

        if upload_path.exists() and not os.access(upload_path, os.W_OK):
            errors.append(f"Upload directory '{upload_path}' is not writable")

        # Check file size limits
        if self.settings.max_file_size < 1024:  # 1KB
            warnings.append("Very small maximum file size may limit functionality")
        elif self.settings.max_file_size > 100 * 1024 * 1024:  # 100MB
            warnings.append("Large maximum file size may impact performance")

        return errors, warnings, recommendations

    def _validate_environment_settings(self) -> Tuple[List[str], List[str], List[str]]:
        """Validate environment-specific settings"""

        errors = []
        warnings = []
        recommendations = []

        if self.settings.environment == "production":
            # Production-specific validations

            if self.settings.debug:
                errors.append("Debug mode must be disabled in production")

            if self.settings.log_level.upper() == "DEBUG":
                warnings.append("Debug logging in production may impact performance")

            if "localhost" in self.settings.database_url:
                warnings.append("Using localhost database in production")

            if not self.settings.enable_metrics:
                recommendations.append("Consider enabling metrics in production")

        elif self.settings.environment == "development":
            # Development-specific recommendations

            if not self.settings.debug:
                recommendations.append("Consider enabling debug mode in development")

            if self.settings.log_level.upper() not in ["DEBUG", "INFO"]:
                recommendations.append(
                    "Consider using DEBUG or INFO log level in development"
                )

        return errors, warnings, recommendations

    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(("localhost", port))
                return result != 0  # Port is available if connection fails
        except Exception:
            return False

    def _is_valid_host(self, host: str) -> bool:
        """Check if a host is valid"""
        if not host:
            return False

        # Check for valid IP address
        try:
            socket.inet_aton(host)
            return True
        except socket.error:
            pass

        # Check for valid hostname
        hostname_pattern = re.compile(r"^(?!-)[A-Z\d-]{1,63}(?<!-)$", re.IGNORECASE)

        if host == "localhost":
            return True

        if all(hostname_pattern.match(part) for part in host.split(".")):
            return True

        return False

    def _is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False


def validate_configuration() -> ValidationResult:
    """Validate application configuration"""
    validator = ConfigValidator()
    return validator.validate_all()


def print_validation_results(result: ValidationResult):
    """Print validation results in a formatted way"""

    if result.is_valid:
        logger.info("✅ Configuration validation passed")
    else:
        logger.error("❌ Configuration validation failed")

    if result.errors:
        logger.error("Configuration Errors:")
        for error in result.errors:
            logger.error(f"  • {error}")

    if result.warnings:
        logger.warning("Configuration Warnings:")
        for warning in result.warnings:
            logger.warning(f"  • {warning}")

    if result.recommendations:
        logger.info("Configuration Recommendations:")
        for rec in result.recommendations:
            logger.info(f"  • {rec}")


# Export commonly used functions
__all__ = [
    "ValidationResult",
    "ConfigValidator",
    "validate_configuration",
    "print_validation_results",
]
