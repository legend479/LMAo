"""
Validation Middleware
Comprehensive request validation and sanitization
"""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import json
import re
from typing import Any
import uuid

from src.shared.logging import get_logger

logger = get_logger(__name__)


class ValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request validation"""

    # Maximum request sizes (in bytes)
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_JSON_DEPTH = 10
    MAX_ARRAY_LENGTH = 1000
    MAX_STRING_LENGTH = 100000

    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        # SQL Injection patterns
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(\b(UNION|OR|AND)\s+\d+\s*=\s*\d+)",
        r"(\'|\")(\s*;\s*)",
        # NoSQL Injection patterns
        r"(\$where|\$ne|\$gt|\$lt|\$regex)",
        # Command Injection patterns
        r"(;|\||&|`|\$\(|\${)",
        r"(\b(rm|cat|ls|ps|kill|wget|curl|nc|netcat)\b)",
        # Path Traversal patterns
        r"(\.\.\/|\.\.\\)",
        r"(%2e%2e%2f|%2e%2e%5c)",
        # XSS patterns (additional to security middleware)
        r"(<script|<iframe|<object|<embed)",
        r"(javascript:|data:|vbscript:)",
        r"(on\w+\s*=)",
        # LDAP Injection patterns
        r"(\*|\(|\)|&|\|)",
        # XML/XXE patterns
        r"(<!ENTITY|<!DOCTYPE)",
        r"(SYSTEM|PUBLIC)",
        # Template Injection patterns
        r"(\{\{|\}\}|\{%|%\})",
        r"(__.*__)",
    ]

    def __init__(self, app):
        super().__init__(app)
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.DANGEROUS_PATTERNS
        ]

    async def dispatch(self, request: Request, call_next):
        """Process request through comprehensive validation"""

        # Generate unique request ID for tracking
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        try:
            # Validate request size
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.MAX_REQUEST_SIZE:
                logger.warning(
                    "Request too large",
                    request_id=request_id,
                    size=content_length,
                    max_size=self.MAX_REQUEST_SIZE,
                )
                raise HTTPException(status_code=413, detail="Request entity too large")

            # Validate headers
            self._validate_headers(request, request_id)

            # Validate query parameters
            self._validate_query_params(request, request_id)

            # Validate request body if present
            if request.method in ["POST", "PUT", "PATCH"]:
                await self._validate_request_body(request, request_id)

            # Process request
            response = await call_next(request)

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "Validation middleware error",
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise HTTPException(status_code=500, detail="Internal validation error")

    def _validate_headers(self, request: Request, request_id: str):
        """Validate request headers"""

        # Check for suspicious headers
        suspicious_headers = ["x-forwarded-for", "x-real-ip", "x-originating-ip"]

        for header in suspicious_headers:
            value = request.headers.get(header)
            if value and self._contains_dangerous_content(value):
                logger.warning(
                    "Suspicious header detected",
                    request_id=request_id,
                    header=header,
                    value=value[:100],
                )  # Log only first 100 chars
                raise HTTPException(status_code=400, detail="Invalid header content")

        # Validate Content-Type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if content_type and not self._is_valid_content_type(content_type):
                logger.warning(
                    "Invalid content type",
                    request_id=request_id,
                    content_type=content_type,
                )
                raise HTTPException(status_code=415, detail="Unsupported media type")

    def _validate_query_params(self, request: Request, request_id: str):
        """Validate query parameters"""

        for key, value in request.query_params.items():
            # Check parameter name
            if not self._is_valid_param_name(key):
                logger.warning(
                    "Invalid parameter name", request_id=request_id, param_name=key
                )
                raise HTTPException(
                    status_code=400, detail=f"Invalid parameter name: {key}"
                )

            # Check parameter value
            if self._contains_dangerous_content(value):
                logger.warning(
                    "Dangerous content in query parameter",
                    request_id=request_id,
                    param_name=key,
                    param_value=value[:100],
                )
                raise HTTPException(status_code=400, detail="Invalid parameter content")

            # Check parameter length
            if len(value) > self.MAX_STRING_LENGTH:
                logger.warning(
                    "Parameter value too long",
                    request_id=request_id,
                    param_name=key,
                    length=len(value),
                )
                raise HTTPException(status_code=400, detail="Parameter value too long")

    async def _validate_request_body(self, request: Request, request_id: str):
        """Validate request body content"""

        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            await self._validate_json_body(request, request_id)
        elif "multipart/form-data" in content_type:
            await self._validate_form_data(request, request_id)
        elif "application/x-www-form-urlencoded" in content_type:
            await self._validate_form_body(request, request_id)

    async def _validate_json_body(self, request: Request, request_id: str):
        """Validate JSON request body"""

        try:
            # Read body
            body = await request.body()

            if not body:
                return

            # Parse JSON
            try:
                json_data = json.loads(body)
            except json.JSONDecodeError as e:
                logger.warning(
                    "Invalid JSON in request body", request_id=request_id, error=str(e)
                )
                raise HTTPException(status_code=400, detail="Invalid JSON format")

            # Validate JSON structure
            self._validate_json_structure(json_data, request_id)

            # Validate JSON content
            self._validate_json_content(json_data, request_id)

        except HTTPException:
            raise
        except Exception as e:
            logger.error("JSON validation error", request_id=request_id, error=str(e))
            raise HTTPException(status_code=400, detail="JSON validation failed")

    async def _validate_form_data(self, request: Request, request_id: str):
        """Validate multipart form data"""

        try:
            form = await request.form()

            for key, value in form.items():
                # Validate field name
                if not self._is_valid_param_name(key):
                    logger.warning(
                        "Invalid form field name", request_id=request_id, field_name=key
                    )
                    raise HTTPException(
                        status_code=400, detail=f"Invalid form field name: {key}"
                    )

                # Validate field value (if it's a string)
                if isinstance(value, str):
                    if self._contains_dangerous_content(value):
                        logger.warning(
                            "Dangerous content in form field",
                            request_id=request_id,
                            field_name=key,
                        )
                        raise HTTPException(
                            status_code=400, detail="Invalid form field content"
                        )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "Form data validation error", request_id=request_id, error=str(e)
            )
            raise HTTPException(status_code=400, detail="Form data validation failed")

    async def _validate_form_body(self, request: Request, request_id: str):
        """Validate URL-encoded form body"""

        try:
            form = await request.form()

            for key, value in form.items():
                if self._contains_dangerous_content(str(value)):
                    logger.warning(
                        "Dangerous content in form body",
                        request_id=request_id,
                        field_name=key,
                    )
                    raise HTTPException(status_code=400, detail="Invalid form content")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "Form body validation error", request_id=request_id, error=str(e)
            )
            raise HTTPException(status_code=400, detail="Form body validation failed")

    def _validate_json_structure(self, data: Any, request_id: str, depth: int = 0):
        """Validate JSON structure (depth, array lengths, etc.)"""

        if depth > self.MAX_JSON_DEPTH:
            logger.warning(
                "JSON depth exceeded",
                request_id=request_id,
                depth=depth,
                max_depth=self.MAX_JSON_DEPTH,
            )
            raise HTTPException(status_code=400, detail="JSON structure too deep")

        if isinstance(data, dict):
            if len(data) > self.MAX_ARRAY_LENGTH:
                logger.warning(
                    "JSON object too large", request_id=request_id, size=len(data)
                )
                raise HTTPException(status_code=400, detail="JSON object too large")

            for key, value in data.items():
                if not isinstance(key, str) or len(key) > 1000:
                    raise HTTPException(status_code=400, detail="Invalid JSON key")
                self._validate_json_structure(value, request_id, depth + 1)

        elif isinstance(data, list):
            if len(data) > self.MAX_ARRAY_LENGTH:
                logger.warning(
                    "JSON array too large", request_id=request_id, size=len(data)
                )
                raise HTTPException(status_code=400, detail="JSON array too large")

            for item in data:
                self._validate_json_structure(item, request_id, depth + 1)

    def _validate_json_content(self, data: Any, request_id: str):
        """Validate JSON content for dangerous patterns"""

        if isinstance(data, str):
            if len(data) > self.MAX_STRING_LENGTH:
                logger.warning(
                    "JSON string too long", request_id=request_id, length=len(data)
                )
                raise HTTPException(status_code=400, detail="JSON string too long")

            if self._contains_dangerous_content(data):
                logger.warning(
                    "Dangerous content in JSON string",
                    request_id=request_id,
                    content=data[:100],
                )
                raise HTTPException(status_code=400, detail="Invalid JSON content")

        elif isinstance(data, dict):
            for key, value in data.items():
                self._validate_json_content(value, request_id)

        elif isinstance(data, list):
            for item in data:
                self._validate_json_content(item, request_id)

    def _contains_dangerous_content(self, text: str) -> bool:
        """Check if text contains dangerous patterns"""

        if not isinstance(text, str):
            return False

        # Check against compiled patterns
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return True

        # Check for excessive special characters
        special_char_count = sum(
            1 for c in text if not c.isalnum() and c not in " .-_@"
        )
        if len(text) > 0 and special_char_count / len(text) > 0.5:
            return True

        return False

    def _is_valid_param_name(self, name: str) -> bool:
        """Check if parameter name is valid"""

        if not name or len(name) > 100:
            return False

        # Allow alphanumeric, underscore, dash, dot
        if not re.match(r"^[a-zA-Z0-9_.-]+$", name):
            return False

        return True

    def _is_valid_content_type(self, content_type: str) -> bool:
        """Check if content type is valid and allowed"""

        allowed_types = [
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain",
            "application/octet-stream",
        ]

        # Extract main content type (ignore charset, boundary, etc.)
        main_type = content_type.split(";")[0].strip().lower()

        return main_type in allowed_types
