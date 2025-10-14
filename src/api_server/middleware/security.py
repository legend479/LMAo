"""
Security Middleware
Comprehensive input sanitization, prompt injection prevention, and output moderation
"""

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import re
import json
from typing import List, Dict, Any
import hashlib
import time

from ...shared.config import get_settings
from ...shared.logging import get_logger

logger = get_logger(__name__)


class PromptInjectionDetector:
    """Advanced prompt injection detection system"""

    # Prompt injection patterns
    PROMPT_INJECTION_PATTERNS = [
        # Direct instruction attempts
        r"(?i)(ignore|forget|disregard)\s+(previous|above|all|your)\s+(instructions?|prompts?|rules?)",
        r"(?i)(system|admin|root|developer)\s+(mode|access|override|command)",
        r"(?i)(act\s+as|pretend\s+to\s+be|roleplay\s+as)\s+(admin|root|system|developer)",
        # Jailbreak attempts
        r"(?i)(jailbreak|break\s+out|escape\s+from)\s+(system|constraints?|limitations?)",
        r"(?i)(bypass|circumvent|override)\s+(safety|security|filters?|restrictions?)",
        r"(?i)do\s+anything\s+now|DAN\s+mode",
        # Prompt leakage attempts
        r"(?i)(show|reveal|display|print)\s+(your|the)\s+(prompt|instructions?|system\s+message)",
        r"(?i)(what\s+(is|are)\s+your|tell\s+me\s+your)\s+(instructions?|prompts?|rules?)",
        r"(?i)repeat\s+(your|the)\s+(initial|original|system)\s+(prompt|instructions?)",
        # Context manipulation
        r"(?i)(new\s+session|start\s+over|reset\s+context|clear\s+memory)",
        r"(?i)(end\s+of|finish)\s+(conversation|chat|session)",
        r"(?i)(switch\s+to|change\s+to)\s+(different|new)\s+(mode|personality|character)",
        # Encoding/obfuscation attempts
        r"(?i)(base64|hex|rot13|unicode|ascii)\s+(decode|encode)",
        r"(?i)(translate|convert)\s+(from|to)\s+(binary|hex|base64)",
        # Social engineering
        r"(?i)(emergency|urgent|critical|important)\s+(override|access|bypass)",
        r"(?i)(authorized|permission|allowed)\s+(by|from)\s+(admin|developer|system)",
        r"(?i)(test|debug|maintenance)\s+(mode|access|command)",
        # Code injection in prompts
        r"(?i)(execute|run|eval)\s+(code|script|command)",
        r"(?i)(import|require|include)\s+[a-zA-Z_][a-zA-Z0-9_]*",
        # Template/format string injection
        r"(?i)\{[^}]*\}.*\{[^}]*\}",  # Multiple template variables
        r"(?i)%[sdxo]|%\([^)]+\)[sdxo]",  # Python format strings
        # Multi-language injection attempts
        r"(?i)(python|javascript|sql|bash|powershell)\s+(code|script)",
        r"(?i)```\s*(python|js|sql|bash|sh)",  # Code blocks
    ]

    # Suspicious keywords that might indicate injection
    SUSPICIOUS_KEYWORDS = {
        "system_override",
        "admin_access",
        "root_mode",
        "developer_mode",
        "bypass_security",
        "ignore_rules",
        "jailbreak",
        "prompt_leak",
        "system_prompt",
        "initial_instructions",
        "base_prompt",
    }

    def __init__(self):
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.PROMPT_INJECTION_PATTERNS
        ]
        self.detection_cache = {}  # Cache for performance
        self.cache_ttl = 3600  # 1 hour cache TTL

    def detect_injection(self, text: str) -> Dict[str, Any]:
        """Detect prompt injection attempts in text"""

        if not text or not isinstance(text, str):
            return {"is_injection": False, "confidence": 0.0, "patterns": []}

        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.detection_cache:
            cache_entry = self.detection_cache[text_hash]
            if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                return cache_entry["result"]

        detected_patterns = []
        confidence_score = 0.0

        # Check against compiled patterns
        for i, pattern in enumerate(self.compiled_patterns):
            matches = pattern.findall(text)
            if matches:
                pattern_info = {
                    "pattern_id": i,
                    "pattern_type": self._get_pattern_type(i),
                    "matches": matches[:3],  # Limit matches for logging
                    "confidence": self._calculate_pattern_confidence(i, matches),
                }
                detected_patterns.append(pattern_info)
                confidence_score += pattern_info["confidence"]

        # Check for suspicious keyword combinations
        keyword_score = self._check_suspicious_keywords(text)
        confidence_score += keyword_score

        # Check for encoding/obfuscation
        encoding_score = self._check_encoding_attempts(text)
        confidence_score += encoding_score

        # Normalize confidence score (0-1)
        confidence_score = min(confidence_score, 1.0)

        result = {
            "is_injection": confidence_score > 0.3,  # Threshold for detection
            "confidence": confidence_score,
            "patterns": detected_patterns,
            "keyword_score": keyword_score,
            "encoding_score": encoding_score,
        }

        # Cache result
        self.detection_cache[text_hash] = {"result": result, "timestamp": time.time()}

        return result

    def _get_pattern_type(self, pattern_id: int) -> str:
        """Get pattern type based on pattern ID"""
        if pattern_id < 3:
            return "direct_instruction"
        elif pattern_id < 6:
            return "jailbreak"
        elif pattern_id < 9:
            return "prompt_leakage"
        elif pattern_id < 12:
            return "context_manipulation"
        elif pattern_id < 14:
            return "encoding_obfuscation"
        elif pattern_id < 17:
            return "social_engineering"
        else:
            return "code_injection"

    def _calculate_pattern_confidence(self, pattern_id: int, matches: List) -> float:
        """Calculate confidence score for pattern matches"""
        base_confidence = {
            "direct_instruction": 0.8,
            "jailbreak": 0.9,
            "prompt_leakage": 0.7,
            "context_manipulation": 0.6,
            "encoding_obfuscation": 0.5,
            "social_engineering": 0.4,
            "code_injection": 0.6,
        }

        pattern_type = self._get_pattern_type(pattern_id)
        confidence = base_confidence.get(pattern_type, 0.3)

        # Increase confidence based on number of matches
        match_multiplier = min(len(matches) * 0.1, 0.3)
        return min(confidence + match_multiplier, 1.0)

    def _check_suspicious_keywords(self, text: str) -> float:
        """Check for suspicious keyword combinations"""
        text_lower = text.lower()
        found_keywords = [kw for kw in self.SUSPICIOUS_KEYWORDS if kw in text_lower]

        if not found_keywords:
            return 0.0

        # Score based on number and rarity of keywords
        return min(len(found_keywords) * 0.15, 0.4)

    def _check_encoding_attempts(self, text: str) -> float:
        """Check for potential encoding/obfuscation attempts"""
        score = 0.0

        # Check for base64-like patterns
        if re.search(r"[A-Za-z0-9+/]{20,}={0,2}", text):
            score += 0.2

        # Check for hex patterns
        if re.search(r"(?:0x)?[0-9a-fA-F]{16,}", text):
            score += 0.2

        # Check for excessive special characters
        special_chars = sum(1 for c in text if not c.isalnum() and c not in " .,!?-")
        if len(text) > 0 and special_chars / len(text) > 0.3:
            score += 0.2

        return min(score, 0.4)


class OutputModerator:
    """Output content moderation system"""

    # Harmful content patterns
    HARMFUL_PATTERNS = [
        # Personal information patterns
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email (be careful)
        # Potentially harmful instructions
        r"(?i)(delete|remove|destroy)\s+(all|everything|files?|data)",
        r"(?i)(format|wipe)\s+(hard\s+drive|disk|computer)",
        r"(?i)(download|install)\s+(malware|virus|trojan)",
        # Inappropriate content indicators
        r"(?i)(explicit|graphic|violent)\s+(content|material|images?)",
        r"(?i)(illegal|unlawful|criminal)\s+(activity|activities|actions?)",
        # System information leakage
        r"(?i)(api\s+key|secret\s+key|password|token)\s*[:=]\s*[A-Za-z0-9+/]{10,}",
        r"(?i)(database|db)\s+(connection|url|string)",
    ]

    # Sensitive information patterns
    SENSITIVE_PATTERNS = [
        r"(?i)(internal|private|confidential)\s+(information|data|details)",
        r"(?i)(system\s+prompt|initial\s+instructions|base\s+configuration)",
        r"(?i)(training\s+data|model\s+weights|proprietary\s+algorithm)",
    ]

    def __init__(self):
        self.harmful_compiled = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.HARMFUL_PATTERNS
        ]
        self.sensitive_compiled = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.SENSITIVE_PATTERNS
        ]

    def moderate_output(self, text: str) -> Dict[str, Any]:
        """Moderate output content for harmful or sensitive information"""

        if not text or not isinstance(text, str):
            return {"is_safe": True, "issues": [], "filtered_text": text}

        issues = []
        filtered_text = text

        # Check for harmful content
        for i, pattern in enumerate(self.harmful_compiled):
            matches = pattern.finditer(text)
            for match in matches:
                issues.append(
                    {
                        "type": "harmful_content",
                        "pattern_id": i,
                        "start": match.start(),
                        "end": match.end(),
                        "matched_text": match.group()[:50],  # Limit for logging
                    }
                )
                # Replace with placeholder
                filtered_text = filtered_text.replace(match.group(), "[FILTERED]")

        # Check for sensitive information
        for i, pattern in enumerate(self.sensitive_compiled):
            matches = pattern.finditer(text)
            for match in matches:
                issues.append(
                    {
                        "type": "sensitive_info",
                        "pattern_id": i,
                        "start": match.start(),
                        "end": match.end(),
                        "matched_text": match.group()[:50],
                    }
                )
                # Replace with placeholder
                filtered_text = filtered_text.replace(match.group(), "[REDACTED]")

        return {
            "is_safe": len(issues) == 0,
            "issues": issues,
            "filtered_text": filtered_text,
            "original_length": len(text),
            "filtered_length": len(filtered_text),
        }


class SecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive security middleware with prompt injection prevention and output moderation"""

    # Traditional injection patterns (XSS, SQL, etc.)
    TRADITIONAL_INJECTION_PATTERNS = [
        r"<script[^>]*>.*?</script>",  # XSS
        r"javascript:",  # JavaScript injection
        r"on\w+\s*=",  # Event handlers
        r"eval\s*\(",  # Code evaluation
        r"exec\s*\(",  # Code execution
        r"(\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b)",  # SQL injection
        r"(\bUNION\b|\bOR\b|\bAND\b)\s+\d+\s*=\s*\d+",  # SQL injection
        r"(\'|\")(\s*;\s*)",  # SQL termination
    ]

    def __init__(self, app):
        super().__init__(app)
        self.traditional_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.TRADITIONAL_INJECTION_PATTERNS
        ]
        self.prompt_detector = PromptInjectionDetector()
        self.output_moderator = OutputModerator()
        self.settings = get_settings()

        # Security metrics
        self.blocked_requests = 0
        self.injection_attempts = 0
        self.moderated_responses = 0

    async def dispatch(self, request: Request, call_next):
        """Process request through comprehensive security checks"""

        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")

        try:
            # Traditional security checks
            await self._check_traditional_security(request, request_id)

            # Prompt injection detection for chat endpoints
            if self._is_chat_endpoint(request):
                await self._check_prompt_injection(request, request_id)

            # Process request
            response = await call_next(request)

            # Output moderation for responses
            if self._should_moderate_output(request, response):
                response = await self._moderate_response(response, request_id)

            # Add comprehensive security headers
            self._add_security_headers(response)

            # Log security metrics
            processing_time = time.time() - start_time
            logger.debug(
                "Security check completed",
                request_id=request_id,
                processing_time=processing_time,
                path=request.url.path,
            )

            return response

        except HTTPException as e:
            self.blocked_requests += 1
            logger.warning(
                "Security check blocked request",
                request_id=request_id,
                status_code=e.status_code,
                detail=e.detail,
                path=request.url.path,
                method=request.method,
            )
            raise
        except Exception as e:
            logger.error(
                "Security middleware error",
                request_id=request_id,
                error=str(e),
                path=request.url.path,
            )
            raise HTTPException(status_code=500, detail="Security processing error")

    async def _check_traditional_security(self, request: Request, request_id: str):
        """Check for traditional security threats"""

        # Check query parameters
        for key, value in request.query_params.items():
            if self._contains_traditional_injection(value):
                logger.warning(
                    "Traditional injection detected in query params",
                    request_id=request_id,
                    param=key,
                    value=value[:100],
                )
                raise HTTPException(
                    status_code=400, detail="Potentially malicious content detected"
                )

        # Check headers for injection
        suspicious_headers = ["user-agent", "referer", "x-forwarded-for"]
        for header in suspicious_headers:
            value = request.headers.get(header)
            if value and self._contains_traditional_injection(value):
                logger.warning(
                    "Traditional injection detected in headers",
                    request_id=request_id,
                    header=header,
                    value=value[:100],
                )
                raise HTTPException(
                    status_code=400, detail="Malicious header content detected"
                )

    async def _check_prompt_injection(self, request: Request, request_id: str):
        """Check for prompt injection attempts in chat requests"""

        if request.method not in ["POST", "PUT", "PATCH"]:
            return

        try:
            # Read and parse request body
            body = await request.body()
            if not body:
                return

            content_type = request.headers.get("content-type", "")

            if "application/json" in content_type:
                try:
                    json_data = json.loads(body)
                    await self._check_json_for_injection(json_data, request_id)
                except json.JSONDecodeError:
                    pass  # Invalid JSON will be caught by validation middleware

        except Exception as e:
            logger.error(
                "Prompt injection check error", request_id=request_id, error=str(e)
            )
            # Don't block on errors, let validation middleware handle it

    async def _check_json_for_injection(self, data: Any, request_id: str):
        """Recursively check JSON data for prompt injection"""

        if isinstance(data, str):
            detection_result = self.prompt_detector.detect_injection(data)

            if detection_result["is_injection"]:
                self.injection_attempts += 1
                logger.warning(
                    "Prompt injection attempt detected",
                    request_id=request_id,
                    confidence=detection_result["confidence"],
                    patterns=len(detection_result["patterns"]),
                    text_preview=data[:100],
                )

                raise HTTPException(
                    status_code=400, detail="Prompt injection attempt detected"
                )

        elif isinstance(data, dict):
            for key, value in data.items():
                await self._check_json_for_injection(value, request_id)

        elif isinstance(data, list):
            for item in data:
                await self._check_json_for_injection(item, request_id)

    async def _moderate_response(self, response: Response, request_id: str) -> Response:
        """Moderate response content"""

        try:
            # Get response content
            content = b""
            async for chunk in response.body_iterator:
                content += chunk

            if not content:
                return response

            # Decode content
            try:
                text_content = content.decode("utf-8")
            except UnicodeDecodeError:
                # Binary content, skip moderation
                return response

            # Moderate content
            moderation_result = self.output_moderator.moderate_output(text_content)

            if not moderation_result["is_safe"]:
                self.moderated_responses += 1
                logger.warning(
                    "Response content moderated",
                    request_id=request_id,
                    issues=len(moderation_result["issues"]),
                    original_length=moderation_result["original_length"],
                    filtered_length=moderation_result["filtered_length"],
                )

                # Create new response with filtered content
                filtered_content = moderation_result["filtered_text"].encode("utf-8")

                new_response = Response(
                    content=filtered_content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )

                # Update content length
                new_response.headers["content-length"] = str(len(filtered_content))
                new_response.headers["x-content-moderated"] = "true"

                return new_response

            return response

        except Exception as e:
            logger.error(
                "Response moderation error", request_id=request_id, error=str(e)
            )
            return response  # Return original response on error

    def _contains_traditional_injection(self, text: str) -> bool:
        """Check for traditional injection patterns"""
        if not isinstance(text, str):
            return False

        for pattern in self.traditional_patterns:
            if pattern.search(text):
                return True
        return False

    def _is_chat_endpoint(self, request: Request) -> bool:
        """Check if request is to a chat endpoint"""
        chat_paths = ["/api/v1/chat", "/chat"]
        return any(request.url.path.startswith(path) for path in chat_paths)

    def _should_moderate_output(self, request: Request, response: Response) -> bool:
        """Determine if response should be moderated"""

        # Only moderate successful responses
        if response.status_code >= 400:
            return False

        # Only moderate text content
        content_type = response.headers.get("content-type", "")
        if not any(
            ct in content_type for ct in ["application/json", "text/plain", "text/html"]
        ):
            return False

        # Moderate chat and document endpoints
        moderate_paths = ["/api/v1/chat", "/api/v1/documents", "/chat", "/documents"]
        return any(request.url.path.startswith(path) for path in moderate_paths)

    def _add_security_headers(self, response: Response):
        """Add comprehensive security headers"""

        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "X-Permitted-Cross-Domain-Policies": "none",
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Resource-Policy": "same-origin",
        }

        for header, value in security_headers.items():
            response.headers[header] = value

    def get_security_metrics(self) -> Dict[str, int]:
        """Get security metrics"""
        return {
            "blocked_requests": self.blocked_requests,
            "injection_attempts": self.injection_attempts,
            "moderated_responses": self.moderated_responses,
        }
