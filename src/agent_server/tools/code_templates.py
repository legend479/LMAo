"""
Code Templates and Template Management
Advanced template system for different tool types and patterns with optimization and best practices
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime

from src.shared.logging import get_logger

logger = get_logger(__name__)


class TemplateCategory(Enum):
    BASIC = "basic"
    ADVANCED = "advanced"
    ENTERPRISE = "enterprise"
    SPECIALIZED = "specialized"


@dataclass
class TemplateMetadata:
    name: str
    category: TemplateCategory
    description: str
    version: str
    author: str
    created_at: datetime
    supported_features: List[str]
    complexity_level: str
    performance_optimized: bool
    security_hardened: bool


class TemplateManager:
    """Advanced template management with optimization and best practices"""

    def __init__(self):
        self.templates = self._initialize_templates()
        self.template_metadata = self._initialize_metadata()
        self.optimization_rules = self._initialize_optimization_rules()
        self.best_practices = self._initialize_best_practices()

    def _initialize_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize comprehensive template library"""
        return {
            "data_processor_basic": {
                "template": '''
"""
{description}
Generated data processing tool with basic functionality
"""

from typing import Dict, Any, List, Optional, Union
import json
import time
import logging
{additional_imports}

from src.shared.logging import get_logger
from src.agent_server.tools.registry import BaseTool, ToolCapabilities, ResourceRequirements, ToolResult, ExecutionContext, ToolCapability

logger = get_logger(__name__)


class {class_name}(BaseTool):
    """Generated data processing tool with basic functionality"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.processing_stats = {{
            "total_processed": 0,
            "success_count": 0,
            "error_count": 0
        }}
    
    async def execute(self, parameters: Dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Execute data processing operation"""
        
        start_time = time.time()
        
        try:
            # Input validation
            validation_result = await self._validate_inputs(parameters)
            if not validation_result["valid"]:
                return self._create_error_result(
                    "Input validation failed",
                    validation_result["errors"],
                    time.time() - start_time
                )
            
            # Extract and sanitize parameters
{parameter_extraction}
            
            # Pre-processing hooks
            await self._pre_process_hook(parameters)
            
            # Main processing logic
            result = await self._process_data({parameter_names})
            
            # Post-processing hooks
            result = await self._post_process_hook(result, parameters)
            
            # Update statistics
            self.processing_stats["total_processed"] += 1
            self.processing_stats["success_count"] += 1
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                data=result,
                metadata={{
                    "tool_type": "data_processor",
                    "execution_time": execution_time,
                    "processing_stats": self.processing_stats.copy(),
                    "parameters_used": list(parameters.keys())
                }},
                execution_time=execution_time,
                success=True,
                quality_score=self._calculate_quality_score(result),
                confidence_score=0.9
            )
            
        except Exception as e:
            self.processing_stats["error_count"] += 1
            logger.error("Data processing failed", 
                        tool_name=self.__class__.__name__,
                        error=str(e),
                        parameters=parameters)
            
            return self._create_error_result(
                "Processing failed",
                [str(e)],
                time.time() - start_time
            )
    
    async def _validate_inputs(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive input validation"""
        
        validation_result = {{
            "valid": True,
            "errors": [],
            "warnings": []
        }}
        
        # Schema validation
        is_valid, schema_errors = self.validate_parameters(parameters)
        if not is_valid:
            validation_result["valid"] = False
            validation_result["errors"].extend(schema_errors)
        
        # Custom validation rules
{validation_rules}
        
        return validation_result
    
    async def _pre_process_hook(self, parameters: Dict[str, Any]):
        """Pre-processing hook for custom logic"""
        pass
    
    async def _post_process_hook(self, result: Any, parameters: Dict[str, Any]) -> Any:
        """Post-processing hook for result transformation"""
        return result
    
    async def _process_data(self, {parameter_signature}) -> Any:
        """Main data processing logic - implement specific functionality here"""
        
        # Default implementation - override in specific tools
        processed_data = {{
            "status": "processed",
            "input_parameters": {{{parameter_dict}}},
            "processing_timestamp": datetime.utcnow().isoformat(),
            "result": "Data processed successfully"
        }}
        
        return processed_data
    
    def _calculate_quality_score(self, result: Any) -> float:
        """Calculate quality score for the processing result"""
        
        if not result:
            return 0.0
        
        # Basic quality metrics
        quality_score = 0.8  # Base score
        
        # Check result completeness
        if isinstance(result, dict):
            if "status" in result and result["status"] == "processed":
                quality_score += 0.1
            if "result" in result and result["result"]:
                quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _create_error_result(self, error_message: str, errors: List[str], execution_time: float) -> ToolResult:
        """Create standardized error result"""
        
        return ToolResult(
            data=None,
            metadata={{
                "error": error_message,
                "errors": errors,
                "tool_type": "data_processor",
                "execution_time": execution_time
            }},
            execution_time=execution_time,
            success=False,
            error_message=error_message,
            quality_score=0.0,
            confidence_score=0.0
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get comprehensive tool schema"""
        return {{
            "name": "{tool_name}",
            "description": "{description}",
            "version": "1.0.0",
            "category": "data_processor",
            "parameters": {parameter_schema},
            "required_params": {required_params},
            "output_schema": {output_schema},
            "examples": {examples},
            "performance_characteristics": {{
                "average_execution_time": "< 30s",
                "memory_usage": "< 512MB",
                "concurrent_limit": 5
            }}
        }}
    
    def get_capabilities(self) -> ToolCapabilities:
        """Get tool capabilities"""
        return ToolCapabilities(
            primary_capability=ToolCapability.DATA_ANALYSIS,
            secondary_capabilities=[ToolCapability.TRANSFORMATION, ToolCapability.VALIDATION],
            input_types=["object", "string", "array"],
            output_types=["object", "string"],
            supported_formats=["json", "csv", "text", "xml"],
            language_support=["python"]
        )
    
    def get_resource_requirements(self) -> ResourceRequirements:
        """Get resource requirements"""
        return ResourceRequirements(
            cpu_cores=1.0,
            memory_mb=512,
            network_bandwidth_mbps=0.0,
            storage_mb=100,
            max_execution_time=30,
            concurrent_limit=5
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check with processing statistics"""
        
        base_health = await super().health_check()
        
        # Add processing-specific health metrics
        success_rate = (
            self.processing_stats["success_count"] / 
            max(self.processing_stats["total_processed"], 1)
        )
        
        base_health.update({{
            "processing_stats": self.processing_stats.copy(),
            "success_rate": success_rate,
            "health_status": "healthy" if success_rate > 0.8 else "degraded"
        }})
        
        return base_health
''',
                "features": [
                    "input_validation",
                    "error_handling",
                    "statistics",
                    "hooks",
                    "quality_scoring",
                ],
                "complexity": "basic",
                "performance_optimized": True,
            },
            "api_client_advanced": {
                "template": '''
"""
{description}
Advanced API client tool with retry logic, caching, and comprehensive error handling
"""

from typing import Dict, Any, List, Optional, Union
import json
import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
import hashlib
{additional_imports}

from src.shared.logging import get_logger
from src.agent_server.tools.registry import BaseTool, ToolCapabilities, ResourceRequirements, ToolResult, ExecutionContext, ToolCapability

logger = get_logger(__name__)


class {class_name}(BaseTool):
    """Advanced API client tool with enterprise features"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.session = None
        self.base_url = config.get("base_url", "") if config else ""
        self.timeout = config.get("timeout", 30) if config else 30
        self.max_retries = config.get("max_retries", 3) if config else 3
        self.retry_delay = config.get("retry_delay", 1.0) if config else 1.0
        self.cache_enabled = config.get("cache_enabled", True) if config else True
        self.cache_ttl = config.get("cache_ttl", 300) if config else 300  # 5 minutes
        
        # Request cache
        self.request_cache: Dict[str, Dict[str, Any]] = {{}}
        
        # Rate limiting
        self.rate_limit_requests = config.get("rate_limit_requests", 100) if config else 100
        self.rate_limit_window = config.get("rate_limit_window", 60) if config else 60  # 1 minute
        self.request_timestamps: List[datetime] = []
        
        # Statistics
        self.api_stats = {{
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cached_responses": 0,
            "average_response_time": 0.0
        }}
    
    async def initialize(self):
        """Initialize HTTP session with advanced configuration"""
        
        if not self.session:
            # Configure timeout
            timeout = aiohttp.ClientTimeout(
                total=self.timeout,
                connect=10,
                sock_read=self.timeout
            )
            
            # Configure connector
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            # Create session
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={{
                    "User-Agent": "SE-SME-Agent/1.0",
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }}
            )
        
        await super().initialize()
    
    async def cleanup(self):
        """Cleanup HTTP session and resources"""
        
        if self.session:
            await self.session.close()
            self.session = None
        
        # Clear cache
        self.request_cache.clear()
        self.request_timestamps.clear()
        
        await super().cleanup()
    
    async def execute(self, parameters: Dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Execute API client operation with advanced features"""
        
        start_time = time.time()
        
        try:
            # Ensure session is initialized
            if not self.session:
                await self.initialize()
            
            # Input validation
            validation_result = await self._validate_inputs(parameters)
            if not validation_result["valid"]:
                return self._create_error_result(
                    "Input validation failed",
                    validation_result["errors"],
                    time.time() - start_time
                )
            
            # Rate limiting check
            if not await self._check_rate_limit():
                return self._create_error_result(
                    "Rate limit exceeded",
                    ["Too many requests within the time window"],
                    time.time() - start_time
                )
            
            # Extract parameters
{parameter_extraction}
            
            # Check cache first
            cache_key = self._generate_cache_key(parameters)
            cached_response = await self._get_cached_response(cache_key)
            
            if cached_response:
                self.api_stats["cached_responses"] += 1
                execution_time = time.time() - start_time
                
                return ToolResult(
                    data=cached_response["data"],
                    metadata={{
                        "tool_type": "api_client",
                        "execution_time": execution_time,
                        "cached": True,
                        "cache_age": (datetime.utcnow() - cached_response["timestamp"]).total_seconds()
                    }},
                    execution_time=execution_time,
                    success=True,
                    quality_score=cached_response.get("quality_score", 0.9),
                    confidence_score=0.9
                )
            
            # Make API request with retry logic
            result = await self._make_api_request_with_retry({parameter_names})
            
            # Cache successful responses
            if result.get("success", False) and self.cache_enabled:
                await self._cache_response(cache_key, result)
            
            # Update statistics
            self.api_stats["total_requests"] += 1
            if result.get("success", False):
                self.api_stats["successful_requests"] += 1
            else:
                self.api_stats["failed_requests"] += 1
            
            execution_time = time.time() - start_time
            self._update_average_response_time(execution_time)
            
            return ToolResult(
                data=result,
                metadata={{
                    "tool_type": "api_client",
                    "execution_time": execution_time,
                    "cached": False,
                    "api_stats": self.api_stats.copy()
                }},
                execution_time=execution_time,
                success=result.get("success", False),
                quality_score=self._calculate_quality_score(result),
                confidence_score=0.9 if result.get("success", False) else 0.3
            )
            
        except Exception as e:
            self.api_stats["failed_requests"] += 1
            logger.error("API client execution failed",
                        tool_name=self.__class__.__name__,
                        error=str(e),
                        parameters=parameters)
            
            return self._create_error_result(
                "API client execution failed",
                [str(e)],
                time.time() - start_time
            )
    
    async def _validate_inputs(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate API client inputs"""
        
        validation_result = {{
            "valid": True,
            "errors": [],
            "warnings": []
        }}
        
        # Schema validation
        is_valid, schema_errors = self.validate_parameters(parameters)
        if not is_valid:
            validation_result["valid"] = False
            validation_result["errors"].extend(schema_errors)
        
        # URL validation
        url = parameters.get("url", "")
        if url and not self._is_valid_url(url):
            validation_result["valid"] = False
            validation_result["errors"].append("Invalid URL format")
        
        return validation_result
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format and security"""
        
        # Basic URL format check
        if not url.startswith(("http://", "https://")):
            return False
        
        # Security checks - prevent SSRF
        forbidden_hosts = ["localhost", "127.0.0.1", "0.0.0.0", "::1"]
        for host in forbidden_hosts:
            if host in url.lower():
                return False
        
        return True
    
    async def _check_rate_limit(self) -> bool:
        """Check if request is within rate limits"""
        
        now = datetime.utcnow()
        
        # Remove old timestamps
        cutoff = now - timedelta(seconds=self.rate_limit_window)
        self.request_timestamps = [
            ts for ts in self.request_timestamps if ts > cutoff
        ]
        
        # Check if we're within limits
        if len(self.request_timestamps) >= self.rate_limit_requests:
            return False
        
        # Add current timestamp
        self.request_timestamps.append(now)
        return True
    
    def _generate_cache_key(self, parameters: Dict[str, Any]) -> str:
        """Generate cache key for request parameters"""
        
        # Create deterministic key from parameters
        param_str = json.dumps(parameters, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    async def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired"""
        
        if not self.cache_enabled or cache_key not in self.request_cache:
            return None
        
        cached_data = self.request_cache[cache_key]
        
        # Check if cache is expired
        if datetime.utcnow() - cached_data["timestamp"] > timedelta(seconds=self.cache_ttl):
            del self.request_cache[cache_key]
            return None
        
        return cached_data
    
    async def _cache_response(self, cache_key: str, response_data: Dict[str, Any]):
        """Cache successful response"""
        
        self.request_cache[cache_key] = {{
            "data": response_data,
            "timestamp": datetime.utcnow(),
            "quality_score": self._calculate_quality_score(response_data)
        }}
        
        # Limit cache size
        if len(self.request_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(
                self.request_cache.keys(),
                key=lambda k: self.request_cache[k]["timestamp"]
            )[:100]
            
            for key in oldest_keys:
                del self.request_cache[key]
    
    async def _make_api_request_with_retry(self, {parameter_signature}) -> Dict[str, Any]:
        """Make API request with retry logic"""
        
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await self._make_api_request({parameter_names})
                
                if result.get("success", False):
                    return result
                
                last_error = result.get("error", "Unknown error")
                
                # If not the last attempt, wait before retry
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(delay)
                    logger.warning("API request failed, retrying",
                                 attempt=attempt + 1,
                                 delay=delay,
                                 error=last_error)
                
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
        
        # All retries failed
        return {{
            "success": False,
            "error": f"All retry attempts failed. Last error: {{last_error}}",
            "attempts": self.max_retries + 1
        }}
    
    async def _make_api_request(self, {parameter_signature}) -> Dict[str, Any]:
        """Make actual API request - implement specific logic here"""
        
        try:
            # Default implementation - override in specific tools
            url = f"{{self.base_url}}/api/endpoint"
            
            async with self.session.get(url) as response:
                response_data = await response.text()
                
                if response.status == 200:
                    try:
                        json_data = json.loads(response_data)
                        return {{
                            "success": True,
                            "data": json_data,
                            "status_code": response.status,
                            "headers": dict(response.headers)
                        }}
                    except json.JSONDecodeError:
                        return {{
                            "success": True,
                            "data": response_data,
                            "status_code": response.status,
                            "headers": dict(response.headers)
                        }}
                else:
                    return {{
                        "success": False,
                        "error": f"HTTP {{response.status}}: {{response_data}}",
                        "status_code": response.status,
                        "headers": dict(response.headers)
                    }}
                    
        except Exception as e:
            return {{
                "success": False,
                "error": str(e),
                "status_code": None
            }}
    
    def _calculate_quality_score(self, result: Dict[str, Any]) -> float:
        """Calculate quality score for API response"""
        
        if not result:
            return 0.0
        
        quality_score = 0.5  # Base score
        
        # Success bonus
        if result.get("success", False):
            quality_score += 0.3
        
        # Data completeness
        if result.get("data"):
            quality_score += 0.2
        
        return min(1.0, quality_score)
    
    def _update_average_response_time(self, execution_time: float):
        """Update average response time statistics"""
        
        total_requests = self.api_stats["total_requests"]
        current_avg = self.api_stats["average_response_time"]
        
        # Calculate new average
        new_avg = ((current_avg * (total_requests - 1)) + execution_time) / total_requests
        self.api_stats["average_response_time"] = new_avg
    
    def _create_error_result(self, error_message: str, errors: List[str], execution_time: float) -> ToolResult:
        """Create standardized error result"""
        
        return ToolResult(
            data=None,
            metadata={{
                "error": error_message,
                "errors": errors,
                "tool_type": "api_client",
                "execution_time": execution_time,
                "api_stats": self.api_stats.copy()
            }},
            execution_time=execution_time,
            success=False,
            error_message=error_message,
            quality_score=0.0,
            confidence_score=0.0
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get comprehensive tool schema"""
        return {{
            "name": "{tool_name}",
            "description": "{description}",
            "version": "1.0.0",
            "category": "api_client",
            "parameters": {parameter_schema},
            "required_params": {required_params},
            "output_schema": {output_schema},
            "examples": {examples},
            "features": [
                "retry_logic",
                "response_caching",
                "rate_limiting",
                "error_handling",
                "statistics_tracking"
            ]
        }}
    
    def get_capabilities(self) -> ToolCapabilities:
        """Get tool capabilities"""
        return ToolCapabilities(
            primary_capability=ToolCapability.COMMUNICATION,
            secondary_capabilities=[ToolCapability.DATA_ANALYSIS, ToolCapability.VALIDATION],
            input_types=["object", "string"],
            output_types=["object"],
            supported_formats=["json", "xml", "text", "html"],
            language_support=["python"]
        )
    
    def get_resource_requirements(self) -> ResourceRequirements:
        """Get resource requirements"""
        return ResourceRequirements(
            cpu_cores=0.5,
            memory_mb=256,
            network_bandwidth_mbps=10.0,
            storage_mb=100,  # For caching
            max_execution_time=self.timeout + (self.max_retries * self.retry_delay * 4),
            concurrent_limit=10
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check with API statistics"""
        
        base_health = await super().health_check()
        
        # Calculate success rate
        total_requests = self.api_stats["total_requests"]
        success_rate = (
            self.api_stats["successful_requests"] / max(total_requests, 1)
        )
        
        # Determine health status
        if success_rate > 0.9:
            health_status = "healthy"
        elif success_rate > 0.7:
            health_status = "degraded"
        else:
            health_status = "unhealthy"
        
        base_health.update({{
            "api_stats": self.api_stats.copy(),
            "success_rate": success_rate,
            "cache_size": len(self.request_cache),
            "rate_limit_status": {{
                "requests_in_window": len(self.request_timestamps),
                "limit": self.rate_limit_requests,
                "window_seconds": self.rate_limit_window
            }},
            "health_status": health_status
        }})
        
        return base_health
''',
                "features": [
                    "retry_logic",
                    "caching",
                    "rate_limiting",
                    "statistics",
                    "health_monitoring",
                ],
                "complexity": "advanced",
                "performance_optimized": True,
            },
        }

    def _initialize_metadata(self) -> Dict[str, TemplateMetadata]:
        """Initialize template metadata"""
        return {
            "data_processor_basic": TemplateMetadata(
                name="Basic Data Processor",
                category=TemplateCategory.BASIC,
                description="Basic data processing template with validation and error handling",
                version="1.0.0",
                author="SE-SME-Agent",
                created_at=datetime.utcnow(),
                supported_features=[
                    "input_validation",
                    "error_handling",
                    "statistics",
                    "hooks",
                ],
                complexity_level="basic",
                performance_optimized=True,
                security_hardened=True,
            ),
            "api_client_advanced": TemplateMetadata(
                name="Advanced API Client",
                category=TemplateCategory.ADVANCED,
                description="Advanced API client with retry logic, caching, and rate limiting",
                version="1.0.0",
                author="SE-SME-Agent",
                created_at=datetime.utcnow(),
                supported_features=[
                    "retry_logic",
                    "caching",
                    "rate_limiting",
                    "statistics",
                ],
                complexity_level="advanced",
                performance_optimized=True,
                security_hardened=True,
            ),
        }

    def _initialize_optimization_rules(self) -> Dict[str, List[str]]:
        """Initialize code optimization rules"""
        return {
            "performance": [
                "Use async/await for I/O operations",
                "Implement connection pooling for network requests",
                "Add caching for frequently accessed data",
                "Use batch processing for multiple items",
                "Implement lazy loading for large datasets",
            ],
            "memory": [
                "Use generators for large data processing",
                "Implement memory-efficient data structures",
                "Clear unused variables and references",
                "Use streaming for file operations",
                "Implement garbage collection hints",
            ],
            "security": [
                "Validate all input parameters",
                "Sanitize file paths and URLs",
                "Implement rate limiting",
                "Use secure random generators",
                "Add input size limits",
            ],
            "reliability": [
                "Implement comprehensive error handling",
                "Add retry logic with exponential backoff",
                "Use circuit breaker pattern for external services",
                "Implement health checks",
                "Add logging and monitoring",
            ],
        }

    def _initialize_best_practices(self) -> Dict[str, List[str]]:
        """Initialize coding best practices"""
        return {
            "code_structure": [
                "Use clear and descriptive variable names",
                "Keep functions small and focused",
                "Add comprehensive docstrings",
                "Use type hints for better code clarity",
                "Follow PEP 8 style guidelines",
            ],
            "error_handling": [
                "Use specific exception types",
                "Provide meaningful error messages",
                "Log errors with context information",
                "Implement graceful degradation",
                "Use try-except blocks appropriately",
            ],
            "testing": [
                "Write unit tests for core functionality",
                "Include edge case testing",
                "Use mocking for external dependencies",
                "Implement integration tests",
                "Add performance benchmarks",
            ],
            "documentation": [
                "Document all public methods",
                "Include usage examples",
                "Explain complex algorithms",
                "Document configuration options",
                "Maintain changelog",
            ],
        }

    def get_template(
        self, template_name: str, tool_type: str = None
    ) -> Optional[Dict[str, Any]]:
        """Get template by name or tool type"""

        if template_name in self.templates:
            return self.templates[template_name]

        # Try to find by tool type
        if tool_type:
            for name, template in self.templates.items():
                if tool_type.lower() in name.lower():
                    return template

        return None

    def get_best_template(
        self, requirements: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Select best template based on requirements"""

        tool_type = requirements.get("tool_type", "")
        complexity = requirements.get("complexity_level", "basic")
        features = requirements.get("required_features", [])

        best_match = None
        best_score = 0

        for template_name, template_data in self.templates.items():
            score = self._calculate_template_score(
                template_name, template_data, tool_type, complexity, features
            )

            if score > best_score:
                best_score = score
                best_match = template_data

        return best_match

    def _calculate_template_score(
        self,
        template_name: str,
        template_data: Dict[str, Any],
        tool_type: str,
        complexity: str,
        features: List[str],
    ) -> float:
        """Calculate template matching score"""

        score = 0.0

        # Tool type match
        if tool_type.lower() in template_name.lower():
            score += 0.4

        # Complexity match
        template_complexity = template_data.get("complexity", "basic")
        if template_complexity == complexity:
            score += 0.3
        elif (
            abs(
                ["basic", "advanced", "enterprise"].index(template_complexity)
                - ["basic", "advanced", "enterprise"].index(complexity)
            )
            == 1
        ):
            score += 0.2

        # Feature match
        template_features = template_data.get("features", [])
        matching_features = set(features) & set(template_features)
        if features:
            feature_score = len(matching_features) / len(features)
            score += 0.3 * feature_score

        return score

    def optimize_template(
        self, template: str, optimization_type: str = "performance"
    ) -> str:
        """Apply optimization rules to template"""

        optimized_template = template

        if optimization_type in self.optimization_rules:
            rules = self.optimization_rules[optimization_type]

            # Apply specific optimizations based on rules
            if optimization_type == "performance":
                optimized_template = self._apply_performance_optimizations(
                    optimized_template
                )
            elif optimization_type == "security":
                optimized_template = self._apply_security_optimizations(
                    optimized_template
                )
            elif optimization_type == "memory":
                optimized_template = self._apply_memory_optimizations(
                    optimized_template
                )

        return optimized_template

    def _apply_performance_optimizations(self, template: str) -> str:
        """Apply performance optimizations to template"""

        # Add async/await patterns
        if "def execute(" in template and "async def execute(" not in template:
            template = template.replace("def execute(", "async def execute(")

        # Add caching hints
        if "# TODO: Implement caching" not in template:
            template = template.replace(
                "# Main processing logic",
                "# Main processing logic\n        # TODO: Implement caching for frequently accessed data",
            )

        return template

    def _apply_security_optimizations(self, template: str) -> str:
        """Apply security optimizations to template"""

        # Add input validation
        if "_validate_inputs" not in template:
            validation_code = '''
    async def _validate_inputs(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate inputs for security"""
        validation_result = {
            "valid": True,
            "errors": []
        }
        
        # Add specific validation logic here
        
        return validation_result
'''
            template = template.replace(
                "async def execute(", validation_code + "\n    async def execute("
            )

        return template

    def _apply_memory_optimizations(self, template: str) -> str:
        """Apply memory optimizations to template"""

        # Add memory management hints
        if "# Memory optimization" not in template:
            template = template.replace(
                "return result",
                "# Memory optimization: clear large variables\n        return result",
            )

        return template

    def validate_template(self, template: str) -> Dict[str, Any]:
        """Validate template for completeness and correctness"""

        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "completeness_score": 0.0,
        }

        required_elements = [
            "class.*BaseTool",
            "async def execute",
            "def get_schema",
            "def get_capabilities",
            "def get_resource_requirements",
        ]

        completeness_score = 0
        for element in required_elements:
            if re.search(element, template):
                completeness_score += 1
            else:
                validation_result["errors"].append(
                    f"Missing required element: {element}"
                )

        validation_result["completeness_score"] = completeness_score / len(
            required_elements
        )

        if validation_result["errors"]:
            validation_result["valid"] = False

        # Check for best practices
        best_practice_checks = [
            ("docstring", '""".*"""'),
            ("error_handling", "try:.*except"),
            ("logging", "logger\."),
            ("type_hints", "->.*:"),
        ]

        for practice_name, pattern in best_practice_checks:
            if not re.search(pattern, template, re.DOTALL):
                validation_result["warnings"].append(
                    f"Missing best practice: {practice_name}"
                )

        return validation_result

    def get_template_metadata(self, template_name: str) -> Optional[TemplateMetadata]:
        """Get metadata for specific template"""
        return self.template_metadata.get(template_name)

    def list_templates(
        self, category: Optional[TemplateCategory] = None
    ) -> List[Dict[str, Any]]:
        """List available templates with optional category filter"""

        templates = []

        for name, metadata in self.template_metadata.items():
            if category is None or metadata.category == category:
                templates.append(
                    {
                        "name": name,
                        "category": metadata.category.value,
                        "description": metadata.description,
                        "complexity_level": metadata.complexity_level,
                        "supported_features": metadata.supported_features,
                        "performance_optimized": metadata.performance_optimized,
                        "security_hardened": metadata.security_hardened,
                    }
                )

        return templates
