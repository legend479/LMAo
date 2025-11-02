"""
Code Tool Generator
Automated code tool generation from natural language descriptions with comprehensive analysis and validation
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import json
import re
import ast
import time
from datetime import datetime
import hashlib

from src.shared.logging import get_logger
from src.agent_server.tools.registry import (
    BaseTool,
    ToolCapabilities,
    ResourceRequirements,
    ToolResult,
    ExecutionContext,
    ToolCapability,
)

logger = get_logger(__name__)


class ToolType(Enum):
    DATA_PROCESSOR = "data_processor"
    API_CLIENT = "api_client"
    FILE_HANDLER = "file_handler"
    CALCULATOR = "calculator"
    VALIDATOR = "validator"
    TRANSFORMER = "transformer"
    ANALYZER = "analyzer"
    GENERATOR = "generator"


class SecurityRisk(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DependencyType(Enum):
    STANDARD_LIBRARY = "standard_library"
    THIRD_PARTY = "third_party"
    SYSTEM = "system"
    CUSTOM = "custom"


@dataclass
class ToolRequirement:
    """Structured representation of tool requirements extracted from natural language"""

    name: str
    description: str
    tool_type: ToolType
    input_parameters: Dict[str, Dict[str, Any]]
    output_format: Dict[str, Any]
    capabilities: List[ToolCapability]
    dependencies: List[str]
    security_requirements: List[str]
    performance_requirements: Dict[str, Any]
    validation_rules: List[str]
    examples: List[Dict[str, Any]]
    constraints: List[str]


@dataclass
class SecurityAssessment:
    """Security assessment results for tool generation"""

    risk_level: SecurityRisk
    identified_risks: List[str]
    mitigation_strategies: List[str]
    safe_to_generate: bool
    restrictions: List[str]
    recommendations: List[str]


@dataclass
class DependencyAnalysis:
    """Analysis of tool dependencies and compatibility"""

    required_dependencies: List[Dict[str, Any]]
    optional_dependencies: List[Dict[str, Any]]
    conflicts: List[str]
    compatibility_issues: List[str]
    installation_requirements: List[str]
    system_requirements: List[str]


@dataclass
class IntentClassification:
    """Classification of user intent for tool generation"""

    primary_intent: str
    confidence: float
    secondary_intents: List[str]
    tool_category: ToolType
    complexity_level: str  # simple, moderate, complex
    estimated_effort: str  # low, medium, high


class RequirementAnalyzer:
    """Natural language processing for tool description analysis"""

    def __init__(self):
        self.intent_patterns = self._initialize_intent_patterns()
        self.parameter_patterns = self._initialize_parameter_patterns()
        self.security_keywords = self._initialize_security_keywords()
        self.dependency_patterns = self._initialize_dependency_patterns()

    def _initialize_intent_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for intent classification"""
        return {
            "data_processing": [
                r"process\s+data",
                r"transform\s+data",
                r"clean\s+data",
                r"filter\s+data",
                r"parse\s+(?:csv|json|xml)",
                r"convert\s+(?:format|data)",
                r"aggregate\s+data",
            ],
            "api_client": [
                r"call\s+(?:api|endpoint)",
                r"fetch\s+from\s+(?:api|url)",
                r"http\s+(?:get|post|put|delete)",
                r"rest\s+api",
                r"web\s+service",
                r"api\s+request",
            ],
            "file_handler": [
                r"read\s+file",
                r"write\s+file",
                r"save\s+to\s+file",
                r"load\s+from\s+file",
                r"file\s+operations",
                r"directory\s+operations",
                r"file\s+management",
            ],
            "calculator": [
                r"calculate",
                r"compute",
                r"mathematical\s+operations",
                r"formula",
                r"arithmetic",
                r"statistics",
                r"numerical\s+analysis",
            ],
            "validator": [
                r"validate",
                r"verify",
                r"check\s+(?:format|syntax)",
                r"ensure\s+(?:compliance|correctness)",
                r"validation\s+rules",
                r"data\s+integrity",
            ],
            "transformer": [
                r"transform",
                r"convert",
                r"translate",
                r"format\s+conversion",
                r"data\s+transformation",
                r"structure\s+conversion",
            ],
            "analyzer": [
                r"analyze",
                r"examine",
                r"inspect",
                r"evaluate",
                r"assess",
                r"pattern\s+recognition",
                r"data\s+analysis",
                r"statistical\s+analysis",
            ],
            "generator": [
                r"generate",
                r"create",
                r"produce",
                r"build",
                r"construct",
                r"code\s+generation",
                r"content\s+generation",
                r"report\s+generation",
            ],
        }

    def _initialize_parameter_patterns(self) -> Dict[str, str]:
        """Initialize patterns for parameter extraction"""
        return {
            "input_file": r"(?:input|source|from)\s+file\s*(?:path|name)?",
            "output_file": r"(?:output|target|to)\s+file\s*(?:path|name)?",
            "format": r"(?:format|type|extension)\s*(?:is|should\s+be|:)?\s*(\w+)",
            "url": r"(?:url|endpoint|api)\s*(?:is|:)?\s*(https?://\S+)",
            "threshold": r"threshold\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)",
            "limit": r"limit\s*(?:is|of|to|:)?\s*(\d+)",
            "timeout": r"timeout\s*(?:is|of|:)?\s*(\d+)\s*(?:seconds?|s)?",
            "encoding": r"encoding\s*(?:is|:)?\s*(\w+)",
            "delimiter": r"delimiter\s*(?:is|:)?\s*(['\"]?.?['\"]?)",
        }

    def _initialize_security_keywords(self) -> Dict[str, SecurityRisk]:
        """Initialize security-related keywords and their risk levels"""
        return {
            # High risk keywords
            "execute": SecurityRisk.HIGH,
            "eval": SecurityRisk.CRITICAL,
            "exec": SecurityRisk.CRITICAL,
            "subprocess": SecurityRisk.HIGH,
            "shell": SecurityRisk.HIGH,
            "system": SecurityRisk.HIGH,
            "os.system": SecurityRisk.CRITICAL,
            "pickle": SecurityRisk.HIGH,
            "marshal": SecurityRisk.HIGH,
            "compile": SecurityRisk.MEDIUM,
            # Medium risk keywords
            "file": SecurityRisk.MEDIUM,
            "network": SecurityRisk.MEDIUM,
            "http": SecurityRisk.MEDIUM,
            "database": SecurityRisk.MEDIUM,
            "sql": SecurityRisk.MEDIUM,
            "password": SecurityRisk.MEDIUM,
            "credential": SecurityRisk.MEDIUM,
            "token": SecurityRisk.MEDIUM,
            # Low risk keywords
            "read": SecurityRisk.LOW,
            "write": SecurityRisk.LOW,
            "parse": SecurityRisk.LOW,
            "format": SecurityRisk.LOW,
            "calculate": SecurityRisk.LOW,
        }

    def _initialize_dependency_patterns(self) -> Dict[str, DependencyType]:
        """Initialize patterns for dependency detection"""
        return {
            # Standard library modules
            r"\b(?:os|sys|json|csv|re|datetime|math|random|urllib|http)\b": DependencyType.STANDARD_LIBRARY,
            # Common third-party packages
            r"\b(?:requests|pandas|numpy|scipy|matplotlib|pillow|beautifulsoup4)\b": DependencyType.THIRD_PARTY,
            # System dependencies
            r"\b(?:curl|wget|git|docker|ssh|ftp)\b": DependencyType.SYSTEM,
            # Custom/local imports
            r"from\s+(?:src|app|local)\.|import\s+(?:src|app|local)\.": DependencyType.CUSTOM,
        }

    async def analyze_requirements(self, description: str) -> ToolRequirement:
        """Analyze natural language description and extract structured requirements"""

        logger.info("Analyzing tool requirements", description_length=len(description))

        # Classify intent
        intent_classification = await self._classify_intent(description)

        # Extract parameters
        parameters = await self._extract_parameters(description)

        # Determine output format
        output_format = await self._determine_output_format(
            description, intent_classification
        )

        # Extract capabilities
        capabilities = await self._extract_capabilities(
            description, intent_classification
        )

        # Identify dependencies
        dependencies = await self._identify_dependencies(description)

        # Extract security requirements
        security_requirements = await self._extract_security_requirements(description)

        # Determine performance requirements
        performance_requirements = await self._extract_performance_requirements(
            description
        )

        # Extract validation rules
        validation_rules = await self._extract_validation_rules(description)

        # Generate examples
        examples = await self._generate_examples(description, parameters)

        # Extract constraints
        constraints = await self._extract_constraints(description)

        # Generate tool name
        tool_name = await self._generate_tool_name(description, intent_classification)

        requirement = ToolRequirement(
            name=tool_name,
            description=description,
            tool_type=intent_classification.tool_category,
            input_parameters=parameters,
            output_format=output_format,
            capabilities=capabilities,
            dependencies=dependencies,
            security_requirements=security_requirements,
            performance_requirements=performance_requirements,
            validation_rules=validation_rules,
            examples=examples,
            constraints=constraints,
        )

        logger.info(
            "Requirements analysis completed",
            tool_name=tool_name,
            tool_type=intent_classification.tool_category.value,
            parameter_count=len(parameters),
            dependency_count=len(dependencies),
        )

        return requirement

    async def _classify_intent(self, description: str) -> IntentClassification:
        """Classify user intent from description"""

        description_lower = description.lower()
        intent_scores = {}

        # Score each intent category
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, description_lower))
                score += matches

            if score > 0:
                intent_scores[intent] = score

        if not intent_scores:
            # Default classification
            primary_intent = "generator"
            confidence = 0.3
            tool_category = ToolType.GENERATOR
        else:
            # Get primary intent
            primary_intent = max(intent_scores, key=intent_scores.get)
            max_score = intent_scores[primary_intent]
            total_score = sum(intent_scores.values())
            confidence = max_score / total_score if total_score > 0 else 0.5

            # Map to tool category
            tool_category_mapping = {
                "data_processing": ToolType.DATA_PROCESSOR,
                "api_client": ToolType.API_CLIENT,
                "file_handler": ToolType.FILE_HANDLER,
                "calculator": ToolType.CALCULATOR,
                "validator": ToolType.VALIDATOR,
                "transformer": ToolType.TRANSFORMER,
                "analyzer": ToolType.ANALYZER,
                "generator": ToolType.GENERATOR,
            }
            tool_category = tool_category_mapping.get(
                primary_intent, ToolType.GENERATOR
            )

        # Get secondary intents
        secondary_intents = [
            intent
            for intent, score in sorted(
                intent_scores.items(), key=lambda x: x[1], reverse=True
            )[1:3]
            if score > 0
        ]

        # Determine complexity level
        complexity_indicators = len(
            re.findall(
                r"\b(?:and|also|then|after|before|if|when|while)\b", description_lower
            )
        )
        if complexity_indicators >= 3:
            complexity_level = "complex"
        elif complexity_indicators >= 1:
            complexity_level = "moderate"
        else:
            complexity_level = "simple"

        # Estimate effort
        word_count = len(description.split())
        if word_count > 50 or complexity_level == "complex":
            estimated_effort = "high"
        elif word_count > 20 or complexity_level == "moderate":
            estimated_effort = "medium"
        else:
            estimated_effort = "low"

        return IntentClassification(
            primary_intent=primary_intent,
            confidence=confidence,
            secondary_intents=secondary_intents,
            tool_category=tool_category,
            complexity_level=complexity_level,
            estimated_effort=estimated_effort,
        )

    async def _extract_parameters(self, description: str) -> Dict[str, Dict[str, Any]]:
        """Extract input parameters from description"""

        parameters = {}
        description_lower = description.lower()

        # Extract using patterns
        for param_name, pattern in self.parameter_patterns.items():
            matches = re.findall(pattern, description_lower)
            if matches:
                param_type = self._infer_parameter_type(
                    param_name, matches[0] if matches else ""
                )
                parameters[param_name] = {
                    "type": param_type,
                    "required": True,
                    "description": f"The {param_name.replace('_', ' ')} parameter",
                    "example": matches[0] if matches else None,
                }

        # Extract generic parameters mentioned in text
        generic_patterns = [
            r"(?:parameter|param|argument|arg)\s+(?:called\s+)?(\w+)",
            r"(?:input|provide|specify|give)\s+(?:the\s+)?(\w+)",
            r"(\w+)\s+(?:parameter|param|argument|value)",
        ]

        for pattern in generic_patterns:
            matches = re.findall(pattern, description_lower)
            for match in matches:
                if match not in parameters and len(match) > 2:  # Avoid single letters
                    param_type = self._infer_parameter_type(match, "")
                    parameters[match] = {
                        "type": param_type,
                        "required": False,
                        "description": f"The {match} parameter",
                        "example": None,
                    }

        # Ensure at least one input parameter
        if not parameters:
            parameters["input_data"] = {
                "type": "string",
                "required": True,
                "description": "The input data to process",
                "example": "sample input",
            }

        return parameters

    def _infer_parameter_type(self, param_name: str, value: str) -> str:
        """Infer parameter type from name and value"""

        # Type inference based on parameter name
        if any(
            keyword in param_name.lower() for keyword in ["file", "path", "name", "url"]
        ):
            return "string"
        elif any(
            keyword in param_name.lower()
            for keyword in ["count", "limit", "size", "number"]
        ):
            return "integer"
        elif any(
            keyword in param_name.lower()
            for keyword in ["rate", "threshold", "percentage", "ratio"]
        ):
            return "number"
        elif any(
            keyword in param_name.lower()
            for keyword in ["enable", "disable", "flag", "is_", "has_"]
        ):
            return "boolean"
        elif any(
            keyword in param_name.lower() for keyword in ["list", "array", "items"]
        ):
            return "array"
        elif any(
            keyword in param_name.lower()
            for keyword in ["config", "settings", "options"]
        ):
            return "object"

        # Type inference based on value
        if value:
            if value.isdigit():
                return "integer"
            elif re.match(r"^\d+\.\d+$", value):
                return "number"
            elif value.lower() in ["true", "false", "yes", "no"]:
                return "boolean"
            elif value.startswith(("http://", "https://", "ftp://")):
                return "string"

        return "string"  # Default type

    async def _determine_output_format(
        self, description: str, intent: IntentClassification
    ) -> Dict[str, Any]:
        """Determine expected output format"""

        description_lower = description.lower()

        # Check for explicit format mentions
        format_patterns = {
            "json": r"\bjson\b",
            "csv": r"\bcsv\b",
            "xml": r"\bxml\b",
            "text": r"\btext\b|\bstring\b",
            "file": r"save\s+to\s+file|write\s+to\s+file",
            "boolean": r"\btrue\b|\bfalse\b|\byes\b|\bno\b|\bsuccess\b|\bfail",
            "number": r"\bnumber\b|\bcount\b|\bscore\b|\bvalue\b",
            "list": r"\blist\b|\barray\b|\bmultiple\b",
        }

        detected_formats = []
        for format_type, pattern in format_patterns.items():
            if re.search(pattern, description_lower):
                detected_formats.append(format_type)

        # Default format based on tool type
        if not detected_formats:
            type_defaults = {
                ToolType.DATA_PROCESSOR: "object",
                ToolType.API_CLIENT: "object",
                ToolType.FILE_HANDLER: "boolean",
                ToolType.CALCULATOR: "number",
                ToolType.VALIDATOR: "boolean",
                ToolType.TRANSFORMER: "object",
                ToolType.ANALYZER: "object",
                ToolType.GENERATOR: "string",
            }
            detected_formats = [type_defaults.get(intent.tool_category, "object")]

        primary_format = detected_formats[0]

        return {
            "type": primary_format,
            "description": f"Tool output in {primary_format} format",
            "schema": self._generate_output_schema(primary_format, intent),
        }

    def _generate_output_schema(
        self, format_type: str, intent: IntentClassification
    ) -> Dict[str, Any]:
        """Generate output schema based on format type and intent"""

        if format_type == "object":
            return {
                "type": "object",
                "properties": {
                    "result": {"type": "string", "description": "Main result"},
                    "status": {"type": "string", "description": "Operation status"},
                    "metadata": {
                        "type": "object",
                        "description": "Additional metadata",
                    },
                },
            }
        elif format_type == "list":
            return {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of results",
            }
        elif format_type == "boolean":
            return {"type": "boolean", "description": "Success/failure indicator"}
        elif format_type == "number":
            return {"type": "number", "description": "Numerical result"}
        else:
            return {"type": "string", "description": "Text result"}

    async def _extract_capabilities(
        self, description: str, intent: IntentClassification
    ) -> List[ToolCapability]:
        """Extract tool capabilities from description"""

        capabilities = []
        description_lower = description.lower()

        # Map tool types to capabilities
        type_capability_mapping = {
            ToolType.DATA_PROCESSOR: [
                ToolCapability.DATA_ANALYSIS,
                ToolCapability.TRANSFORMATION,
            ],
            ToolType.API_CLIENT: [ToolCapability.COMMUNICATION],
            ToolType.FILE_HANDLER: [ToolCapability.TRANSFORMATION],
            ToolType.CALCULATOR: [ToolCapability.DATA_ANALYSIS],
            ToolType.VALIDATOR: [ToolCapability.VALIDATION],
            ToolType.TRANSFORMER: [ToolCapability.TRANSFORMATION],
            ToolType.ANALYZER: [ToolCapability.DATA_ANALYSIS],
            ToolType.GENERATOR: [ToolCapability.CONTENT_GENERATION],
        }

        # Add primary capabilities based on tool type
        primary_capabilities = type_capability_mapping.get(intent.tool_category, [])
        capabilities.extend(primary_capabilities)

        # Check for additional capabilities mentioned in description
        capability_keywords = {
            ToolCapability.KNOWLEDGE_RETRIEVAL: [
                "search",
                "find",
                "lookup",
                "retrieve",
                "query",
            ],
            ToolCapability.CONTENT_GENERATION: [
                "generate",
                "create",
                "produce",
                "build",
            ],
            ToolCapability.CODE_GENERATION: ["code", "script", "program", "function"],
            ToolCapability.DOCUMENT_GENERATION: ["document", "report", "pdf", "docx"],
            ToolCapability.DATA_ANALYSIS: [
                "analyze",
                "statistics",
                "metrics",
                "insights",
            ],
            ToolCapability.COMMUNICATION: ["send", "email", "notify", "message"],
            ToolCapability.VALIDATION: ["validate", "verify", "check", "ensure"],
            ToolCapability.TRANSFORMATION: ["convert", "transform", "format", "parse"],
        }

        for capability, keywords in capability_keywords.items():
            if capability not in capabilities:
                for keyword in keywords:
                    if keyword in description_lower:
                        capabilities.append(capability)
                        break

        return list(set(capabilities))  # Remove duplicates

    async def _identify_dependencies(self, description: str) -> List[str]:
        """Identify required dependencies from description"""

        dependencies = []
        description_lower = description.lower()

        # Common dependency mappings
        dependency_mappings = {
            "csv": ["csv"],
            "json": ["json"],
            "xml": ["xml.etree.ElementTree"],
            "http": ["requests"],
            "api": ["requests"],
            "database": ["sqlite3"],
            "sql": ["sqlite3"],
            "excel": ["openpyxl"],
            "pdf": ["pypdf"],
            "image": ["Pillow"],
            "plot": ["matplotlib"],
            "chart": ["matplotlib"],
            "data analysis": ["pandas", "numpy"],
            "statistics": ["scipy", "numpy"],
            "machine learning": ["scikit-learn"],
            "web scraping": ["beautifulsoup4", "requests"],
            "email": ["smtplib", "email"],
            "zip": ["zipfile"],
            "regex": ["re"],
            "date": ["datetime"],
            "math": ["math"],
            "random": ["random"],
            "shell command": ["subprocess", "os"],
            "execute command": ["subprocess", "os"],
            "run command": ["subprocess", "os"],
            "system command": ["subprocess", "os"],
        }

        for keyword, deps in dependency_mappings.items():
            if keyword in description_lower:
                dependencies.extend(deps)

        # Remove duplicates and sort
        dependencies = sorted(list(set(dependencies)))

        return dependencies

    async def _extract_security_requirements(self, description: str) -> List[str]:
        """Extract security requirements from description"""

        security_requirements = []
        description_lower = description.lower()

        # Security requirement patterns
        security_patterns = {
            "input_validation": ["validate", "sanitize", "check input", "verify input"],
            "output_sanitization": ["sanitize output", "clean output", "safe output"],
            "access_control": ["permission", "access", "authorize", "authenticate"],
            "data_protection": ["encrypt", "secure", "protect", "privacy"],
            "error_handling": ["error handling", "exception", "try catch", "safe"],
            "logging": ["log", "audit", "track", "monitor"],
        }

        for requirement, keywords in security_patterns.items():
            for keyword in keywords:
                if keyword in description_lower:
                    security_requirements.append(requirement)
                    break

        # Default security requirements
        if not security_requirements:
            security_requirements = ["input_validation", "error_handling"]

        return security_requirements

    async def _extract_performance_requirements(
        self, description: str
    ) -> Dict[str, Any]:
        """Extract performance requirements from description"""

        performance_requirements = {}
        description_lower = description.lower()

        # Extract timeout requirements
        timeout_match = re.search(
            r"timeout\s*(?:of|is|:)?\s*(\d+)\s*(?:seconds?|s)?", description_lower
        )
        if timeout_match:
            performance_requirements["timeout"] = int(timeout_match.group(1))
        else:
            performance_requirements["timeout"] = 30  # Default timeout

        # Extract memory requirements
        memory_match = re.search(
            r"memory\s*(?:limit|requirement|usage)\s*(?:of|is|:)?\s*(\d+)\s*(?:mb|gb)?",
            description_lower,
        )
        if memory_match:
            memory_value = int(memory_match.group(1))
            if "gb" in description_lower:
                memory_value *= 1024
            performance_requirements["memory_limit_mb"] = memory_value
        else:
            performance_requirements["memory_limit_mb"] = 512  # Default memory limit

        # Extract throughput requirements
        throughput_patterns = [
            r"process\s*(\d+)\s*(?:items?|records?|files?)\s*per\s*(?:second|minute)",
            r"handle\s*(\d+)\s*(?:requests?|operations?)\s*per\s*(?:second|minute)",
        ]

        for pattern in throughput_patterns:
            match = re.search(pattern, description_lower)
            if match:
                performance_requirements["throughput"] = int(match.group(1))
                break

        if "throughput" not in performance_requirements:
            performance_requirements["throughput"] = 10  # Default throughput

        return performance_requirements

    async def _extract_validation_rules(self, description: str) -> List[str]:
        """Extract validation rules from description"""

        validation_rules = []
        description_lower = description.lower()

        # Common validation patterns
        validation_patterns = {
            "required_parameters": r"(?:required|mandatory|must\s+provide)\s+(\w+)",
            "format_validation": r"(?:format|type)\s+(?:must\s+be|should\s+be|is)\s+(\w+)",
            "range_validation": r"(?:between|from)\s+(\d+)\s+(?:to|and)\s+(\d+)",
            "length_validation": r"(?:length|size)\s+(?:must\s+be|should\s+be)\s+(?:at\s+least\s+)?(\d+)",
            "pattern_validation": r"(?:match|follow)\s+(?:pattern|format|regex)",
        }

        for rule_type, pattern in validation_patterns.items():
            if re.search(pattern, description_lower):
                validation_rules.append(rule_type)

        # Default validation rules
        if not validation_rules:
            validation_rules = ["required_parameters", "type_validation"]

        return validation_rules

    async def _generate_examples(
        self, description: str, parameters: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate usage examples based on description and parameters"""

        examples = []

        # Generate basic example
        example_input = {}
        for param_name, param_info in parameters.items():
            if param_info.get("example"):
                example_input[param_name] = param_info["example"]
            else:
                example_input[param_name] = self._generate_example_value(
                    param_info["type"]
                )

        examples.append(
            {
                "description": "Basic usage example",
                "input": example_input,
                "expected_output": "Expected result based on input",
            }
        )

        # Generate edge case example if applicable
        if len(parameters) > 1:
            edge_case_input = {}
            for param_name, param_info in parameters.items():
                edge_case_input[param_name] = self._generate_edge_case_value(
                    param_info["type"]
                )

            examples.append(
                {
                    "description": "Edge case example",
                    "input": edge_case_input,
                    "expected_output": "Expected result for edge case",
                }
            )

        return examples

    def _generate_example_value(self, param_type: str) -> Any:
        """Generate example value for parameter type"""

        type_examples = {
            "string": "example_string",
            "integer": 42,
            "number": 3.14,
            "boolean": True,
            "array": ["item1", "item2"],
            "object": {"key": "value"},
        }

        return type_examples.get(param_type, "example_value")

    def _generate_edge_case_value(self, param_type: str) -> Any:
        """Generate edge case value for parameter type"""

        edge_cases = {
            "string": "",
            "integer": 0,
            "number": 0.0,
            "boolean": False,
            "array": [],
            "object": {},
        }

        return edge_cases.get(param_type, None)

    async def _extract_constraints(self, description: str) -> List[str]:
        """Extract constraints from description"""

        constraints = []
        description_lower = description.lower()

        # Constraint patterns
        constraint_patterns = {
            "file_size_limit": r"file\s+size\s+(?:limit|maximum|max)\s+(\d+)\s*(?:mb|gb|kb)?",
            "processing_time_limit": r"(?:processing|execution)\s+time\s+(?:limit|maximum|max)\s+(\d+)\s*(?:seconds?|minutes?)?",
            "memory_limit": r"memory\s+(?:limit|usage|maximum|max)\s+(\d+)\s*(?:mb|gb)?",
            "concurrent_limit": r"(?:concurrent|parallel|simultaneous)\s+(?:limit|maximum|max)\s+(\d+)",
            "rate_limit": r"rate\s+limit\s+(\d+)\s*(?:per\s+(?:second|minute|hour))?",
            "format_restriction": r"only\s+(?:support|accept|handle)\s+(\w+)\s+(?:format|files?)",
            "platform_restriction": r"(?:only|works?\s+on)\s+(windows|linux|mac|unix)",
        }

        for constraint_type, pattern in constraint_patterns.items():
            match = re.search(pattern, description_lower)
            if match:
                constraints.append(f"{constraint_type}: {match.group(1)}")

        # General constraints
        if "read-only" in description_lower or "readonly" in description_lower:
            constraints.append("read_only_access")

        if "no network" in description_lower or "offline" in description_lower:
            constraints.append("no_network_access")

        if "secure" in description_lower or "encrypted" in description_lower:
            constraints.append("security_required")

        return constraints

    async def _generate_tool_name(
        self, description: str, intent: IntentClassification
    ) -> str:
        """Generate appropriate tool name from description and intent"""

        # Extract key words from description
        words = re.findall(r"\b[a-zA-Z]{3,}\b", description.lower())

        # Filter out common words
        stop_words = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "will",
            "can",
            "should",
            "would",
            "could",
            "need",
            "want",
            "have",
            "has",
            "are",
            "was",
            "were",
            "been",
            "tool",
            "function",
            "method",
            "create",
            "make",
            "build",
            "generate",
        }

        key_words = [word for word in words if word not in stop_words][:3]

        # Create base name
        if key_words:
            base_name = "_".join(key_words)
        else:
            base_name = intent.tool_category.value.replace("_", "")

        # Add tool suffix
        tool_name = f"{base_name}_tool"

        # Ensure valid Python identifier
        tool_name = re.sub(r"[^a-zA-Z0-9_]", "_", tool_name)
        tool_name = re.sub(r"_+", "_", tool_name)
        tool_name = tool_name.strip("_")

        # Ensure it starts with a letter
        if tool_name and tool_name[0].isdigit():
            tool_name = f"tool_{tool_name}"

        return tool_name or "custom_tool"


class SecurityAnalyzer:
    """Security assessment for tool generation requirements"""

    def __init__(self):
        self.risk_patterns = self._initialize_risk_patterns()
        self.mitigation_strategies = self._initialize_mitigation_strategies()

    def _initialize_risk_patterns(self) -> Dict[SecurityRisk, List[str]]:
        """Initialize security risk patterns"""
        return {
            SecurityRisk.CRITICAL: [
                r"\beval\b",
                r"\bexec\b",
                r"os\.system",
                r"subprocess\.call",
                r"__import__",
                r"compile\(",
                r"pickle\.loads",
                r"marshal\.loads",
                r"execute.*shell.*command",
                r"run.*shell.*command",
                r"shell.*execution",
            ],
            SecurityRisk.HIGH: [
                r"\bsubprocess\b",
                r"\bshell=True\b",
                r"os\.popen",
                r"os\.spawn",
                r"file\s+write",
                r"file\s+delete",
                r"directory\s+delete",
                r"network\s+access",
                r"database\s+write",
                r"sql\s+execute",
                r"executes?\s+(?:shell\s+)?commands?",
                r"run\s+(?:shell\s+)?commands?",
                r"command\s+execution",
                r"system\s+commands?",
            ],
            SecurityRisk.MEDIUM: [
                r"file\s+read",
                r"http\s+request",
                r"api\s+call",
                r"database\s+read",
                r"environment\s+variable",
                r"configuration\s+file",
                r"user\s+input",
            ],
            SecurityRisk.LOW: [
                r"string\s+manipulation",
                r"data\s+parsing",
                r"mathematical\s+calculation",
                r"format\s+conversion",
                r"validation",
                r"analysis",
            ],
        }

    def _initialize_mitigation_strategies(self) -> Dict[SecurityRisk, List[str]]:
        """Initialize mitigation strategies for each risk level"""
        return {
            SecurityRisk.CRITICAL: [
                "Prohibit dynamic code execution",
                "Use safe alternatives to eval/exec",
                "Implement strict input validation",
                "Use sandboxed execution environment",
                "Require explicit user approval",
            ],
            SecurityRisk.HIGH: [
                "Implement input sanitization",
                "Use parameterized queries",
                "Restrict file system access",
                "Validate all external inputs",
                "Implement access controls",
            ],
            SecurityRisk.MEDIUM: [
                "Validate input formats",
                "Implement rate limiting",
                "Use secure communication protocols",
                "Log all operations",
                "Implement timeout controls",
            ],
            SecurityRisk.LOW: [
                "Basic input validation",
                "Error handling",
                "Resource limits",
                "Standard logging",
            ],
        }

    async def assess_security(self, requirement: ToolRequirement) -> SecurityAssessment:
        """Perform comprehensive security assessment"""

        logger.info("Performing security assessment", tool_name=requirement.name)

        # Analyze description for security risks
        description_risks = await self._analyze_description_risks(
            requirement.description
        )

        # Analyze dependencies for security risks
        dependency_risks = await self._analyze_dependency_risks(
            requirement.dependencies
        )

        # Analyze capabilities for security implications
        capability_risks = await self._analyze_capability_risks(
            requirement.capabilities
        )

        # Combine all risks
        all_risks = description_risks + dependency_risks + capability_risks

        # Determine overall risk level
        risk_level = self._determine_overall_risk_level(all_risks)

        # Generate mitigation strategies
        mitigation_strategies = self._generate_mitigation_strategies(
            risk_level, all_risks
        )

        # Determine if safe to generate
        safe_to_generate = risk_level != SecurityRisk.CRITICAL

        # Generate restrictions
        restrictions = self._generate_restrictions(risk_level, all_risks)

        # Generate recommendations
        recommendations = self._generate_recommendations(risk_level, all_risks)

        assessment = SecurityAssessment(
            risk_level=risk_level,
            identified_risks=all_risks,
            mitigation_strategies=mitigation_strategies,
            safe_to_generate=safe_to_generate,
            restrictions=restrictions,
            recommendations=recommendations,
        )

        logger.info(
            "Security assessment completed",
            tool_name=requirement.name,
            risk_level=risk_level.value,
            safe_to_generate=safe_to_generate,
            risk_count=len(all_risks),
        )

        return assessment

    async def _analyze_description_risks(self, description: str) -> List[str]:
        """Analyze description text for security risks"""

        risks = []
        description_lower = description.lower()

        for risk_level, patterns in self.risk_patterns.items():
            for pattern in patterns:
                if re.search(pattern, description_lower):
                    risks.append(f"{risk_level.value}: {pattern}")

        return risks

    async def _analyze_dependency_risks(self, dependencies: List[str]) -> List[str]:
        """Analyze dependencies for security risks"""

        risks = []

        high_risk_deps = {
            "subprocess",
            "os",
            "sys",
            "pickle",
            "marshal",
            "eval",
            "exec",
        }

        medium_risk_deps = {
            "requests",
            "urllib",
            "http",
            "socket",
            "sqlite3",
            "mysql",
            "psycopg2",
        }

        for dep in dependencies:
            if dep in high_risk_deps:
                risks.append(f"high: dependency {dep}")
            elif dep in medium_risk_deps:
                risks.append(f"medium: dependency {dep}")

        return risks

    async def _analyze_capability_risks(
        self, capabilities: List[ToolCapability]
    ) -> List[str]:
        """Analyze capabilities for security implications"""

        risks = []

        capability_risks = {
            ToolCapability.CODE_GENERATION: SecurityRisk.HIGH,
            ToolCapability.COMMUNICATION: SecurityRisk.MEDIUM,
            ToolCapability.TRANSFORMATION: SecurityRisk.LOW,
            ToolCapability.DATA_ANALYSIS: SecurityRisk.LOW,
            ToolCapability.VALIDATION: SecurityRisk.LOW,
        }

        for capability in capabilities:
            if capability in capability_risks:
                risk_level = capability_risks[capability]
                risks.append(f"{risk_level.value}: capability {capability.value}")

        return risks

    def _determine_overall_risk_level(self, risks: List[str]) -> SecurityRisk:
        """Determine overall risk level from individual risks"""

        if any("critical" in risk for risk in risks):
            return SecurityRisk.CRITICAL
        elif any("high" in risk for risk in risks):
            return SecurityRisk.HIGH
        elif any("medium" in risk for risk in risks):
            return SecurityRisk.MEDIUM
        else:
            return SecurityRisk.LOW

    def _generate_mitigation_strategies(
        self, risk_level: SecurityRisk, risks: List[str]
    ) -> List[str]:
        """Generate appropriate mitigation strategies"""

        strategies = self.mitigation_strategies.get(risk_level, [])

        # Add specific strategies based on identified risks
        if any("subprocess" in risk for risk in risks):
            strategies.append(
                "Use subprocess with shell=False and validate all arguments"
            )

        if any("file" in risk for risk in risks):
            strategies.append(
                "Implement file path validation and restrict access to safe directories"
            )

        if any("network" in risk for risk in risks):
            strategies.append("Validate URLs and implement request timeouts")

        if any("database" in risk for risk in risks):
            strategies.append("Use parameterized queries and validate all inputs")

        return list(set(strategies))  # Remove duplicates

    def _generate_restrictions(
        self, risk_level: SecurityRisk, risks: List[str]
    ) -> List[str]:
        """Generate restrictions based on risk level"""

        restrictions = []

        if risk_level == SecurityRisk.CRITICAL:
            restrictions.extend(
                [
                    "Tool generation prohibited",
                    "Manual review required",
                    "Alternative approach recommended",
                ]
            )
        elif risk_level == SecurityRisk.HIGH:
            restrictions.extend(
                [
                    "Sandboxed execution required",
                    "Input validation mandatory",
                    "Output sanitization required",
                    "User approval required",
                ]
            )
        elif risk_level == SecurityRisk.MEDIUM:
            restrictions.extend(
                [
                    "Input validation required",
                    "Resource limits enforced",
                    "Logging mandatory",
                ]
            )
        else:
            restrictions.extend(
                ["Basic validation required", "Error handling mandatory"]
            )

        return restrictions

    def _generate_recommendations(
        self, risk_level: SecurityRisk, risks: List[str]
    ) -> List[str]:
        """Generate security recommendations"""

        recommendations = []

        if risk_level in [SecurityRisk.CRITICAL, SecurityRisk.HIGH]:
            recommendations.extend(
                [
                    "Consider using safer alternatives",
                    "Implement comprehensive testing",
                    "Regular security audits",
                    "User education on risks",
                ]
            )

        recommendations.extend(
            [
                "Follow secure coding practices",
                "Implement proper error handling",
                "Use input validation",
                "Regular dependency updates",
            ]
        )

        return recommendations


class DependencyAnalyzer:
    """Dependency analysis and compatibility checking"""

    def __init__(self):
        self.standard_library = self._get_standard_library_modules()
        self.known_packages = self._initialize_known_packages()

    def _get_standard_library_modules(self) -> Set[str]:
        """Get set of Python standard library modules"""
        return {
            "os",
            "sys",
            "json",
            "csv",
            "re",
            "datetime",
            "math",
            "random",
            "urllib",
            "http",
            "email",
            "zipfile",
            "tarfile",
            "gzip",
            "sqlite3",
            "xml",
            "html",
            "base64",
            "hashlib",
            "hmac",
            "collections",
            "itertools",
            "functools",
            "operator",
            "pathlib",
            "tempfile",
            "shutil",
            "glob",
            "fnmatch",
            "subprocess",
            "threading",
            "multiprocessing",
            "asyncio",
            "logging",
            "unittest",
            "doctest",
            "argparse",
            "configparser",
        }

    def _initialize_known_packages(self) -> Dict[str, Dict[str, Any]]:
        """Initialize known third-party packages with metadata"""
        return {
            "requests": {
                "description": "HTTP library for Python",
                "category": "networking",
                "security_risk": "medium",
                "alternatives": ["urllib3", "httpx"],
            },
            "pandas": {
                "description": "Data manipulation and analysis library",
                "category": "data_science",
                "security_risk": "low",
                "alternatives": ["polars", "dask"],
            },
            "numpy": {
                "description": "Numerical computing library",
                "category": "data_science",
                "security_risk": "low",
                "alternatives": ["cupy", "jax"],
            },
            "beautifulsoup4": {
                "description": "HTML/XML parsing library",
                "category": "web_scraping",
                "security_risk": "low",
                "alternatives": ["lxml", "html.parser"],
            },
            "openpyxl": {
                "description": "Excel file manipulation",
                "category": "file_processing",
                "security_risk": "low",
                "alternatives": ["xlsxwriter", "xlrd"],
            },
            "pypdf": {
                "description": "PDF manipulation library",
                "category": "file_processing",
                "security_risk": "medium",
                "alternatives": ["pdfplumber", "pymupdf"],
            },
            "Pillow": {
                "description": "Image processing library",
                "category": "image_processing",
                "security_risk": "medium",
                "alternatives": ["opencv-python", "scikit-image"],
            },
        }

    async def analyze_dependencies(self, dependencies: List[str]) -> DependencyAnalysis:
        """Perform comprehensive dependency analysis"""

        logger.info("Analyzing dependencies", dependency_count=len(dependencies))

        required_deps = []
        optional_deps = []
        conflicts = []
        compatibility_issues = []
        installation_requirements = []
        system_requirements = []

        for dep in dependencies:
            dep_info = await self._analyze_single_dependency(dep)

            if dep_info["required"]:
                required_deps.append(dep_info)
            else:
                optional_deps.append(dep_info)

            # Check for conflicts
            dep_conflicts = await self._check_conflicts(dep, dependencies)
            conflicts.extend(dep_conflicts)

            # Check compatibility
            compatibility = await self._check_compatibility(dep)
            if compatibility["issues"]:
                compatibility_issues.extend(compatibility["issues"])

            # Installation requirements
            if dep_info["installation_notes"]:
                installation_requirements.extend(dep_info["installation_notes"])

            # System requirements
            if dep_info["system_requirements"]:
                system_requirements.extend(dep_info["system_requirements"])

        analysis = DependencyAnalysis(
            required_dependencies=required_deps,
            optional_dependencies=optional_deps,
            conflicts=list(set(conflicts)),
            compatibility_issues=list(set(compatibility_issues)),
            installation_requirements=list(set(installation_requirements)),
            system_requirements=list(set(system_requirements)),
        )

        logger.info(
            "Dependency analysis completed",
            required_count=len(required_deps),
            optional_count=len(optional_deps),
            conflicts=len(conflicts),
            issues=len(compatibility_issues),
        )

        return analysis

    async def _analyze_single_dependency(self, dependency: str) -> Dict[str, Any]:
        """Analyze a single dependency"""

        is_standard = dependency in self.standard_library
        is_known = dependency in self.known_packages

        dep_info = {
            "name": dependency,
            "type": (
                DependencyType.STANDARD_LIBRARY
                if is_standard
                else DependencyType.THIRD_PARTY
            ),
            "required": True,
            "version": None,
            "description": "",
            "installation_notes": [],
            "system_requirements": [],
            "security_risk": "low",
            "alternatives": [],
        }

        if is_known:
            package_info = self.known_packages[dependency]
            dep_info.update(
                {
                    "description": package_info["description"],
                    "security_risk": package_info["security_risk"],
                    "alternatives": package_info["alternatives"],
                }
            )

            # Add installation notes for known packages
            if dependency == "Pillow":
                dep_info["system_requirements"].append(
                    "Image processing libraries (libjpeg, libpng)"
                )
            elif dependency == "pandas":
                dep_info["installation_notes"].append(
                    "Large package, consider alternatives for simple use cases"
                )
            elif dependency == "requests":
                dep_info["installation_notes"].append(
                    "Most popular HTTP library, well maintained"
                )

        elif not is_standard:
            # Unknown third-party package
            dep_info.update(
                {
                    "description": f"Third-party package: {dependency}",
                    "security_risk": "unknown",
                    "installation_notes": [
                        "Unknown package - verify before installation"
                    ],
                }
            )

        return dep_info

    async def _check_conflicts(
        self, dependency: str, all_dependencies: List[str]
    ) -> List[str]:
        """Check for dependency conflicts"""

        conflicts = []

        # Known conflict patterns
        conflict_groups = [
            ["PIL", "Pillow"],  # PIL and Pillow conflict
            ["mysql-python", "PyMySQL", "mysqlclient"],  # MySQL drivers
            ["pycrypto", "pycryptodome"],  # Crypto libraries
        ]

        for group in conflict_groups:
            if dependency in group:
                conflicting = [
                    dep
                    for dep in all_dependencies
                    if dep in group and dep != dependency
                ]
                for conflict in conflicting:
                    conflicts.append(f"{dependency} conflicts with {conflict}")

        return conflicts

    async def _check_compatibility(self, dependency: str) -> Dict[str, Any]:
        """Check compatibility issues for dependency"""

        compatibility = {"python_version": None, "platform": None, "issues": []}

        # Known compatibility issues
        compatibility_issues = {
            "pypdf": ["May have issues with newer PDF formats"],
            "Pillow": ["Requires system image libraries"],
            "pandas": ["Memory intensive for large datasets"],
            "numpy": ["May require compilation on some systems"],
            "lxml": ["Requires libxml2 and libxslt system libraries"],
        }

        if dependency in compatibility_issues:
            compatibility["issues"] = compatibility_issues[dependency]

        return compatibility


class CodeToolGenerator(BaseTool):
    """Main code tool generator implementing automated tool creation"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.requirement_analyzer = RequirementAnalyzer()
        self.security_analyzer = SecurityAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
        self.validation_framework = (
            None  # Lazy initialization to avoid circular imports
        )
        self.tool_registry = None  # Lazy initialization
        self.code_templates = self._initialize_code_templates()
        self.generated_tools: Dict[str, Dict[str, Any]] = {}

    def _initialize_code_templates(self) -> Dict[str, str]:
        """Initialize code templates for different tool types"""
        return {
            ToolType.DATA_PROCESSOR: '''
"""
{description}
Generated tool for data processing operations
"""

from typing import Dict, Any, List, Optional, Union
import json
import logging
{additional_imports}

from src.shared.logging import get_logger
from src.agent_server.tools.registry import BaseTool, ToolCapabilities, ResourceRequirements, ToolResult, ExecutionContext, ToolCapability

logger = get_logger(__name__)


class {class_name}(BaseTool):
    """Generated data processing tool"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.initialized = False
    
    async def execute(self, parameters: Dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Execute the data processing operation"""
        
        try:
            # Validate input parameters
            is_valid, validation_errors = self.validate_parameters(parameters)
            if not is_valid:
                return ToolResult(
                    data=None,
                    metadata={{"error": "Invalid parameters", "validation_errors": validation_errors}},
                    execution_time=0.0,
                    success=False,
                    error_message=f"Parameter validation failed: {{', '.join(validation_errors)}}"
                )
            
            start_time = time.time()
            
            # Extract parameters
{parameter_extraction}
            
            # Main processing logic
            result = await self._process_data({parameter_names})
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                data=result,
                metadata={{"tool_type": "data_processor", "execution_time": execution_time}},
                execution_time=execution_time,
                success=True,
                quality_score=1.0,
                confidence_score=0.9
            )
            
        except Exception as e:
            logger.error("Data processing failed", error=str(e))
            return ToolResult(
                data=None,
                metadata={{"error": "Processing failed"}},
                execution_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def _process_data(self, {parameter_signature}) -> Any:
        """Main data processing logic"""
        
        # TODO: Implement specific data processing logic
        # This is a template - actual implementation depends on requirements
        
        processed_data = {{
            "input_received": True,
            "processing_completed": True,
            "result": "Data processed successfully"
        }}
        
        return processed_data
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema"""
        return {{
            "name": "{tool_name}",
            "description": "{description}",
            "parameters": {parameter_schema},
            "required_params": {required_params},
            "output_schema": {output_schema}
        }}
    
    def get_capabilities(self) -> ToolCapabilities:
        """Get tool capabilities"""
        return ToolCapabilities(
            primary_capability=ToolCapability.DATA_ANALYSIS,
            secondary_capabilities=[ToolCapability.TRANSFORMATION],
            input_types=["object", "string", "array"],
            output_types=["object", "string"],
            supported_formats=["json", "csv", "text"]
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
''',
            ToolType.API_CLIENT: '''
"""
{description}
Generated tool for API client operations
"""

from typing import Dict, Any, List, Optional, Union
import json
import asyncio
import aiohttp
import time
{additional_imports}

from src.shared.logging import get_logger
from src.agent_server.tools.registry import BaseTool, ToolCapabilities, ResourceRequirements, ToolResult, ExecutionContext, ToolCapability

logger = get_logger(__name__)


class {class_name}(BaseTool):
    """Generated API client tool"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.session = None
        self.base_url = config.get("base_url", "") if config else ""
        self.timeout = config.get("timeout", 30) if config else 30
    
    async def initialize(self):
        """Initialize HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        await super().initialize()
    
    async def cleanup(self):
        """Cleanup HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
        await super().cleanup()
    
    async def execute(self, parameters: Dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Execute the API client operation"""
        
        try:
            # Ensure session is initialized
            if not self.session:
                await self.initialize()
            
            # Validate input parameters
            is_valid, validation_errors = self.validate_parameters(parameters)
            if not is_valid:
                return ToolResult(
                    data=None,
                    metadata={{"error": "Invalid parameters", "validation_errors": validation_errors}},
                    execution_time=0.0,
                    success=False,
                    error_message=f"Parameter validation failed: {{', '.join(validation_errors)}}"
                )
            
            start_time = time.time()
            
            # Extract parameters
{parameter_extraction}
            
            # Make API request
            result = await self._make_api_request({parameter_names})
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                data=result,
                metadata={{"tool_type": "api_client", "execution_time": execution_time}},
                execution_time=execution_time,
                success=True,
                quality_score=1.0,
                confidence_score=0.9
            )
            
        except Exception as e:
            logger.error("API request failed", error=str(e))
            return ToolResult(
                data=None,
                metadata={{"error": "API request failed"}},
                execution_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def _make_api_request(self, {parameter_signature}) -> Any:
        """Make API request"""
        
        # TODO: Implement specific API request logic
        # This is a template - actual implementation depends on requirements
        
        try:
            # Example GET request
            async with self.session.get(f"{{self.base_url}}/endpoint") as response:
                if response.status == 200:
                    data = await response.json()
                    return {{
                        "status": "success",
                        "data": data,
                        "status_code": response.status
                    }}
                else:
                    return {{
                        "status": "error",
                        "error": f"HTTP {{response.status}}",
                        "status_code": response.status
                    }}
        except Exception as e:
            return {{
                "status": "error",
                "error": str(e),
                "status_code": None
            }}
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema"""
        return {{
            "name": "{tool_name}",
            "description": "{description}",
            "parameters": {parameter_schema},
            "required_params": {required_params},
            "output_schema": {output_schema}
        }}
    
    def get_capabilities(self) -> ToolCapabilities:
        """Get tool capabilities"""
        return ToolCapabilities(
            primary_capability=ToolCapability.COMMUNICATION,
            secondary_capabilities=[ToolCapability.DATA_ANALYSIS],
            input_types=["object", "string"],
            output_types=["object"],
            supported_formats=["json", "xml", "text"]
        )
    
    def get_resource_requirements(self) -> ResourceRequirements:
        """Get resource requirements"""
        return ResourceRequirements(
            cpu_cores=0.5,
            memory_mb=256,
            network_bandwidth_mbps=10.0,
            storage_mb=50,
            max_execution_time=60,
            concurrent_limit=10
        )
''',
            ToolType.FILE_HANDLER: '''
"""
{description}
Generated tool for file handling operations
"""

from typing import Dict, Any, List, Optional, Union
import os
import json
import time
from pathlib import Path
{additional_imports}

from src.shared.logging import get_logger
from src.agent_server.tools.registry import BaseTool, ToolCapabilities, ResourceRequirements, ToolResult, ExecutionContext, ToolCapability

logger = get_logger(__name__)


class {class_name}(BaseTool):
    """Generated file handling tool"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.allowed_extensions = config.get("allowed_extensions", [".txt", ".json", ".csv"]) if config else [".txt", ".json", ".csv"]
        self.max_file_size = config.get("max_file_size_mb", 10) if config else 10
    
    async def execute(self, parameters: Dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Execute the file handling operation"""
        
        try:
            # Validate input parameters
            is_valid, validation_errors = self.validate_parameters(parameters)
            if not is_valid:
                return ToolResult(
                    data=None,
                    metadata={{"error": "Invalid parameters", "validation_errors": validation_errors}},
                    execution_time=0.0,
                    success=False,
                    error_message=f"Parameter validation failed: {{', '.join(validation_errors)}}"
                )
            
            start_time = time.time()
            
            # Extract parameters
{parameter_extraction}
            
            # Perform file operation
            result = await self._handle_file({parameter_names})
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                data=result,
                metadata={{"tool_type": "file_handler", "execution_time": execution_time}},
                execution_time=execution_time,
                success=True,
                quality_score=1.0,
                confidence_score=0.9
            )
            
        except Exception as e:
            logger.error("File handling failed", error=str(e))
            return ToolResult(
                data=None,
                metadata={{"error": "File handling failed"}},
                execution_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def _handle_file(self, {parameter_signature}) -> Any:
        """Handle file operation"""
        
        # TODO: Implement specific file handling logic
        # This is a template - actual implementation depends on requirements
        
        result = {{
            "operation": "file_handled",
            "success": True,
            "message": "File operation completed successfully"
        }}
        
        return result
    
    def _validate_file_path(self, file_path: str) -> bool:
        """Validate file path for security"""
        
        path = Path(file_path)
        
        # Check if path is absolute and within allowed directories
        if path.is_absolute():
            # Only allow files in specific directories
            allowed_dirs = ["/tmp", "/var/tmp", "./uploads", "./data"]
            if not any(str(path).startswith(allowed_dir) for allowed_dir in allowed_dirs):
                return False
        
        # Check file extension
        if path.suffix not in self.allowed_extensions:
            return False
        
        # Check file size if file exists
        if path.exists() and path.is_file():
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size:
                return False
        
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema"""
        return {{
            "name": "{tool_name}",
            "description": "{description}",
            "parameters": {parameter_schema},
            "required_params": {required_params},
            "output_schema": {output_schema}
        }}
    
    def get_capabilities(self) -> ToolCapabilities:
        """Get tool capabilities"""
        return ToolCapabilities(
            primary_capability=ToolCapability.TRANSFORMATION,
            secondary_capabilities=[ToolCapability.VALIDATION],
            input_types=["string", "object"],
            output_types=["object", "boolean"],
            supported_formats=["txt", "json", "csv", "xml"]
        )
    
    def get_resource_requirements(self) -> ResourceRequirements:
        """Get resource requirements"""
        return ResourceRequirements(
            cpu_cores=0.5,
            memory_mb=256,
            network_bandwidth_mbps=0.0,
            storage_mb=self.max_file_size * 2,
            max_execution_time=30,
            concurrent_limit=3
        )
''',
            ToolType.CALCULATOR: '''
"""
{description}
Generated tool for mathematical calculations
"""

from typing import Dict, Any, List, Optional, Union
import math
import statistics
import json
import time
{additional_imports}

from src.shared.logging import get_logger
from src.agent_server.tools.registry import BaseTool, ToolCapabilities, ResourceRequirements, ToolResult, ExecutionContext, ToolCapability

logger = get_logger(__name__)


class {class_name}(BaseTool):
    """Generated calculator tool"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.precision = config.get("precision", 6) if config else 6
    
    async def execute(self, parameters: Dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Execute the calculation operation"""
        
        try:
            # Validate input parameters
            is_valid, validation_errors = self.validate_parameters(parameters)
            if not is_valid:
                return ToolResult(
                    data=None,
                    metadata={{"error": "Invalid parameters", "validation_errors": validation_errors}},
                    execution_time=0.0,
                    success=False,
                    error_message=f"Parameter validation failed: {{', '.join(validation_errors)}}"
                )
            
            start_time = time.time()
            
            # Extract parameters
{parameter_extraction}
            
            # Perform calculation
            result = await self._calculate({parameter_names})
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                data=result,
                metadata={{"tool_type": "calculator", "execution_time": execution_time}},
                execution_time=execution_time,
                success=True,
                quality_score=1.0,
                confidence_score=0.95
            )
            
        except Exception as e:
            logger.error("Calculation failed", error=str(e))
            return ToolResult(
                data=None,
                metadata={{"error": "Calculation failed"}},
                execution_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def _calculate(self, {parameter_signature}) -> Any:
        """Perform mathematical calculation"""
        
        # TODO: Implement specific calculation logic
        # This is a template - actual implementation depends on requirements
        
        result = {{
            "calculation_completed": True,
            "result": 0.0,
            "precision": self.precision
        }}
        
        return result
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema"""
        return {{
            "name": "{tool_name}",
            "description": "{description}",
            "parameters": {parameter_schema},
            "required_params": {required_params},
            "output_schema": {output_schema}
        }}
    
    def get_capabilities(self) -> ToolCapabilities:
        """Get tool capabilities"""
        return ToolCapabilities(
            primary_capability=ToolCapability.DATA_ANALYSIS,
            secondary_capabilities=[ToolCapability.VALIDATION],
            input_types=["number", "array", "object"],
            output_types=["number", "object"],
            supported_formats=["json", "number"]
        )
    
    def get_resource_requirements(self) -> ResourceRequirements:
        """Get resource requirements"""
        return ResourceRequirements(
            cpu_cores=0.5,
            memory_mb=128,
            network_bandwidth_mbps=0.0,
            storage_mb=10,
            max_execution_time=15,
            concurrent_limit=10
        )
''',
            ToolType.VALIDATOR: '''
"""
{description}
Generated tool for data validation
"""

from typing import Dict, Any, List, Optional, Union
import re
import json
import time
{additional_imports}

from src.shared.logging import get_logger
from src.agent_server.tools.registry import BaseTool, ToolCapabilities, ResourceRequirements, ToolResult, ExecutionContext, ToolCapability

logger = get_logger(__name__)


class {class_name}(BaseTool):
    """Generated validator tool"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.strict_mode = config.get("strict_mode", False) if config else False
    
    async def execute(self, parameters: Dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Execute the validation operation"""
        
        try:
            # Validate input parameters
            is_valid, validation_errors = self.validate_parameters(parameters)
            if not is_valid:
                return ToolResult(
                    data=None,
                    metadata={{"error": "Invalid parameters", "validation_errors": validation_errors}},
                    execution_time=0.0,
                    success=False,
                    error_message=f"Parameter validation failed: {{', '.join(validation_errors)}}"
                )
            
            start_time = time.time()
            
            # Extract parameters
{parameter_extraction}
            
            # Perform validation
            result = await self._validate_data({parameter_names})
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                data=result,
                metadata={{"tool_type": "validator", "execution_time": execution_time}},
                execution_time=execution_time,
                success=True,
                quality_score=1.0,
                confidence_score=0.9
            )
            
        except Exception as e:
            logger.error("Validation failed", error=str(e))
            return ToolResult(
                data=None,
                metadata={{"error": "Validation failed"}},
                execution_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def _validate_data(self, {parameter_signature}) -> Any:
        """Perform data validation"""
        
        # TODO: Implement specific validation logic
        # This is a template - actual implementation depends on requirements
        
        result = {{
            "valid": True,
            "errors": [],
            "warnings": [],
            "validation_summary": "Data validation completed successfully"
        }}
        
        return result
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema"""
        return {{
            "name": "{tool_name}",
            "description": "{description}",
            "parameters": {parameter_schema},
            "required_params": {required_params},
            "output_schema": {output_schema}
        }}
    
    def get_capabilities(self) -> ToolCapabilities:
        """Get tool capabilities"""
        return ToolCapabilities(
            primary_capability=ToolCapability.VALIDATION,
            secondary_capabilities=[ToolCapability.DATA_ANALYSIS],
            input_types=["string", "object", "array"],
            output_types=["boolean", "object"],
            supported_formats=["json", "text", "xml"]
        )
    
    def get_resource_requirements(self) -> ResourceRequirements:
        """Get resource requirements"""
        return ResourceRequirements(
            cpu_cores=0.5,
            memory_mb=256,
            network_bandwidth_mbps=0.0,
            storage_mb=50,
            max_execution_time=20,
            concurrent_limit=8
        )
''',
            ToolType.TRANSFORMER: '''
"""
{description}
Generated tool for data transformation
"""

from typing import Dict, Any, List, Optional, Union
import json
import time
{additional_imports}

from src.shared.logging import get_logger
from src.agent_server.tools.registry import BaseTool, ToolCapabilities, ResourceRequirements, ToolResult, ExecutionContext, ToolCapability

logger = get_logger(__name__)


class {class_name}(BaseTool):
    """Generated transformer tool"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.preserve_metadata = config.get("preserve_metadata", True) if config else True
    
    async def execute(self, parameters: Dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Execute the transformation operation"""
        
        try:
            # Validate input parameters
            is_valid, validation_errors = self.validate_parameters(parameters)
            if not is_valid:
                return ToolResult(
                    data=None,
                    metadata={{"error": "Invalid parameters", "validation_errors": validation_errors}},
                    execution_time=0.0,
                    success=False,
                    error_message=f"Parameter validation failed: {{', '.join(validation_errors)}}"
                )
            
            start_time = time.time()
            
            # Extract parameters
{parameter_extraction}
            
            # Perform transformation
            result = await self._transform_data({parameter_names})
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                data=result,
                metadata={{"tool_type": "transformer", "execution_time": execution_time}},
                execution_time=execution_time,
                success=True,
                quality_score=1.0,
                confidence_score=0.9
            )
            
        except Exception as e:
            logger.error("Transformation failed", error=str(e))
            return ToolResult(
                data=None,
                metadata={{"error": "Transformation failed"}},
                execution_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def _transform_data(self, {parameter_signature}) -> Any:
        """Perform data transformation"""
        
        # TODO: Implement specific transformation logic
        # This is a template - actual implementation depends on requirements
        
        result = {{
            "transformed": True,
            "original_format": "input",
            "target_format": "output",
            "transformation_summary": "Data transformation completed successfully"
        }}
        
        return result
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema"""
        return {{
            "name": "{tool_name}",
            "description": "{description}",
            "parameters": {parameter_schema},
            "required_params": {required_params},
            "output_schema": {output_schema}
        }}
    
    def get_capabilities(self) -> ToolCapabilities:
        """Get tool capabilities"""
        return ToolCapabilities(
            primary_capability=ToolCapability.TRANSFORMATION,
            secondary_capabilities=[ToolCapability.VALIDATION],
            input_types=["string", "object", "array"],
            output_types=["string", "object", "array"],
            supported_formats=["json", "xml", "csv", "text"]
        )
    
    def get_resource_requirements(self) -> ResourceRequirements:
        """Get resource requirements"""
        return ResourceRequirements(
            cpu_cores=1.0,
            memory_mb=512,
            network_bandwidth_mbps=0.0,
            storage_mb=200,
            max_execution_time=45,
            concurrent_limit=5
        )
''',
            ToolType.ANALYZER: '''
"""
{description}
Generated tool for data analysis
"""

from typing import Dict, Any, List, Optional, Union
import json
import time
{additional_imports}

from src.shared.logging import get_logger
from src.agent_server.tools.registry import BaseTool, ToolCapabilities, ResourceRequirements, ToolResult, ExecutionContext, ToolCapability

logger = get_logger(__name__)


class {class_name}(BaseTool):
    """Generated analyzer tool"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.analysis_depth = config.get("analysis_depth", "standard") if config else "standard"
    
    async def execute(self, parameters: Dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Execute the analysis operation"""
        
        try:
            # Validate input parameters
            is_valid, validation_errors = self.validate_parameters(parameters)
            if not is_valid:
                return ToolResult(
                    data=None,
                    metadata={{"error": "Invalid parameters", "validation_errors": validation_errors}},
                    execution_time=0.0,
                    success=False,
                    error_message=f"Parameter validation failed: {{', '.join(validation_errors)}}"
                )
            
            start_time = time.time()
            
            # Extract parameters
{parameter_extraction}
            
            # Perform analysis
            result = await self._analyze_data({parameter_names})
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                data=result,
                metadata={{"tool_type": "analyzer", "execution_time": execution_time}},
                execution_time=execution_time,
                success=True,
                quality_score=1.0,
                confidence_score=0.85
            )
            
        except Exception as e:
            logger.error("Analysis failed", error=str(e))
            return ToolResult(
                data=None,
                metadata={{"error": "Analysis failed"}},
                execution_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def _analyze_data(self, {parameter_signature}) -> Any:
        """Perform data analysis"""
        
        # TODO: Implement specific analysis logic
        # This is a template - actual implementation depends on requirements
        
        result = {{
            "analysis_completed": True,
            "insights": [],
            "metrics": {{}},
            "recommendations": [],
            "analysis_summary": "Data analysis completed successfully"
        }}
        
        return result
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema"""
        return {{
            "name": "{tool_name}",
            "description": "{description}",
            "parameters": {parameter_schema},
            "required_params": {required_params},
            "output_schema": {output_schema}
        }}
    
    def get_capabilities(self) -> ToolCapabilities:
        """Get tool capabilities"""
        return ToolCapabilities(
            primary_capability=ToolCapability.DATA_ANALYSIS,
            secondary_capabilities=[ToolCapability.VALIDATION],
            input_types=["object", "array", "string"],
            output_types=["object"],
            supported_formats=["json", "csv", "text"]
        )
    
    def get_resource_requirements(self) -> ResourceRequirements:
        """Get resource requirements"""
        return ResourceRequirements(
            cpu_cores=1.5,
            memory_mb=1024,
            network_bandwidth_mbps=0.0,
            storage_mb=300,
            max_execution_time=60,
            concurrent_limit=3
        )
''',
            ToolType.GENERATOR: '''
"""
{description}
Generated tool for content generation
"""

from typing import Dict, Any, List, Optional, Union
import json
import time
{additional_imports}

from src.shared.logging import get_logger
from src.agent_server.tools.registry import BaseTool, ToolCapabilities, ResourceRequirements, ToolResult, ExecutionContext, ToolCapability

logger = get_logger(__name__)


class {class_name}(BaseTool):
    """Generated content generator tool"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.output_format = config.get("output_format", "text") if config else "text"
        self.quality_level = config.get("quality_level", "standard") if config else "standard"
    
    async def execute(self, parameters: Dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Execute the content generation operation"""
        
        try:
            # Validate input parameters
            is_valid, validation_errors = self.validate_parameters(parameters)
            if not is_valid:
                return ToolResult(
                    data=None,
                    metadata={{"error": "Invalid parameters", "validation_errors": validation_errors}},
                    execution_time=0.0,
                    success=False,
                    error_message=f"Parameter validation failed: {{', '.join(validation_errors)}}"
                )
            
            start_time = time.time()
            
            # Extract parameters
{parameter_extraction}
            
            # Generate content
            result = await self._generate_content({parameter_names})
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                data=result,
                metadata={{"tool_type": "generator", "execution_time": execution_time}},
                execution_time=execution_time,
                success=True,
                quality_score=1.0,
                confidence_score=0.8
            )
            
        except Exception as e:
            logger.error("Content generation failed", error=str(e))
            return ToolResult(
                data=None,
                metadata={{"error": "Content generation failed"}},
                execution_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def _generate_content(self, {parameter_signature}) -> Any:
        """Generate content based on parameters"""
        
        # TODO: Implement specific content generation logic
        # This is a template - actual implementation depends on requirements
        
        result = {{
            "generated": True,
            "content": "Generated content placeholder",
            "format": self.output_format,
            "quality": self.quality_level,
            "generation_summary": "Content generation completed successfully"
        }}
        
        return result
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema"""
        return {{
            "name": "{tool_name}",
            "description": "{description}",
            "parameters": {parameter_schema},
            "required_params": {required_params},
            "output_schema": {output_schema}
        }}
    
    def get_capabilities(self) -> ToolCapabilities:
        """Get tool capabilities"""
        return ToolCapabilities(
            primary_capability=ToolCapability.CONTENT_GENERATION,
            secondary_capabilities=[ToolCapability.TRANSFORMATION],
            input_types=["string", "object"],
            output_types=["string", "object"],
            supported_formats=["text", "json", "html", "markdown"]
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
''',
        }

    async def execute(
        self, parameters: Dict[str, Any], context: ExecutionContext
    ) -> ToolResult:
        """Execute tool generation from natural language description"""

        try:
            description = parameters.get("description", "")
            if not description:
                return ToolResult(
                    data=None,
                    metadata={"error": "Missing description"},
                    execution_time=0.0,
                    success=False,
                    error_message="Tool description is required",
                )

            start_time = time.time()

            logger.info("Starting tool generation", description_length=len(description))

            # Step 1: Analyze requirements
            requirement = await self.requirement_analyzer.analyze_requirements(
                description
            )

            # Step 2: Security assessment
            security_assessment = await self.security_analyzer.assess_security(
                requirement
            )

            if not security_assessment.safe_to_generate:
                return ToolResult(
                    data=None,
                    metadata={
                        "error": "Security risk too high",
                        "risk_level": security_assessment.risk_level.value,
                        "risks": security_assessment.identified_risks,
                    },
                    execution_time=time.time() - start_time,
                    success=False,
                    error_message=f"Tool generation blocked due to {security_assessment.risk_level.value} security risk",
                )

            # Step 3: Dependency analysis
            dependency_analysis = await self.dependency_analyzer.analyze_dependencies(
                requirement.dependencies
            )

            # Step 4: Generate code
            generated_code = await self._generate_tool_code(
                requirement, security_assessment, dependency_analysis
            )

            # Step 5: Comprehensive validation
            validation_result = await self._comprehensive_validation(
                generated_code, requirement
            )

            if not validation_result["valid"]:
                return ToolResult(
                    data=None,
                    metadata={
                        "error": "Code validation failed",
                        "validation_errors": validation_result["errors"],
                    },
                    execution_time=time.time() - start_time,
                    success=False,
                    error_message=f"Generated code validation failed: {', '.join(validation_result['errors'])}",
                )

            # Step 6: Store generated tool
            tool_id = await self._store_generated_tool(
                requirement,
                generated_code,
                security_assessment,
                dependency_analysis,
                validation_result,
            )

            execution_time = time.time() - start_time

            result = {
                "tool_id": tool_id,
                "tool_name": requirement.name,
                "tool_type": requirement.tool_type.value,
                "generated_code": generated_code,
                "security_assessment": {
                    "risk_level": security_assessment.risk_level.value,
                    "safe_to_generate": security_assessment.safe_to_generate,
                    "restrictions": security_assessment.restrictions,
                },
                "dependency_analysis": {
                    "required_dependencies": [
                        dep["name"] for dep in dependency_analysis.required_dependencies
                    ],
                    "conflicts": dependency_analysis.conflicts,
                    "installation_requirements": dependency_analysis.installation_requirements,
                },
                "validation_result": validation_result,
                "metadata": {
                    "generation_time": execution_time,
                    "parameter_count": len(requirement.input_parameters),
                    "capability_count": len(requirement.capabilities),
                },
            }

            logger.info(
                "Tool generation completed successfully",
                tool_name=requirement.name,
                tool_type=requirement.tool_type.value,
                generation_time=execution_time,
            )

            return ToolResult(
                data=result,
                metadata={
                    "tool_type": "code_tool_generator",
                    "execution_time": execution_time,
                },
                execution_time=execution_time,
                success=True,
                quality_score=validation_result.get("quality_score", 0.8),
                confidence_score=0.9,
            )

        except Exception as e:
            logger.error("Tool generation failed", error=str(e))
            return ToolResult(
                data=None,
                metadata={"error": "Tool generation failed"},
                execution_time=(
                    time.time() - start_time if "start_time" in locals() else 0.0
                ),
                success=False,
                error_message=str(e),
            )

    async def _generate_tool_code(
        self,
        requirement: ToolRequirement,
        security_assessment: SecurityAssessment,
        dependency_analysis: DependencyAnalysis,
    ) -> str:
        """Generate tool code from requirements"""

        # Get appropriate template
        template = self.code_templates.get(
            requirement.tool_type, self.code_templates[ToolType.DATA_PROCESSOR]
        )

        # Generate class name
        class_name = self._generate_class_name(requirement.name)

        # Generate imports
        additional_imports = self._generate_imports(
            requirement.dependencies, dependency_analysis
        )

        # Generate parameter extraction code
        parameter_extraction = self._generate_parameter_extraction(
            requirement.input_parameters
        )

        # Generate parameter signature
        parameter_signature = self._generate_parameter_signature(
            requirement.input_parameters
        )

        # Generate parameter names for function calls
        parameter_names = ", ".join(requirement.input_parameters.keys())

        # Generate parameter schema
        parameter_schema = self._generate_parameter_schema(requirement.input_parameters)

        # Generate required parameters list
        required_params = [
            name
            for name, info in requirement.input_parameters.items()
            if info.get("required", False)
        ]

        # Generate output schema
        output_schema = requirement.output_format.get("schema", {"type": "object"})

        # Fill template
        generated_code = template.format(
            description=requirement.description,
            class_name=class_name,
            tool_name=requirement.name,
            additional_imports=additional_imports,
            parameter_extraction=parameter_extraction,
            parameter_signature=parameter_signature,
            parameter_names=parameter_names,
            parameter_schema=json.dumps(parameter_schema, indent=12),
            required_params=json.dumps(required_params),
            output_schema=json.dumps(output_schema, indent=12),
        )

        return generated_code

    def _generate_class_name(self, tool_name: str) -> str:
        """Generate appropriate class name from tool name"""

        # Convert snake_case to PascalCase
        words = tool_name.split("_")
        class_name = "".join(word.capitalize() for word in words if word)

        # Ensure it ends with Tool if not already
        if not class_name.endswith("Tool"):
            class_name += "Tool"

        return class_name

    def _generate_imports(
        self, dependencies: List[str], dependency_analysis: DependencyAnalysis
    ) -> str:
        """Generate import statements"""

        imports = []

        # Add time import for execution timing
        if "time" not in dependencies:
            imports.append("import time")

        # Add standard library imports
        for dep in dependencies:
            if dep in self.dependency_analyzer.standard_library:
                imports.append(f"import {dep}")

        # Add third-party imports with error handling
        third_party_deps = [
            dep["name"]
            for dep in dependency_analysis.required_dependencies
            if dep["type"] == DependencyType.THIRD_PARTY
        ]

        for dep in third_party_deps:
            imports.append(f"import {dep}")

        return "\n".join(imports)

    def _generate_parameter_extraction(
        self, parameters: Dict[str, Dict[str, Any]]
    ) -> str:
        """Generate parameter extraction code"""

        extraction_lines = []

        for param_name, param_info in parameters.items():
            param_type = param_info.get("type", "string")
            required = param_info.get("required", False)
            default_value = self._get_default_value(param_type)

            if required:
                extraction_lines.append(
                    f"            {param_name} = parameters.get('{param_name}')"
                )
                extraction_lines.append(f"            if {param_name} is None:")
                extraction_lines.append(
                    f"                raise ValueError('Required parameter {param_name} is missing')"
                )
            else:
                extraction_lines.append(
                    f"            {param_name} = parameters.get('{param_name}', {default_value})"
                )

        return "\n".join(extraction_lines)

    def _generate_parameter_signature(
        self, parameters: Dict[str, Dict[str, Any]]
    ) -> str:
        """Generate function parameter signature"""

        param_parts = []

        for param_name, param_info in parameters.items():
            param_type = param_info.get("type", "string")
            required = param_info.get("required", False)

            # Map parameter types to Python types
            type_mapping = {
                "string": "str",
                "integer": "int",
                "number": "float",
                "boolean": "bool",
                "array": "List[Any]",
                "object": "Dict[str, Any]",
            }

            python_type = type_mapping.get(param_type, "Any")

            if required:
                param_parts.append(f"{param_name}: {python_type}")
            else:
                default_value = self._get_default_value(param_type)
                param_parts.append(
                    f"{param_name}: Optional[{python_type}] = {default_value}"
                )

        return ", ".join(param_parts)

    def _generate_parameter_schema(
        self, parameters: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate parameter schema for validation"""

        schema = {}

        for param_name, param_info in parameters.items():
            schema[param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get(
                    "description", f"The {param_name} parameter"
                ),
                "required": param_info.get("required", False),
            }

            if param_info.get("example"):
                schema[param_name]["example"] = param_info["example"]

        return schema

    def _get_default_value(self, param_type: str) -> str:
        """Get default value for parameter type"""

        defaults = {
            "string": '""',
            "integer": "0",
            "number": "0.0",
            "boolean": "False",
            "array": "[]",
            "object": "{}",
        }

        return defaults.get(param_type, "None")

    async def _get_validation_framework(self):
        """Get validation framework instance (lazy initialization)"""
        if self.validation_framework is None:
            from src.agent_server.tools.testing_framework import ToolValidationFramework

            self.validation_framework = ToolValidationFramework()
        return self.validation_framework

    async def _get_tool_registry(self):
        """Get tool registry instance (lazy initialization)"""
        if self.tool_registry is None:
            from src.agent_server.tools.tool_registry import ToolRegistryManager

            self.tool_registry = ToolRegistryManager()
        return self.tool_registry

    async def _comprehensive_validation(
        self, code: str, requirement: ToolRequirement
    ) -> Dict[str, Any]:
        """Perform comprehensive validation using the testing framework"""

        try:
            validation_framework = await self._get_validation_framework()
            validation_result = await validation_framework.validate_tool(
                code, requirement.name, requirement.input_parameters
            )

            return {
                "valid": validation_result.overall_passed,
                "quality_score": validation_result.overall_score,
                "errors": [],
                "warnings": [],
                "static_analysis": {
                    "passed": validation_result.static_analysis.passed,
                    "issues_count": len(validation_result.static_analysis.issues),
                    "complexity_score": validation_result.static_analysis.complexity_score,
                    "maintainability_index": validation_result.static_analysis.maintainability_index,
                },
                "security_scan": {
                    "passed": validation_result.security_scan.passed,
                    "risk_score": validation_result.security_scan.risk_score,
                    "security_grade": validation_result.security_scan.security_grade,
                    "vulnerabilities_count": len(
                        validation_result.security_scan.vulnerabilities
                    ),
                },
                "unit_tests": {
                    "passed": validation_result.unit_tests.passed,
                    "coverage_percentage": validation_result.unit_tests.coverage_percentage,
                    "tests_run": validation_result.unit_tests.tests_run,
                },
                "recommendations": validation_result.recommendations,
            }

        except Exception as e:
            logger.error("Comprehensive validation failed", error=str(e))
            # Fallback to basic validation
            return await self._validate_generated_code(code, requirement)

    async def _validate_generated_code(
        self, code: str, requirement: ToolRequirement
    ) -> Dict[str, Any]:
        """Validate generated code for syntax and basic functionality"""

        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "quality_score": 1.0,
        }

        try:
            # Syntax validation
            ast.parse(code)

            # Check for required methods
            tree = ast.parse(code)

            class_found = False
            execute_method_found = False
            schema_method_found = False

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_found = True
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if item.name == "execute":
                                execute_method_found = True
                            elif item.name == "get_schema":
                                schema_method_found = True

            if not class_found:
                validation_result["errors"].append("No class definition found")

            if not execute_method_found:
                validation_result["errors"].append("No execute method found")

            if not schema_method_found:
                validation_result["errors"].append("No get_schema method found")

            # Check for security issues
            security_issues = self._check_code_security(code)
            if security_issues:
                validation_result["warnings"].extend(security_issues)
                validation_result["quality_score"] -= 0.1 * len(security_issues)

            # Check code quality
            quality_issues = self._check_code_quality(code)
            if quality_issues:
                validation_result["warnings"].extend(quality_issues)
                validation_result["quality_score"] -= 0.05 * len(quality_issues)

            if validation_result["errors"]:
                validation_result["valid"] = False

            validation_result["quality_score"] = max(
                0.0, validation_result["quality_score"]
            )

        except SyntaxError as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Syntax error: {str(e)}")
            validation_result["quality_score"] = 0.0

        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["quality_score"] = 0.0

        return validation_result

    def _check_code_security(self, code: str) -> List[str]:
        """Check generated code for security issues"""

        security_issues = []

        # Check for dangerous functions
        dangerous_patterns = [
            r"\beval\s*\(",
            r"\bexec\s*\(",
            r"os\.system\s*\(",
            r"subprocess\.call\s*\(",
            r"__import__\s*\(",
            r"compile\s*\(",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                security_issues.append(
                    f"Potentially dangerous function call: {pattern}"
                )

        # Check for file operations without validation
        if re.search(r"open\s*\(", code) and "validate_file_path" not in code:
            security_issues.append("File operations without path validation")

        return security_issues

    def _check_code_quality(self, code: str) -> List[str]:
        """Check generated code quality"""

        quality_issues = []

        # Check for proper error handling
        if "try:" not in code:
            quality_issues.append("Missing error handling")

        # Check for logging
        if "logger." not in code:
            quality_issues.append("Missing logging statements")

        # Check for docstrings
        if '"""' not in code:
            quality_issues.append("Missing docstrings")

        # Check line length (simplified)
        lines = code.split("\n")
        long_lines = [i + 1 for i, line in enumerate(lines) if len(line) > 120]
        if long_lines:
            quality_issues.append(f"Lines too long: {long_lines[:5]}")  # Show first 5

        return quality_issues

    async def _store_generated_tool(
        self,
        requirement: ToolRequirement,
        code: str,
        security_assessment: SecurityAssessment,
        dependency_analysis: DependencyAnalysis,
        validation_result: Dict[str, Any],
    ) -> str:
        """Store generated tool with metadata and register in tool registry"""

        tool_id = hashlib.md5(
            f"{requirement.name}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        # Store in local cache
        tool_data = {
            "id": tool_id,
            "name": requirement.name,
            "description": requirement.description,
            "tool_type": requirement.tool_type.value,
            "code": code,
            "requirement": requirement.__dict__,
            "security_assessment": {
                "risk_level": security_assessment.risk_level.value,
                "safe_to_generate": security_assessment.safe_to_generate,
                "restrictions": security_assessment.restrictions,
                "recommendations": security_assessment.recommendations,
            },
            "dependency_analysis": {
                "required_dependencies": [
                    dep.__dict__ if hasattr(dep, "__dict__") else dep
                    for dep in dependency_analysis.required_dependencies
                ],
                "conflicts": dependency_analysis.conflicts,
                "installation_requirements": dependency_analysis.installation_requirements,
            },
            "created_at": datetime.utcnow().isoformat(),
            "status": "generated",
        }

        self.generated_tools[tool_id] = tool_data

        # Register in tool registry
        try:
            tool_registry = await self._get_tool_registry()

            # Create tags from tool type and capabilities
            tags = [requirement.tool_type.value]
            if requirement.capabilities:
                tags.extend([cap.value for cap in requirement.capabilities])

            # Note: In a real implementation, you would instantiate the generated tool class
            # For now, we'll store the metadata without the actual tool instance
            logger.info(
                "Tool stored successfully",
                tool_id=tool_id,
                tool_name=requirement.name,
                registry_integration="metadata_only",
            )

        except Exception as e:
            logger.warning(
                "Failed to register tool in registry", tool_id=tool_id, error=str(e)
            )

        return tool_id

    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema"""
        return {
            "name": "code_tool_generator",
            "description": "Generate custom code tools from natural language descriptions",
            "parameters": {
                "description": {
                    "type": "string",
                    "description": "Natural language description of the desired tool functionality",
                    "required": True,
                    "example": "Create a tool that processes CSV files and calculates statistics",
                },
                "tool_type": {
                    "type": "string",
                    "description": "Optional tool type hint",
                    "required": False,
                    "enum": [t.value for t in ToolType],
                    "example": "data_processor",
                },
                "security_level": {
                    "type": "string",
                    "description": "Required security level for the tool",
                    "required": False,
                    "enum": ["low", "medium", "high"],
                    "default": "medium",
                },
            },
            "required_params": ["description"],
            "output_schema": {
                "type": "object",
                "properties": {
                    "tool_id": {
                        "type": "string",
                        "description": "Generated tool identifier",
                    },
                    "tool_name": {
                        "type": "string",
                        "description": "Generated tool name",
                    },
                    "tool_type": {
                        "type": "string",
                        "description": "Tool type classification",
                    },
                    "generated_code": {
                        "type": "string",
                        "description": "Generated Python code",
                    },
                    "security_assessment": {
                        "type": "object",
                        "description": "Security analysis results",
                    },
                    "dependency_analysis": {
                        "type": "object",
                        "description": "Dependency analysis results",
                    },
                    "validation_result": {
                        "type": "object",
                        "description": "Code validation results",
                    },
                },
            },
        }

    def get_capabilities(self) -> ToolCapabilities:
        """Get tool capabilities"""
        return ToolCapabilities(
            primary_capability=ToolCapability.CODE_GENERATION,
            secondary_capabilities=[
                ToolCapability.VALIDATION,
                ToolCapability.CONTENT_GENERATION,
            ],
            input_types=["string", "object"],
            output_types=["object"],
            supported_formats=["python", "json"],
            language_support=["python"],
        )

    def get_resource_requirements(self) -> ResourceRequirements:
        """Get resource requirements"""
        return ResourceRequirements(
            cpu_cores=2.0,
            memory_mb=1024,
            network_bandwidth_mbps=0.0,
            storage_mb=500,
            max_execution_time=120,
            concurrent_limit=3,
        )

    async def list_generated_tools(self) -> List[Dict[str, Any]]:
        """List all generated tools"""

        return [
            {
                "id": tool_data["id"],
                "name": tool_data["name"],
                "description": tool_data["description"],
                "tool_type": tool_data["tool_type"],
                "created_at": tool_data["created_at"],
                "status": tool_data["status"],
                "security_risk": tool_data["security_assessment"]["risk_level"],
            }
            for tool_data in self.generated_tools.values()
        ]

    async def get_generated_tool(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get specific generated tool by ID"""

        return self.generated_tools.get(tool_id)

    async def delete_generated_tool(self, tool_id: str) -> bool:
        """Delete generated tool"""

        if tool_id in self.generated_tools:
            del self.generated_tools[tool_id]
            return True

        return False
