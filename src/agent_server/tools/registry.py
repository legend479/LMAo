"""
Tool Registry
Dynamic tool registration, discovery, and lifecycle management with advanced selection and execution
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import concurrent.futures
from datetime import datetime
from enum import Enum

from src.shared.logging import get_logger

logger = get_logger(__name__)


class ToolCapability(Enum):
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    CONTENT_GENERATION = "content_generation"
    CODE_GENERATION = "code_generation"
    DOCUMENT_GENERATION = "document_generation"
    DATA_ANALYSIS = "data_analysis"
    COMMUNICATION = "communication"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"


class ExecutionPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    GPU = "gpu"


@dataclass
class ToolCapabilities:
    primary_capability: ToolCapability
    secondary_capabilities: List[ToolCapability] = field(default_factory=list)
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    supported_formats: List[str] = field(default_factory=list)
    language_support: List[str] = field(default_factory=list)


@dataclass
class ResourceRequirements:
    cpu_cores: float = 1.0
    memory_mb: int = 512
    network_bandwidth_mbps: float = 10.0
    storage_mb: int = 100
    gpu_memory_mb: int = 0
    max_execution_time: int = 300  # seconds
    concurrent_limit: int = 5


@dataclass
class PerformanceMetrics:
    average_execution_time: float = 0.0
    success_rate: float = 1.0
    error_rate: float = 0.0
    throughput_per_minute: float = 0.0
    resource_efficiency: float = 1.0
    user_satisfaction: float = 1.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ToolMetadata:
    name: str
    description: str
    version: str
    author: str
    category: str
    capabilities: ToolCapabilities
    resource_requirements: ResourceRequirements
    performance_metrics: PerformanceMetrics
    parameters: Dict[str, Any]
    required_params: List[str]
    created_at: datetime
    last_used: Optional[datetime] = None
    usage_count: int = 0
    failure_count: int = 0
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ToolResult:
    data: Any
    metadata: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0
    confidence_score: float = 1.0


@dataclass
class ExecutionContext:
    session_id: str
    user_id: Optional[str] = None
    priority: ExecutionPriority = ExecutionPriority.NORMAL
    timeout: int = 300
    retry_count: int = 0
    max_retries: int = 3
    resource_constraints: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    context_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSelectionCriteria:
    required_capabilities: List[ToolCapability]
    preferred_capabilities: List[ToolCapability] = field(default_factory=list)
    input_type: Optional[str] = None
    output_type: Optional[str] = None
    max_execution_time: Optional[int] = None
    min_success_rate: float = 0.8
    resource_constraints: Dict[str, Any] = field(default_factory=dict)
    exclude_tools: List[str] = field(default_factory=list)
    prefer_tools: List[str] = field(default_factory=list)


@dataclass
class ToolScore:
    tool_name: str
    overall_score: float
    capability_score: float
    performance_score: float
    resource_score: float
    preference_score: float
    confidence: float
    rationale: str


class BaseTool(ABC):
    """Enhanced base class for all tools"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metadata: Optional[ToolMetadata] = None
        self._initialized = False
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._failure_count = 0

    @abstractmethod
    async def execute(
        self, parameters: Dict[str, Any], context: ExecutionContext
    ) -> ToolResult:
        """Execute the tool with given parameters and context"""
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for validation and documentation"""
        pass

    @abstractmethod
    def get_capabilities(self) -> ToolCapabilities:
        """Get tool capabilities"""
        pass

    @abstractmethod
    def get_resource_requirements(self) -> ResourceRequirements:
        """Get resource requirements"""
        pass

    async def initialize(self):
        """Initialize tool resources"""
        self._initialized = True

    async def cleanup(self):
        """Cleanup tool resources"""
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate input parameters and return validation result with errors"""
        schema = self.get_schema()
        required_params = schema.get("required_params", [])
        errors = []

        for param in required_params:
            if param not in parameters:
                errors.append(f"Missing required parameter: {param}")

        # Additional validation based on schema
        param_definitions = schema.get("parameters", {})
        for param_name, param_value in parameters.items():
            if param_name in param_definitions:
                param_def = param_definitions[param_name]
                if not self._validate_parameter_type(param_value, param_def):
                    errors.append(f"Invalid type for parameter {param_name}")

        return len(errors) == 0, errors

    def _validate_parameter_type(self, value: Any, param_def: Dict[str, Any]) -> bool:
        """Validate parameter type against definition"""
        expected_type = param_def.get("type", "string")

        if expected_type == "string" and not isinstance(value, str):
            return False
        elif expected_type == "integer" and not isinstance(value, int):
            return False
        elif expected_type == "number" and not isinstance(value, (int, float)):
            return False
        elif expected_type == "boolean" and not isinstance(value, bool):
            return False
        elif expected_type == "array" and not isinstance(value, list):
            return False
        elif expected_type == "object" and not isinstance(value, dict):
            return False

        return True

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the tool"""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "execution_count": self._execution_count,
            "failure_rate": self._failure_count / max(self._execution_count, 1),
            "average_execution_time": self._total_execution_time
            / max(self._execution_count, 1),
        }

    def update_metrics(self, execution_time: float, success: bool):
        """Update tool performance metrics"""
        self._execution_count += 1
        self._total_execution_time += execution_time
        if not success:
            self._failure_count += 1


class ToolExecutionPool:
    """Manages concurrent tool execution with resource management"""

    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.resource_usage: Dict[ResourceType, float] = {
            ResourceType.CPU: 0.0,
            ResourceType.MEMORY: 0.0,
            ResourceType.NETWORK: 0.0,
            ResourceType.STORAGE: 0.0,
            ResourceType.GPU: 0.0,
        }
        self.execution_queue: List[Dict[str, Any]] = []

    async def execute_tool(
        self, tool: BaseTool, parameters: Dict[str, Any], context: ExecutionContext
    ) -> ToolResult:
        """Execute tool with resource management and monitoring"""

        execution_id = (
            f"{context.session_id}_{tool.metadata.name}_{datetime.utcnow().timestamp()}"
        )

        # Check resource availability
        if not await self._check_resource_availability(tool, context):
            return ToolResult(
                data=None,
                metadata={"error": "Insufficient resources"},
                execution_time=0.0,
                success=False,
                error_message="Insufficient resources available for execution",
            )

        # Reserve resources
        await self._reserve_resources(tool, execution_id)

        try:
            # Track execution
            self.active_executions[execution_id] = {
                "tool_name": tool.metadata.name,
                "start_time": datetime.utcnow(),
                "context": context,
                "status": "running",
            }

            # Execute with timeout
            result = await asyncio.wait_for(
                tool.execute(parameters, context), timeout=context.timeout
            )

            # Update metrics
            tool.update_metrics(result.execution_time, result.success)

            return result

        except asyncio.TimeoutError:
            logger.error(
                "Tool execution timeout",
                tool_name=tool.metadata.name,
                execution_id=execution_id,
            )
            return ToolResult(
                data=None,
                metadata={"error": "Execution timeout"},
                execution_time=context.timeout,
                success=False,
                error_message=f"Tool execution exceeded timeout of {context.timeout} seconds",
            )

        except Exception as e:
            logger.error(
                "Tool execution failed", tool_name=tool.metadata.name, error=str(e)
            )
            tool.update_metrics(0.0, False)
            return ToolResult(
                data=None,
                metadata={"error": "Execution failed"},
                execution_time=0.0,
                success=False,
                error_message=str(e),
            )

        finally:
            # Release resources
            await self._release_resources(tool, execution_id)

            # Remove from active executions
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]

    async def _check_resource_availability(
        self, tool: BaseTool, context: ExecutionContext
    ) -> bool:
        """Check if resources are available for tool execution"""

        requirements = tool.get_resource_requirements()

        # Check CPU availability
        if (
            self.resource_usage[ResourceType.CPU] + requirements.cpu_cores
            > self.max_workers
        ):
            return False

        # Check memory availability (simplified check)
        if (
            self.resource_usage[ResourceType.MEMORY] + requirements.memory_mb > 8192
        ):  # 8GB limit
            return False

        return True

    async def _reserve_resources(self, tool: BaseTool, execution_id: str):
        """Reserve resources for tool execution"""

        requirements = tool.get_resource_requirements()

        self.resource_usage[ResourceType.CPU] += requirements.cpu_cores
        self.resource_usage[ResourceType.MEMORY] += requirements.memory_mb
        self.resource_usage[ResourceType.NETWORK] += requirements.network_bandwidth_mbps
        self.resource_usage[ResourceType.STORAGE] += requirements.storage_mb
        self.resource_usage[ResourceType.GPU] += requirements.gpu_memory_mb

    async def _release_resources(self, tool: BaseTool, execution_id: str):
        """Release resources after tool execution"""

        requirements = tool.get_resource_requirements()

        self.resource_usage[ResourceType.CPU] = max(
            0, self.resource_usage[ResourceType.CPU] - requirements.cpu_cores
        )
        self.resource_usage[ResourceType.MEMORY] = max(
            0, self.resource_usage[ResourceType.MEMORY] - requirements.memory_mb
        )
        self.resource_usage[ResourceType.NETWORK] = max(
            0,
            self.resource_usage[ResourceType.NETWORK]
            - requirements.network_bandwidth_mbps,
        )
        self.resource_usage[ResourceType.STORAGE] = max(
            0, self.resource_usage[ResourceType.STORAGE] - requirements.storage_mb
        )
        self.resource_usage[ResourceType.GPU] = max(
            0, self.resource_usage[ResourceType.GPU] - requirements.gpu_memory_mb
        )

    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        return {
            resource_type.value: usage
            for resource_type, usage in self.resource_usage.items()
        }

    def get_active_executions(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active executions"""
        return self.active_executions.copy()


class ToolSelector:
    """Advanced tool selection with multi-criteria optimization"""

    def __init__(self):
        self.selection_history: List[Dict[str, Any]] = []
        self.performance_cache: Dict[str, PerformanceMetrics] = {}

    async def select_best_tool(
        self,
        criteria: ToolSelectionCriteria,
        available_tools: Dict[str, ToolMetadata],
        context: ExecutionContext,
    ) -> Optional[ToolScore]:
        """Select the best tool based on multi-criteria optimization"""

        # Filter tools by capabilities
        candidate_tools = self._filter_by_capabilities(available_tools, criteria)

        if not candidate_tools:
            return None

        # Score each candidate tool
        tool_scores = []
        for tool_name, metadata in candidate_tools.items():
            score = await self._calculate_tool_score(
                tool_name, metadata, criteria, context
            )
            tool_scores.append(score)

        # Sort by overall score
        tool_scores.sort(key=lambda x: x.overall_score, reverse=True)

        # Log selection decision
        best_tool = tool_scores[0]
        logger.info(
            "Tool selected",
            tool_name=best_tool.tool_name,
            score=best_tool.overall_score,
            rationale=best_tool.rationale,
        )

        # Store selection history
        self.selection_history.append(
            {
                "timestamp": datetime.utcnow(),
                "criteria": criteria,
                "selected_tool": best_tool.tool_name,
                "score": best_tool.overall_score,
                "alternatives": [
                    {"name": score.tool_name, "score": score.overall_score}
                    for score in tool_scores[1:5]
                ],
            }
        )

        return best_tool

    def _filter_by_capabilities(
        self, available_tools: Dict[str, ToolMetadata], criteria: ToolSelectionCriteria
    ) -> Dict[str, ToolMetadata]:
        """Filter tools by required capabilities"""

        filtered_tools = {}

        for tool_name, metadata in available_tools.items():
            # Skip excluded tools
            if tool_name in criteria.exclude_tools:
                continue

            # Check required capabilities
            tool_capabilities = [
                metadata.capabilities.primary_capability
            ] + metadata.capabilities.secondary_capabilities

            has_required_capabilities = all(
                req_cap in tool_capabilities
                for req_cap in criteria.required_capabilities
            )

            if has_required_capabilities:
                # Check input/output type compatibility
                if (
                    criteria.input_type
                    and criteria.input_type not in metadata.capabilities.input_types
                ):
                    continue

                if (
                    criteria.output_type
                    and criteria.output_type not in metadata.capabilities.output_types
                ):
                    continue

                # Check execution time constraint
                if (
                    criteria.max_execution_time
                    and metadata.resource_requirements.max_execution_time
                    > criteria.max_execution_time
                ):
                    continue

                # Check success rate constraint
                if (
                    metadata.performance_metrics.success_rate
                    < criteria.min_success_rate
                ):
                    continue

                filtered_tools[tool_name] = metadata

        return filtered_tools

    async def _calculate_tool_score(
        self,
        tool_name: str,
        metadata: ToolMetadata,
        criteria: ToolSelectionCriteria,
        context: ExecutionContext,
    ) -> ToolScore:
        """Calculate comprehensive score for a tool"""

        # Capability score (0-1)
        capability_score = self._calculate_capability_score(metadata, criteria)

        # Performance score (0-1)
        performance_score = self._calculate_performance_score(metadata, criteria)

        # Resource efficiency score (0-1)
        resource_score = self._calculate_resource_score(metadata, criteria, context)

        # User preference score (0-1)
        preference_score = self._calculate_preference_score(
            tool_name, criteria, context
        )

        # Weighted overall score
        weights = {
            "capability": 0.35,
            "performance": 0.25,
            "resource": 0.20,
            "preference": 0.20,
        }

        overall_score = (
            capability_score * weights["capability"]
            + performance_score * weights["performance"]
            + resource_score * weights["resource"]
            + preference_score * weights["preference"]
        )

        # Calculate confidence based on data availability
        confidence = self._calculate_confidence(metadata)

        # Generate rationale
        rationale = self._generate_selection_rationale(
            tool_name,
            capability_score,
            performance_score,
            resource_score,
            preference_score,
        )

        return ToolScore(
            tool_name=tool_name,
            overall_score=overall_score,
            capability_score=capability_score,
            performance_score=performance_score,
            resource_score=resource_score,
            preference_score=preference_score,
            confidence=confidence,
            rationale=rationale,
        )

    def _calculate_capability_score(
        self, metadata: ToolMetadata, criteria: ToolSelectionCriteria
    ) -> float:
        """Calculate capability match score"""

        tool_capabilities = [
            metadata.capabilities.primary_capability
        ] + metadata.capabilities.secondary_capabilities

        # Required capabilities score
        required_score = 1.0  # All required capabilities are already filtered

        # Preferred capabilities bonus
        preferred_matches = sum(
            1
            for pref_cap in criteria.preferred_capabilities
            if pref_cap in tool_capabilities
        )
        preferred_score = preferred_matches / max(
            len(criteria.preferred_capabilities), 1
        )

        # Primary capability bonus
        primary_bonus = (
            0.2
            if metadata.capabilities.primary_capability
            in criteria.required_capabilities
            else 0.0
        )

        return min(1.0, required_score + preferred_score * 0.3 + primary_bonus)

    def _calculate_performance_score(
        self, metadata: ToolMetadata, criteria: ToolSelectionCriteria
    ) -> float:
        """Calculate performance score based on historical metrics"""

        metrics = metadata.performance_metrics

        # Success rate component (0-1)
        success_component = metrics.success_rate

        # Speed component (inverse of execution time, normalized)
        max_acceptable_time = criteria.max_execution_time or 300
        speed_component = max(
            0, 1 - (metrics.average_execution_time / max_acceptable_time)
        )

        # Throughput component
        throughput_component = min(
            1.0, metrics.throughput_per_minute / 10.0
        )  # Normalize to 10 per minute

        # Resource efficiency component
        efficiency_component = metrics.resource_efficiency

        # Weighted performance score
        performance_score = (
            success_component * 0.4
            + speed_component * 0.3
            + throughput_component * 0.2
            + efficiency_component * 0.1
        )

        return performance_score

    def _calculate_resource_score(
        self,
        metadata: ToolMetadata,
        criteria: ToolSelectionCriteria,
        context: ExecutionContext,
    ) -> float:
        """Calculate resource efficiency score"""

        requirements = metadata.resource_requirements
        constraints = context.resource_constraints

        # CPU efficiency
        max_cpu = constraints.get("max_cpu_cores", 4.0)
        cpu_score = max(0, 1 - (requirements.cpu_cores / max_cpu))

        # Memory efficiency
        max_memory = constraints.get("max_memory_mb", 2048)
        memory_score = max(0, 1 - (requirements.memory_mb / max_memory))

        # Time efficiency
        max_time = constraints.get("max_execution_time", 300)
        time_score = max(0, 1 - (requirements.max_execution_time / max_time))

        # Overall resource score
        resource_score = (cpu_score + memory_score + time_score) / 3

        return resource_score

    def _calculate_preference_score(
        self, tool_name: str, criteria: ToolSelectionCriteria, context: ExecutionContext
    ) -> float:
        """Calculate user preference score"""

        # Preferred tools bonus
        if tool_name in criteria.prefer_tools:
            return 1.0

        # User context preferences
        user_preferences = context.preferences
        preferred_categories = user_preferences.get("preferred_tool_categories", [])

        # Historical usage preference (simplified)
        usage_preference = 0.5  # Default neutral preference

        return usage_preference

    def _calculate_confidence(self, metadata: ToolMetadata) -> float:
        """Calculate confidence in the scoring based on data availability"""

        confidence = 0.5  # Base confidence

        # Usage history confidence
        if metadata.usage_count > 10:
            confidence += 0.2
        elif metadata.usage_count > 100:
            confidence += 0.3

        # Recent usage confidence
        if metadata.last_used and (datetime.utcnow() - metadata.last_used).days < 7:
            confidence += 0.1

        # Performance data confidence
        if (
            metadata.performance_metrics.last_updated
            and (datetime.utcnow() - metadata.performance_metrics.last_updated).days < 1
        ):
            confidence += 0.2

        return min(1.0, confidence)

    def _generate_selection_rationale(
        self,
        tool_name: str,
        capability_score: float,
        performance_score: float,
        resource_score: float,
        preference_score: float,
    ) -> str:
        """Generate human-readable rationale for tool selection"""

        reasons = []

        if capability_score > 0.8:
            reasons.append("excellent capability match")
        elif capability_score > 0.6:
            reasons.append("good capability match")

        if performance_score > 0.8:
            reasons.append("high performance metrics")
        elif performance_score > 0.6:
            reasons.append("acceptable performance")

        if resource_score > 0.8:
            reasons.append("efficient resource usage")

        if preference_score > 0.8:
            reasons.append("user preference alignment")

        if not reasons:
            reasons.append("best available option")

        return f"Selected {tool_name} due to: {', '.join(reasons)}"


# Legacy compatibility - use ToolRegistryManager from tool_registry.py instead
from .tool_registry import ToolRegistryManager as ToolExecutionRegistry

__all__ = [
    "BaseTool",
    "ToolCapabilities",
    "ResourceRequirements",
    "ToolResult",
    "ToolMetadata",
    "ExecutionContext",
    "ToolSelectionCriteria",
    "ToolScore",
    "ToolCapability",
    "ExecutionPriority",
    "ResourceType",
    "PerformanceMetrics",
    "ToolExecutionPool",
    "ToolSelector",
    "ToolExecutionRegistry",
]
