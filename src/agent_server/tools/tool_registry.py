"""
Tool Registry and Management System
Centralized registry for tool discovery, lifecycle management, and performance monitoring
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import json
import hashlib
import time
import uuid
import sqlite3

from src.shared.logging import get_logger
from .registry import (
    BaseTool,
    ToolCapabilities,
    ResourceRequirements,
    ToolResult,
    ToolMetadata as BaseToolMetadata,
    ToolCapability,
    PerformanceMetrics,
)

logger = get_logger(__name__)


class ToolStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    MAINTENANCE = "maintenance"
    FAILED = "failed"


class ToolVersion(Enum):
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


@dataclass
class ToolMetadata(BaseToolMetadata):
    """Extended tool metadata for registry management"""

    id: str = ""
    status: ToolStatus = ToolStatus.ACTIVE
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Performance metrics
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_execution_time: float = 0.0
    last_execution: Optional[datetime] = None

    # Quality metrics
    validation_score: float = 0.0
    user_rating: float = 0.0
    usage_frequency: float = 0.0

    # Lifecycle information
    deprecation_date: Optional[datetime] = None
    replacement_tool_id: Optional[str] = None
    maintenance_notes: str = ""

    # Schema field for compatibility
    schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolUsageStats:
    """Tool usage statistics"""

    tool_id: str
    execution_count: int
    success_rate: float
    avg_execution_time: float
    peak_usage_time: str
    error_patterns: List[str]
    user_feedback: List[Dict[str, Any]]
    performance_trend: str  # improving, stable, declining


@dataclass
class ToolRecommendation:
    """Tool recommendation based on usage patterns and capabilities"""

    tool_id: str
    tool_name: str
    confidence_score: float
    reason: str
    alternative_tools: List[str]
    estimated_performance: Dict[str, float]


class ToolDatabase:
    """SQLite database for tool registry persistence"""

    def __init__(self, db_path: str = "data/tools.db"):
        self.db_path = db_path
        # Ensure data directory exists
        import os

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database schema"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Tools table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tools (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    description TEXT,
                    author TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    status TEXT,
                    tags TEXT,
                    category TEXT,
                    dependencies TEXT,
                    capabilities TEXT,
                    resource_requirements TEXT,
                    schema TEXT,
                    total_executions INTEGER DEFAULT 0,
                    successful_executions INTEGER DEFAULT 0,
                    failed_executions INTEGER DEFAULT 0,
                    avg_execution_time REAL DEFAULT 0.0,
                    last_execution TEXT,
                    validation_score REAL DEFAULT 0.0,
                    user_rating REAL DEFAULT 0.0,
                    usage_frequency REAL DEFAULT 0.0,
                    deprecation_date TEXT,
                    replacement_tool_id TEXT,
                    maintenance_notes TEXT
                )
            """
            )

            # Tool executions table for detailed tracking
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_id TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    execution_time REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    parameters TEXT,
                    result_size INTEGER,
                    timestamp TEXT,
                    FOREIGN KEY (tool_id) REFERENCES tools (id)
                )
            """
            )

            # Tool ratings table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_id TEXT,
                    user_id TEXT,
                    rating INTEGER,
                    feedback TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (tool_id) REFERENCES tools (id)
                )
            """
            )

            # Tool versions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_id TEXT,
                    version TEXT,
                    code TEXT,
                    changelog TEXT,
                    created_at TEXT,
                    FOREIGN KEY (tool_id) REFERENCES tools (id)
                )
            """
            )

            conn.commit()

    async def save_tool(self, metadata: ToolMetadata) -> bool:
        """Save tool metadata to database"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Convert complex objects to JSON
                capabilities_json = (
                    json.dumps(self._serialize_capabilities(metadata.capabilities))
                    if metadata.capabilities
                    else None
                )
                resource_requirements_json = (
                    json.dumps(asdict(metadata.resource_requirements))
                    if metadata.resource_requirements
                    else None
                )

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO tools (
                        id, name, version, description, author, created_at, updated_at,
                        status, tags, category, dependencies, capabilities, resource_requirements,
                        schema, total_executions, successful_executions, failed_executions,
                        avg_execution_time, last_execution, validation_score, user_rating,
                        usage_frequency, deprecation_date, replacement_tool_id, maintenance_notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        metadata.id,
                        metadata.name,
                        metadata.version,
                        metadata.description,
                        metadata.author,
                        metadata.created_at.isoformat(),
                        metadata.updated_at.isoformat(),
                        metadata.status.value,
                        json.dumps(metadata.tags),
                        metadata.category,
                        json.dumps(metadata.dependencies),
                        capabilities_json,
                        resource_requirements_json,
                        json.dumps(metadata.schema),
                        metadata.total_executions,
                        metadata.successful_executions,
                        metadata.failed_executions,
                        metadata.avg_execution_time,
                        (
                            metadata.last_execution.isoformat()
                            if metadata.last_execution
                            else None
                        ),
                        metadata.validation_score,
                        metadata.user_rating,
                        metadata.usage_frequency,
                        (
                            metadata.deprecation_date.isoformat()
                            if metadata.deprecation_date
                            else None
                        ),
                        metadata.replacement_tool_id,
                        metadata.maintenance_notes,
                    ),
                )

                conn.commit()
                return True

        except Exception as e:
            logger.error(
                "Failed to save tool to database", tool_id=metadata.id, error=str(e)
            )
            return False

    async def get_tool(self, tool_id: str) -> Optional[ToolMetadata]:
        """Get tool metadata from database"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM tools WHERE id = ?", (tool_id,))
                row = cursor.fetchone()

                if row:
                    return self._row_to_metadata(row)
                return None

        except Exception as e:
            logger.error(
                "Failed to get tool from database", tool_id=tool_id, error=str(e)
            )
            return None

    async def list_tools(
        self,
        status: Optional[ToolStatus] = None,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> List[ToolMetadata]:
        """List tools with optional filtering"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = "SELECT * FROM tools"
                params = []
                conditions = []

                if status:
                    conditions.append("status = ?")
                    params.append(status.value)

                if category:
                    conditions.append("category = ?")
                    params.append(category)

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY updated_at DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                return [self._row_to_metadata(row) for row in rows]

        except Exception as e:
            logger.error("Failed to list tools from database", error=str(e))
            return []

    async def delete_tool(self, tool_id: str) -> bool:
        """Delete tool from database"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # First check if tool exists and get rowcount from tools table
                cursor.execute("DELETE FROM tools WHERE id = ?", (tool_id,))
                tools_deleted = cursor.rowcount

                # Delete related records
                cursor.execute(
                    "DELETE FROM tool_executions WHERE tool_id = ?", (tool_id,)
                )
                cursor.execute("DELETE FROM tool_ratings WHERE tool_id = ?", (tool_id,))
                cursor.execute(
                    "DELETE FROM tool_versions WHERE tool_id = ?", (tool_id,)
                )
                conn.commit()

                return tools_deleted > 0

        except Exception as e:
            logger.error(
                "Failed to delete tool from database", tool_id=tool_id, error=str(e)
            )
            return False

    async def record_execution(
        self,
        tool_id: str,
        user_id: str,
        session_id: str,
        execution_time: float,
        success: bool,
        error_message: str = "",
        parameters: Dict[str, Any] = None,
        result_size: int = 0,
    ):
        """Record tool execution for analytics"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO tool_executions (
                        tool_id, user_id, session_id, execution_time, success,
                        error_message, parameters, result_size, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        tool_id,
                        user_id,
                        session_id,
                        execution_time,
                        success,
                        error_message,
                        json.dumps(parameters or {}),
                        result_size,
                        datetime.utcnow().isoformat(),
                    ),
                )
                conn.commit()

        except Exception as e:
            logger.error("Failed to record execution", tool_id=tool_id, error=str(e))

    def _serialize_capabilities(self, capabilities: ToolCapabilities) -> Dict[str, Any]:
        """Serialize ToolCapabilities to JSON-serializable dict"""
        return {
            "primary_capability": capabilities.primary_capability.value,
            "secondary_capabilities": [
                cap.value for cap in capabilities.secondary_capabilities
            ],
            "input_types": capabilities.input_types,
            "output_types": capabilities.output_types,
            "supported_formats": capabilities.supported_formats,
            "language_support": capabilities.language_support,
        }

    def _row_to_metadata(self, row) -> ToolMetadata:
        """Convert database row to ToolMetadata object"""

        # Parse JSON fields
        tags = json.loads(row[8]) if row[8] else []
        dependencies = json.loads(row[10]) if row[10] else []

        # Reconstruct capabilities
        capabilities = None
        if row[11]:
            cap_data = json.loads(row[11])
            if isinstance(cap_data, dict) and "primary_capability" in cap_data:
                try:
                    primary_cap = ToolCapability(cap_data["primary_capability"])
                    secondary_caps = [
                        ToolCapability(cap)
                        for cap in cap_data.get("secondary_capabilities", [])
                    ]
                    capabilities = ToolCapabilities(
                        primary_capability=primary_cap,
                        secondary_capabilities=secondary_caps,
                        input_types=cap_data.get("input_types", []),
                        output_types=cap_data.get("output_types", []),
                        supported_formats=cap_data.get("supported_formats", []),
                        language_support=cap_data.get("language_support", []),
                    )
                except (ValueError, KeyError):
                    capabilities = None

        # Reconstruct resource requirements
        resource_requirements = None
        if row[12]:
            req_data = json.loads(row[12])
            if isinstance(req_data, dict):
                try:
                    resource_requirements = ResourceRequirements(
                        cpu_cores=req_data.get("cpu_cores", 1.0),
                        memory_mb=req_data.get("memory_mb", 512),
                        network_bandwidth_mbps=req_data.get(
                            "network_bandwidth_mbps", 10.0
                        ),
                        storage_mb=req_data.get("storage_mb", 100),
                        gpu_memory_mb=req_data.get("gpu_memory_mb", 0),
                        max_execution_time=req_data.get("max_execution_time", 300),
                        concurrent_limit=req_data.get("concurrent_limit", 5),
                    )
                except (ValueError, KeyError):
                    resource_requirements = ResourceRequirements()

        schema = json.loads(row[13]) if row[13] else {}

        # Create performance metrics placeholder
        performance_metrics = PerformanceMetrics()

        return ToolMetadata(
            id=row[0],
            name=row[1],
            version=row[2],
            description=row[3] or "",
            author=row[4] or "",
            created_at=datetime.fromisoformat(row[5]),
            updated_at=datetime.fromisoformat(row[6]),
            status=ToolStatus(row[7]),
            tags=tags,
            category=row[9] or "",
            dependencies=dependencies,
            capabilities=capabilities,
            resource_requirements=resource_requirements or ResourceRequirements(),
            performance_metrics=performance_metrics,
            parameters=schema.get("parameters", {}),
            required_params=schema.get("required_params", []),
            schema=schema,
            total_executions=row[14],
            successful_executions=row[15],
            failed_executions=row[16],
            avg_execution_time=row[17],
            last_execution=datetime.fromisoformat(row[18]) if row[18] else None,
            validation_score=row[19],
            user_rating=row[20],
            usage_frequency=row[21],
            deprecation_date=datetime.fromisoformat(row[22]) if row[22] else None,
            replacement_tool_id=row[23],
            maintenance_notes=row[24] or "",
        )


class ToolRegistryManager:
    """Enhanced tool registry with database persistence and lifecycle management"""

    def __init__(self, db_path: str = "data/tools.db"):
        self.db = ToolDatabase(db_path)
        self.active_tools: Dict[str, BaseTool] = {}
        self.tool_cache: Dict[str, ToolMetadata] = {}
        self.performance_monitor = PerformanceMonitor(self.db)
        self.recommendation_engine = RecommendationEngine(self.db)
        self.version_manager = VersionManager(self.db)
        self.analytics_engine = AnalyticsEngine(self.db)
        self._initialized = False

    async def initialize(self):
        """Initialize the tool registry"""
        if self._initialized:
            return

        logger.info("Initializing Tool Registry Manager")

        # Load existing tools from database
        try:
            existing_tools = await self.db.list_tools(limit=1000)
            for tool_metadata in existing_tools:
                self.tool_cache[tool_metadata.id] = tool_metadata

            logger.info(f"Loaded {len(existing_tools)} tools from database")
        except Exception as e:
            logger.warning(f"Failed to load tools from database: {str(e)}")

        self._initialized = True
        logger.info("Tool Registry Manager initialized successfully")

    async def shutdown(self):
        """Shutdown the tool registry and cleanup resources"""
        logger.info("Shutting down Tool Registry Manager")

        # Cleanup active tools
        for tool_id, tool in self.active_tools.items():
            try:
                if hasattr(tool, "cleanup"):
                    await tool.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up tool {tool_id}: {str(e)}")

        self.active_tools.clear()
        self.tool_cache.clear()

        logger.info("Tool Registry Manager shutdown complete")

    async def list_tools(
        self,
        status: Optional[ToolStatus] = None,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """List all available tools for API response with optional filtering"""
        try:
            tools_metadata = await self.list_tools_metadata(
                status=status, category=category, limit=limit
            )

            tools_list = []
            for metadata in tools_metadata:
                tools_list.append(
                    {
                        "id": metadata.id,
                        "name": metadata.name,
                        "description": metadata.description,
                        "category": metadata.category,
                        "version": metadata.version,
                        "status": metadata.status.value,
                        "tags": metadata.tags,
                        "capabilities": {
                            "primary": (
                                metadata.capabilities.primary_capability.value
                                if metadata.capabilities
                                else None
                            ),
                            "secondary": (
                                [
                                    cap.value
                                    for cap in metadata.capabilities.secondary_capabilities
                                ]
                                if metadata.capabilities
                                else []
                            ),
                        },
                        "usage_count": metadata.total_executions,
                        "success_rate": (
                            (
                                metadata.successful_executions
                                / max(metadata.total_executions, 1)
                            )
                            if metadata.total_executions > 0
                            else 0.0
                        ),
                        "parameters": metadata.parameters,
                        "required_params": metadata.required_params,
                        "schema": metadata.schema,
                    }
                )

            return {
                "tools": tools_list,
                "total_count": len(tools_list),
                "active_count": len([t for t in tools_list if t["status"] == "active"]),
            }
        except Exception as e:
            logger.error(f"Failed to list tools: {str(e)}")
            return {"tools": [], "total_count": 0, "active_count": 0}

    async def list_tools_metadata(
        self,
        status: Optional[ToolStatus] = None,
        category: Optional[str] = None,
        tags: List[str] = None,
        limit: int = 100,
    ) -> List[ToolMetadata]:
        """List tools with filtering options (renamed from list_tools)"""

        tools = await self.db.list_tools(status, category, limit)

        # Filter by tags if specified
        if tags:
            tools = [tool for tool in tools if any(tag in tool.tags for tag in tags)]

        return tools

    async def register_tool(
        self,
        tool: BaseTool,
        code: str = "",
        author: str = "system",
        tags: List[str] = None,
        validation_score: float = 0.0,
    ) -> str:
        """Register a new tool in the registry"""

        logger.info("Registering new tool", tool_name=tool.__class__.__name__)

        try:
            # Generate tool ID
            tool_id = self._generate_tool_id(tool.__class__.__name__)

            # Get tool information
            schema = tool.get_schema()
            capabilities = tool.get_capabilities()
            resource_requirements = tool.get_resource_requirements()

            # Create performance metrics placeholder
            performance_metrics = PerformanceMetrics()

            # Create metadata
            metadata = ToolMetadata(
                id=tool_id,
                name=schema.get("name", tool.__class__.__name__),
                version="1.0.0",
                description=schema.get("description", ""),
                author=author,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                status=ToolStatus.ACTIVE,
                tags=tags or [],
                category=self._infer_category(capabilities),
                dependencies=self._extract_dependencies(code),
                capabilities=capabilities,
                resource_requirements=resource_requirements,
                performance_metrics=performance_metrics,
                parameters=schema.get("parameters", {}),
                required_params=schema.get("required_params", []),
                schema=schema,
                validation_score=validation_score,
            )

            # Save to database
            success = await self.db.save_tool(metadata)

            if success:
                # Cache the tool
                self.active_tools[tool_id] = tool
                self.tool_cache[tool_id] = metadata

                logger.info(
                    "Tool registered successfully",
                    tool_id=tool_id,
                    tool_name=metadata.name,
                )

                return tool_id
            else:
                raise Exception("Failed to save tool to database")

        except Exception as e:
            logger.error(
                "Failed to register tool",
                tool_name=tool.__class__.__name__,
                error=str(e),
            )
            raise

    async def get_tool(self, tool_id: str) -> Optional[BaseTool]:
        """Get tool instance by ID"""

        # Check active tools cache first
        if tool_id in self.active_tools:
            return self.active_tools[tool_id]

        # Check if tool exists in database
        metadata = await self.db.get_tool(tool_id)
        if metadata and metadata.status == ToolStatus.ACTIVE:
            # Tool exists but not loaded - would need to load from code storage
            logger.warning("Tool exists but not loaded in memory", tool_id=tool_id)

        return None

    async def get_tool_metadata(self, tool_id: str) -> Optional[ToolMetadata]:
        """Get tool metadata by ID"""

        # Check cache first
        if tool_id in self.tool_cache:
            return self.tool_cache[tool_id]

        # Load from database
        metadata = await self.db.get_tool(tool_id)
        if metadata:
            self.tool_cache[tool_id] = metadata

        return metadata

    async def search_tools(self, query: str, limit: int = 20) -> List[ToolMetadata]:
        """Search tools by name, description, or tags"""

        all_tools = await self.db.list_tools(limit=1000)  # Get more tools for search

        query_lower = query.lower()
        matching_tools = []

        for tool in all_tools:
            score = 0

            # Name match (highest weight)
            if query_lower in tool.name.lower():
                score += 10

            # Description match
            if query_lower in tool.description.lower():
                score += 5

            # Tag match
            for tag in tool.tags:
                if query_lower in tag.lower():
                    score += 3

            # Category match
            if query_lower in tool.category.lower():
                score += 2

            if score > 0:
                matching_tools.append((tool, score))

        # Sort by score and return top results
        matching_tools.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, score in matching_tools[:limit]]

    async def update_tool_status(
        self, tool_id: str, status: ToolStatus, maintenance_notes: str = ""
    ) -> bool:
        """Update tool status"""

        metadata = await self.get_tool_metadata(tool_id)
        if not metadata:
            return False

        metadata.status = status
        metadata.updated_at = datetime.utcnow()
        metadata.maintenance_notes = maintenance_notes

        if status == ToolStatus.INACTIVE:
            # Remove from active tools
            self.active_tools.pop(tool_id, None)

        success = await self.db.save_tool(metadata)
        if success:
            self.tool_cache[tool_id] = metadata

        return success

    async def deprecate_tool(
        self,
        tool_id: str,
        replacement_tool_id: str = None,
        deprecation_date: datetime = None,
    ) -> bool:
        """Deprecate a tool"""

        metadata = await self.get_tool_metadata(tool_id)
        if not metadata:
            return False

        metadata.status = ToolStatus.DEPRECATED
        metadata.updated_at = datetime.utcnow()
        metadata.deprecation_date = deprecation_date or datetime.utcnow()
        metadata.replacement_tool_id = replacement_tool_id

        # Remove from active tools
        self.active_tools.pop(tool_id, None)

        success = await self.db.save_tool(metadata)
        if success:
            self.tool_cache[tool_id] = metadata

        logger.info(
            "Tool deprecated", tool_id=tool_id, replacement_tool_id=replacement_tool_id
        )

        return success

    async def delete_tool(self, tool_id: str) -> bool:
        """Delete a tool from the registry"""

        # Remove from caches
        self.active_tools.pop(tool_id, None)
        self.tool_cache.pop(tool_id, None)

        # Delete from database
        success = await self.db.delete_tool(tool_id)

        if success:
            logger.info("Tool deleted", tool_id=tool_id)

        return success

    async def record_execution(
        self,
        tool_id: str,
        user_id: str,
        session_id: str,
        result: ToolResult,
        parameters: Dict[str, Any] = None,
    ):
        """Record tool execution for analytics"""

        await self.db.record_execution(
            tool_id=tool_id,
            user_id=user_id,
            session_id=session_id,
            execution_time=result.execution_time,
            success=result.success,
            error_message=result.error_message or "",
            parameters=parameters,
            result_size=len(str(result.data)) if result.data else 0,
        )

        # Update tool metadata
        await self._update_tool_performance_metrics(tool_id, result)

    async def get_tool_usage_stats(
        self, tool_id: str, days: int = 30
    ) -> Optional[ToolUsageStats]:
        """Get tool usage statistics"""

        return await self.performance_monitor.get_usage_stats(tool_id, days)

    async def get_tool_recommendations(
        self, user_id: str, context: Dict[str, Any] = None, limit: int = 5
    ) -> List[ToolRecommendation]:
        """Get tool recommendations for user"""

        return await self.recommendation_engine.get_recommendations(
            user_id, context, limit
        )

    async def cleanup_deprecated_tools(self, days_old: int = 90):
        """Clean up old deprecated tools"""

        cutoff_date = datetime.utcnow() - timedelta(days=days_old)

        deprecated_tools = await self.db.list_tools(status=ToolStatus.DEPRECATED)

        for tool in deprecated_tools:
            if tool.deprecation_date and tool.deprecation_date < cutoff_date:
                await self.delete_tool(tool.id)
                logger.info(
                    "Cleaned up deprecated tool", tool_id=tool.id, tool_name=tool.name
                )

    async def update_tool_version(
        self,
        tool_id: str,
        version_type: ToolVersion,
        code: str = "",
        changelog: str = "",
    ) -> bool:
        """Update tool to a new version"""

        metadata = await self.get_tool_metadata(tool_id)
        if not metadata:
            return False

        # Generate new version number
        new_version = self.version_manager.increment_version(
            metadata.version, version_type
        )

        # Create version record
        success = await self.version_manager.create_version(
            tool_id, new_version, code, changelog
        )

        if success:
            logger.info(
                "Tool version updated",
                tool_id=tool_id,
                old_version=metadata.version,
                new_version=new_version,
            )

        return success

    async def get_system_analytics(self) -> Dict[str, Any]:
        """Get system-wide analytics and insights"""

        try:
            # Get performance overview
            performance_overview = (
                await self.performance_monitor.get_system_performance_overview()
            )

            # Get usage report for last 30 days
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)
            usage_report = await self.analytics_engine.generate_usage_report(
                start_date, end_date
            )

            # Get performance anomalies
            anomalies = await self.analytics_engine.detect_performance_anomalies()

            # Get tool status distribution
            all_tools = await self.db.list_tools(limit=1000)
            status_distribution = {}
            for tool in all_tools:
                status = tool.status.value
                status_distribution[status] = status_distribution.get(status, 0) + 1

            return {
                "performance_overview": performance_overview,
                "usage_report": usage_report,
                "anomalies": anomalies,
                "tool_status_distribution": status_distribution,
                "total_tools": len(all_tools),
                "generated_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Failed to get system analytics", error=str(e))
            return {}

    async def rate_tool(
        self, tool_id: str, user_id: str, rating: int, feedback: str = ""
    ) -> bool:
        """Rate a tool (1-5 stars)"""

        if not (1 <= rating <= 5):
            return False

        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO tool_ratings (tool_id, user_id, rating, feedback, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (tool_id, user_id, rating, feedback, datetime.utcnow().isoformat()),
                )

                conn.commit()

                # Update tool metadata with new average rating
                await self._update_tool_rating(tool_id)

                logger.info(
                    "Tool rated", tool_id=tool_id, user_id=user_id, rating=rating
                )
                return True

        except Exception as e:
            logger.error("Failed to rate tool", tool_id=tool_id, error=str(e))
            return False

    async def _update_tool_rating(self, tool_id: str):
        """Update tool's average rating"""

        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT AVG(rating) as avg_rating
                    FROM tool_ratings
                    WHERE tool_id = ?
                """,
                    (tool_id,),
                )

                result = cursor.fetchone()
                if result and result[0]:
                    avg_rating = result[0]

                    metadata = await self.get_tool_metadata(tool_id)
                    if metadata:
                        metadata.user_rating = avg_rating
                        metadata.updated_at = datetime.utcnow()
                        await self.db.save_tool(metadata)

                        # Update cache
                        self.tool_cache[tool_id] = metadata

        except Exception as e:
            logger.error("Failed to update tool rating", tool_id=tool_id, error=str(e))

    async def discover_tools_by_capability(
        self, capabilities: List[ToolCapability], limit: int = 20
    ) -> List[ToolMetadata]:
        """Discover tools by specific capabilities"""

        all_tools = await self.db.list_tools(status=ToolStatus.ACTIVE, limit=1000)

        matching_tools = []
        for tool in all_tools:
            if tool.capabilities:
                tool_caps = [
                    tool.capabilities.primary_capability
                ] + tool.capabilities.secondary_capabilities

                # Check if tool has any of the required capabilities
                if any(cap in tool_caps for cap in capabilities):
                    matching_tools.append(tool)

        # Sort by relevance (number of matching capabilities and performance)
        def relevance_score(tool):
            if not tool.capabilities:
                return 0

            tool_caps = [
                tool.capabilities.primary_capability
            ] + tool.capabilities.secondary_capabilities
            matching_caps = sum(1 for cap in capabilities if cap in tool_caps)

            # Performance bonus
            success_rate = tool.successful_executions / max(tool.total_executions, 1)
            performance_bonus = success_rate * 0.5

            return matching_caps + performance_bonus

        matching_tools.sort(key=relevance_score, reverse=True)
        return matching_tools[:limit]

    def _generate_tool_id(self, tool_name: str) -> str:
        """Generate unique tool ID"""

        timestamp = str(int(time.time() * 1000000))  # Microsecond precision
        random_suffix = str(uuid.uuid4().hex[:4])  # Add random component
        hash_input = f"{tool_name}_{timestamp}_{random_suffix}"
        tool_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]

        return f"tool_{tool_hash}"

    def _infer_category(self, capabilities: ToolCapabilities) -> str:
        """Infer tool category from capabilities"""

        if not capabilities:
            return "general"

        primary_cap = capabilities.primary_capability.value

        category_mapping = {
            "knowledge_retrieval": "information",
            "content_generation": "generation",
            "document_generation": "generation",
            "code_generation": "development",
            "data_analysis": "analysis",
            "communication": "communication",
            "validation": "validation",
            "transformation": "processing",
        }

        return category_mapping.get(primary_cap, "general")

    def _extract_dependencies(self, code: str) -> List[str]:
        """Extract dependencies from tool code"""

        dependencies = []

        if not code:
            return dependencies

        # Simple regex-based extraction
        import re

        # Find import statements
        import_patterns = [
            r"^import\s+(\w+)",
            r"^from\s+(\w+)",
        ]

        for line in code.split("\n"):
            line = line.strip()
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    dep = match.group(1)
                    if dep not in ["src", "typing", "__future__"]:
                        dependencies.append(dep)

        return list(set(dependencies))  # Remove duplicates

    async def _update_tool_performance_metrics(self, tool_id: str, result: ToolResult):
        """Update tool performance metrics"""

        metadata = await self.get_tool_metadata(tool_id)
        if not metadata:
            return

        # Update execution counts
        metadata.total_executions += 1
        if result.success:
            metadata.successful_executions += 1
        else:
            metadata.failed_executions += 1

        # Update average execution time
        if metadata.total_executions == 1:
            metadata.avg_execution_time = result.execution_time
        else:
            # Running average
            metadata.avg_execution_time = (
                metadata.avg_execution_time * (metadata.total_executions - 1)
                + result.execution_time
            ) / metadata.total_executions

        metadata.last_execution = datetime.utcnow()
        metadata.updated_at = datetime.utcnow()

        # Save updated metadata
        await self.db.save_tool(metadata)
        self.tool_cache[tool_id] = metadata

    async def get_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        """Get tool instance by its friendly name from the ACTIVE cache."""

        # Iterate through the live, cached tool instances
        for tool_id, tool_instance in self.active_tools.items():

            # Check the metadata cache for the corresponding tool_id
            # We get metadata from tool_cache, not the instance, to get the name
            metadata = self.tool_cache.get(tool_id)

            # Check if metadata exists and the name matches
            if metadata and metadata.name == tool_name:
                return tool_instance  # Found it

        # If the loop finishes, the tool is not active or doesn't exist
        logger.error(
            "Tool not found in active_tools cache",
            tool_name=tool_name,
            active_tools_count=len(self.active_tools),
            available_tools=list(self.active_tools.keys()),
        )
        return None


class PerformanceMonitor:
    """Tool performance monitoring and analytics"""

    def __init__(self, db: ToolDatabase):
        self.db = db
        self.metrics_cache: Dict[str, ToolUsageStats] = {}
        self.cache_ttl = timedelta(minutes=15)
        self.last_cache_update: Dict[str, datetime] = {}

    async def get_usage_stats(
        self, tool_id: str, days: int = 30
    ) -> Optional[ToolUsageStats]:
        """Get comprehensive usage statistics for a tool"""

        # Check cache first
        if (
            tool_id in self.metrics_cache
            and tool_id in self.last_cache_update
            and datetime.utcnow() - self.last_cache_update[tool_id] < self.cache_ttl
        ):
            return self.metrics_cache[tool_id]

        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()

                # Get execution statistics
                cursor.execute(
                    """
                    SELECT 
                        COUNT(*) as total_executions,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_executions,
                        AVG(execution_time) as avg_execution_time,
                        MIN(timestamp) as first_execution,
                        MAX(timestamp) as last_execution
                    FROM tool_executions 
                    WHERE tool_id = ? AND timestamp >= datetime('now', '-{} days')
                """.format(
                        days
                    ),
                    (tool_id,),
                )

                stats_row = cursor.fetchone()

                if not stats_row or stats_row[0] == 0:
                    return None

                total_executions = stats_row[0]
                successful_executions = stats_row[1] or 0
                avg_execution_time = stats_row[2] or 0.0
                success_rate = (
                    successful_executions / total_executions
                    if total_executions > 0
                    else 0.0
                )

                # Get error patterns
                cursor.execute(
                    """
                    SELECT error_message, COUNT(*) as count
                    FROM tool_executions 
                    WHERE tool_id = ? AND success = 0 AND timestamp >= datetime('now', '-{} days')
                    GROUP BY error_message
                    ORDER BY count DESC
                    LIMIT 5
                """.format(
                        days
                    ),
                    (tool_id,),
                )

                error_patterns = [row[0] for row in cursor.fetchall() if row[0]]

                # Get peak usage time
                cursor.execute(
                    """
                    SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
                    FROM tool_executions 
                    WHERE tool_id = ? AND timestamp >= datetime('now', '-{} days')
                    GROUP BY hour
                    ORDER BY count DESC
                    LIMIT 1
                """.format(
                        days
                    ),
                    (tool_id,),
                )

                peak_hour_row = cursor.fetchone()
                peak_usage_time = (
                    f"{peak_hour_row[0]}:00-{int(peak_hour_row[0])+1}:00"
                    if peak_hour_row
                    else "N/A"
                )

                # Get user feedback
                cursor.execute(
                    """
                    SELECT rating, feedback, timestamp
                    FROM tool_ratings 
                    WHERE tool_id = ? AND timestamp >= datetime('now', '-{} days')
                    ORDER BY timestamp DESC
                    LIMIT 10
                """.format(
                        days
                    ),
                    (tool_id,),
                )

                user_feedback = [
                    {"rating": row[0], "feedback": row[1], "timestamp": row[2]}
                    for row in cursor.fetchall()
                ]

                # Determine performance trend
                performance_trend = await self._calculate_performance_trend(
                    tool_id, days
                )

                stats = ToolUsageStats(
                    tool_id=tool_id,
                    execution_count=total_executions,
                    success_rate=success_rate,
                    avg_execution_time=avg_execution_time,
                    peak_usage_time=peak_usage_time,
                    error_patterns=error_patterns,
                    user_feedback=user_feedback,
                    performance_trend=performance_trend,
                )

                # Cache the results
                self.metrics_cache[tool_id] = stats
                self.last_cache_update[tool_id] = datetime.utcnow()

                return stats

        except Exception as e:
            logger.error("Failed to get usage stats", tool_id=tool_id, error=str(e))
            return None

    async def _calculate_performance_trend(self, tool_id: str, days: int) -> str:
        """Calculate performance trend over time"""

        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()

                # Get performance data for two periods
                half_days = days // 2

                # Recent period
                cursor.execute(
                    """
                    SELECT AVG(execution_time), AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END)
                    FROM tool_executions 
                    WHERE tool_id = ? AND timestamp >= datetime('now', '-{} days')
                """.format(
                        half_days
                    ),
                    (tool_id,),
                )

                recent_row = cursor.fetchone()

                # Earlier period
                cursor.execute(
                    """
                    SELECT AVG(execution_time), AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END)
                    FROM tool_executions 
                    WHERE tool_id = ? 
                    AND timestamp >= datetime('now', '-{} days')
                    AND timestamp < datetime('now', '-{} days')
                """.format(
                        days, half_days
                    ),
                    (tool_id,),
                )

                earlier_row = cursor.fetchone()

                if not recent_row or not earlier_row:
                    return "stable"

                recent_time, recent_success = recent_row
                earlier_time, earlier_success = earlier_row

                if not all(
                    [recent_time, recent_success, earlier_time, earlier_success]
                ):
                    return "stable"

                # Compare metrics
                time_improvement = (
                    (earlier_time - recent_time) / earlier_time
                    if earlier_time > 0
                    else 0
                )
                success_improvement = recent_success - earlier_success

                # Determine trend
                if time_improvement > 0.1 or success_improvement > 0.05:
                    return "improving"
                elif time_improvement < -0.1 or success_improvement < -0.05:
                    return "declining"
                else:
                    return "stable"

        except Exception as e:
            logger.error(
                "Failed to calculate performance trend", tool_id=tool_id, error=str(e)
            )
            return "stable"

    async def get_system_performance_overview(self) -> Dict[str, Any]:
        """Get system-wide performance overview"""

        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()

                # Overall statistics
                cursor.execute(
                    """
                    SELECT 
                        COUNT(DISTINCT tool_id) as active_tools,
                        COUNT(*) as total_executions,
                        AVG(execution_time) as avg_execution_time,
                        AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as overall_success_rate
                    FROM tool_executions 
                    WHERE timestamp >= datetime('now', '-7 days')
                """
                )

                overview_row = cursor.fetchone()

                # Top performing tools
                cursor.execute(
                    """
                    SELECT 
                        tool_id,
                        COUNT(*) as executions,
                        AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
                        AVG(execution_time) as avg_time
                    FROM tool_executions 
                    WHERE timestamp >= datetime('now', '-7 days')
                    GROUP BY tool_id
                    ORDER BY success_rate DESC, executions DESC
                    LIMIT 5
                """
                )

                top_tools = [
                    {
                        "tool_id": row[0],
                        "executions": row[1],
                        "success_rate": row[2],
                        "avg_execution_time": row[3],
                    }
                    for row in cursor.fetchall()
                ]

                return {
                    "active_tools": overview_row[0] or 0,
                    "total_executions": overview_row[1] or 0,
                    "avg_execution_time": overview_row[2] or 0.0,
                    "overall_success_rate": overview_row[3] or 0.0,
                    "top_performing_tools": top_tools,
                    "timestamp": datetime.utcnow().isoformat(),
                }

        except Exception as e:
            logger.error("Failed to get system performance overview", error=str(e))
            return {}


class RecommendationEngine:
    """Tool recommendation engine based on usage patterns and capabilities"""

    def __init__(self, db: ToolDatabase):
        self.db = db
        self.recommendation_cache: Dict[str, List[ToolRecommendation]] = {}
        self.cache_ttl = timedelta(hours=1)
        self.last_cache_update: Dict[str, datetime] = {}

    async def get_recommendations(
        self, user_id: str, context: Dict[str, Any] = None, limit: int = 5
    ) -> List[ToolRecommendation]:
        """Generate tool recommendations for user"""

        cache_key = f"{user_id}_{hash(str(context))}"

        # Check cache first
        if (
            cache_key in self.recommendation_cache
            and cache_key in self.last_cache_update
            and datetime.utcnow() - self.last_cache_update[cache_key] < self.cache_ttl
        ):
            return self.recommendation_cache[cache_key][:limit]

        try:
            recommendations = []

            # Get user's tool usage history
            user_history = await self._get_user_tool_history(user_id)

            # Get all active tools
            all_tools = await self.db.list_tools(status=ToolStatus.ACTIVE, limit=1000)

            # Generate recommendations based on different strategies

            # 1. Collaborative filtering - tools used by similar users
            collaborative_recs = await self._collaborative_filtering_recommendations(
                user_id, user_history, all_tools, limit // 2
            )
            recommendations.extend(collaborative_recs)

            # 2. Content-based filtering - tools similar to user's preferences
            content_recs = await self._content_based_recommendations(
                user_history, all_tools, context, limit // 2
            )
            recommendations.extend(content_recs)

            # 3. Trending tools - popular tools with good performance
            if len(recommendations) < limit:
                trending_recs = await self._trending_tools_recommendations(
                    all_tools, limit - len(recommendations)
                )
                recommendations.extend(trending_recs)

            # Remove duplicates and sort by confidence
            seen_tools = set()
            unique_recommendations = []
            for rec in recommendations:
                if rec.tool_id not in seen_tools:
                    seen_tools.add(rec.tool_id)
                    unique_recommendations.append(rec)

            unique_recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
            final_recommendations = unique_recommendations[:limit]

            # Cache the results
            self.recommendation_cache[cache_key] = final_recommendations
            self.last_cache_update[cache_key] = datetime.utcnow()

            return final_recommendations

        except Exception as e:
            logger.error(
                "Failed to generate recommendations", user_id=user_id, error=str(e)
            )
            return []

    async def _get_user_tool_history(
        self, user_id: str, days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get user's tool usage history"""

        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT 
                        tool_id,
                        COUNT(*) as usage_count,
                        AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
                        MAX(timestamp) as last_used
                    FROM tool_executions 
                    WHERE user_id = ? AND timestamp >= datetime('now', '-{} days')
                    GROUP BY tool_id
                    ORDER BY usage_count DESC
                """.format(
                        days
                    ),
                    (user_id,),
                )

                return [
                    {
                        "tool_id": row[0],
                        "usage_count": row[1],
                        "success_rate": row[2],
                        "last_used": row[3],
                    }
                    for row in cursor.fetchall()
                ]

        except Exception as e:
            logger.error(
                "Failed to get user tool history", user_id=user_id, error=str(e)
            )
            return []

    async def _collaborative_filtering_recommendations(
        self,
        user_id: str,
        user_history: List[Dict[str, Any]],
        all_tools: List[ToolMetadata],
        limit: int,
    ) -> List[ToolRecommendation]:
        """Generate recommendations using collaborative filtering"""

        if not user_history:
            return []

        try:
            user_tools = {item["tool_id"] for item in user_history}

            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()

                # Find users with similar tool usage patterns
                similar_users_query = """
                    SELECT 
                        user_id,
                        COUNT(*) as common_tools
                    FROM tool_executions 
                    WHERE tool_id IN ({}) AND user_id != ?
                    GROUP BY user_id
                    HAVING common_tools >= 2
                    ORDER BY common_tools DESC
                    LIMIT 10
                """.format(
                    ",".join(["?" for _ in user_tools])
                )

                cursor.execute(similar_users_query, list(user_tools) + [user_id])
                similar_users = [row[0] for row in cursor.fetchall()]

                if not similar_users:
                    return []

                # Get tools used by similar users but not by current user
                recommended_tools_query = """
                    SELECT 
                        tool_id,
                        COUNT(DISTINCT user_id) as user_count,
                        AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate
                    FROM tool_executions 
                    WHERE user_id IN ({}) AND tool_id NOT IN ({})
                    GROUP BY tool_id
                    ORDER BY user_count DESC, success_rate DESC
                    LIMIT ?
                """.format(
                    ",".join(["?" for _ in similar_users]),
                    ",".join(["?" for _ in user_tools]),
                )

                cursor.execute(
                    recommended_tools_query, similar_users + list(user_tools) + [limit]
                )

                recommendations = []
                for row in cursor.fetchall():
                    tool_id, user_count, success_rate = row

                    # Find tool metadata
                    tool_metadata = next(
                        (t for t in all_tools if t.id == tool_id), None
                    )
                    if tool_metadata:
                        confidence = min(
                            0.9, (user_count / len(similar_users)) * success_rate
                        )

                        recommendations.append(
                            ToolRecommendation(
                                tool_id=tool_id,
                                tool_name=tool_metadata.name,
                                confidence_score=confidence,
                                reason=f"Used by {user_count} users with similar preferences",
                                alternative_tools=[],
                                estimated_performance={
                                    "success_rate": success_rate,
                                    "execution_time": tool_metadata.avg_execution_time,
                                },
                            )
                        )

                return recommendations

        except Exception as e:
            logger.error(
                "Failed to generate collaborative filtering recommendations",
                error=str(e),
            )
            return []

    async def _content_based_recommendations(
        self,
        user_history: List[Dict[str, Any]],
        all_tools: List[ToolMetadata],
        context: Dict[str, Any],
        limit: int,
    ) -> List[ToolRecommendation]:
        """Generate recommendations using content-based filtering"""

        if not user_history:
            return []

        try:
            # Get user's preferred tool categories and capabilities
            user_tool_ids = {item["tool_id"] for item in user_history}
            user_tools = [t for t in all_tools if t.id in user_tool_ids]

            if not user_tools:
                return []

            # Analyze user preferences
            preferred_categories = {}
            preferred_capabilities = {}

            for tool in user_tools:
                # Category preferences
                category = tool.category
                preferred_categories[category] = (
                    preferred_categories.get(category, 0) + 1
                )

                # Capability preferences
                if tool.capabilities:
                    cap = tool.capabilities.primary_capability.value
                    preferred_capabilities[cap] = preferred_capabilities.get(cap, 0) + 1

            # Find similar tools not used by user
            recommendations = []
            for tool in all_tools:
                if tool.id in user_tool_ids or tool.status != ToolStatus.ACTIVE:
                    continue

                score = 0.0
                reasons = []

                # Category similarity
                if tool.category in preferred_categories:
                    category_score = preferred_categories[tool.category] / len(
                        user_tools
                    )
                    score += category_score * 0.4
                    reasons.append(f"matches your preferred category: {tool.category}")

                # Capability similarity
                if tool.capabilities:
                    cap = tool.capabilities.primary_capability.value
                    if cap in preferred_capabilities:
                        cap_score = preferred_capabilities[cap] / len(user_tools)
                        score += cap_score * 0.4
                        reasons.append(f"has similar capabilities: {cap}")

                # Performance bonus
                if tool.avg_execution_time > 0 and tool.successful_executions > 0:
                    success_rate = tool.successful_executions / tool.total_executions
                    if success_rate > 0.8:
                        score += 0.2
                        reasons.append("has high success rate")

                # Context matching
                if context:
                    context_score = self._calculate_context_match(tool, context)
                    score += context_score * 0.3
                    if context_score > 0.5:
                        reasons.append("matches current context")

                if score > 0.3:  # Minimum threshold
                    recommendations.append(
                        ToolRecommendation(
                            tool_id=tool.id,
                            tool_name=tool.name,
                            confidence_score=min(0.95, score),
                            reason=f"Recommended because it {', '.join(reasons[:2])}",
                            alternative_tools=[],
                            estimated_performance={
                                "success_rate": tool.successful_executions
                                / max(tool.total_executions, 1),
                                "execution_time": tool.avg_execution_time,
                            },
                        )
                    )

            # Sort by confidence and return top results
            recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
            return recommendations[:limit]

        except Exception as e:
            logger.error(
                "Failed to generate content-based recommendations", error=str(e)
            )
            return []

    async def _trending_tools_recommendations(
        self, all_tools: List[ToolMetadata], limit: int
    ) -> List[ToolRecommendation]:
        """Generate recommendations based on trending/popular tools"""

        try:
            # Filter active tools with good performance
            trending_tools = []

            for tool in all_tools:
                if (
                    tool.status == ToolStatus.ACTIVE
                    and tool.total_executions > 10
                    and tool.successful_executions / max(tool.total_executions, 1) > 0.7
                ):
                    # Calculate trending score
                    usage_score = min(1.0, tool.total_executions / 100)
                    success_score = tool.successful_executions / tool.total_executions
                    recency_score = 1.0

                    if tool.last_execution:
                        days_since_use = (datetime.utcnow() - tool.last_execution).days
                        recency_score = max(0.1, 1.0 - (days_since_use / 30))

                    trending_score = (
                        usage_score * 0.4 + success_score * 0.4 + recency_score * 0.2
                    )

                    trending_tools.append((tool, trending_score))

            # Sort by trending score
            trending_tools.sort(key=lambda x: x[1], reverse=True)

            recommendations = []
            for tool, score in trending_tools[:limit]:
                recommendations.append(
                    ToolRecommendation(
                        tool_id=tool.id,
                        tool_name=tool.name,
                        confidence_score=min(0.8, score),
                        reason="Popular tool with good performance",
                        alternative_tools=[],
                        estimated_performance={
                            "success_rate": tool.successful_executions
                            / max(tool.total_executions, 1),
                            "execution_time": tool.avg_execution_time,
                        },
                    )
                )

            return recommendations

        except Exception as e:
            logger.error(
                "Failed to generate trending tools recommendations", error=str(e)
            )
            return []

    def _calculate_context_match(
        self, tool: ToolMetadata, context: Dict[str, Any]
    ) -> float:
        """Calculate how well a tool matches the current context"""

        score = 0.0

        # Match based on context keywords
        context_text = " ".join(
            str(v).lower() for v in context.values() if isinstance(v, str)
        )

        # Check tool name and description
        tool_text = f"{tool.name} {tool.description}".lower()

        # Simple keyword matching
        common_words = set(context_text.split()) & set(tool_text.split())
        if common_words:
            score += len(common_words) / max(len(context_text.split()), 1)

        # Check tags
        if tool.tags:
            tag_text = " ".join(tool.tags).lower()
            tag_matches = set(context_text.split()) & set(tag_text.split())
            if tag_matches:
                score += len(tag_matches) / len(tool.tags)

        return min(1.0, score)


class VersionManager:
    """Tool version management and lifecycle"""

    def __init__(self, db: ToolDatabase):
        self.db = db

    async def create_version(
        self, tool_id: str, version: str, code: str, changelog: str = ""
    ) -> bool:
        """Create a new version of a tool"""

        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO tool_versions (tool_id, version, code, changelog, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (tool_id, version, code, changelog, datetime.utcnow().isoformat()),
                )

                conn.commit()

                # Update tool metadata with new version
                tool_metadata = await self.db.get_tool(tool_id)
                if tool_metadata:
                    tool_metadata.version = version
                    tool_metadata.updated_at = datetime.utcnow()
                    await self.db.save_tool(tool_metadata)

                logger.info("Tool version created", tool_id=tool_id, version=version)
                return True

        except Exception as e:
            logger.error("Failed to create tool version", tool_id=tool_id, error=str(e))
            return False

    async def get_version_history(self, tool_id: str) -> List[Dict[str, Any]]:
        """Get version history for a tool"""

        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT version, changelog, created_at
                    FROM tool_versions
                    WHERE tool_id = ?
                    ORDER BY created_at DESC
                """,
                    (tool_id,),
                )

                return [
                    {"version": row[0], "changelog": row[1], "created_at": row[2]}
                    for row in cursor.fetchall()
                ]

        except Exception as e:
            logger.error("Failed to get version history", tool_id=tool_id, error=str(e))
            return []

    async def rollback_version(self, tool_id: str, target_version: str) -> bool:
        """Rollback tool to a previous version"""

        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()

                # Get the target version code
                cursor.execute(
                    """
                    SELECT code FROM tool_versions
                    WHERE tool_id = ? AND version = ?
                """,
                    (tool_id, target_version),
                )

                version_row = cursor.fetchone()
                if not version_row:
                    logger.error(
                        "Target version not found",
                        tool_id=tool_id,
                        version=target_version,
                    )
                    return False

                # Update tool metadata
                tool_metadata = await self.db.get_tool(tool_id)
                if tool_metadata:
                    tool_metadata.version = target_version
                    tool_metadata.updated_at = datetime.utcnow()
                    tool_metadata.maintenance_notes = (
                        f"Rolled back to version {target_version}"
                    )
                    await self.db.save_tool(tool_metadata)

                logger.info(
                    "Tool rolled back", tool_id=tool_id, target_version=target_version
                )
                return True

        except Exception as e:
            logger.error(
                "Failed to rollback tool version", tool_id=tool_id, error=str(e)
            )
            return False

    def parse_version(self, version: str) -> Tuple[int, int, int]:
        """Parse semantic version string"""

        try:
            parts = version.split(".")
            major = int(parts[0]) if len(parts) > 0 else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
            return major, minor, patch
        except (ValueError, IndexError):
            return 0, 0, 0

    def increment_version(self, current_version: str, version_type: ToolVersion) -> str:
        """Increment version based on type"""

        major, minor, patch = self.parse_version(current_version)

        if version_type == ToolVersion.MAJOR:
            return f"{major + 1}.0.0"
        elif version_type == ToolVersion.MINOR:
            return f"{major}.{minor + 1}.0"
        elif version_type == ToolVersion.PATCH:
            return f"{major}.{minor}.{patch + 1}"

        return current_version


class AnalyticsEngine:
    """Advanced analytics for tool usage and performance"""

    def __init__(self, db: ToolDatabase):
        self.db = db

    async def generate_usage_report(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive usage report"""

        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()

                # Overall statistics
                cursor.execute(
                    """
                    SELECT 
                        COUNT(DISTINCT tool_id) as unique_tools_used,
                        COUNT(DISTINCT user_id) as unique_users,
                        COUNT(*) as total_executions,
                        AVG(execution_time) as avg_execution_time,
                        AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as overall_success_rate
                    FROM tool_executions 
                    WHERE timestamp BETWEEN ? AND ?
                """,
                    (start_date.isoformat(), end_date.isoformat()),
                )

                overall_stats = cursor.fetchone()

                # Top tools by usage
                cursor.execute(
                    """
                    SELECT 
                        te.tool_id,
                        t.name,
                        COUNT(*) as executions,
                        AVG(CASE WHEN te.success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
                        AVG(te.execution_time) as avg_time
                    FROM tool_executions te
                    JOIN tools t ON te.tool_id = t.id
                    WHERE te.timestamp BETWEEN ? AND ?
                    GROUP BY te.tool_id, t.name
                    ORDER BY executions DESC
                    LIMIT 10
                """,
                    (start_date.isoformat(), end_date.isoformat()),
                )

                top_tools = [
                    {
                        "tool_id": row[0],
                        "tool_name": row[1],
                        "executions": row[2],
                        "success_rate": row[3],
                        "avg_execution_time": row[4],
                    }
                    for row in cursor.fetchall()
                ]

                # Usage by day
                cursor.execute(
                    """
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as executions,
                        COUNT(DISTINCT user_id) as unique_users
                    FROM tool_executions 
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """,
                    (start_date.isoformat(), end_date.isoformat()),
                )

                daily_usage = [
                    {"date": row[0], "executions": row[1], "unique_users": row[2]}
                    for row in cursor.fetchall()
                ]

                # Error analysis
                cursor.execute(
                    """
                    SELECT 
                        error_message,
                        COUNT(*) as count,
                        COUNT(DISTINCT tool_id) as affected_tools
                    FROM tool_executions 
                    WHERE timestamp BETWEEN ? AND ? AND success = 0
                    GROUP BY error_message
                    ORDER BY count DESC
                    LIMIT 10
                """,
                    (start_date.isoformat(), end_date.isoformat()),
                )

                error_analysis = [
                    {"error_message": row[0], "count": row[1], "affected_tools": row[2]}
                    for row in cursor.fetchall()
                ]

                return {
                    "report_period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                    },
                    "overall_statistics": {
                        "unique_tools_used": overall_stats[0] or 0,
                        "unique_users": overall_stats[1] or 0,
                        "total_executions": overall_stats[2] or 0,
                        "avg_execution_time": overall_stats[3] or 0.0,
                        "overall_success_rate": overall_stats[4] or 0.0,
                    },
                    "top_tools": top_tools,
                    "daily_usage": daily_usage,
                    "error_analysis": error_analysis,
                    "generated_at": datetime.utcnow().isoformat(),
                }

        except Exception as e:
            logger.error("Failed to generate usage report", error=str(e))
            return {}

    async def detect_performance_anomalies(
        self, tool_id: str = None
    ) -> List[Dict[str, Any]]:
        """Detect performance anomalies in tool usage"""

        anomalies = []

        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()

                # Query for tools to analyze
                if tool_id:
                    tools_query = "SELECT id, name FROM tools WHERE id = ?"
                    tools_params = (tool_id,)
                else:
                    tools_query = "SELECT id, name FROM tools WHERE status = 'active'"
                    tools_params = ()

                cursor.execute(tools_query, tools_params)
                tools = cursor.fetchall()

                for tool_id, tool_name in tools:
                    # Check for sudden increase in execution time
                    cursor.execute(
                        """
                        SELECT 
                            AVG(execution_time) as recent_avg,
                            (SELECT AVG(execution_time) 
                             FROM tool_executions 
                             WHERE tool_id = ? 
                             AND timestamp BETWEEN datetime('now', '-14 days') AND datetime('now', '-7 days')
                            ) as previous_avg
                        FROM tool_executions 
                        WHERE tool_id = ? AND timestamp >= datetime('now', '-7 days')
                    """,
                        (tool_id, tool_id),
                    )

                    time_result = cursor.fetchone()
                    if time_result and time_result[0] and time_result[1]:
                        recent_avg, previous_avg = time_result
                        if recent_avg > previous_avg * 1.5:  # 50% increase
                            anomalies.append(
                                {
                                    "tool_id": tool_id,
                                    "tool_name": tool_name,
                                    "type": "performance_degradation",
                                    "description": f"Execution time increased from {previous_avg:.2f}s to {recent_avg:.2f}s",
                                    "severity": (
                                        "medium"
                                        if recent_avg > previous_avg * 2
                                        else "low"
                                    ),
                                }
                            )

                    # Check for sudden increase in error rate
                    cursor.execute(
                        """
                        SELECT 
                            AVG(CASE WHEN success = 0 THEN 1.0 ELSE 0.0 END) as recent_error_rate,
                            (SELECT AVG(CASE WHEN success = 0 THEN 1.0 ELSE 0.0 END)
                             FROM tool_executions 
                             WHERE tool_id = ? 
                             AND timestamp BETWEEN datetime('now', '-14 days') AND datetime('now', '-7 days')
                            ) as previous_error_rate
                        FROM tool_executions 
                        WHERE tool_id = ? AND timestamp >= datetime('now', '-7 days')
                    """,
                        (tool_id, tool_id),
                    )

                    error_result = cursor.fetchone()
                    if (
                        error_result
                        and error_result[0] is not None
                        and error_result[1] is not None
                    ):
                        recent_error_rate, previous_error_rate = error_result
                        if (
                            recent_error_rate > previous_error_rate + 0.1
                        ):  # 10% increase
                            anomalies.append(
                                {
                                    "tool_id": tool_id,
                                    "tool_name": tool_name,
                                    "type": "error_rate_increase",
                                    "description": f"Error rate increased from {previous_error_rate:.1%} to {recent_error_rate:.1%}",
                                    "severity": (
                                        "high" if recent_error_rate > 0.3 else "medium"
                                    ),
                                }
                            )

                return anomalies

        except Exception as e:
            logger.error("Failed to detect performance anomalies", error=str(e))
            return []

    async def get_tool_health_score(self, tool_id: str) -> Dict[str, Any]:
        """Calculate comprehensive health score for a tool"""

        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()

                # Get recent performance metrics
                cursor.execute(
                    """
                    SELECT 
                        COUNT(*) as total_executions,
                        AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
                        AVG(execution_time) as avg_execution_time,
                        COUNT(DISTINCT user_id) as unique_users
                    FROM tool_executions 
                    WHERE tool_id = ? AND timestamp >= datetime('now', '-30 days')
                """,
                    (tool_id,),
                )

                metrics = cursor.fetchone()
                if not metrics or metrics[0] == 0:
                    return {"health_score": 0.0, "status": "no_data"}

                (
                    total_executions,
                    success_rate,
                    avg_execution_time,
                    unique_users,
                ) = metrics

                # Calculate component scores (0-1)

                # Reliability score (based on success rate)
                reliability_score = success_rate or 0.0

                # Performance score (based on execution time - lower is better)
                # Assume 5 seconds is the baseline, anything faster gets higher score
                performance_score = max(
                    0.0, min(1.0, 5.0 / max(avg_execution_time, 0.1))
                )

                # Usage score (based on execution frequency)
                usage_score = min(
                    1.0, total_executions / 100.0
                )  # 100 executions = full score

                # Adoption score (based on unique users)
                adoption_score = min(1.0, unique_users / 10.0)  # 10 users = full score

                # Get user ratings
                cursor.execute(
                    """
                    SELECT AVG(rating) as avg_rating
                    FROM tool_ratings 
                    WHERE tool_id = ? AND timestamp >= datetime('now', '-30 days')
                """,
                    (tool_id,),
                )

                rating_result = cursor.fetchone()
                user_rating_score = (
                    (rating_result[0] / 5.0) if rating_result[0] else 0.5
                )

                # Calculate weighted health score
                weights = {
                    "reliability": 0.3,
                    "performance": 0.25,
                    "usage": 0.2,
                    "adoption": 0.15,
                    "user_rating": 0.1,
                }

                health_score = (
                    reliability_score * weights["reliability"]
                    + performance_score * weights["performance"]
                    + usage_score * weights["usage"]
                    + adoption_score * weights["adoption"]
                    + user_rating_score * weights["user_rating"]
                )

                # Determine status
                if health_score >= 0.8:
                    status = "excellent"
                elif health_score >= 0.6:
                    status = "good"
                elif health_score >= 0.4:
                    status = "fair"
                else:
                    status = "poor"

                return {
                    "health_score": round(health_score, 3),
                    "status": status,
                    "component_scores": {
                        "reliability": round(reliability_score, 3),
                        "performance": round(performance_score, 3),
                        "usage": round(usage_score, 3),
                        "adoption": round(adoption_score, 3),
                        "user_rating": round(user_rating_score, 3),
                    },
                    "metrics": {
                        "total_executions": total_executions,
                        "success_rate": success_rate,
                        "avg_execution_time": avg_execution_time,
                        "unique_users": unique_users,
                    },
                    "calculated_at": datetime.utcnow().isoformat(),
                }

        except Exception as e:
            logger.error(
                "Failed to calculate tool health score", tool_id=tool_id, error=str(e)
            )
            return {"health_score": 0.0, "status": "error"}


# Enhanced registry manager methods
async def get_tool_analytics(self, tool_id: str) -> Dict[str, Any]:
    """Get comprehensive analytics for a tool"""

    analytics = {}

    try:
        # Get usage statistics
        usage_stats = await self.performance_monitor.get_usage_stats(tool_id)
        if usage_stats:
            analytics["usage_stats"] = asdict(usage_stats)

        # Get health score
        health_score = await self.analytics_engine.get_tool_health_score(tool_id)
        analytics["health_score"] = health_score

        # Get version history
        version_history = await self.version_manager.get_version_history(tool_id)
        analytics["version_history"] = version_history

        # Get performance anomalies
        anomalies = await self.analytics_engine.detect_performance_anomalies(tool_id)
        analytics["anomalies"] = anomalies

        return analytics

    except Exception as e:
        logger.error("Failed to get tool analytics", tool_id=tool_id, error=str(e))
        return {}


# Add the new method to ToolRegistryManager class
ToolRegistryManager.get_tool_analytics = get_tool_analytics
