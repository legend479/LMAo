"""
Test Suite for Tool Registry and Management System
Comprehensive tests for tool registry functionality
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta

from src.agent_server.tools.tool_registry import (
    ToolRegistryManager,
    ToolDatabase,
    ToolStatus,
    ToolVersion,
    ToolCapability,
    ToolMetadata,
    PerformanceMonitor,
    RecommendationEngine,
    VersionManager,
    AnalyticsEngine,
)
from src.agent_server.tools.registry import (
    BaseTool,
    ToolCapabilities,
    ResourceRequirements,
    ToolResult,
    ExecutionContext,
)


class MockTool(BaseTool):
    """Mock tool for testing"""

    def __init__(self, name: str = "test_tool", config: dict = None):
        super().__init__(config)
        self.name = name
        self.execution_count = 0

    async def execute(self, parameters: dict, context: ExecutionContext) -> ToolResult:
        self.execution_count += 1
        return ToolResult(
            data={"result": f"executed {self.name}"},
            metadata={"execution_count": self.execution_count},
            execution_time=1.0,
            success=True,
        )

    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": f"Test tool {self.name}",
            "parameters": {
                "input": {"type": "string", "description": "Input parameter"}
            },
            "required_params": ["input"],
        }

    def get_capabilities(self) -> ToolCapabilities:
        return ToolCapabilities(
            primary_capability=ToolCapability.CONTENT_GENERATION,
            secondary_capabilities=[ToolCapability.VALIDATION],
            input_types=["string"],
            output_types=["json"],
        )

    def get_resource_requirements(self) -> ResourceRequirements:
        return ResourceRequirements(cpu_cores=1.0, memory_mb=256, max_execution_time=30)


@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
async def registry_manager(temp_db):
    """Create registry manager with temporary database"""
    manager = ToolRegistryManager(temp_db)
    return manager


@pytest.fixture
def mock_tool():
    """Create mock tool for testing"""
    return MockTool("test_tool")


@pytest.fixture
def sample_metadata():
    """Create sample tool metadata"""
    return ToolMetadata(
        id="test_tool_123",
        name="Test Tool",
        version="1.0.0",
        description="A test tool for unit testing",
        author="test_author",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        status=ToolStatus.ACTIVE,
        tags=["test", "mock"],
        category="testing",
        dependencies=[],
        capabilities=ToolCapabilities(
            primary_capability=ToolCapability.CONTENT_GENERATION,
            secondary_capabilities=[ToolCapability.VALIDATION],
        ),
        resource_requirements=ResourceRequirements(),
        schema={"name": "Test Tool"},
        total_executions=10,
        successful_executions=9,
        failed_executions=1,
        avg_execution_time=1.5,
    )


class TestToolDatabase:
    """Test tool database operations"""

    @pytest.mark.asyncio
    async def test_database_initialization(self, temp_db):
        """Test database initialization"""
        db = ToolDatabase(temp_db)

        # Check that tables were created
        import sqlite3

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            expected_tables = [
                "tools",
                "tool_executions",
                "tool_ratings",
                "tool_versions",
            ]
            for table in expected_tables:
                assert table in tables

    @pytest.mark.asyncio
    async def test_save_and_get_tool(self, temp_db, sample_metadata):
        """Test saving and retrieving tool metadata"""
        db = ToolDatabase(temp_db)

        # Save tool
        success = await db.save_tool(sample_metadata)
        assert success

        # Retrieve tool
        retrieved = await db.get_tool(sample_metadata.id)
        assert retrieved is not None
        assert retrieved.id == sample_metadata.id
        assert retrieved.name == sample_metadata.name
        assert retrieved.version == sample_metadata.version

    @pytest.mark.asyncio
    async def test_list_tools(self, temp_db, sample_metadata):
        """Test listing tools with filters"""
        db = ToolDatabase(temp_db)

        # Save multiple tools
        tools = []
        for i in range(3):
            tool = ToolMetadata(
                id=f"tool_{i}",
                name=f"Tool {i}",
                version="1.0.0",
                description=f"Test tool {i}",
                author="test",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                status=ToolStatus.ACTIVE if i < 2 else ToolStatus.DEPRECATED,
                category="test",
                tags=[],
                dependencies=[],
                capabilities=None,
                resource_requirements=None,
                schema={},
            )
            tools.append(tool)
            await db.save_tool(tool)

        # List all tools
        all_tools = await db.list_tools()
        assert len(all_tools) == 3

        # List active tools only
        active_tools = await db.list_tools(status=ToolStatus.ACTIVE)
        assert len(active_tools) == 2

        # List deprecated tools only
        deprecated_tools = await db.list_tools(status=ToolStatus.DEPRECATED)
        assert len(deprecated_tools) == 1

    @pytest.mark.asyncio
    async def test_delete_tool(self, temp_db, sample_metadata):
        """Test deleting tool"""
        db = ToolDatabase(temp_db)

        # Save tool
        await db.save_tool(sample_metadata)

        # Verify it exists
        retrieved = await db.get_tool(sample_metadata.id)
        assert retrieved is not None

        # Delete tool
        success = await db.delete_tool(sample_metadata.id)
        assert success

        # Verify it's gone
        retrieved = await db.get_tool(sample_metadata.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_record_execution(self, temp_db, sample_metadata):
        """Test recording tool execution"""
        db = ToolDatabase(temp_db)

        # Save tool first
        await db.save_tool(sample_metadata)

        # Record execution
        await db.record_execution(
            tool_id=sample_metadata.id,
            user_id="test_user",
            session_id="test_session",
            execution_time=2.5,
            success=True,
            parameters={"input": "test"},
        )

        # Verify execution was recorded
        import sqlite3

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM tool_executions WHERE tool_id = ?",
                (sample_metadata.id,),
            )
            count = cursor.fetchone()[0]
            assert count == 1


class TestToolRegistryManager:
    """Test tool registry manager"""

    @pytest.mark.asyncio
    async def test_register_tool(self, registry_manager, mock_tool):
        """Test tool registration"""
        tool_id = await registry_manager.register_tool(
            mock_tool,
            code="# Mock tool code",
            author="test_author",
            tags=["test", "mock"],
        )

        assert tool_id is not None
        assert tool_id.startswith("tool_")

        # Verify tool is in registry
        retrieved_tool = await registry_manager.get_tool(tool_id)
        assert retrieved_tool is not None

        # Verify metadata
        metadata = await registry_manager.get_tool_metadata(tool_id)
        assert metadata is not None
        assert metadata.name == mock_tool.name
        assert metadata.author == "test_author"
        assert "test" in metadata.tags

    @pytest.mark.asyncio
    async def test_get_tool_metadata(self, registry_manager, mock_tool):
        """Test getting tool metadata"""
        tool_id = await registry_manager.register_tool(mock_tool)

        metadata = await registry_manager.get_tool_metadata(tool_id)
        assert metadata is not None
        assert metadata.id == tool_id
        assert metadata.status == ToolStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_update_tool_status(self, registry_manager, mock_tool):
        """Test updating tool status"""
        tool_id = await registry_manager.register_tool(mock_tool)

        # Update to inactive
        success = await registry_manager.update_tool_status(
            tool_id, ToolStatus.INACTIVE, "Testing status update"
        )
        assert success

        # Verify status changed
        metadata = await registry_manager.get_tool_metadata(tool_id)
        assert metadata.status == ToolStatus.INACTIVE
        assert metadata.maintenance_notes == "Testing status update"

    @pytest.mark.asyncio
    async def test_deprecate_tool(self, registry_manager, mock_tool):
        """Test tool deprecation"""
        tool_id = await registry_manager.register_tool(mock_tool)

        # Deprecate tool
        success = await registry_manager.deprecate_tool(tool_id, "replacement_tool_id")
        assert success

        # Verify deprecation
        metadata = await registry_manager.get_tool_metadata(tool_id)
        assert metadata.status == ToolStatus.DEPRECATED
        assert metadata.replacement_tool_id == "replacement_tool_id"
        assert metadata.deprecation_date is not None

    @pytest.mark.asyncio
    async def test_search_tools(self, registry_manager):
        """Test tool search functionality"""
        # Register multiple tools
        tools = []
        for i in range(3):
            tool = MockTool(f"search_tool_{i}")
            tool_id = await registry_manager.register_tool(
                tool, tags=["search", f"category_{i % 2}"]
            )
            tools.append(tool_id)

        # Search by name
        results = await registry_manager.search_tools("search_tool")
        assert len(results) >= 3

        # Search by partial name
        results = await registry_manager.search_tools("search")
        assert len(results) >= 3

    @pytest.mark.asyncio
    async def test_list_tools_with_filters(self, registry_manager):
        """Test listing tools with various filters"""
        # Register tools with different statuses
        active_tool = MockTool("active_tool")
        active_id = await registry_manager.register_tool(active_tool)

        inactive_tool = MockTool("inactive_tool")
        inactive_id = await registry_manager.register_tool(inactive_tool)
        await registry_manager.update_tool_status(inactive_id, ToolStatus.INACTIVE)

        # List all tools
        all_tools = await registry_manager.list_tools()
        assert len(all_tools) >= 2

        # List only active tools
        active_tools = await registry_manager.list_tools(status=ToolStatus.ACTIVE)
        active_ids = [t.id for t in active_tools]
        assert active_id in active_ids
        assert inactive_id not in active_ids

        # List only inactive tools
        inactive_tools = await registry_manager.list_tools(status=ToolStatus.INACTIVE)
        inactive_ids = [t.id for t in inactive_tools]
        assert inactive_id in inactive_ids
        assert active_id not in inactive_ids

    @pytest.mark.asyncio
    async def test_record_and_get_execution(self, registry_manager, mock_tool):
        """Test recording and retrieving execution data"""
        tool_id = await registry_manager.register_tool(mock_tool)

        # Create mock result
        result = ToolResult(
            data={"test": "data"}, metadata={}, execution_time=1.5, success=True
        )

        # Record execution
        await registry_manager.record_execution(
            tool_id, "test_user", "test_session", result, {"input": "test"}
        )

        # Get usage stats
        stats = await registry_manager.get_tool_usage_stats(tool_id)
        assert stats is not None
        assert stats.execution_count >= 1

    @pytest.mark.asyncio
    async def test_rate_tool(self, registry_manager, mock_tool):
        """Test tool rating functionality"""
        tool_id = await registry_manager.register_tool(mock_tool)

        # Rate the tool
        success = await registry_manager.rate_tool(
            tool_id, "test_user", 5, "Excellent tool!"
        )
        assert success

        # Verify rating was recorded
        metadata = await registry_manager.get_tool_metadata(tool_id)
        assert metadata.user_rating > 0


class TestPerformanceMonitor:
    """Test performance monitoring functionality"""

    @pytest.mark.asyncio
    async def test_get_usage_stats(self, temp_db, sample_metadata):
        """Test getting usage statistics"""
        db = ToolDatabase(temp_db)
        monitor = PerformanceMonitor(db)

        # Save tool and record some executions
        await db.save_tool(sample_metadata)

        for i in range(5):
            await db.record_execution(
                tool_id=sample_metadata.id,
                user_id=f"user_{i}",
                session_id=f"session_{i}",
                execution_time=1.0 + i * 0.1,
                success=i < 4,  # 4 successes, 1 failure
            )

        # Get usage stats
        stats = await monitor.get_usage_stats(sample_metadata.id)
        assert stats is not None
        assert stats.execution_count == 5
        assert stats.success_rate == 0.8  # 4/5

    @pytest.mark.asyncio
    async def test_system_performance_overview(self, temp_db):
        """Test system performance overview"""
        db = ToolDatabase(temp_db)
        monitor = PerformanceMonitor(db)

        # Create and save multiple tools with executions
        for i in range(3):
            metadata = ToolMetadata(
                id=f"perf_tool_{i}",
                name=f"Performance Tool {i}",
                version="1.0.0",
                description="Performance test tool",
                author="test",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                status=ToolStatus.ACTIVE,
                category="performance",
                tags=[],
                dependencies=[],
                capabilities=None,
                resource_requirements=None,
                schema={},
            )
            await db.save_tool(metadata)

            # Record executions
            for j in range(5):
                await db.record_execution(
                    tool_id=metadata.id,
                    user_id=f"user_{j}",
                    session_id=f"session_{j}",
                    execution_time=1.0,
                    success=True,
                )

        # Get overview
        overview = await monitor.get_system_performance_overview()
        assert overview["active_tools"] >= 3
        assert overview["total_executions"] >= 15


class TestRecommendationEngine:
    """Test recommendation engine functionality"""

    @pytest.mark.asyncio
    async def test_get_recommendations(self, temp_db):
        """Test getting tool recommendations"""
        db = ToolDatabase(temp_db)
        engine = RecommendationEngine(db)

        # Create some tools
        for i in range(3):
            metadata = ToolMetadata(
                id=f"rec_tool_{i}",
                name=f"Recommendation Tool {i}",
                version="1.0.0",
                description="Recommendation test tool",
                author="test",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                status=ToolStatus.ACTIVE,
                category="recommendation",
                tags=[f"tag_{i}"],
                dependencies=[],
                capabilities=None,
                resource_requirements=None,
                schema={},
                total_executions=10 + i,
                successful_executions=9 + i,
                user_rating=4.0 + i * 0.2,
            )
            await db.save_tool(metadata)

        # Get recommendations
        recommendations = await engine.get_recommendations("test_user", {}, 5)
        assert isinstance(recommendations, list)
        # Note: Recommendations might be empty if no usage history exists


class TestVersionManager:
    """Test version management functionality"""

    @pytest.mark.asyncio
    async def test_create_version(self, temp_db, sample_metadata):
        """Test creating tool versions"""
        db = ToolDatabase(temp_db)
        version_manager = VersionManager(db)

        # Save tool
        await db.save_tool(sample_metadata)

        # Create new version
        success = await version_manager.create_version(
            sample_metadata.id, "1.1.0", "# Updated code", "Bug fixes"
        )
        assert success

        # Get version history
        history = await version_manager.get_version_history(sample_metadata.id)
        assert len(history) >= 1
        assert any(v["version"] == "1.1.0" for v in history)

    def test_parse_version(self, temp_db):
        """Test version parsing"""
        db = ToolDatabase(temp_db)
        version_manager = VersionManager(db)

        # Test valid versions
        assert version_manager.parse_version("1.2.3") == (1, 2, 3)
        assert version_manager.parse_version("0.1.0") == (0, 1, 0)
        assert version_manager.parse_version("10.20.30") == (10, 20, 30)

        # Test invalid versions
        assert version_manager.parse_version("invalid") == (0, 0, 0)
        assert version_manager.parse_version("1.2") == (1, 2, 0)

    def test_increment_version(self, temp_db):
        """Test version incrementing"""
        db = ToolDatabase(temp_db)
        version_manager = VersionManager(db)

        # Test major increment
        assert version_manager.increment_version("1.2.3", ToolVersion.MAJOR) == "2.0.0"

        # Test minor increment
        assert version_manager.increment_version("1.2.3", ToolVersion.MINOR) == "1.3.0"

        # Test patch increment
        assert version_manager.increment_version("1.2.3", ToolVersion.PATCH) == "1.2.4"


class TestAnalyticsEngine:
    """Test analytics engine functionality"""

    @pytest.mark.asyncio
    async def test_generate_usage_report(self, temp_db, sample_metadata):
        """Test generating usage reports"""
        db = ToolDatabase(temp_db)
        analytics = AnalyticsEngine(db)

        # Save tool and record executions
        await db.save_tool(sample_metadata)

        for i in range(10):
            await db.record_execution(
                tool_id=sample_metadata.id,
                user_id=f"user_{i % 3}",  # 3 unique users
                session_id=f"session_{i}",
                execution_time=1.0,
                success=i < 8,  # 8 successes, 2 failures
            )

        # Generate report
        start_date = datetime.utcnow() - timedelta(days=7)
        end_date = datetime.utcnow()

        report = await analytics.generate_usage_report(start_date, end_date)

        assert "overall_statistics" in report
        assert "top_tools" in report
        assert "daily_usage" in report
        assert report["overall_statistics"]["total_executions"] >= 10

    @pytest.mark.asyncio
    async def test_get_tool_health_score(self, temp_db, sample_metadata):
        """Test calculating tool health scores"""
        db = ToolDatabase(temp_db)
        analytics = AnalyticsEngine(db)

        # Save tool and record executions
        await db.save_tool(sample_metadata)

        for i in range(20):
            await db.record_execution(
                tool_id=sample_metadata.id,
                user_id=f"user_{i % 5}",  # 5 unique users
                session_id=f"session_{i}",
                execution_time=1.0,
                success=True,
            )

        # Get health score
        health = await analytics.get_tool_health_score(sample_metadata.id)

        assert "health_score" in health
        assert "status" in health
        assert "component_scores" in health
        assert 0.0 <= health["health_score"] <= 1.0


class TestIntegration:
    """Integration tests for the complete tool registry system"""

    @pytest.mark.asyncio
    async def test_complete_tool_lifecycle(self, registry_manager):
        """Test complete tool lifecycle from registration to deletion"""
        mock_tool = MockTool("lifecycle_tool")

        # 1. Register tool
        tool_id = await registry_manager.register_tool(
            mock_tool, author="integration_test", tags=["integration", "lifecycle"]
        )
        assert tool_id is not None

        # 2. Verify registration
        metadata = await registry_manager.get_tool_metadata(tool_id)
        assert metadata is not None
        assert metadata.status == ToolStatus.ACTIVE

        # 3. Record some executions
        for i in range(5):
            result = ToolResult(
                data={"iteration": i}, metadata={}, execution_time=1.0, success=True
            )
            await registry_manager.record_execution(
                tool_id, f"user_{i}", f"session_{i}", result
            )

        # 4. Rate the tool
        await registry_manager.rate_tool(tool_id, "test_user", 5, "Great tool!")

        # 5. Update version
        success = await registry_manager.update_tool_version(
            tool_id, ToolVersion.MINOR, "# Updated code", "Added new features"
        )
        assert success

        # 6. Get analytics
        analytics = await registry_manager.get_tool_analytics(tool_id)
        assert analytics is not None

        # 7. Deprecate tool
        success = await registry_manager.deprecate_tool(tool_id)
        assert success

        # 8. Verify deprecation
        metadata = await registry_manager.get_tool_metadata(tool_id)
        assert metadata.status == ToolStatus.DEPRECATED

        # 9. Delete tool
        success = await registry_manager.delete_tool(tool_id)
        assert success

        # 10. Verify deletion
        metadata = await registry_manager.get_tool_metadata(tool_id)
        assert metadata is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
