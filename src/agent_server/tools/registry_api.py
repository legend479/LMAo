"""
Tool Registry API Interface
RESTful API endpoints for tool registry management
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from datetime import datetime

from src.shared.logging import get_logger
from src.shared.models import APIResponse, PaginationParams, PaginatedResponse
from .tool_registry import (
    ToolRegistryManager,
    ToolStatus,
    ToolVersion,
    ToolCapability,
)

logger = get_logger(__name__)

# Pydantic models for API requests/responses


class ToolRegistrationRequest(BaseModel):
    """Request model for tool registration"""

    name: str = Field(..., description="Tool name")
    description: str = Field("", description="Tool description")
    author: str = Field("system", description="Tool author")
    tags: List[str] = Field(default_factory=list, description="Tool tags")
    code: str = Field("", description="Tool implementation code")
    validation_score: float = Field(0.0, ge=0.0, le=1.0, description="Validation score")


class ToolUpdateRequest(BaseModel):
    """Request model for tool updates"""

    status: Optional[ToolStatus] = None
    maintenance_notes: str = ""
    replacement_tool_id: Optional[str] = None


class ToolVersionRequest(BaseModel):
    """Request model for tool version updates"""

    version_type: ToolVersion
    code: str = ""
    changelog: str = ""


class ToolRatingRequest(BaseModel):
    """Request model for tool rating"""

    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    feedback: str = Field("", description="Optional feedback")


class ToolSearchRequest(BaseModel):
    """Request model for tool search"""

    query: str = Field(..., description="Search query")
    categories: List[str] = Field(
        default_factory=list, description="Filter by categories"
    )
    capabilities: List[ToolCapability] = Field(
        default_factory=list, description="Filter by capabilities"
    )
    tags: List[str] = Field(default_factory=list, description="Filter by tags")
    min_rating: float = Field(0.0, ge=0.0, le=5.0, description="Minimum user rating")


class ToolRecommendationRequest(BaseModel):
    """Request model for tool recommendations"""

    user_id: str
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Context for recommendations"
    )
    limit: int = Field(5, ge=1, le=20, description="Number of recommendations")


class AnalyticsRequest(BaseModel):
    """Request model for analytics"""

    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    tool_ids: List[str] = Field(
        default_factory=list, description="Specific tools to analyze"
    )


# API Router
router = APIRouter(prefix="/tools", tags=["Tool Registry"])


# Dependency to get registry manager
async def get_registry_manager() -> ToolRegistryManager:
    """Get tool registry manager instance"""
    # In a real implementation, this would be injected or retrieved from app state
    return ToolRegistryManager()


@router.post("/register", response_model=APIResponse)
async def register_tool(
    request: ToolRegistrationRequest,
    registry: ToolRegistryManager = Depends(get_registry_manager),
):
    """Register a new tool in the registry"""

    try:
        # Note: In a real implementation, you would need to create a BaseTool instance
        # from the provided code and metadata. This is a simplified example.

        logger.info("Registering new tool", tool_name=request.name)

        # This would involve code compilation/validation and tool instantiation
        # For now, we'll return a success response

        return APIResponse(
            success=True,
            data={"message": "Tool registration initiated", "tool_name": request.name},
            message="Tool registration request received and is being processed",
        )

    except Exception as e:
        logger.error("Failed to register tool", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=PaginatedResponse)
async def list_tools(
    pagination: PaginationParams = Depends(),
    status: Optional[ToolStatus] = Query(None, description="Filter by status"),
    category: Optional[str] = Query(None, description="Filter by category"),
    registry: ToolRegistryManager = Depends(get_registry_manager),
):
    """List tools with optional filtering and pagination"""

    try:
        # Get tools with filtering
        all_tools = await registry.list_tools(
            status=status,
            category=category,
            limit=1000,  # Get more for proper pagination
        )

        # Apply pagination
        start_idx = pagination.offset
        end_idx = start_idx + pagination.size
        paginated_tools = all_tools[start_idx:end_idx]

        # Convert to dict format for response
        tools_data = [
            {
                "id": tool.id,
                "name": tool.name,
                "version": tool.version,
                "description": tool.description,
                "author": tool.author,
                "status": tool.status.value,
                "category": tool.category,
                "tags": tool.tags,
                "created_at": tool.created_at.isoformat(),
                "updated_at": tool.updated_at.isoformat(),
                "total_executions": tool.total_executions,
                "success_rate": tool.successful_executions
                / max(tool.total_executions, 1),
                "avg_execution_time": tool.avg_execution_time,
                "user_rating": tool.user_rating,
            }
            for tool in paginated_tools
        ]

        return PaginatedResponse.create(
            items=tools_data, total=len(all_tools), pagination=pagination
        )

    except Exception as e:
        logger.error("Failed to list tools", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{tool_id}", response_model=APIResponse)
async def get_tool_details(
    tool_id: str, registry: ToolRegistryManager = Depends(get_registry_manager)
):
    """Get detailed information about a specific tool"""

    try:
        metadata = await registry.get_tool_metadata(tool_id)

        if not metadata:
            raise HTTPException(status_code=404, detail="Tool not found")

        # Get additional analytics
        analytics = await registry.get_tool_analytics(tool_id)

        tool_data = {
            "id": metadata.id,
            "name": metadata.name,
            "version": metadata.version,
            "description": metadata.description,
            "author": metadata.author,
            "status": metadata.status.value,
            "category": metadata.category,
            "tags": metadata.tags,
            "dependencies": metadata.dependencies,
            "created_at": metadata.created_at.isoformat(),
            "updated_at": metadata.updated_at.isoformat(),
            "total_executions": metadata.total_executions,
            "successful_executions": metadata.successful_executions,
            "failed_executions": metadata.failed_executions,
            "success_rate": metadata.successful_executions
            / max(metadata.total_executions, 1),
            "avg_execution_time": metadata.avg_execution_time,
            "user_rating": metadata.user_rating,
            "validation_score": metadata.validation_score,
            "usage_frequency": metadata.usage_frequency,
            "last_execution": (
                metadata.last_execution.isoformat() if metadata.last_execution else None
            ),
            "deprecation_date": (
                metadata.deprecation_date.isoformat()
                if metadata.deprecation_date
                else None
            ),
            "replacement_tool_id": metadata.replacement_tool_id,
            "maintenance_notes": metadata.maintenance_notes,
            "analytics": analytics,
        }

        return APIResponse(
            success=True, data=tool_data, message="Tool details retrieved successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get tool details", tool_id=tool_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{tool_id}/status", response_model=APIResponse)
async def update_tool_status(
    tool_id: str,
    request: ToolUpdateRequest,
    registry: ToolRegistryManager = Depends(get_registry_manager),
):
    """Update tool status and maintenance information"""

    try:
        if request.status:
            success = await registry.update_tool_status(
                tool_id, request.status, request.maintenance_notes
            )

            if not success:
                raise HTTPException(status_code=404, detail="Tool not found")

            return APIResponse(
                success=True,
                data={"tool_id": tool_id, "new_status": request.status.value},
                message="Tool status updated successfully",
            )
        else:
            raise HTTPException(status_code=400, detail="Status is required")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update tool status", tool_id=tool_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{tool_id}/version", response_model=APIResponse)
async def update_tool_version(
    tool_id: str,
    request: ToolVersionRequest,
    registry: ToolRegistryManager = Depends(get_registry_manager),
):
    """Update tool to a new version"""

    try:
        success = await registry.update_tool_version(
            tool_id, request.version_type, request.code, request.changelog
        )

        if not success:
            raise HTTPException(status_code=404, detail="Tool not found")

        return APIResponse(
            success=True,
            data={"tool_id": tool_id, "version_type": request.version_type.value},
            message="Tool version updated successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update tool version", tool_id=tool_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{tool_id}/deprecate", response_model=APIResponse)
async def deprecate_tool(
    tool_id: str,
    replacement_tool_id: Optional[str] = Query(
        None, description="ID of replacement tool"
    ),
    registry: ToolRegistryManager = Depends(get_registry_manager),
):
    """Deprecate a tool"""

    try:
        success = await registry.deprecate_tool(tool_id, replacement_tool_id)

        if not success:
            raise HTTPException(status_code=404, detail="Tool not found")

        return APIResponse(
            success=True,
            data={"tool_id": tool_id, "replacement_tool_id": replacement_tool_id},
            message="Tool deprecated successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to deprecate tool", tool_id=tool_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{tool_id}", response_model=APIResponse)
async def delete_tool(
    tool_id: str, registry: ToolRegistryManager = Depends(get_registry_manager)
):
    """Delete a tool from the registry"""

    try:
        success = await registry.delete_tool(tool_id)

        if not success:
            raise HTTPException(status_code=404, detail="Tool not found")

        return APIResponse(
            success=True, data={"tool_id": tool_id}, message="Tool deleted successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete tool", tool_id=tool_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=APIResponse)
async def search_tools(
    request: ToolSearchRequest,
    registry: ToolRegistryManager = Depends(get_registry_manager),
):
    """Search tools by various criteria"""

    try:
        # Basic text search
        tools = await registry.search_tools(request.query, limit=100)

        # Apply additional filters
        if request.categories:
            tools = [t for t in tools if t.category in request.categories]

        if request.capabilities:
            tools = [
                t
                for t in tools
                if t.capabilities
                and any(
                    cap
                    in [t.capabilities.primary_capability]
                    + t.capabilities.secondary_capabilities
                    for cap in request.capabilities
                )
            ]

        if request.tags:
            tools = [t for t in tools if any(tag in t.tags for tag in request.tags)]

        if request.min_rating > 0:
            tools = [t for t in tools if t.user_rating >= request.min_rating]

        # Convert to response format
        tools_data = [
            {
                "id": tool.id,
                "name": tool.name,
                "description": tool.description,
                "category": tool.category,
                "tags": tool.tags,
                "user_rating": tool.user_rating,
                "success_rate": tool.successful_executions
                / max(tool.total_executions, 1),
                "total_executions": tool.total_executions,
            }
            for tool in tools
        ]

        return APIResponse(
            success=True,
            data={"tools": tools_data, "total_found": len(tools_data)},
            message=f"Found {len(tools_data)} tools matching search criteria",
        )

    except Exception as e:
        logger.error("Failed to search tools", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommendations", response_model=APIResponse)
async def get_tool_recommendations(
    request: ToolRecommendationRequest,
    registry: ToolRegistryManager = Depends(get_registry_manager),
):
    """Get personalized tool recommendations"""

    try:
        recommendations = await registry.get_tool_recommendations(
            request.user_id, request.context, request.limit
        )

        recommendations_data = [
            {
                "tool_id": rec.tool_id,
                "tool_name": rec.tool_name,
                "confidence_score": rec.confidence_score,
                "reason": rec.reason,
                "alternative_tools": rec.alternative_tools,
                "estimated_performance": rec.estimated_performance,
            }
            for rec in recommendations
        ]

        return APIResponse(
            success=True,
            data={"recommendations": recommendations_data},
            message=f"Generated {len(recommendations_data)} recommendations",
        )

    except Exception as e:
        logger.error("Failed to get recommendations", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{tool_id}/rate", response_model=APIResponse)
async def rate_tool(
    tool_id: str,
    request: ToolRatingRequest,
    user_id: str = Query(..., description="User ID"),
    registry: ToolRegistryManager = Depends(get_registry_manager),
):
    """Rate a tool"""

    try:
        success = await registry.rate_tool(
            tool_id, user_id, request.rating, request.feedback
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to rate tool")

        return APIResponse(
            success=True,
            data={"tool_id": tool_id, "rating": request.rating},
            message="Tool rated successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to rate tool", tool_id=tool_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{tool_id}/analytics", response_model=APIResponse)
async def get_tool_analytics(
    tool_id: str, registry: ToolRegistryManager = Depends(get_registry_manager)
):
    """Get comprehensive analytics for a tool"""

    try:
        analytics = await registry.get_tool_analytics(tool_id)

        if not analytics:
            raise HTTPException(
                status_code=404, detail="Tool not found or no analytics available"
            )

        return APIResponse(
            success=True,
            data=analytics,
            message="Tool analytics retrieved successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get tool analytics", tool_id=tool_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/system", response_model=APIResponse)
async def get_system_analytics(
    registry: ToolRegistryManager = Depends(get_registry_manager),
):
    """Get system-wide analytics and insights"""

    try:
        analytics = await registry.get_system_analytics()

        return APIResponse(
            success=True,
            data=analytics,
            message="System analytics retrieved successfully",
        )

    except Exception as e:
        logger.error("Failed to get system analytics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/discover", response_model=APIResponse)
async def discover_tools_by_capability(
    capabilities: List[ToolCapability] = Query(
        ..., description="Required capabilities"
    ),
    limit: int = Query(
        20, ge=1, le=100, description="Maximum number of tools to return"
    ),
    registry: ToolRegistryManager = Depends(get_registry_manager),
):
    """Discover tools by specific capabilities"""

    try:
        tools = await registry.discover_tools_by_capability(capabilities, limit)

        tools_data = [
            {
                "id": tool.id,
                "name": tool.name,
                "description": tool.description,
                "capabilities": {
                    "primary": (
                        tool.capabilities.primary_capability.value
                        if tool.capabilities
                        else None
                    ),
                    "secondary": (
                        [cap.value for cap in tool.capabilities.secondary_capabilities]
                        if tool.capabilities
                        else []
                    ),
                },
                "success_rate": tool.successful_executions
                / max(tool.total_executions, 1),
                "user_rating": tool.user_rating,
            }
            for tool in tools
        ]

        return APIResponse(
            success=True,
            data={"tools": tools_data, "total_found": len(tools_data)},
            message=f"Found {len(tools_data)} tools with matching capabilities",
        )

    except Exception as e:
        logger.error("Failed to discover tools by capability", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup", response_model=APIResponse)
async def cleanup_deprecated_tools(
    days_old: int = Query(90, ge=1, description="Age in days for cleanup"),
    registry: ToolRegistryManager = Depends(get_registry_manager),
):
    """Clean up old deprecated tools"""

    try:
        await registry.cleanup_deprecated_tools(days_old)

        return APIResponse(
            success=True,
            data={"days_old": days_old},
            message=f"Cleanup completed for tools deprecated more than {days_old} days ago",
        )

    except Exception as e:
        logger.error("Failed to cleanup deprecated tools", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
