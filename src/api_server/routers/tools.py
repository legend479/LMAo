"""
Tools Router
Handles tool management and execution endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from .auth import get_current_active_user, User
import src.shared.services as services
from src.shared.logging import get_logger

router = APIRouter()

logger = get_logger(__name__)


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    required_params: List[str]
    category: Optional[str] = None
    tags: List[str] = []
    usage_count: int = 0
    success_rate: float = 0.0


class ToolExecutionRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]
    session_id: Optional[str] = None


class ToolInvocationRequest(BaseModel):
    parameters: Dict[str, Any]
    session_id: Optional[str] = None


class ToolExecutionResponse(BaseModel):
    tool_name: str
    result: Any
    status: str
    execution_time: float
    metadata: Dict[str, Any] = {}


@router.get("/", response_model=List[ToolSchema])
async def list_tools():
    """List all available tools"""
    # TODO: Get tools from tool registry

    # Prefer real tools from the Agent service when available
    try:
        agent_client = await services.get_agent_client()
        tools_data = await agent_client.get_available_tools()

        if isinstance(tools_data, dict):
            raw_tools = tools_data.get("tools", []) or []
        else:
            raw_tools = tools_data or []

        tools: List[ToolSchema] = []
        for tool in raw_tools:
            if not isinstance(tool, dict):
                continue

            name = tool.get("name")
            if not name:
                continue

            description = tool.get("description", "")
            schema = tool.get("schema", {}) if isinstance(tool, dict) else {}

            parameters = schema.get("parameters") or tool.get("parameters") or {}
            required_params = (
                schema.get("required_params") or tool.get("required_params") or []
            )

            category = tool.get("category")
            tags = tool.get("tags") or []
            usage_count = int(tool.get("usage_count", 0) or 0)
            success_rate = float(tool.get("success_rate", 0.0) or 0.0)

            tools.append(
                ToolSchema(
                    name=name,
                    description=description,
                    parameters=parameters,
                    required_params=required_params,
                    category=category,
                    tags=tags,
                    usage_count=usage_count,
                    success_rate=success_rate,
                )
            )

        if tools:
            # Sort by usage_count (desc), then success_rate (desc) for better discovery
            tools.sort(key=lambda t: (t.usage_count, t.success_rate), reverse=True)
            return tools

    except Exception as e:
        logger.error("Failed to fetch tools from agent service", error=str(e))

    # Fallback to a static knowledge retrieval tool definition
    return [
        ToolSchema(
            name="knowledge_retrieval",
            description="Retrieve knowledge from the RAG pipeline",
            parameters={"query": "string", "filters": "object"},
            required_params=["query"],
        )
    ]


@router.post("/execute", response_model=ToolExecutionResponse)
async def execute_tool(
    request: ToolExecutionRequest,
    current_user: User = Depends(get_current_active_user),
):
    """Execute a specific tool"""
    # TODO: Integrate with tool execution system

    session_id = request.session_id or f"tool_session_{current_user.id}"

    try:
        agent_client = await services.get_agent_client()
        agent_response = await agent_client.execute_tool(
            request.tool_name,
            request.parameters,
            session_id,
        )

        return ToolExecutionResponse(
            tool_name=agent_response.get("tool_name", request.tool_name),
            result=agent_response.get("result"),
            status=agent_response.get("status", "success"),
            execution_time=float(agent_response.get("execution_time", 0.0)),
            metadata=agent_response.get("metadata", {}),
        )

    except Exception as e:
        logger.error(
            "Tool execution via /execute failed",
            tool_name=request.tool_name,
            session_id=session_id,
            error=str(e),
        )

        # Safe fallback so tests and non-agent environments still work
        return ToolExecutionResponse(
            tool_name=request.tool_name,
            result={
                "message": "Tool execution not available",
                "error": str(e),
            },
            status="error",
            execution_time=0.0,
            metadata={"error": True},
        )


@router.post("/{tool_name}/execute", response_model=ToolExecutionResponse)
async def execute_named_tool(
    tool_name: str,
    request: ToolInvocationRequest,
    current_user: User = Depends(get_current_active_user),
):
    """Execute a specific tool by name"""

    session_id = request.session_id or f"tool_session_{current_user.id}"

    try:
        agent_client = await services.get_agent_client()
        agent_response = await agent_client.execute_tool(
            tool_name,
            request.parameters,
            session_id,
        )

        return ToolExecutionResponse(
            tool_name=agent_response.get("tool_name", tool_name),
            result=agent_response.get("result"),
            status=agent_response.get("status", "success"),
            execution_time=float(agent_response.get("execution_time", 0.0)),
            metadata=agent_response.get("metadata", {}),
        )

    except Exception as e:
        logger.error(
            "Tool execution via /{tool_name}/execute failed",
            tool_name=tool_name,
            session_id=session_id,
            error=str(e),
        )

        # Align behavior with /execute: always return ToolExecutionResponse
        return ToolExecutionResponse(
            tool_name=tool_name,
            result={
                "message": "Tool execution not available",
                "error": str(e),
            },
            status="error",
            execution_time=0.0,
            metadata={"error": True},
        )


@router.get("/{tool_name}/schema", response_model=ToolSchema)
async def get_tool_schema(tool_name: str):
    """Get schema for a specific tool"""

    try:
        agent_client = await services.get_agent_client()
        tools_data = await agent_client.get_available_tools()

        if isinstance(tools_data, dict):
            raw_tools = tools_data.get("tools", []) or []
        else:
            raw_tools = tools_data or []

        for tool in raw_tools:
            if not isinstance(tool, dict):
                continue

            if tool.get("name") != tool_name:
                continue

            description = tool.get("description", "")
            schema = tool.get("schema", {}) if isinstance(tool, dict) else {}

            parameters = schema.get("parameters") or tool.get("parameters") or {}
            required_params = (
                schema.get("required_params") or tool.get("required_params") or []
            )

            return ToolSchema(
                name=tool_name,
                description=description,
                parameters=parameters,
                required_params=required_params,
            )

    except Exception as e:
        logger.error(
            "Failed to fetch tool schema from agent service",
            tool_name=tool_name,
            error=str(e),
        )

    return ToolSchema(
        name=tool_name,
        description=f"Schema for {tool_name}",
        parameters={},
        required_params=[],
    )
