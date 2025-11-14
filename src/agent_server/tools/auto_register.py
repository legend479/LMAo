"""
Auto-registration of available tools
"""

from typing import List
from src.shared.logging import get_logger
from .registry import BaseTool, ToolCapabilities, ResourceRequirements, ToolCapability
from .knowledge_retrieval import KnowledgeRetrievalTool
from .compiler_runtime import CompilerRuntimeTool
from .readability_scoring import ReadabilityScoringTool
from .document_generation import DocumentGenerationTool
from .email_automation import EmailAutomationTool

logger = get_logger(__name__)


async def register_default_tools(registry) -> List[str]:
    """
    Register all available default tools

    Args:
        registry: ToolRegistryManager instance

    Returns:
        List of registered tool IDs
    """
    registered_tools = []

    # Check if tools are already registered to avoid duplicates
    existing_tools = await registry.list_tools_metadata()
    existing_tool_names = {tool.name for tool in existing_tools}

    logger.info(f"Found {len(existing_tool_names)} existing tools in registry")

    tools_to_register = [
        (
            KnowledgeRetrievalTool,
            "Knowledge retrieval and search",
            "knowledge_retrieval",
        ),
        (CompilerRuntimeTool, "Code compilation and execution", "compiler_runtime"),
        (ReadabilityScoringTool, "Text readability analysis", "readability_scoring"),
        (
            DocumentGenerationTool,
            "Document generation and conversion",
            "document_generation",
        ),
        (EmailAutomationTool, "Email composition and automation", "email_automation"),
    ]

    for tool_class, description, tool_name in tools_to_register:
        try:
            # Skip if tool already exists
            if tool_name in existing_tool_names:
                logger.info(f"Tool {tool_name} already registered, skipping")
                continue

            # Create tool instance
            tool = tool_class()

            # Initialize if needed
            if hasattr(tool, "initialize"):
                await tool.initialize()

            # Register the tool
            tool_id = await registry.register_tool(
                tool=tool,
                author="system",
                tags=["default", "built-in"],
                validation_score=1.0,
            )

            registered_tools.append(tool_id)
            logger.info(f"Registered tool: {tool_class.__name__} with ID: {tool_id}")

        except Exception as e:
            logger.warning(f"Failed to register {tool_class.__name__}: {str(e)}")
            continue

    if len(registered_tools) > 0:
        logger.info(
            f"Successfully registered {len(registered_tools)} new default tools"
        )
    else:
        logger.info("All default tools already registered")

    return registered_tools


async def register_custom_tools(registry, tools: List[BaseTool]) -> List[str]:
    """
    Register custom tools

    Args:
        registry: ToolRegistryManager instance
        tools: List of tool instances to register

    Returns:
        List of registered tool IDs
    """
    registered_tools = []

    for tool in tools:
        try:
            # Initialize if needed
            if hasattr(tool, "initialize"):
                await tool.initialize()

            # Register the tool
            tool_id = await registry.register_tool(
                tool=tool, author="custom", tags=["custom"], validation_score=0.8
            )

            registered_tools.append(tool_id)
            logger.info(
                f"Registered custom tool: {tool.__class__.__name__} with ID: {tool_id}"
            )

        except Exception as e:
            logger.warning(
                f"Failed to register custom tool {tool.__class__.__name__}: {str(e)}"
            )
            continue

    logger.info(f"Successfully registered {len(registered_tools)} custom tools")
    return registered_tools
