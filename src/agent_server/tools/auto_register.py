"""
Auto-registration of tools with configuration-based enablement
"""

from typing import List
from src.shared.config import get_settings
from src.shared.logging import get_logger
from .registry import BaseTool

logger = get_logger(__name__)


async def register_default_tools(registry) -> List[str]:
    """
    Register default tools based on configuration

    Args:
        registry: ToolRegistryManager instance

    Returns:
        List of registered tool IDs
    """
    settings = get_settings()
    registered_tools = []

    logger.info("Starting tool registration with configuration-based enablement")

    # Check if tools are already registered to avoid duplicates
    existing_tools = await registry.list_tools_metadata()
    existing_tool_names = {tool.name for tool in existing_tools}

    logger.info(f"Found {len(existing_tool_names)} existing tools in registry")

    # Always register safe, core tools
    try:
        if "knowledge_retrieval" not in existing_tool_names:
            from .knowledge_retrieval import KnowledgeRetrievalTool

            tool = KnowledgeRetrievalTool()
            await tool.initialize()
            tool_id = await registry.register_tool(
                tool=tool,
                author="system",
                tags=["default", "built-in", "core"],
                validation_score=1.0,
            )
            registered_tools.append(tool_id)
            logger.info("Registered knowledge_retrieval tool")
        else:
            logger.info("knowledge_retrieval tool already registered")
    except Exception as e:
        logger.error(f"Failed to register knowledge_retrieval tool: {e}")

    try:
        if "readability_scoring" not in existing_tool_names:
            from .readability_scoring import ReadabilityScoringTool

            tool = ReadabilityScoringTool()
            await tool.initialize()
            tool_id = await registry.register_tool(
                tool=tool,
                author="system",
                tags=["default", "built-in", "analysis"],
                validation_score=1.0,
            )
            registered_tools.append(tool_id)
            logger.info("Registered readability_scoring tool")
        else:
            logger.info("readability_scoring tool already registered")
    except Exception as e:
        logger.error(f"Failed to register readability_scoring tool: {e}")

    try:
        if "document_generation" not in existing_tool_names:
            from .document_generation import DocumentGenerationTool

            tool = DocumentGenerationTool()
            await tool.initialize()
            tool_id = await registry.register_tool(
                tool=tool,
                author="system",
                tags=["default", "built-in", "generation"],
                validation_score=1.0,
            )
            registered_tools.append(tool_id)
            logger.info("Registered document_generation tool")
        else:
            logger.info("document_generation tool already registered")
    except Exception as e:
        logger.error(f"Failed to register document_generation tool: {e}")

    # Conditionally register code execution tools
    if settings.enable_code_execution:
        try:
            if "compiler_runtime" not in existing_tool_names:
                from .compiler_runtime import CompilerRuntimeTool

                tool = CompilerRuntimeTool()
                await tool.initialize()
                tool_id = await registry.register_tool(
                    tool=tool,
                    author="system",
                    tags=["default", "built-in", "code-execution"],
                    validation_score=1.0,
                )
                registered_tools.append(tool_id)
                logger.info("Registered compiler_runtime tool (code execution enabled)")
            else:
                logger.info("compiler_runtime tool already registered")
        except Exception as e:
            logger.error(f"Failed to register compiler_runtime tool: {e}")
    else:
        logger.info(
            "Code execution tools disabled by configuration (ENABLE_CODE_EXECUTION=false)"
        )

    # Conditionally register email tools
    if settings.enable_email_tools:
        try:
            if "email_automation" not in existing_tool_names:
                from .email_automation import EmailAutomationTool

                tool = EmailAutomationTool()
                await tool.initialize()
                tool_id = await registry.register_tool(
                    tool=tool,
                    author="system",
                    tags=["default", "built-in", "communication"],
                    validation_score=1.0,
                )
                registered_tools.append(tool_id)
                logger.info("Registered email_automation tool (email tools enabled)")
            else:
                logger.info("email_automation tool already registered")
        except Exception as e:
            logger.error(f"Failed to register email_automation tool: {e}")
    else:
        logger.info("Email tools disabled by configuration (ENABLE_EMAIL_TOOLS=false)")

    logger.info(
        f"Tool registration complete. Registered {len(registered_tools)} new tools"
    )

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
