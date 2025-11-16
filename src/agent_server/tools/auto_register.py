"""
Auto-registration of tools with configuration-based enablement
"""

from typing import List
from src.shared.config import get_settings
from src.shared.logging import get_logger
from .registry import BaseTool

logger = get_logger(__name__)


# In auto_register.py

async def register_default_tools(registry) -> List[str]:
    """
    Register default tools based on configuration.
    This function ensures tools are in the database AND loaded into the
    active_tools cache for execution.
    """
    settings = get_settings()
    registered_tools = []

    logger.info("Starting tool registration and loading active cache...")

    # Get a map of tool metadata already in the database (Name -> Metadata)
    existing_tools_metadata = await registry.list_tools_metadata()
    existing_tool_map = {tool.name: tool for tool in existing_tools_metadata}
    logger.info(f"Found {len(existing_tool_map)} existing tool metadata records.")

    # A helper function to handle the registration/loading logic
    async def load_or_register_tool(
        tool_instance: BaseTool,
        tool_name: str,  # The canonical, snake_case name
        author: str,
        tags: List[str],
        validation_score: float = 1.0,
    ):
        try:
            await tool_instance.initialize()
            
            if tool_name not in existing_tool_map:
                # --- Tool is NOT in the database ---
                # Register it for the first time.
                # register_tool() will save metadata AND add to active_tools.
                logger.info(f"Registering new tool: {tool_name}")
                tool_id = await registry.register_tool(
                    tool=tool_instance,
                    author=author,
                    tags=tags,
                    validation_score=validation_score,
                )
                registered_tools.append(tool_id)
            
            else:
                # --- Tool IS in the database ---
                # We MUST manually add its instance to the active_tools cache.
                logger.info(f"Loading existing tool into active cache: {tool_name}")
                existing_metadata = existing_tool_map[tool_name]
                
                # Manually add the instance to the live cache
                registry.active_tools[existing_metadata.id] = tool_instance
                
                # Also update the metadata cache to ensure it's fresh
                # (in case the tool's schema changed)
                registry.tool_cache[existing_metadata.id] = existing_metadata
                
                registered_tools.append(existing_metadata.id)

        except Exception as e:
            logger.error(f"Failed to load or register tool '{tool_name}': {e}")

    # --- Register Core Tools ---
    from .knowledge_retrieval import KnowledgeRetrievalTool
    await load_or_register_tool(
        tool_instance=KnowledgeRetrievalTool(),
        tool_name="knowledge_retrieval",
        author="system",
        tags=["default", "built-in", "core"],
    )

    from .readability_scoring import ReadabilityScoringTool
    await load_or_register_tool(
        tool_instance=ReadabilityScoringTool(),
        tool_name="readability_scoring",
        author="system",
        tags=["default", "built-in", "analysis"],
    )

    from .document_generation import DocumentGenerationTool
    await load_or_register_tool(
        tool_instance=DocumentGenerationTool(),
        tool_name="document_generation",
        author="system",
        tags=["default", "built-in", "generation"],
    )

    # --- Conditionally Register Code Execution ---
    if settings.enable_code_execution:
        from .compiler_runtime import CompilerRuntimeTool
        await load_or_register_tool(
            tool_instance=CompilerRuntimeTool(),
            tool_name="compiler_runtime",
            author="system",
            tags=["default", "built-in", "code-execution"],
        )
    else:
        logger.info("Code execution tools disabled by configuration.")

    # --- Conditionally Register Email ---
    if settings.enable_email_tools:
        from .email_automation import EmailAutomationTool
        await load_or_register_tool(
            tool_instance=EmailAutomationTool(),
            tool_name="email_automation",
            author="system",
            tags=["default", "built-in", "communication"],
        )
    else:
        logger.info("Email tools disabled by configuration.")

    logger.info(
        f"Tool registration complete. {len(registered_tools)} tools loaded into active cache."
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
