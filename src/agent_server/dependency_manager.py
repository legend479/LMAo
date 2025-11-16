"""
Dependency Manager
Advanced dependency context management for tool chaining
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .tools.tool_output_schema import (
    StructuredToolOutput,
    OutputSummarizer,
    KnowledgeOutput,
    CodeOutput,
    AnalysisOutput,
)
from src.shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DependencyContext:
    """Enhanced dependency context for tool execution"""

    # Previous tool outputs
    previous_outputs: List[StructuredToolOutput] = field(default_factory=list)

    # Accumulated knowledge
    accumulated_knowledge: List[str] = field(default_factory=list)
    entities_discovered: List[Dict[str, Any]] = field(default_factory=list)

    # Execution metadata
    execution_chain: List[str] = field(default_factory=list)
    total_execution_time: float = 0.0

    # Context for current tool
    relevant_context: Dict[str, Any] = field(default_factory=dict)

    # Quality metrics
    average_confidence: float = 1.0
    average_quality: float = 1.0


class DependencyManager:
    """Manages dependencies and context flow between tools"""

    def __init__(self):
        self._initialized = False
        self.output_summarizer = OutputSummarizer()

        # Cache for dependency contexts
        self.context_cache: Dict[str, DependencyContext] = {}

    async def initialize(self):
        """Initialize dependency manager"""
        if self._initialized:
            return

        logger.info("Initializing Dependency Manager")
        self._initialized = True
        logger.info("Dependency Manager initialized")

    def build_dependency_context(
        self,
        task_id: str,
        dependencies: List[str],
        task_results: Dict[str, Any],
        task_config: Dict[str, Any],
    ) -> DependencyContext:
        """
        Build comprehensive dependency context for a task

        Args:
            task_id: Current task ID
            dependencies: List of dependency task IDs
            task_results: All task results so far
            task_config: Configuration for current task

        Returns:
            DependencyContext with all relevant information
        """
        logger.info(
            "Building dependency context", task_id=task_id, dependencies=dependencies
        )

        # Collect outputs from dependencies
        previous_outputs = []
        for dep_id in dependencies:
            if dep_id in task_results:
                result = task_results[dep_id]

                # Convert to StructuredToolOutput if not already
                if isinstance(result, StructuredToolOutput):
                    previous_outputs.append(result)
                elif isinstance(result, dict) and "data" in result:
                    # Try to reconstruct structured output
                    structured = self._reconstruct_structured_output(result, dep_id)
                    if structured:
                        previous_outputs.append(structured)

        # Extract accumulated knowledge
        accumulated_knowledge = []
        entities_discovered = []
        execution_chain = []
        total_execution_time = 0.0

        for output in previous_outputs:
            accumulated_knowledge.extend(output.key_findings)
            entities_discovered.extend(output.entities_extracted)
            execution_chain.append(output.metadata.tool_name)
            total_execution_time += output.metadata.execution_time

        # Calculate quality metrics
        if previous_outputs:
            average_confidence = sum(
                o.metadata.confidence_score for o in previous_outputs
            ) / len(previous_outputs)
            average_quality = sum(
                o.metadata.quality_score for o in previous_outputs
            ) / len(previous_outputs)
        else:
            average_confidence = 1.0
            average_quality = 1.0

        # Build relevant context for current task
        relevant_context = self._extract_relevant_context(previous_outputs, task_config)

        context = DependencyContext(
            previous_outputs=previous_outputs,
            accumulated_knowledge=list(set(accumulated_knowledge)),
            entities_discovered=entities_discovered,
            execution_chain=execution_chain,
            total_execution_time=total_execution_time,
            relevant_context=relevant_context,
            average_confidence=average_confidence,
            average_quality=average_quality,
        )

        # Cache the context
        self.context_cache[task_id] = context

        logger.info(
            "Dependency context built",
            task_id=task_id,
            previous_tools=len(previous_outputs),
            knowledge_items=len(accumulated_knowledge),
            entities=len(entities_discovered),
        )

        return context

    def _reconstruct_structured_output(
        self, result: Dict[str, Any], tool_name: str
    ) -> Optional[StructuredToolOutput]:
        """Reconstruct StructuredToolOutput from dict result"""
        try:
            from .tools.tool_output_schema import (
                ToolOutputMetadata,
                OutputType,
                DataFormat,
            )

            # Create metadata
            metadata = ToolOutputMetadata(
                tool_name=tool_name,
                execution_time=result.get("execution_time", 0.0),
                timestamp=datetime.utcnow().isoformat(),
                output_type=OutputType.STRUCTURED_DATA,
                data_format=DataFormat.JSON,
                confidence_score=result.get("confidence_score", 1.0),
                quality_score=result.get("quality_score", 1.0),
            )

            # Create structured output
            return StructuredToolOutput(
                data=result.get("data"),
                summary=result.get("summary", "No summary available"),
                metadata=metadata,
                key_findings=result.get("key_findings", []),
                entities_extracted=result.get("entities_extracted", []),
                recommendations=result.get("recommendations", []),
                context_for_next_tool=result.get("context_for_next_tool", {}),
                success=result.get("success", True),
                error_message=result.get("error_message"),
            )
        except Exception as e:
            logger.warning(f"Failed to reconstruct structured output: {e}")
            return None

    def _extract_relevant_context(
        self, previous_outputs: List[StructuredToolOutput], task_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract context relevant to the current task"""

        task_type = task_config.get("type", "general")

        relevant_context = {
            "task_type": task_type,
            "previous_summaries": [],
            "relevant_data": {},
            "recommendations": [],
        }

        for output in previous_outputs:
            # Add summary
            relevant_context["previous_summaries"].append(
                {
                    "tool": output.metadata.tool_name,
                    "summary": output.summary,
                    "confidence": output.metadata.confidence_score,
                }
            )

            # Extract type-specific data
            if task_type == "code_generation" and isinstance(output, KnowledgeOutput):
                # For code generation, extract code examples and patterns
                relevant_context["relevant_data"]["knowledge_base"] = {
                    "top_chunks": output.get_top_chunks(3),
                    "sources": output.get_sources_summary(),
                }

            elif task_type == "content_generation":
                # For content generation, extract key findings and structure
                relevant_context["relevant_data"]["key_points"] = output.key_findings
                relevant_context["relevant_data"][
                    "entities"
                ] = output.entities_extracted

            elif task_type == "analysis":
                # For analysis, extract metrics and insights
                if isinstance(output, AnalysisOutput):
                    relevant_context["relevant_data"]["metrics"] = output.metrics
                    relevant_context["relevant_data"]["insights"] = output.insights

            # Add recommendations
            relevant_context["recommendations"].extend(output.recommendations)

        # Deduplicate recommendations
        relevant_context["recommendations"] = list(
            set(relevant_context["recommendations"])
        )

        return relevant_context

    def inject_dependency_context(
        self, task_parameters: Dict[str, Any], dependency_context: DependencyContext
    ) -> Dict[str, Any]:
        """
        Inject dependency context into task parameters

        Args:
            task_parameters: Original task parameters
            dependency_context: Built dependency context

        Returns:
            Enhanced parameters with dependency context
        """
        enhanced_params = task_parameters.copy()

        # Add dependency context as a special parameter
        enhanced_params["_dependency_context"] = {
            "previous_tools": dependency_context.execution_chain,
            "accumulated_knowledge": dependency_context.accumulated_knowledge[
                :10
            ],  # Top 10
            "entities": dependency_context.entities_discovered[:20],  # Top 20
            "relevant_context": dependency_context.relevant_context,
            "quality_metrics": {
                "average_confidence": dependency_context.average_confidence,
                "average_quality": dependency_context.average_quality,
            },
        }

        # Add summarized context for LLM
        if dependency_context.previous_outputs:
            enhanced_params["_context_summary"] = (
                self.output_summarizer.summarize_for_llm(
                    dependency_context.previous_outputs, max_total_length=1500
                )
            )

        # Add specific context based on task type
        task_type = task_parameters.get("type", "general")

        if task_type == "code_generation":
            enhanced_params["_code_context"] = self._build_code_context(
                dependency_context
            )
        elif task_type == "content_generation":
            enhanced_params["_content_context"] = self._build_content_context(
                dependency_context
            )

        logger.debug(
            "Injected dependency context",
            original_params=len(task_parameters),
            enhanced_params=len(enhanced_params),
        )

        return enhanced_params

    def _build_code_context(
        self, dependency_context: DependencyContext
    ) -> Dict[str, Any]:
        """Build code-specific context"""

        code_context = {"examples": [], "patterns": [], "dependencies": []}

        for output in dependency_context.previous_outputs:
            if isinstance(output, CodeOutput):
                code_context["examples"].append(
                    {
                        "code": output.code,
                        "language": output.language,
                        "explanation": output.explanation,
                    }
                )
                code_context["dependencies"].extend(output.dependencies)

            elif isinstance(output, KnowledgeOutput):
                # Extract code snippets from knowledge
                for chunk in output.retrieved_chunks:
                    if chunk.get("chunk_type") == "code":
                        code_context["examples"].append(
                            {
                                "code": chunk.get("content", ""),
                                "source": chunk.get("document_id", "unknown"),
                            }
                        )

        return code_context

    def _build_content_context(
        self, dependency_context: DependencyContext
    ) -> Dict[str, Any]:
        """Build content-specific context"""

        content_context = {
            "key_points": dependency_context.accumulated_knowledge,
            "entities": dependency_context.entities_discovered,
            "structure_suggestions": [],
            "tone_guidance": [],
        }

        # Extract structure suggestions from previous outputs
        for output in dependency_context.previous_outputs:
            if output.recommendations:
                content_context["structure_suggestions"].extend(
                    [r for r in output.recommendations if "structure" in r.lower()]
                )

        return content_context

    def create_result_summary(
        self, tool_output: StructuredToolOutput, for_downstream: bool = True
    ) -> Dict[str, Any]:
        """
        Create a summary of tool result for downstream consumption

        Args:
            tool_output: The tool output to summarize
            for_downstream: Whether this is for downstream tools

        Returns:
            Summarized result
        """
        summary = {
            "tool": tool_output.metadata.tool_name,
            "summary": tool_output.summary,
            "success": tool_output.success,
            "confidence": tool_output.metadata.confidence_score,
            "quality": tool_output.metadata.quality_score,
        }

        if for_downstream:
            # Add context for next tool
            summary["context_for_next_tool"] = tool_output.context_for_next_tool
            summary["key_findings"] = tool_output.key_findings[:5]  # Top 5
            summary["recommendations"] = tool_output.recommendations[:3]  # Top 3
        else:
            # Full summary for final output
            summary["key_findings"] = tool_output.key_findings
            summary["recommendations"] = tool_output.recommendations
            summary["entities"] = tool_output.entities_extracted

        return summary

    def get_context_for_task(self, task_id: str) -> Optional[DependencyContext]:
        """Get cached context for a task"""
        return self.context_cache.get(task_id)

    def clear_context_cache(self, task_id: Optional[str] = None):
        """Clear context cache"""
        if task_id:
            if task_id in self.context_cache:
                del self.context_cache[task_id]
        else:
            self.context_cache.clear()

    async def health_check(self) -> Dict[str, Any]:
        """Health check for dependency manager"""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "cached_contexts": len(self.context_cache),
            "components": {"output_summarizer": "operational"},
        }
