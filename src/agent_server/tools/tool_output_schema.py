"""
Tool Output Schema
Standardized output formats for better inter-tool communication
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import json


class OutputType(Enum):
    """Standard output types for tools"""

    TEXT = "text"
    STRUCTURED_DATA = "structured_data"
    CODE = "code"
    DOCUMENT = "document"
    ANALYSIS = "analysis"
    KNOWLEDGE = "knowledge"
    EXECUTION_RESULT = "execution_result"
    ERROR = "error"


class DataFormat(Enum):
    """Data format specifications"""

    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    PLAIN_TEXT = "plain_text"
    CODE_BLOCK = "code_block"
    BINARY = "binary"


@dataclass
class ToolOutputMetadata:
    """Metadata for tool outputs"""

    tool_name: str
    execution_time: float
    timestamp: str
    output_type: OutputType
    data_format: DataFormat
    confidence_score: float = 1.0
    quality_score: float = 1.0
    token_count: Optional[int] = None
    source_references: List[str] = field(default_factory=list)
    dependencies_used: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["output_type"] = self.output_type.value
        data["data_format"] = self.data_format.value
        return data


@dataclass
class StructuredToolOutput:
    """Standardized tool output format"""

    # Core data
    data: Any  # Main output data
    summary: str  # Brief summary for downstream tools

    # Metadata
    metadata: ToolOutputMetadata

    # Structured fields for common use cases
    key_findings: List[str] = field(default_factory=list)
    entities_extracted: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Context for downstream tools
    context_for_next_tool: Dict[str, Any] = field(default_factory=dict)

    # Error handling
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "data": self.data,
            "summary": self.summary,
            "metadata": self.metadata.to_dict(),
            "key_findings": self.key_findings,
            "entities_extracted": self.entities_extracted,
            "recommendations": self.recommendations,
            "context_for_next_tool": self.context_for_next_tool,
            "success": self.success,
            "error_message": self.error_message,
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def get_summary_for_llm(self, max_length: int = 500) -> str:
        """Get concise summary for LLM context"""
        summary_parts = [self.summary]

        if self.key_findings:
            summary_parts.append(f"Key findings: {', '.join(self.key_findings[:3])}")

        if self.recommendations:
            summary_parts.append(
                f"Recommendations: {', '.join(self.recommendations[:2])}"
            )

        full_summary = " | ".join(summary_parts)

        if len(full_summary) > max_length:
            return full_summary[: max_length - 3] + "..."

        return full_summary

    def extract_for_dependency(self, dependency_type: str) -> Dict[str, Any]:
        """Extract relevant data for a specific dependency type"""

        if dependency_type == "knowledge":
            return {
                "summary": self.summary,
                "key_findings": self.key_findings,
                "entities": self.entities_extracted,
                "confidence": self.metadata.confidence_score,
            }

        elif dependency_type == "code":
            return {
                "code": self.data if isinstance(self.data, str) else None,
                "summary": self.summary,
                "recommendations": self.recommendations,
            }

        elif dependency_type == "analysis":
            return {
                "findings": self.key_findings,
                "recommendations": self.recommendations,
                "quality_score": self.metadata.quality_score,
            }

        else:
            # Generic extraction
            return {
                "summary": self.summary,
                "data": self.data,
                "context": self.context_for_next_tool,
            }


@dataclass
class KnowledgeOutput(StructuredToolOutput):
    """Specialized output for knowledge retrieval"""

    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    total_sources: int = 0
    relevance_scores: List[float] = field(default_factory=list)
    query_reformulation: Optional[Dict[str, Any]] = None

    def get_top_chunks(self, n: int = 3) -> List[Dict[str, Any]]:
        """Get top N most relevant chunks"""
        return sorted(
            self.retrieved_chunks, key=lambda x: x.get("score", 0.0), reverse=True
        )[:n]

    def get_sources_summary(self) -> str:
        """Get summary of sources"""
        if not self.retrieved_chunks:
            return "No sources found"

        sources = set()
        for chunk in self.retrieved_chunks:
            source = chunk.get("metadata", {}).get("document_title", "Unknown")
            sources.add(source)

        return f"Retrieved from {len(sources)} sources: {', '.join(list(sources)[:3])}"


@dataclass
class CodeOutput(StructuredToolOutput):
    """Specialized output for code generation"""

    code: str = ""
    language: str = "python"
    explanation: str = ""
    test_cases: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    def get_executable_code(self) -> str:
        """Get code ready for execution"""
        return self.code

    def get_code_with_comments(self) -> str:
        """Get code with explanation as comments"""
        if not self.explanation:
            return self.code

        comment_prefix = "#" if self.language in ["python", "ruby"] else "//"
        explanation_lines = [
            f"{comment_prefix} {line}" for line in self.explanation.split("\n")
        ]

        return "\n".join(explanation_lines) + "\n\n" + self.code


@dataclass
class AnalysisOutput(StructuredToolOutput):
    """Specialized output for analysis tasks"""

    metrics: Dict[str, float] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    visualizations: List[Dict[str, Any]] = field(default_factory=list)

    def get_metrics_summary(self) -> str:
        """Get summary of key metrics"""
        if not self.metrics:
            return "No metrics available"

        top_metrics = sorted(
            self.metrics.items(), key=lambda x: abs(x[1]), reverse=True
        )[:3]

        return ", ".join([f"{k}: {v:.2f}" for k, v in top_metrics])


@dataclass
class DocumentOutput(StructuredToolOutput):
    """Specialized output for document generation"""

    document_path: Optional[str] = None
    document_format: str = "pdf"
    page_count: int = 0
    sections: List[str] = field(default_factory=list)

    def get_document_info(self) -> str:
        """Get document information summary"""
        return f"{self.document_format.upper()} document with {self.page_count} pages, {len(self.sections)} sections"


class OutputSchemaValidator:
    """Validator for tool outputs"""

    @staticmethod
    def validate_output(output: StructuredToolOutput) -> tuple[bool, List[str]]:
        """Validate tool output structure"""
        errors = []

        # Check required fields
        if not output.summary:
            errors.append("Summary is required")

        if not output.metadata:
            errors.append("Metadata is required")

        if output.metadata and not output.metadata.tool_name:
            errors.append("Tool name is required in metadata")

        # Check data consistency
        if not output.success and not output.error_message:
            errors.append("Error message required when success=False")

        # Check confidence scores
        if output.metadata:
            if not 0 <= output.metadata.confidence_score <= 1:
                errors.append("Confidence score must be between 0 and 1")

            if not 0 <= output.metadata.quality_score <= 1:
                errors.append("Quality score must be between 0 and 1")

        return len(errors) == 0, errors

    @staticmethod
    def validate_for_dependency(
        output: StructuredToolOutput, required_fields: List[str]
    ) -> tuple[bool, List[str]]:
        """Validate output has required fields for dependency"""
        errors = []

        output_dict = output.to_dict()

        for field in required_fields:
            if field not in output_dict or output_dict[field] is None:
                errors.append(f"Required field '{field}' is missing")

        return len(errors) == 0, errors


class OutputSummarizer:
    """Summarize tool outputs for downstream consumption"""

    @staticmethod
    def summarize_for_llm(
        outputs: List[StructuredToolOutput], max_total_length: int = 2000
    ) -> str:
        """Summarize multiple tool outputs for LLM context"""

        if not outputs:
            return ""

        summaries = []
        length_per_output = max_total_length // len(outputs)

        for output in outputs:
            summary = output.get_summary_for_llm(max_length=length_per_output)
            tool_name = output.metadata.tool_name
            summaries.append(f"[{tool_name}] {summary}")

        return "\n\n".join(summaries)

    @staticmethod
    def extract_key_information(outputs: List[StructuredToolOutput]) -> Dict[str, Any]:
        """Extract key information from multiple outputs"""

        all_findings = []
        all_recommendations = []
        all_entities = []

        for output in outputs:
            all_findings.extend(output.key_findings)
            all_recommendations.extend(output.recommendations)
            all_entities.extend(output.entities_extracted)

        return {
            "key_findings": list(set(all_findings)),
            "recommendations": list(set(all_recommendations)),
            "entities": all_entities,
            "total_tools": len(outputs),
            "success_rate": sum(1 for o in outputs if o.success) / len(outputs),
        }

    @staticmethod
    def create_dependency_context(
        outputs: List[StructuredToolOutput], target_tool: str
    ) -> Dict[str, Any]:
        """Create context for a target tool from previous outputs"""

        context = {
            "previous_tools": [],
            "accumulated_knowledge": [],
            "entities_discovered": [],
            "recommendations": [],
        }

        for output in outputs:
            context["previous_tools"].append(
                {
                    "tool": output.metadata.tool_name,
                    "summary": output.summary,
                    "confidence": output.metadata.confidence_score,
                }
            )

            context["accumulated_knowledge"].extend(output.key_findings)
            context["entities_discovered"].extend(output.entities_extracted)
            context["recommendations"].extend(output.recommendations)

        # Deduplicate
        context["accumulated_knowledge"] = list(set(context["accumulated_knowledge"]))
        context["recommendations"] = list(set(context["recommendations"]))

        return context


# Factory functions for creating specialized outputs
def create_knowledge_output(
    data: Any,
    summary: str,
    tool_name: str,
    retrieved_chunks: List[Dict[str, Any]],
    **kwargs,
) -> KnowledgeOutput:
    """Factory for creating knowledge outputs"""

    metadata = ToolOutputMetadata(
        tool_name=tool_name,
        execution_time=kwargs.get("execution_time", 0.0),
        timestamp=datetime.utcnow().isoformat(),
        output_type=OutputType.KNOWLEDGE,
        data_format=DataFormat.JSON,
        confidence_score=kwargs.get("confidence_score", 1.0),
        quality_score=kwargs.get("quality_score", 1.0),
    )

    return KnowledgeOutput(
        data=data,
        summary=summary,
        metadata=metadata,
        retrieved_chunks=retrieved_chunks,
        total_sources=len(retrieved_chunks),
        relevance_scores=[c.get("score", 0.0) for c in retrieved_chunks],
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ["execution_time", "confidence_score", "quality_score"]
        },
    )


def create_code_output(
    code: str, summary: str, tool_name: str, language: str = "python", **kwargs
) -> CodeOutput:
    """Factory for creating code outputs"""

    metadata = ToolOutputMetadata(
        tool_name=tool_name,
        execution_time=kwargs.get("execution_time", 0.0),
        timestamp=datetime.utcnow().isoformat(),
        output_type=OutputType.CODE,
        data_format=DataFormat.CODE_BLOCK,
        confidence_score=kwargs.get("confidence_score", 1.0),
        quality_score=kwargs.get("quality_score", 1.0),
    )

    return CodeOutput(
        data=code,
        code=code,
        summary=summary,
        metadata=metadata,
        language=language,
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ["execution_time", "confidence_score", "quality_score"]
        },
    )


def create_analysis_output(
    data: Any, summary: str, tool_name: str, metrics: Dict[str, float], **kwargs
) -> AnalysisOutput:
    """Factory for creating analysis outputs"""

    metadata = ToolOutputMetadata(
        tool_name=tool_name,
        execution_time=kwargs.get("execution_time", 0.0),
        timestamp=datetime.utcnow().isoformat(),
        output_type=OutputType.ANALYSIS,
        data_format=DataFormat.JSON,
        confidence_score=kwargs.get("confidence_score", 1.0),
        quality_score=kwargs.get("quality_score", 1.0),
    )

    return AnalysisOutput(
        data=data,
        summary=summary,
        metadata=metadata,
        metrics=metrics,
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ["execution_time", "confidence_score", "quality_score"]
        },
    )
