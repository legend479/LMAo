"""
Knowledge Retrieval Tool
Enhanced tool integrating with RAG pipeline for advanced knowledge retrieval with scope detection
"""

from typing import Dict, Any, List
import time
import re
from dataclasses import dataclass

from .registry import (
    BaseTool,
    ToolResult,
    ExecutionContext,
    ToolCapabilities,
    ResourceRequirements,
    ToolCapability,
)

# Import RAG pipeline dynamically to avoid circular imports
# from src.rag_pipeline.main import get_rag_pipeline
from src.shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScopeDetectionResult:
    """Result of scope detection analysis"""

    is_in_scope: bool
    confidence: float
    domain_classification: str
    detected_entities: List[str]
    reasoning: str
    suggested_alternatives: List[str] = None


class ScopeDetector:
    """Detects if queries are within software engineering domain"""

    def __init__(self):
        # Software engineering domain keywords and patterns
        self.se_keywords = {
            # Programming languages
            "languages": [
                "python",
                "java",
                "javascript",
                "typescript",
                "c++",
                "c#",
                "go",
                "rust",
                "php",
                "ruby",
                "swift",
                "kotlin",
                "scala",
                "r",
                "matlab",
                "sql",
            ],
            # Frameworks and libraries
            "frameworks": [
                "react",
                "angular",
                "vue",
                "django",
                "flask",
                "spring",
                "express",
                "laravel",
                "rails",
                "tensorflow",
                "pytorch",
                "pandas",
                "numpy",
            ],
            # Software engineering concepts
            "concepts": [
                "algorithm",
                "data structure",
                "design pattern",
                "architecture",
                "database",
                "api",
                "microservices",
                "testing",
                "debugging",
                "optimization",
                "performance",
                "security",
                "authentication",
                "authorization",
                "encryption",
                "deployment",
                "devops",
                "ci/cd",
                "version control",
                "git",
                "docker",
                "kubernetes",
                "cloud",
                "aws",
                "azure",
                "gcp",
            ],
            # Development practices
            "practices": [
                "agile",
                "scrum",
                "tdd",
                "bdd",
                "code review",
                "refactoring",
                "documentation",
                "unit test",
                "integration test",
                "load test",
                "continuous integration",
                "continuous deployment",
            ],
            # Technical terms
            "technical": [
                "function",
                "method",
                "class",
                "object",
                "variable",
                "array",
                "list",
                "dictionary",
                "hash",
                "tree",
                "graph",
                "queue",
                "stack",
                "heap",
                "recursion",
                "iteration",
                "complexity",
                "big o",
                "sorting",
                "searching",
            ],
        }

        # Out-of-scope indicators
        self.out_of_scope_indicators = {
            "medical": [
                "medicine",
                "doctor",
                "patient",
                "disease",
                "treatment",
                "diagnosis",
            ],
            "legal": ["law", "lawyer", "court", "legal", "contract", "lawsuit"],
            "finance": [
                "stock",
                "investment",
                "trading",
                "banking",
                "loan",
                "mortgage",
            ],
            "cooking": ["recipe", "cooking", "ingredient", "kitchen", "food", "meal"],
            "sports": ["football", "basketball", "soccer", "tennis", "game", "player"],
            "travel": ["vacation", "hotel", "flight", "tourism", "destination", "trip"],
        }

        # Ambiguous terms that could be in or out of scope
        self.ambiguous_terms = [
            "network",
            "system",
            "process",
            "service",
            "client",
            "server",
        ]

    def classify_query_domain(self, query: str) -> ScopeDetectionResult:
        """Classify if query is within software engineering domain"""
        query_lower = query.lower()
        words = re.findall(r"\b\w+\b", query_lower)

        # Count matches in different categories
        se_matches = self._count_se_matches(words)
        out_of_scope_matches = self._count_out_of_scope_matches(words)

        # Calculate confidence and classification
        total_se_score = sum(se_matches.values())
        total_out_of_scope_score = sum(out_of_scope_matches.values())

        # Determine if in scope
        is_in_scope = total_se_score > total_out_of_scope_score

        # Calculate confidence
        if total_se_score + total_out_of_scope_score == 0:
            # No clear indicators - check for ambiguous terms or general tech terms
            confidence = (
                0.3
                if any(term in query_lower for term in self.ambiguous_terms)
                else 0.1
            )
            is_in_scope = True  # Default to in-scope for ambiguous queries
        else:
            confidence = max(total_se_score, total_out_of_scope_score) / (
                total_se_score + total_out_of_scope_score
            )

        # Boost confidence for strong SE indicators
        if any(lang in query_lower for lang in self.se_keywords["languages"]):
            confidence = min(1.0, confidence + 0.3)

        # Domain classification
        if is_in_scope:
            domain_classification = self._get_primary_se_domain(se_matches)
        else:
            domain_classification = self._get_primary_out_of_scope_domain(
                out_of_scope_matches
            )

        # Extract entities
        detected_entities = self._extract_entities(words)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            is_in_scope, se_matches, out_of_scope_matches, confidence
        )

        # Suggest alternatives for out-of-scope queries
        suggested_alternatives = None
        if not is_in_scope:
            suggested_alternatives = self._suggest_alternatives(
                query, domain_classification
            )

        return ScopeDetectionResult(
            is_in_scope=is_in_scope,
            confidence=confidence,
            domain_classification=domain_classification,
            detected_entities=detected_entities,
            reasoning=reasoning,
            suggested_alternatives=suggested_alternatives,
        )

    def _count_se_matches(self, words: List[str]) -> Dict[str, int]:
        """Count matches in software engineering categories"""
        matches = {}
        for category, keywords in self.se_keywords.items():
            matches[category] = sum(1 for word in words if word in keywords)
        return matches

    def _count_out_of_scope_matches(self, words: List[str]) -> Dict[str, int]:
        """Count matches in out-of-scope categories"""
        matches = {}
        for category, keywords in self.out_of_scope_indicators.items():
            matches[category] = sum(1 for word in words if word in keywords)
        return matches

    def _get_primary_se_domain(self, se_matches: Dict[str, int]) -> str:
        """Get primary software engineering domain"""
        if not any(se_matches.values()):
            return "general_software_engineering"

        primary_domain = max(se_matches.items(), key=lambda x: x[1])
        return (
            primary_domain[0]
            if primary_domain[1] > 0
            else "general_software_engineering"
        )

    def _get_primary_out_of_scope_domain(
        self, out_of_scope_matches: Dict[str, int]
    ) -> str:
        """Get primary out-of-scope domain"""
        if not any(out_of_scope_matches.values()):
            return "unknown"

        primary_domain = max(out_of_scope_matches.items(), key=lambda x: x[1])
        return primary_domain[0] if primary_domain[1] > 0 else "unknown"

    def _extract_entities(self, words: List[str]) -> List[str]:
        """Extract relevant entities from query"""
        entities = []

        # Extract programming languages
        for lang in self.se_keywords["languages"]:
            if lang in words:
                entities.append(f"language:{lang}")

        # Extract frameworks
        for framework in self.se_keywords["frameworks"]:
            if framework in words:
                entities.append(f"framework:{framework}")

        # Extract concepts
        for concept in self.se_keywords["concepts"]:
            if concept in words:
                entities.append(f"concept:{concept}")

        return entities

    def _generate_reasoning(
        self,
        is_in_scope: bool,
        se_matches: Dict[str, int],
        out_of_scope_matches: Dict[str, int],
        confidence: float,
    ) -> str:
        """Generate human-readable reasoning for classification"""
        if is_in_scope:
            strong_indicators = [
                category for category, count in se_matches.items() if count > 0
            ]
            if strong_indicators:
                return f"Query classified as software engineering domain due to {', '.join(strong_indicators)} indicators (confidence: {confidence:.2f})"
            else:
                return f"Query classified as software engineering domain by default (confidence: {confidence:.2f})"
        else:
            strong_indicators = [
                category
                for category, count in out_of_scope_matches.items()
                if count > 0
            ]
            return f"Query classified as outside software engineering domain due to {', '.join(strong_indicators)} indicators (confidence: {confidence:.2f})"

    def _suggest_alternatives(self, query: str, domain: str) -> List[str]:
        """Suggest alternative software engineering related queries"""
        suggestions = []

        if domain == "medical":
            suggestions = [
                "How to implement a healthcare management system?",
                "What are best practices for medical data security?",
                "How to design APIs for healthcare applications?",
            ]
        elif domain == "finance":
            suggestions = [
                "How to implement financial calculations in software?",
                "What are security considerations for fintech applications?",
                "How to design trading system architecture?",
            ]
        elif domain == "legal":
            suggestions = [
                "How to implement document management systems?",
                "What are compliance requirements for software systems?",
                "How to design audit trails in applications?",
            ]
        else:
            suggestions = [
                "How to design software architecture?",
                "What are best practices for code organization?",
                "How to implement automated testing?",
            ]

        return suggestions


class KnowledgeRetrievalTool(BaseTool):
    """Enhanced tool for retrieving knowledge from the RAG pipeline with scope detection"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.rag_pipeline = None
        self.scope_detector = ScopeDetector()
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
        self.relevance_threshold = 0.3  # Minimum relevance score for results

    async def initialize(self):
        """Initialize the knowledge retrieval tool"""
        await super().initialize()

        # Initialize RAG pipeline connection (dynamic import to avoid circular imports)
        try:
            from src.rag_pipeline.main import get_rag_pipeline

            self.rag_pipeline = await get_rag_pipeline()
            logger.info("RAG pipeline connected successfully")
        except Exception as e:
            logger.warning(f"Failed to connect to RAG pipeline: {e}")
            self.rag_pipeline = None

        logger.info("Enhanced Knowledge Retrieval Tool initialized")

    async def execute(
        self, parameters: Dict[str, Any], context: ExecutionContext
    ) -> ToolResult:
        """Execute knowledge retrieval with scope detection and RAG integration"""

        start_time = time.time()

        try:
            query = parameters["query"]
            filters = parameters.get("filters", {})
            max_results = parameters.get("max_results", 10)
            include_metadata = parameters.get("include_metadata", True)
            rerank = parameters.get("rerank", True)

            logger.info(
                "Executing knowledge retrieval with scope detection",
                query=query[:100],
                session_id=context.session_id,
                priority=context.priority.value,
            )

            # Step 1: Scope detection
            scope_result = self.scope_detector.classify_query_domain(query)

            # Step 2: Handle out-of-scope queries
            if not scope_result.is_in_scope:
                return self._generate_out_of_scope_response(
                    query, scope_result, time.time() - start_time
                )

            # Step 3: Check cache for in-scope queries
            cache_key = self._generate_cache_key(query, filters, max_results)
            cached_result = self._get_cached_result(cache_key)

            if cached_result:
                logger.info("Returning cached result", cache_key=cache_key)
                execution_time = time.time() - start_time
                cached_result.execution_time = execution_time
                cached_result.metadata["cache_hit"] = True
                return cached_result

            # Step 4: Execute RAG pipeline search
            if self.rag_pipeline:
                search_result = await self._execute_rag_search(
                    query, filters, max_results, rerank
                )
            else:
                # Fallback to mock results if RAG pipeline not available
                logger.warning("RAG pipeline not available, using mock results")
                search_result = self._generate_mock_search_result(query, max_results)

            # Step 5: Assemble hierarchical context
            hierarchical_context = self._assemble_hierarchical_context(
                search_result["results"]
            )

            # Step 6: Calculate relevance and confidence scores
            relevance_score = self._calculate_relevance_score(
                search_result["results"], query
            )
            confidence_score = self._calculate_confidence_score_with_scope(
                search_result, query, scope_result.confidence
            )

            # Step 7: Prepare final result
            result_data = {
                "query": query,
                "results": search_result["results"],
                "hierarchical_context": hierarchical_context,
                "total_results": search_result["total_results"],
                "processing_time": search_result["processing_time"],
                "scope_detection": {
                    "is_in_scope": scope_result.is_in_scope,
                    "confidence": scope_result.confidence,
                    "domain_classification": scope_result.domain_classification,
                    "detected_entities": scope_result.detected_entities,
                    "reasoning": scope_result.reasoning,
                },
                "search_metadata": {
                    "search_type": search_result.get("search_type", "hybrid"),
                    "max_score": search_result.get("max_score", 0.0),
                    "filters_applied": filters,
                    "reranking_applied": rerank,
                    "relevance_threshold": self.relevance_threshold,
                },
            }

            # Include detailed metadata if requested
            if include_metadata:
                result_data["detailed_metadata"] = self._generate_detailed_metadata(
                    search_result, scope_result, query
                )

            execution_time = time.time() - start_time

            result = ToolResult(
                data=result_data,
                metadata={
                    "tool": "knowledge_retrieval",
                    "version": "3.0.0",
                    "query_length": len(query),
                    "filters_applied": len(filters),
                    "max_results": max_results,
                    "session_id": context.session_id,
                    "priority": context.priority.value,
                    "cache_hit": False,
                    "processing_strategy": "rag_with_scope_detection",
                    "scope_confidence": scope_result.confidence,
                    "domain_classification": scope_result.domain_classification,
                },
                execution_time=execution_time,
                success=True,
                resource_usage={
                    "cpu_usage": 0.4,
                    "memory_usage_mb": 256,
                    "network_requests": 1 if self.rag_pipeline else 0,
                },
                quality_score=relevance_score,
                confidence_score=confidence_score,
            )

            # Cache the result
            self._cache_result(cache_key, result)

            return result

        except Exception as e:
            execution_time = time.time() - start_time

            logger.error(
                "Knowledge retrieval failed",
                error=str(e),
                session_id=context.session_id,
            )

            return ToolResult(
                data=None,
                metadata={
                    "tool": "knowledge_retrieval",
                    "session_id": context.session_id,
                    "error_type": type(e).__name__,
                },
                execution_time=execution_time,
                success=False,
                error_message=str(e),
                quality_score=0.0,
                confidence_score=0.0,
            )

    def get_schema(self) -> Dict[str, Any]:
        """Get enhanced tool schema with scope detection"""
        return {
            "name": "knowledge_retrieval",
            "description": "Retrieve relevant knowledge from the software engineering knowledge base with scope detection, hierarchical context assembly, and advanced filtering",
            "version": "3.0.0",
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant information",
                    "required": True,
                    "min_length": 1,
                    "max_length": 1000,
                },
                "filters": {
                    "type": "object",
                    "description": "Optional filters to narrow search results",
                    "properties": {
                        "document_type": {
                            "type": "string",
                            "enum": ["pdf", "docx", "md", "txt", "code"],
                        },
                        "topic": {
                            "type": "string",
                            "description": "Specific topic or domain",
                        },
                        "date_range": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "string", "format": "date"},
                                "end": {"type": "string", "format": "date"},
                            },
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language filter",
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": ["beginner", "intermediate", "advanced"],
                        },
                    },
                    "required": False,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50,
                    "required": False,
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Include detailed metadata in results",
                    "default": True,
                    "required": False,
                },
                "rerank": {
                    "type": "boolean",
                    "description": "Apply reranking for better relevance",
                    "default": True,
                    "required": False,
                },
            },
            "required_params": ["query"],
            "returns": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "score": {"type": "number", "minimum": 0, "maximum": 1},
                                "chunk_id": {"type": "string"},
                                "document_id": {"type": "string"},
                                "chunk_type": {"type": "string"},
                                "parent_chunk_id": {"type": "string"},
                                "highlights": {"type": "object"},
                                "metadata": {"type": "object"},
                            },
                        },
                    },
                    "hierarchical_context": {
                        "type": "object",
                        "properties": {
                            "total_documents": {"type": "integer"},
                            "total_chunks": {"type": "integer"},
                            "parent_chunks": {"type": "integer"},
                            "child_chunks": {"type": "integer"},
                            "relationships": {"type": "array"},
                            "document_coverage": {"type": "object"},
                        },
                    },
                    "total_results": {"type": "integer"},
                    "processing_time": {"type": "number"},
                    "scope_detection": {
                        "type": "object",
                        "properties": {
                            "is_in_scope": {"type": "boolean"},
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "domain_classification": {"type": "string"},
                            "detected_entities": {"type": "array"},
                            "reasoning": {"type": "string"},
                            "suggested_alternatives": {"type": "array"},
                        },
                    },
                    "search_metadata": {"type": "object"},
                    "detailed_metadata": {"type": "object"},
                    "message": {"type": "string"},
                },
            },
            "capabilities": {
                "primary": "knowledge_retrieval",
                "secondary": ["data_analysis", "content_filtering"],
                "input_types": ["text", "query"],
                "output_types": ["structured_data", "ranked_results"],
            },
        }

    def get_capabilities(self) -> ToolCapabilities:
        """Get tool capabilities"""
        return ToolCapabilities(
            primary_capability=ToolCapability.KNOWLEDGE_RETRIEVAL,
            secondary_capabilities=[ToolCapability.DATA_ANALYSIS],
            input_types=["text", "query", "structured_query"],
            output_types=["structured_data", "ranked_results", "json"],
            supported_formats=["json", "text", "markdown"],
            language_support=["en", "code"],
        )

    def get_resource_requirements(self) -> ResourceRequirements:
        """Get resource requirements"""
        return ResourceRequirements(
            cpu_cores=1.0,
            memory_mb=512,
            network_bandwidth_mbps=10.0,
            storage_mb=100,
            gpu_memory_mb=0,
            max_execution_time=30,
            concurrent_limit=10,
        )

    def _generate_cache_key(
        self, query: str, filters: Dict[str, Any], max_results: int
    ) -> str:
        """Generate cache key for the request"""
        import hashlib

        cache_data = f"{query}_{str(sorted(filters.items()))}_{max_results}"
        return hashlib.md5(cache_data.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> ToolResult:
        """Get cached result if available and not expired"""
        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            if time.time() - cached_item["timestamp"] < self.cache_ttl:
                return cached_item["result"]
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: ToolResult):
        """Cache the result"""
        self.cache[cache_key] = {"result": result, "timestamp": time.time()}

        # Simple cache cleanup - remove oldest entries if cache is too large
        if len(self.cache) > 100:
            oldest_key = min(
                self.cache.keys(), key=lambda k: self.cache[k]["timestamp"]
            )
            del self.cache[oldest_key]

    def _calculate_processing_time(
        self, query: str, filters: Dict[str, Any], max_results: int
    ) -> float:
        """Calculate realistic processing time based on query complexity"""

        base_time = 0.1

        # Query complexity factor
        query_complexity = len(query.split()) / 10.0
        complexity_factor = min(query_complexity, 2.0)

        # Filter complexity factor
        filter_factor = len(filters) * 0.05

        # Results factor
        results_factor = max_results * 0.01

        return base_time + complexity_factor + filter_factor + results_factor

    def _generate_mock_results(
        self, query: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """Generate realistic mock results"""

        results = []
        query_lower = query.lower()

        # Generate results based on query content
        if "python" in query_lower:
            results.extend(
                [
                    {
                        "content": "Python is a high-level, interpreted programming language with dynamic semantics. Its high-level built-in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development.",
                        "source": "python_documentation.md",
                        "relevance_score": 0.95,
                        "chunk_id": "python_intro_001",
                        "document_type": "documentation",
                        "topic": "programming_languages",
                        "metadata": {"language": "python", "difficulty": "beginner"},
                    },
                    {
                        "content": "Python's design philosophy emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.",
                        "source": "python_philosophy.md",
                        "relevance_score": 0.88,
                        "chunk_id": "python_design_002",
                        "document_type": "documentation",
                        "topic": "programming_philosophy",
                        "metadata": {
                            "language": "python",
                            "difficulty": "intermediate",
                        },
                    },
                ]
            )

        elif "javascript" in query_lower:
            results.extend(
                [
                    {
                        "content": "JavaScript is a programming language that conforms to the ECMAScript specification. JavaScript is high-level, often just-in-time compiled, and multi-paradigm.",
                        "source": "javascript_basics.md",
                        "relevance_score": 0.92,
                        "chunk_id": "js_intro_001",
                        "document_type": "documentation",
                        "topic": "programming_languages",
                        "metadata": {
                            "language": "javascript",
                            "difficulty": "beginner",
                        },
                    }
                ]
            )

        elif "algorithm" in query_lower:
            results.extend(
                [
                    {
                        "content": "An algorithm is a finite sequence of well-defined, computer-implementable instructions, typically to solve a class of problems or to perform a computation.",
                        "source": "algorithms_introduction.md",
                        "relevance_score": 0.90,
                        "chunk_id": "algo_def_001",
                        "document_type": "textbook",
                        "topic": "algorithms",
                        "metadata": {"difficulty": "intermediate"},
                    }
                ]
            )

        else:
            # Generic software engineering results
            results.extend(
                [
                    {
                        "content": "Software engineering is the systematic application of engineering approaches to the development of software. It involves the application of engineering principles to software development.",
                        "source": "software_engineering_fundamentals.md",
                        "relevance_score": 0.85,
                        "chunk_id": "se_intro_001",
                        "document_type": "textbook",
                        "topic": "software_engineering",
                        "metadata": {"difficulty": "beginner"},
                    },
                    {
                        "content": "The software development life cycle (SDLC) is a process used by the software industry to design, develop and test high quality software. The SDLC aims to produce a high-quality software that meets or exceeds customer expectations.",
                        "source": "sdlc_overview.md",
                        "relevance_score": 0.82,
                        "chunk_id": "sdlc_001",
                        "document_type": "documentation",
                        "topic": "software_development",
                        "metadata": {"difficulty": "intermediate"},
                    },
                ]
            )

        # Return up to max_results
        return results[:max_results]

    def _analyze_query_intent(self, query: str) -> str:
        """Analyze query intent"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["what", "define", "explain"]):
            return "definition"
        elif any(word in query_lower for word in ["how", "implement", "create"]):
            return "implementation"
        elif any(word in query_lower for word in ["why", "reason", "benefit"]):
            return "explanation"
        elif any(word in query_lower for word in ["example", "sample", "demo"]):
            return "example"
        else:
            return "general"

    def _assess_query_complexity(self, query: str) -> str:
        """Assess query complexity"""
        word_count = len(query.split())

        if word_count <= 5:
            return "simple"
        elif word_count <= 15:
            return "moderate"
        else:
            return "complex"

    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from query"""
        entities = []
        query_lower = query.lower()

        # Programming languages
        languages = ["python", "javascript", "java", "c++", "c#", "go", "rust"]
        entities.extend([lang for lang in languages if lang in query_lower])

        # Frameworks
        frameworks = ["react", "angular", "vue", "django", "flask", "spring"]
        entities.extend([fw for fw in frameworks if fw in query_lower])

        # Concepts
        concepts = ["algorithm", "data structure", "design pattern", "api", "database"]
        entities.extend([concept for concept in concepts if concept in query_lower])

        return entities

    def _calculate_quality_score(self, result_data: Dict[str, Any]) -> float:
        """Calculate quality score for the results"""

        if not result_data.get("results"):
            return 0.0

        # Base quality score
        quality = 0.7

        # Boost for multiple results
        if len(result_data["results"]) > 1:
            quality += 0.1

        # Boost for high relevance scores
        avg_relevance = sum(
            r.get("relevance_score", 0) for r in result_data["results"]
        ) / len(result_data["results"])
        quality += avg_relevance * 0.2

        # Boost for metadata richness
        if result_data.get("search_metadata"):
            quality += 0.1

        return min(quality, 1.0)

    def _calculate_confidence_score(
        self, result_data: Dict[str, Any], query: str
    ) -> float:
        """Calculate confidence score for the results"""

        confidence = 0.8  # Base confidence for mock results

        # Boost confidence if query matches known patterns
        if any(
            entity in query.lower() for entity in ["python", "javascript", "algorithm"]
        ):
            confidence += 0.1

        # Reduce confidence for very short queries
        if len(query.split()) < 3:
            confidence -= 0.2

        return max(0.0, min(confidence, 1.0))

    def _generate_out_of_scope_response(
        self, query: str, scope_result: ScopeDetectionResult, execution_time: float
    ) -> ToolResult:
        """Generate response for out-of-scope queries"""

        response_message = (
            f"I specialize in software engineering topics, but your query appears to be about "
            f"{scope_result.domain_classification}. {scope_result.reasoning}"
        )

        result_data = {
            "query": query,
            "results": [],
            "total_results": 0,
            "processing_time": execution_time,
            "scope_detection": {
                "is_in_scope": False,
                "confidence": scope_result.confidence,
                "domain_classification": scope_result.domain_classification,
                "detected_entities": scope_result.detected_entities,
                "reasoning": scope_result.reasoning,
                "suggested_alternatives": scope_result.suggested_alternatives,
            },
            "message": response_message,
        }

        return ToolResult(
            data=result_data,
            metadata={
                "tool": "knowledge_retrieval",
                "version": "3.0.0",
                "out_of_scope": True,
                "domain_classification": scope_result.domain_classification,
            },
            execution_time=execution_time,
            success=True,  # Successfully detected out-of-scope
            quality_score=0.0,
            confidence_score=scope_result.confidence,
        )

    async def _execute_rag_search(
        self, query: str, filters: Dict[str, Any], max_results: int, rerank: bool
    ) -> Dict[str, Any]:
        """Execute search using RAG pipeline"""

        try:
            # Ensure RAG pipeline is available
            if not self.rag_pipeline:
                try:
                    from src.rag_pipeline.main import get_rag_pipeline

                    self.rag_pipeline = await get_rag_pipeline()
                except Exception as e:
                    logger.error(f"Failed to initialize RAG pipeline: {e}")
                    raise RuntimeError("RAG pipeline not available")

            # Determine search type based on filters and preferences
            search_type = "hybrid"  # Default to hybrid search

            # Execute search
            search_result = await self.rag_pipeline.search(
                query=query,
                filters=filters,
                max_results=max_results,
                search_type=search_type,
            )

            # Filter results by relevance threshold
            filtered_results = [
                result
                for result in search_result["results"]
                if result.get("score", 0) >= self.relevance_threshold
            ]

            return {
                "results": filtered_results,
                "total_results": len(filtered_results),
                "processing_time": search_result.get("processing_time", 0.0),
                "search_type": search_result.get("search_type", "hybrid"),
                "max_score": search_result.get("max_score", 0.0),
            }

        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            # Return empty results on failure
            return {
                "results": [],
                "total_results": 0,
                "processing_time": 0.0,
                "search_type": "failed",
                "max_score": 0.0,
                "error": str(e),
            }

    def _generate_mock_search_result(
        self, query: str, max_results: int
    ) -> Dict[str, Any]:
        """Generate mock search results when RAG pipeline is unavailable"""

        mock_results = self._generate_mock_results(query, max_results)

        return {
            "results": [
                {
                    "chunk_id": result["chunk_id"],
                    "content": result["content"],
                    "score": result["relevance_score"],
                    "document_id": f"doc_{result['chunk_id'].split('_')[0]}",
                    "chunk_type": result["document_type"],
                    "parent_chunk_id": None,
                    "highlights": {},
                    "metadata": result["metadata"],
                }
                for result in mock_results
            ],
            "total_results": len(mock_results),
            "processing_time": 0.1,
            "search_type": "mock",
            "max_score": (
                max(r["relevance_score"] for r in mock_results) if mock_results else 0.0
            ),
        }

    def _assemble_hierarchical_context(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assemble hierarchical parent-child context relationships"""

        if not results:
            return {"parent_chunks": [], "child_chunks": [], "relationships": []}

        # Group results by document and chunk hierarchy
        document_groups = {}
        parent_child_relationships = []

        for result in results:
            doc_id = result.get("document_id", "unknown")
            chunk_id = result.get("chunk_id", "unknown")
            parent_chunk_id = result.get("parent_chunk_id")

            if doc_id not in document_groups:
                document_groups[doc_id] = {
                    "document_id": doc_id,
                    "chunks": [],
                    "parent_chunks": [],
                    "child_chunks": [],
                }

            document_groups[doc_id]["chunks"].append(result)

            # Track parent-child relationships
            if parent_chunk_id:
                parent_child_relationships.append(
                    {
                        "parent_chunk_id": parent_chunk_id,
                        "child_chunk_id": chunk_id,
                        "document_id": doc_id,
                    }
                )
                document_groups[doc_id]["child_chunks"].append(result)
            else:
                document_groups[doc_id]["parent_chunks"].append(result)

        # Create hierarchical context summary
        context_summary = {
            "total_documents": len(document_groups),
            "total_chunks": len(results),
            "parent_chunks": sum(
                len(group["parent_chunks"]) for group in document_groups.values()
            ),
            "child_chunks": sum(
                len(group["child_chunks"]) for group in document_groups.values()
            ),
            "relationships": parent_child_relationships,
            "document_coverage": {
                doc_id: {
                    "chunk_count": len(group["chunks"]),
                    "has_hierarchy": len(group["child_chunks"]) > 0,
                }
                for doc_id, group in document_groups.items()
            },
        }

        return context_summary

    def _calculate_relevance_score(
        self, results: List[Dict[str, Any]], query: str
    ) -> float:
        """Calculate aggregate relevance score for quality assessment"""

        if not results:
            return 0.0

        # Calculate weighted average of result scores
        total_score = 0.0
        total_weight = 0.0

        for i, result in enumerate(results):
            score = result.get("score", 0.0)
            # Weight decreases with rank (first result has highest weight)
            weight = 1.0 / (i + 1)
            total_score += score * weight
            total_weight += weight

        average_relevance = total_score / total_weight if total_weight > 0 else 0.0

        # Boost score if we have multiple relevant results
        diversity_bonus = min(0.2, len(results) * 0.05)

        # Boost score if results span multiple documents (indicates comprehensive coverage)
        unique_docs = len(set(r.get("document_id", "") for r in results))
        coverage_bonus = min(0.1, unique_docs * 0.02)

        final_score = min(1.0, average_relevance + diversity_bonus + coverage_bonus)

        return final_score

    def _calculate_confidence_score_with_scope(
        self, search_result: Dict[str, Any], query: str, scope_confidence: float
    ) -> float:
        """Calculate confidence score incorporating scope detection"""

        base_confidence = 0.7

        # Factor in scope detection confidence
        scope_factor = scope_confidence * 0.3

        # Factor in search result quality
        results = search_result.get("results", [])
        if results:
            max_score = search_result.get("max_score", 0.0)
            result_count = len(results)

            # Higher confidence with better scores and more results
            result_factor = min(0.3, max_score * 0.2 + min(result_count, 5) * 0.02)
        else:
            result_factor = -0.2  # Lower confidence with no results

        # Factor in query complexity (more complex queries have lower confidence)
        query_complexity = self._assess_query_complexity(query)
        complexity_factor = {"simple": 0.1, "moderate": 0.0, "complex": -0.1}.get(
            query_complexity, 0.0
        )

        final_confidence = max(
            0.0,
            min(
                1.0, base_confidence + scope_factor + result_factor + complexity_factor
            ),
        )

        return final_confidence

    def _generate_detailed_metadata(
        self,
        search_result: Dict[str, Any],
        scope_result: ScopeDetectionResult,
        query: str,
    ) -> Dict[str, Any]:
        """Generate detailed metadata for the search results"""

        results = search_result.get("results", [])

        # Analyze result distribution
        document_types = {}
        chunk_types = {}
        score_distribution = []

        for result in results:
            # Document type analysis
            doc_type = result.get("metadata", {}).get("document_category", "unknown")
            document_types[doc_type] = document_types.get(doc_type, 0) + 1

            # Chunk type analysis
            chunk_type = result.get("chunk_type", "unknown")
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

            # Score distribution
            score_distribution.append(result.get("score", 0.0))

        # Calculate statistics
        avg_score = (
            sum(score_distribution) / len(score_distribution)
            if score_distribution
            else 0.0
        )
        min_score = min(score_distribution) if score_distribution else 0.0
        max_score = max(score_distribution) if score_distribution else 0.0

        return {
            "query_analysis": {
                "length": len(query),
                "word_count": len(query.split()),
                "complexity": self._assess_query_complexity(query),
                "intent": self._analyze_query_intent(query),
                "entities": scope_result.detected_entities,
            },
            "result_analysis": {
                "total_results": len(results),
                "document_type_distribution": document_types,
                "chunk_type_distribution": chunk_types,
                "score_statistics": {
                    "average": avg_score,
                    "minimum": min_score,
                    "maximum": max_score,
                    "above_threshold": sum(
                        1 for s in score_distribution if s >= self.relevance_threshold
                    ),
                },
            },
            "search_performance": {
                "processing_time": search_result.get("processing_time", 0.0),
                "search_type": search_result.get("search_type", "unknown"),
                "cache_used": False,  # Will be updated if cache is used
            },
        }

    async def cleanup(self):
        """Cleanup tool resources"""
        logger.info("Cleaning up Enhanced Knowledge Retrieval Tool")

        # Clear cache
        self.cache.clear()

        # Cleanup RAG pipeline connections if needed
        if self.rag_pipeline:
            try:
                # RAG pipeline cleanup is handled by the pipeline itself
                pass
            except Exception as e:
                logger.warning(f"Error during RAG pipeline cleanup: {e}")

        self.rag_pipeline = None
