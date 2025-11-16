"""
Query Processor
Advanced query understanding, reformulation, and optimization for improved RAG retrieval
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

from src.shared.logging import get_logger
from src.shared.llm.integration import get_llm_integration

logger = get_logger(__name__)


class QueryType(Enum):
    """Types of queries for different processing strategies"""

    FACTUAL = "factual"  # "What is X?"
    PROCEDURAL = "procedural"  # "How to do X?"
    CONCEPTUAL = "conceptual"  # "Explain X"
    COMPARATIVE = "comparative"  # "X vs Y"
    TROUBLESHOOTING = "troubleshooting"  # "Why doesn't X work?"
    CODE_SPECIFIC = "code_specific"  # Code-related queries
    EXPLORATORY = "exploratory"  # Open-ended exploration


@dataclass
class QueryAnalysis:
    """Comprehensive query analysis result"""

    original_query: str
    query_type: QueryType
    key_entities: List[str]
    key_concepts: List[str]
    intent: str
    complexity_score: float  # 0-1
    ambiguity_score: float  # 0-1
    requires_decomposition: bool
    suggested_filters: Dict[str, Any]
    confidence: float


@dataclass
class ReformulatedQuery:
    """Result of query reformulation"""

    original_query: str
    reformulated_query: str
    expansion_terms: List[str]
    sub_queries: List[str]  # For complex queries
    search_strategy: str  # "hybrid", "vector", "keyword"
    boost_fields: Dict[str, float]  # Fields to boost in search
    filters: Dict[str, Any]
    reasoning: str  # Why this reformulation


class QueryProcessor:
    """Advanced query processing for optimal RAG retrieval"""

    def __init__(self):
        self.llm_integration = None
        self._initialized = False

        # Query patterns for classification
        self.query_patterns = {
            QueryType.FACTUAL: [
                r"what is",
                r"what are",
                r"define",
                r"definition of",
                r"meaning of",
            ],
            QueryType.PROCEDURAL: [
                r"how to",
                r"how do i",
                r"how can i",
                r"steps to",
                r"guide to",
                r"tutorial",
            ],
            QueryType.CONCEPTUAL: [
                r"explain",
                r"describe",
                r"why does",
                r"why is",
                r"concept of",
                r"theory of",
            ],
            QueryType.COMPARATIVE: [
                r"difference between",
                r"compare",
                r"vs",
                r"versus",
                r"better than",
                r"advantages of",
            ],
            QueryType.TROUBLESHOOTING: [
                r"error",
                r"not working",
                r"fails",
                r"issue with",
                r"problem with",
                r"debug",
                r"fix",
            ],
            QueryType.CODE_SPECIFIC: [
                r"function",
                r"method",
                r"class",
                r"implementation",
                r"code for",
                r"syntax",
                r"algorithm",
            ],
        }

        # Technical domain keywords
        self.domain_keywords = {
            "programming": [
                "python",
                "java",
                "javascript",
                "code",
                "function",
                "class",
            ],
            "architecture": [
                "design",
                "pattern",
                "architecture",
                "system",
                "microservice",
            ],
            "database": ["sql", "database", "query", "table", "index"],
            "devops": ["docker", "kubernetes", "ci/cd", "deployment", "pipeline"],
            "testing": ["test", "testing", "unit test", "integration", "qa"],
        }

    async def initialize(self):
        """Initialize query processor"""
        if self._initialized:
            return

        logger.info("Initializing Query Processor")

        # Initialize LLM integration
        self.llm_integration = await get_llm_integration()

        self._initialized = True
        logger.info("Query Processor initialized")

    async def process_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ReformulatedQuery:
        """
        Main entry point: Analyze and reformulate query for optimal retrieval

        Args:
            query: Original user query
            context: Optional context (conversation history, user preferences, etc.)

        Returns:
            ReformulatedQuery with optimized query and metadata
        """
        if not self._initialized:
            await self.initialize()

        logger.info("Processing query for reformulation", query=query[:100])

        # Step 1: Analyze query
        analysis = await self.analyze_query(query, context)

        # Step 2: Decide if reformulation is needed
        if not self._needs_reformulation(analysis):
            logger.info("Query is clear, no reformulation needed")
            return ReformulatedQuery(
                original_query=query,
                reformulated_query=query,
                expansion_terms=[],
                sub_queries=[],
                search_strategy="hybrid",
                boost_fields={},
                filters={},
                reasoning="Query is clear and well-formed",
            )

        # Step 3: Reformulate query using LLM
        reformulated = await self._reformulate_with_llm(query, analysis, context)

        logger.info(
            "Query reformulated",
            original=query[:50],
            reformulated=reformulated.reformulated_query[:50],
            strategy=reformulated.search_strategy,
        )

        return reformulated

    async def analyze_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> QueryAnalysis:
        """Comprehensive query analysis"""

        # Classify query type
        query_type = self._classify_query_type(query)

        # Extract entities and concepts
        entities = self._extract_entities(query)
        concepts = self._extract_concepts(query)

        # Determine intent
        intent = self._determine_intent(query, query_type)

        # Calculate complexity
        complexity_score = self._calculate_complexity(query, entities, concepts)

        # Calculate ambiguity
        ambiguity_score = self._calculate_ambiguity(query, entities)

        # Check if decomposition needed
        requires_decomposition = complexity_score > 0.7 or " and " in query.lower()

        # Suggest filters based on entities
        suggested_filters = self._suggest_filters(entities, concepts)

        # Calculate confidence
        confidence = self._calculate_analysis_confidence(
            query_type, entities, concepts, ambiguity_score
        )

        return QueryAnalysis(
            original_query=query,
            query_type=query_type,
            key_entities=entities,
            key_concepts=concepts,
            intent=intent,
            complexity_score=complexity_score,
            ambiguity_score=ambiguity_score,
            requires_decomposition=requires_decomposition,
            suggested_filters=suggested_filters,
            confidence=confidence,
        )

    def _classify_query_type(self, query: str) -> QueryType:
        """Classify query into type categories"""
        query_lower = query.lower()

        # Check patterns for each type
        for query_type, patterns in self.query_patterns.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                return query_type

        # Default to exploratory
        return QueryType.EXPLORATORY

    def _extract_entities(self, query: str) -> List[str]:
        """Extract key entities (technologies, tools, concepts)"""
        entities = []
        query_lower = query.lower()

        # Programming languages
        languages = [
            "python",
            "java",
            "javascript",
            "typescript",
            "c++",
            "c#",
            "go",
            "rust",
        ]
        entities.extend([lang for lang in languages if lang in query_lower])

        # Frameworks
        frameworks = ["react", "angular", "vue", "django", "flask", "spring", "express"]
        entities.extend([fw for fw in frameworks if fw in query_lower])

        # Tools
        tools = ["docker", "kubernetes", "git", "jenkins", "aws", "azure"]
        entities.extend([tool for tool in tools if tool in query_lower])

        # Capitalize entities
        entities = [e.title() for e in entities]

        return list(set(entities))  # Remove duplicates

    def _extract_concepts(self, query: str) -> List[str]:
        """Extract key concepts"""
        concepts = []
        query_lower = query.lower()

        # Technical concepts
        concept_keywords = [
            "algorithm",
            "data structure",
            "design pattern",
            "architecture",
            "api",
            "database",
            "testing",
            "deployment",
            "security",
            "performance",
        ]

        concepts.extend([c for c in concept_keywords if c in query_lower])

        return list(set(concepts))

    def _determine_intent(self, query: str, query_type: QueryType) -> str:
        """Determine user intent"""
        intent_map = {
            QueryType.FACTUAL: "learn_definition",
            QueryType.PROCEDURAL: "learn_how_to",
            QueryType.CONCEPTUAL: "understand_concept",
            QueryType.COMPARATIVE: "compare_options",
            QueryType.TROUBLESHOOTING: "solve_problem",
            QueryType.CODE_SPECIFIC: "implement_code",
            QueryType.EXPLORATORY: "explore_topic",
        }

        return intent_map.get(query_type, "general_inquiry")

    def _calculate_complexity(
        self, query: str, entities: List[str], concepts: List[str]
    ) -> float:
        """Calculate query complexity score (0-1)"""
        score = 0.0

        # Length factor
        word_count = len(query.split())
        if word_count > 20:
            score += 0.3
        elif word_count > 10:
            score += 0.2
        elif word_count > 5:
            score += 0.1

        # Entity/concept factor
        score += min(0.3, (len(entities) + len(concepts)) * 0.05)

        # Multi-part query factor
        if " and " in query.lower() or " or " in query.lower():
            score += 0.2

        # Question complexity
        if query.count("?") > 1:
            score += 0.2

        return min(1.0, score)

    def _calculate_ambiguity(self, query: str, entities: List[str]) -> float:
        """Calculate query ambiguity score (0-1)"""
        score = 0.0

        # Very short queries are ambiguous
        if len(query.split()) < 3:
            score += 0.4

        # No entities = ambiguous
        if not entities:
            score += 0.3

        # Vague terms
        vague_terms = ["thing", "stuff", "something", "it", "this", "that"]
        if any(term in query.lower() for term in vague_terms):
            score += 0.3

        return min(1.0, score)

    def _suggest_filters(
        self, entities: List[str], concepts: List[str]
    ) -> Dict[str, Any]:
        """Suggest search filters based on entities and concepts"""
        filters = {}

        # Language filter
        languages = [
            "Python",
            "Java",
            "JavaScript",
            "TypeScript",
            "C++",
            "C#",
            "Go",
            "Rust",
        ]
        detected_langs = [e for e in entities if e in languages]
        if detected_langs:
            filters["language"] = detected_langs[0].lower()

        # Topic filter
        if concepts:
            filters["topic"] = concepts[0]

        return filters

    def _calculate_analysis_confidence(
        self,
        query_type: QueryType,
        entities: List[str],
        concepts: List[str],
        ambiguity_score: float,
    ) -> float:
        """Calculate confidence in the analysis"""
        confidence = 0.5  # Base confidence

        # Clear query type increases confidence
        if query_type != QueryType.EXPLORATORY:
            confidence += 0.2

        # Entities increase confidence
        if entities:
            confidence += 0.2

        # Concepts increase confidence
        if concepts:
            confidence += 0.1

        # Ambiguity decreases confidence
        confidence -= ambiguity_score * 0.3

        return max(0.0, min(1.0, confidence))

    def _needs_reformulation(self, analysis: QueryAnalysis) -> bool:
        """Determine if query needs reformulation"""

        # High ambiguity needs reformulation
        if analysis.ambiguity_score > 0.5:
            return True

        # Complex queries need decomposition
        if analysis.requires_decomposition:
            return True

        # Low confidence needs clarification
        if analysis.confidence < 0.6:
            return True

        return False

    async def _reformulate_with_llm(
        self, query: str, analysis: QueryAnalysis, context: Optional[Dict[str, Any]]
    ) -> ReformulatedQuery:
        """Use LLM to reformulate query for better retrieval"""

        # Build prompt for LLM
        system_prompt = """You are an expert at optimizing search queries for a software engineering knowledge base.

Your task is to reformulate user queries to improve retrieval quality. Consider:
1. Expand abbreviations and acronyms
2. Add relevant technical terms
3. Clarify ambiguous references
4. Break down complex multi-part questions
5. Add domain-specific context

Respond in JSON format:
{
    "reformulated_query": "optimized query string",
    "expansion_terms": ["term1", "term2"],
    "sub_queries": ["sub-query 1", "sub-query 2"],
    "search_strategy": "hybrid|vector|keyword",
    "reasoning": "explanation of changes"
}"""

        # Build user prompt with analysis context
        user_prompt = f"""Original Query: "{query}"

Query Analysis:
- Type: {analysis.query_type.value}
- Intent: {analysis.intent}
- Entities: {', '.join(analysis.key_entities) if analysis.key_entities else 'none'}
- Concepts: {', '.join(analysis.key_concepts) if analysis.key_concepts else 'none'}
- Complexity: {analysis.complexity_score:.2f}
- Ambiguity: {analysis.ambiguity_score:.2f}

Please reformulate this query for optimal retrieval from a software engineering knowledge base."""

        try:
            # Call LLM
            response = await self.llm_integration.generate_response(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,  # Low temperature for consistent reformulation
                max_tokens=1000,  # Increased to avoid truncation
            )

            # Parse JSON response - handle both pure JSON and text with JSON
            import json
            import re

            result = None

            # Try to parse as pure JSON first
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from text (LLM might wrap it in markdown or text)
                # Look for JSON object in the response
                json_match = re.search(
                    r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL
                )
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass

            # If we still don't have valid JSON, use fallback
            if not result or not isinstance(result, dict):
                logger.warning(
                    f"Could not parse LLM response as JSON, using fallback. Response: {response[:100]}"
                )
                return self._fallback_reformulation(query, analysis)

            # Build boost fields based on query type
            boost_fields = self._determine_boost_fields(analysis.query_type)

            return ReformulatedQuery(
                original_query=query,
                reformulated_query=result.get("reformulated_query", query),
                expansion_terms=result.get("expansion_terms", []),
                sub_queries=result.get("sub_queries", []),
                search_strategy=result.get("search_strategy", "hybrid"),
                boost_fields=boost_fields,
                filters=analysis.suggested_filters,
                reasoning=result.get("reasoning", "LLM reformulation applied"),
            )

        except Exception as e:
            logger.warning(
                f"LLM reformulation failed with exception, using fallback: {e}"
            )
            return self._fallback_reformulation(query, analysis)

    def _fallback_reformulation(
        self, query: str, analysis: QueryAnalysis
    ) -> ReformulatedQuery:
        """Fallback reformulation without LLM"""

        # Simple expansion with entities and concepts
        expansion_terms = analysis.key_entities + analysis.key_concepts

        # Add expansion terms to query
        reformulated = query
        if expansion_terms:
            reformulated = f"{query} {' '.join(expansion_terms[:3])}"

        # Determine search strategy
        if analysis.query_type == QueryType.CODE_SPECIFIC:
            search_strategy = "hybrid"  # Code needs both semantic and keyword
        elif analysis.ambiguity_score > 0.6:
            search_strategy = "vector"  # Semantic search for ambiguous queries
        else:
            search_strategy = "hybrid"

        # Boost fields
        boost_fields = self._determine_boost_fields(analysis.query_type)

        return ReformulatedQuery(
            original_query=query,
            reformulated_query=reformulated,
            expansion_terms=expansion_terms[:5],
            sub_queries=[],
            search_strategy=search_strategy,
            boost_fields=boost_fields,
            filters=analysis.suggested_filters,
            reasoning="Fallback reformulation with entity/concept expansion",
        )

    def _determine_boost_fields(self, query_type: QueryType) -> Dict[str, float]:
        """Determine which fields to boost based on query type"""

        boost_map = {
            QueryType.FACTUAL: {"document_title": 1.5, "content": 1.0},
            QueryType.PROCEDURAL: {"content": 1.5, "document_title": 1.0},
            QueryType.CONCEPTUAL: {"content": 1.5, "document_title": 1.2},
            QueryType.COMPARATIVE: {"content": 1.5, "document_title": 1.0},
            QueryType.TROUBLESHOOTING: {"content": 1.5, "content.technical": 1.3},
            QueryType.CODE_SPECIFIC: {"content.code": 2.0, "content": 1.0},
            QueryType.EXPLORATORY: {"content": 1.0, "document_title": 1.0},
        }

        return boost_map.get(query_type, {"content": 1.0})

    async def decompose_complex_query(self, query: str) -> List[str]:
        """Decompose complex query into simpler sub-queries"""

        if not self._initialized:
            await self.initialize()

        system_prompt = """You are an expert at breaking down complex queries into simpler sub-queries.

Given a complex query, decompose it into 2-4 simpler, focused sub-queries that together address the original question.

Respond with a JSON array of sub-queries:
["sub-query 1", "sub-query 2", "sub-query 3"]"""

        user_prompt = f"""Complex Query: "{query}"

Please decompose this into simpler sub-queries."""

        try:
            response = await self.llm_integration.generate_response(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=300,
            )

            import json

            sub_queries = json.loads(response)

            logger.info(f"Decomposed query into {len(sub_queries)} sub-queries")
            return sub_queries

        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [query]  # Return original query as fallback

    async def health_check(self) -> Dict[str, Any]:
        """Health check for query processor"""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "llm_available": self.llm_integration is not None,
            "components": {
                "query_classifier": "operational",
                "entity_extractor": "operational",
                "llm_reformulator": (
                    "operational" if self.llm_integration else "unavailable"
                ),
            },
        }
