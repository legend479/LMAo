"""
Adaptive Retrieval Strategy
Dynamic retrieval strategy selection and self-correcting retrieval
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from src.shared.logging import get_logger
from src.shared.llm.integration import get_llm_integration

logger = get_logger(__name__)


class RetrievalStrategy(Enum):
    """Available retrieval strategies"""

    HYBRID = "hybrid"  # BM25 + Vector
    VECTOR_ONLY = "vector"  # Pure semantic search
    KEYWORD_ONLY = "keyword"  # Pure BM25
    MULTI_HOP = "multi_hop"  # Iterative retrieval
    DENSE_FIRST = "dense_first"  # Vector then keyword
    SPARSE_FIRST = "sparse_first"  # Keyword then vector


class QueryComplexity(Enum):
    """Query complexity levels"""

    SIMPLE = "simple"  # Single concept, clear intent
    MODERATE = "moderate"  # Multiple concepts, clear intent
    COMPLEX = "complex"  # Multiple concepts, complex relationships
    VERY_COMPLEX = "very_complex"  # Multi-hop reasoning required


@dataclass
class RetrievalQuality:
    """Assessment of retrieval quality"""

    relevance_score: float  # 0-1
    coverage_score: float  # 0-1
    diversity_score: float  # 0-1
    confidence_score: float  # 0-1
    overall_quality: float  # 0-1
    needs_retry: bool
    retry_strategy: Optional[RetrievalStrategy] = None
    reasoning: str = ""


@dataclass
class AdaptiveRetrievalResult:
    """Result of adaptive retrieval"""

    results: List[Dict[str, Any]]
    strategy_used: RetrievalStrategy
    quality_assessment: RetrievalQuality
    iterations: int
    total_time: float
    metadata: Dict[str, Any]


class AdaptiveRetrievalEngine:
    """Adaptive retrieval with strategy selection and self-correction"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Configuration
        self.quality_threshold = self.config.get("quality_threshold", 0.6)
        self.max_iterations = self.config.get("max_iterations", 3)
        self.enable_self_correction = self.config.get("enable_self_correction", True)
        self.enable_multi_hop = self.config.get("enable_multi_hop", True)

        # Components
        self.llm_integration = None
        self.query_processor = None
        self.search_engine = None

        self._initialized = False

        # Strategy performance tracking
        self.strategy_performance = {
            strategy: {"success": 0, "total": 0, "avg_quality": 0.0}
            for strategy in RetrievalStrategy
        }

    async def initialize(self, search_engine, query_processor=None):
        """Initialize adaptive retrieval engine"""
        if self._initialized:
            return

        logger.info("Initializing Adaptive Retrieval Engine")

        self.search_engine = search_engine
        self.query_processor = query_processor
        self.llm_integration = await get_llm_integration()

        self._initialized = True
        logger.info("Adaptive Retrieval Engine initialized")

    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = 10,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> AdaptiveRetrievalResult:
        """
        Adaptive retrieval with strategy selection and self-correction

        Args:
            query: Search query
            filters: Optional filters
            max_results: Maximum results to return
            query_analysis: Optional pre-computed query analysis

        Returns:
            AdaptiveRetrievalResult with results and metadata
        """
        if not self._initialized:
            raise RuntimeError("AdaptiveRetrievalEngine not initialized")

        start_time = time.time()
        iterations = 0

        logger.info("Starting adaptive retrieval", query=query[:100])

        # Step 1: Select initial strategy
        initial_strategy = await self._select_strategy(query, query_analysis)

        # Step 2: Execute retrieval
        results, strategy_used = await self._execute_retrieval(
            query, filters, max_results, initial_strategy
        )
        iterations += 1

        # Step 3: Assess quality
        quality = await self._assess_quality(query, results, query_analysis)

        # Step 4: Self-correction loop
        if (
            self.enable_self_correction
            and quality.needs_retry
            and iterations < self.max_iterations
        ):
            logger.info(
                "Quality below threshold, attempting self-correction",
                quality=quality.overall_quality,
                retry_strategy=(
                    quality.retry_strategy.value if quality.retry_strategy else None
                ),
            )

            results, strategy_used, quality, iterations = await self._self_correct(
                query, filters, max_results, quality, iterations, query_analysis
            )

        # Update strategy performance
        self._update_strategy_performance(strategy_used, quality.overall_quality)

        total_time = time.time() - start_time

        logger.info(
            "Adaptive retrieval completed",
            strategy=strategy_used.value,
            quality=quality.overall_quality,
            iterations=iterations,
            time=f"{total_time:.2f}s",
        )

        return AdaptiveRetrievalResult(
            results=results,
            strategy_used=strategy_used,
            quality_assessment=quality,
            iterations=iterations,
            total_time=total_time,
            metadata={
                "initial_strategy": initial_strategy.value,
                "final_strategy": strategy_used.value,
                "self_correction_applied": iterations > 1,
            },
        )

    async def _select_strategy(
        self, query: str, query_analysis: Optional[Dict[str, Any]]
    ) -> RetrievalStrategy:
        """Select optimal retrieval strategy based on query characteristics"""

        # If we have query analysis, use it
        if query_analysis:
            complexity = query_analysis.get("complexity_score", 0.5)
            query_type = query_analysis.get("query_type", "exploratory")
            entities = query_analysis.get("key_entities", [])
        else:
            # Simple heuristics
            complexity = self._estimate_complexity(query)
            query_type = "exploratory"
            entities = []

        # Strategy selection logic
        if complexity > 0.7:
            # Complex queries benefit from multi-hop
            if self.enable_multi_hop:
                return RetrievalStrategy.MULTI_HOP
            else:
                return RetrievalStrategy.HYBRID

        elif query_type == "code_specific" or any(
            keyword in query.lower()
            for keyword in ["function", "class", "method", "code"]
        ):
            # Code queries benefit from hybrid search
            return RetrievalStrategy.HYBRID

        elif len(query.split()) < 5:
            # Short queries benefit from semantic search
            return RetrievalStrategy.VECTOR_ONLY

        elif len(entities) > 3:
            # Entity-rich queries benefit from keyword search
            return RetrievalStrategy.KEYWORD_ONLY

        else:
            # Default to hybrid
            return RetrievalStrategy.HYBRID

    def _estimate_complexity(self, query: str) -> float:
        """Estimate query complexity"""
        score = 0.0

        # Length factor
        word_count = len(query.split())
        if word_count > 20:
            score += 0.3
        elif word_count > 10:
            score += 0.2

        # Multi-part query
        if " and " in query.lower() or " or " in query.lower():
            score += 0.3

        # Question complexity
        if query.count("?") > 1:
            score += 0.2

        # Technical terms
        technical_terms = [
            "algorithm",
            "architecture",
            "implementation",
            "optimization",
        ]
        if any(term in query.lower() for term in technical_terms):
            score += 0.2

        return min(1.0, score)

    async def _execute_retrieval(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        max_results: int,
        strategy: RetrievalStrategy,
    ) -> Tuple[List[Dict[str, Any]], RetrievalStrategy]:
        """Execute retrieval with specified strategy"""

        logger.debug(f"Executing retrieval with strategy: {strategy.value}")

        if strategy == RetrievalStrategy.MULTI_HOP:
            results = await self._multi_hop_retrieval(query, filters, max_results)
            actual_strategy = RetrievalStrategy.MULTI_HOP

        elif strategy == RetrievalStrategy.DENSE_FIRST:
            results = await self._dense_first_retrieval(query, filters, max_results)
            actual_strategy = RetrievalStrategy.DENSE_FIRST

        elif strategy == RetrievalStrategy.SPARSE_FIRST:
            results = await self._sparse_first_retrieval(query, filters, max_results)
            actual_strategy = RetrievalStrategy.SPARSE_FIRST

        else:
            # Standard strategies
            search_type = strategy.value
            response = await self.search_engine.search(
                query=query,
                filters=filters or {},
                max_results=max_results,
                search_type=search_type,
            )
            results = (
                response.results
                if hasattr(response, "results")
                else response.get("results", [])
            )
            actual_strategy = strategy

        # Convert SearchResult objects to dictionaries
        results_as_dicts = []
        for r in results:
            if hasattr(r, "__dataclass_fields__"):  # It's a SearchResult dataclass
                results_as_dicts.append(
                    {
                        "chunk_id": r.chunk_id,
                        "content": r.content,
                        "score": r.score,
                        "metadata": r.metadata,
                        "document_id": r.document_id,
                        "chunk_type": r.chunk_type,
                        "parent_chunk_id": r.parent_chunk_id,
                        "highlights": r.highlights,
                    }
                )
            else:  # Already a dict
                results_as_dicts.append(r)

        return results_as_dicts, actual_strategy

    async def _multi_hop_retrieval(
        self, query: str, filters: Optional[Dict[str, Any]], max_results: int
    ) -> List[Dict[str, Any]]:
        """Multi-hop retrieval for complex queries"""

        logger.info("Executing multi-hop retrieval")

        # Step 1: Initial retrieval
        response = await self.search_engine.search(
            query=query,
            filters=filters or {},
            max_results=max_results // 2,
            search_type="hybrid",
        )
        initial_results = (
            response.results
            if hasattr(response, "results")
            else response.get("results", [])
        )

        if not initial_results:
            return []

        # Step 2: Extract key concepts from initial results
        key_concepts = self._extract_concepts_from_results(initial_results)

        # Step 3: Expand query with key concepts
        expanded_query = f"{query} {' '.join(key_concepts[:3])}"

        # Step 4: Second hop retrieval
        response2 = await self.search_engine.search(
            query=expanded_query,
            filters=filters or {},
            max_results=max_results // 2,
            search_type="vector",  # Use semantic search for expansion
        )
        second_hop_results = (
            response2.results
            if hasattr(response2, "results")
            else response2.get("results", [])
        )

        # Step 5: Merge and deduplicate
        all_results = initial_results + second_hop_results
        unique_results = self._deduplicate_results(all_results)

        # Step 6: Re-rank by relevance
        sorted_results = sorted(
            unique_results,
            key=lambda x: x.get("score", 0.0) if isinstance(x, dict) else x.score,
            reverse=True,
        )[:max_results]

        # Convert SearchResult objects to dictionaries
        results_as_dicts = []
        for r in sorted_results:
            if hasattr(r, "__dataclass_fields__"):  # It's a SearchResult dataclass
                results_as_dicts.append(
                    {
                        "chunk_id": r.chunk_id,
                        "content": r.content,
                        "score": r.score,
                        "metadata": r.metadata,
                        "document_id": r.document_id,
                        "chunk_type": r.chunk_type,
                        "parent_chunk_id": r.parent_chunk_id,
                        "highlights": r.highlights,
                    }
                )
            else:  # Already a dict
                results_as_dicts.append(r)

        logger.info(
            f"Multi-hop retrieval: {len(initial_results)} + {len(second_hop_results)} -> {len(results_as_dicts)} results"
        )

        return results_as_dicts

    async def _dense_first_retrieval(
        self, query: str, filters: Optional[Dict[str, Any]], max_results: int
    ) -> List[Dict[str, Any]]:
        """Dense (vector) first, then sparse (keyword) retrieval"""

        # First: Vector search
        response1 = await self.search_engine.search(
            query=query,
            filters=filters or {},
            max_results=max_results,
            search_type="vector",
        )
        vector_results = (
            response1.results
            if hasattr(response1, "results")
            else response1.get("results", [])
        )

        # If vector search is good enough, return
        if len(vector_results) >= max_results // 2:
            return vector_results

        # Second: Keyword search to fill gaps
        response2 = await self.search_engine.search(
            query=query,
            filters=filters or {},
            max_results=max_results - len(vector_results),
            search_type="keyword",
        )
        keyword_results = (
            response2.results
            if hasattr(response2, "results")
            else response2.get("results", [])
        )

        # Merge and deduplicate
        all_results = vector_results + keyword_results
        deduped = self._deduplicate_results(all_results)[:max_results]

        # Convert SearchResult objects to dictionaries
        results_as_dicts = []
        for r in deduped:
            if hasattr(r, "__dataclass_fields__"):  # It's a SearchResult dataclass
                results_as_dicts.append(
                    {
                        "chunk_id": r.chunk_id,
                        "content": r.content,
                        "score": r.score,
                        "metadata": r.metadata,
                        "document_id": r.document_id,
                        "chunk_type": r.chunk_type,
                        "parent_chunk_id": r.parent_chunk_id,
                        "highlights": r.highlights,
                    }
                )
            else:  # Already a dict
                results_as_dicts.append(r)

        return results_as_dicts

    async def _sparse_first_retrieval(
        self, query: str, filters: Optional[Dict[str, Any]], max_results: int
    ) -> List[Dict[str, Any]]:
        """Sparse (keyword) first, then dense (vector) retrieval"""

        # First: Keyword search
        response1 = await self.search_engine.search(
            query=query,
            filters=filters or {},
            max_results=max_results,
            search_type="keyword",
        )
        keyword_results = (
            response1.results
            if hasattr(response1, "results")
            else response1.get("results", [])
        )

        # If keyword search is good enough, convert and return
        if len(keyword_results) >= max_results // 2:
            results_as_dicts = []
            for r in keyword_results:
                if hasattr(r, "__dataclass_fields__"):
                    results_as_dicts.append(
                        {
                            "chunk_id": r.chunk_id,
                            "content": r.content,
                            "score": r.score,
                            "metadata": r.metadata,
                            "document_id": r.document_id,
                            "chunk_type": r.chunk_type,
                            "parent_chunk_id": r.parent_chunk_id,
                            "highlights": r.highlights,
                        }
                    )
                else:
                    results_as_dicts.append(r)
            return results_as_dicts

        # Second: Vector search to fill gaps
        response2 = await self.search_engine.search(
            query=query,
            filters=filters or {},
            max_results=max_results - len(keyword_results),
            search_type="vector",
        )
        vector_results = (
            response2.results
            if hasattr(response2, "results")
            else response2.get("results", [])
        )

        # Merge and deduplicate
        all_results = keyword_results + vector_results
        deduped = self._deduplicate_results(all_results)[:max_results]

        # Convert SearchResult objects to dictionaries
        results_as_dicts = []
        for r in deduped:
            if hasattr(r, "__dataclass_fields__"):
                results_as_dicts.append(
                    {
                        "chunk_id": r.chunk_id,
                        "content": r.content,
                        "score": r.score,
                        "metadata": r.metadata,
                        "document_id": r.document_id,
                        "chunk_type": r.chunk_type,
                        "parent_chunk_id": r.parent_chunk_id,
                        "highlights": r.highlights,
                    }
                )
            else:
                results_as_dicts.append(r)

        return results_as_dicts

    def _extract_concepts_from_results(
        self, results: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract key concepts from results for query expansion"""
        concepts = set()

        for result in results[:3]:  # Top 3 results
            content = (
                result.content
                if hasattr(result, "content")
                else result.get("content", "")
            )
            # Simple concept extraction (can be enhanced with NER)
            words = content.split()
            # Extract capitalized words (likely concepts)
            for word in words:
                if word[0].isupper() and len(word) > 3:
                    concepts.add(word.lower())

        return list(concepts)[:5]

    def _deduplicate_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate results based on chunk_id"""
        seen_ids = set()
        unique_results = []

        for result in results:
            chunk_id = (
                result.chunk_id
                if hasattr(result, "chunk_id")
                else result.get("chunk_id")
            )
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_results.append(result)

        return unique_results

    async def _assess_quality(
        self,
        query: str,
        results: List[Dict[str, Any]],
        query_analysis: Optional[Dict[str, Any]],
    ) -> RetrievalQuality:
        """Assess quality of retrieval results"""

        if not results:
            return RetrievalQuality(
                relevance_score=0.0,
                coverage_score=0.0,
                diversity_score=0.0,
                confidence_score=0.0,
                overall_quality=0.0,
                needs_retry=True,
                retry_strategy=RetrievalStrategy.VECTOR_ONLY,
                reasoning="No results found",
            )

        # Calculate relevance score (average of top results)
        top_scores = [r.get("score", 0.0) for r in results[:5]]
        relevance_score = sum(top_scores) / len(top_scores) if top_scores else 0.0

        # Calculate coverage score (query term coverage)
        coverage_score = self._calculate_coverage(query, results)

        # Calculate diversity score
        diversity_score = self._calculate_diversity(results)

        # Calculate confidence score
        confidence_score = min(
            1.0, len(results) / 10.0
        )  # More results = higher confidence

        # Overall quality (weighted average)
        overall_quality = (
            0.4 * relevance_score
            + 0.3 * coverage_score
            + 0.2 * diversity_score
            + 0.1 * confidence_score
        )

        # Determine if retry is needed
        needs_retry = overall_quality < self.quality_threshold

        # Suggest retry strategy
        retry_strategy = None
        reasoning = ""

        if needs_retry:
            if relevance_score < 0.3:
                retry_strategy = RetrievalStrategy.VECTOR_ONLY
                reasoning = "Low relevance scores, trying semantic search"
            elif coverage_score < 0.3:
                retry_strategy = RetrievalStrategy.KEYWORD_ONLY
                reasoning = "Poor query coverage, trying keyword search"
            elif diversity_score < 0.3:
                retry_strategy = RetrievalStrategy.MULTI_HOP
                reasoning = "Low diversity, trying multi-hop retrieval"
            else:
                retry_strategy = RetrievalStrategy.HYBRID
                reasoning = "General quality issues, trying hybrid approach"
        else:
            reasoning = f"Quality acceptable: {overall_quality:.2f}"

        return RetrievalQuality(
            relevance_score=relevance_score,
            coverage_score=coverage_score,
            diversity_score=diversity_score,
            confidence_score=confidence_score,
            overall_quality=overall_quality,
            needs_retry=needs_retry,
            retry_strategy=retry_strategy,
            reasoning=reasoning,
        )

    def _calculate_coverage(self, query: str, results: List[Dict[str, Any]]) -> float:
        """Calculate how well results cover query terms"""
        query_terms = set(query.lower().split())

        if not query_terms:
            return 1.0

        covered_terms = set()
        for result in results[:5]:
            content = (
                result.content
                if hasattr(result, "content")
                else result.get("content", "")
            ).lower()
            for term in query_terms:
                if term in content:
                    covered_terms.add(term)

        return len(covered_terms) / len(query_terms)

    def _calculate_diversity(self, results: List[Dict[str, Any]]) -> float:
        """Calculate diversity of results"""
        if len(results) < 2:
            return 1.0

        # Simple diversity: check if results come from different documents
        document_ids = set()
        for result in results:
            doc_id = (
                result.document_id
                if hasattr(result, "document_id")
                else result.get("document_id")
            )
            if doc_id:
                document_ids.add(doc_id)

        # Diversity = unique documents / total results
        return len(document_ids) / len(results)

    async def _self_correct(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        max_results: int,
        initial_quality: RetrievalQuality,
        iterations: int,
        query_analysis: Optional[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], RetrievalStrategy, RetrievalQuality, int]:
        """Self-correction loop for poor retrieval"""

        best_results = []
        best_strategy = RetrievalStrategy.HYBRID
        best_quality = initial_quality

        # Try different strategies
        strategies_to_try = (
            [initial_quality.retry_strategy]
            if initial_quality.retry_strategy
            else [
                RetrievalStrategy.VECTOR_ONLY,
                RetrievalStrategy.KEYWORD_ONLY,
                RetrievalStrategy.MULTI_HOP,
            ]
        )

        for strategy in strategies_to_try:
            if iterations >= self.max_iterations:
                break

            logger.info(
                f"Retry iteration {iterations + 1} with strategy: {strategy.value}"
            )

            # Execute retrieval with new strategy
            results, actual_strategy = await self._execute_retrieval(
                query, filters, max_results, strategy
            )
            iterations += 1

            # Assess quality
            quality = await self._assess_quality(query, results, query_analysis)

            # Keep best results
            if quality.overall_quality > best_quality.overall_quality:
                best_results = results
                best_strategy = actual_strategy
                best_quality = quality

                # If quality is good enough, stop
                if quality.overall_quality >= self.quality_threshold:
                    logger.info(f"Quality threshold met: {quality.overall_quality:.2f}")
                    break

        return best_results, best_strategy, best_quality, iterations

    def _update_strategy_performance(self, strategy: RetrievalStrategy, quality: float):
        """Update strategy performance tracking"""
        perf = self.strategy_performance[strategy]
        perf["total"] += 1

        if quality >= self.quality_threshold:
            perf["success"] += 1

        # Update average quality
        current_avg = perf["avg_quality"]
        total = perf["total"]
        perf["avg_quality"] = (current_avg * (total - 1) + quality) / total

    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get strategy performance statistics"""
        return {
            strategy.value: {
                "success_rate": (
                    perf["success"] / perf["total"] if perf["total"] > 0 else 0.0
                ),
                "avg_quality": perf["avg_quality"],
                "total_uses": perf["total"],
            }
            for strategy, perf in self.strategy_performance.items()
        }

    async def health_check(self) -> Dict[str, Any]:
        """Health check for adaptive retrieval"""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "config": {
                "quality_threshold": self.quality_threshold,
                "max_iterations": self.max_iterations,
                "self_correction_enabled": self.enable_self_correction,
                "multi_hop_enabled": self.enable_multi_hop,
            },
            "strategy_performance": self.get_strategy_performance(),
        }
