"""
Hybrid Embedding Selection
Intelligent embedding selection and fusion for optimal semantic search
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from src.shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QueryComposition:
    """Analysis of query composition"""

    code_ratio: float  # 0-1, how much is code-related
    concept_ratio: float  # 0-1, how much is conceptual
    entity_ratio: float  # 0-1, how much is entity-focused
    complexity: float  # 0-1, overall complexity
    recommended_strategy: str  # "general", "domain", "hybrid", "ensemble"


@dataclass
class EmbeddingWeights:
    """Weights for embedding fusion"""

    general_weight: float  # Weight for general embedding
    domain_weight: float  # Weight for domain-specific embedding
    strategy: str  # "general_only", "domain_only", "weighted_fusion", "ensemble"
    reasoning: str  # Explanation of weight selection


class HybridEmbeddingSelector:
    """Intelligent embedding selection and fusion"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Configuration
        self.enable_fusion = self.config.get("enable_fusion", True)
        self.enable_ensemble = self.config.get("enable_ensemble", True)
        self.fusion_threshold = self.config.get("fusion_threshold", 0.3)

        # Code indicators
        self.code_keywords = [
            "function",
            "method",
            "class",
            "variable",
            "algorithm",
            "implementation",
            "code",
            "syntax",
            "compile",
            "debug",
            "def",
            "return",
            "import",
            "for",
            "while",
            "if",
        ]

        # Concept indicators
        self.concept_keywords = [
            "concept",
            "theory",
            "principle",
            "pattern",
            "architecture",
            "design",
            "approach",
            "methodology",
            "paradigm",
            "philosophy",
        ]

        # Entity indicators (proper nouns, frameworks, languages)
        self.entity_keywords = [
            "python",
            "java",
            "javascript",
            "react",
            "django",
            "aws",
            "docker",
            "kubernetes",
            "git",
            "sql",
        ]

        self._initialized = False

    async def initialize(self):
        """Initialize hybrid embedding selector"""
        if self._initialized:
            return

        logger.info("Initializing Hybrid Embedding Selector")
        self._initialized = True
        logger.info("Hybrid Embedding Selector initialized")

    async def analyze_query_composition(
        self, query: str, query_analysis: Optional[Dict[str, Any]] = None
    ) -> QueryComposition:
        """
        Analyze query composition to determine embedding strategy

        Args:
            query: Search query
            query_analysis: Optional pre-computed query analysis

        Returns:
            QueryComposition with ratios and recommendation
        """
        query_lower = query.lower()
        words = query_lower.split()

        # Calculate code ratio
        code_matches = sum(
            1 for keyword in self.code_keywords if keyword in query_lower
        )
        code_ratio = min(1.0, code_matches / 3.0)  # Normalize to 0-1

        # Calculate concept ratio
        concept_matches = sum(
            1 for keyword in self.concept_keywords if keyword in query_lower
        )
        concept_ratio = min(1.0, concept_matches / 3.0)

        # Calculate entity ratio
        entity_matches = sum(
            1 for keyword in self.entity_keywords if keyword in query_lower
        )
        entity_ratio = min(1.0, entity_matches / 2.0)

        # Use query analysis if available
        if query_analysis:
            complexity = query_analysis.get("complexity_score", 0.5)
        else:
            # Simple complexity estimation
            complexity = min(1.0, len(words) / 20.0)

        # Determine recommended strategy
        if code_ratio > 0.6:
            recommended_strategy = "domain"  # Use domain-specific (code) embeddings
        elif concept_ratio > 0.6:
            recommended_strategy = "general"  # Use general embeddings
        elif code_ratio > 0.3 and concept_ratio > 0.3:
            recommended_strategy = "hybrid"  # Use weighted fusion
        elif complexity > 0.7:
            recommended_strategy = "ensemble"  # Use ensemble for complex queries
        else:
            recommended_strategy = "general"  # Default to general

        logger.debug(
            "Query composition analyzed",
            code_ratio=f"{code_ratio:.2f}",
            concept_ratio=f"{concept_ratio:.2f}",
            entity_ratio=f"{entity_ratio:.2f}",
            strategy=recommended_strategy,
        )

        return QueryComposition(
            code_ratio=code_ratio,
            concept_ratio=concept_ratio,
            entity_ratio=entity_ratio,
            complexity=complexity,
            recommended_strategy=recommended_strategy,
        )

    async def select_embedding_weights(
        self, query: str, query_composition: Optional[QueryComposition] = None
    ) -> EmbeddingWeights:
        """
        Select optimal embedding weights based on query composition

        Args:
            query: Search query
            query_composition: Optional pre-computed composition

        Returns:
            EmbeddingWeights with strategy and reasoning
        """
        if not self._initialized:
            await self.initialize()

        # Analyze composition if not provided
        if not query_composition:
            query_composition = await self.analyze_query_composition(query)

        strategy = query_composition.recommended_strategy

        if strategy == "domain":
            # Pure domain-specific embeddings
            return EmbeddingWeights(
                general_weight=0.0,
                domain_weight=1.0,
                strategy="domain_only",
                reasoning="Query is code-focused, using domain-specific embeddings",
            )

        elif strategy == "general":
            # Pure general embeddings
            return EmbeddingWeights(
                general_weight=1.0,
                domain_weight=0.0,
                strategy="general_only",
                reasoning="Query is concept-focused, using general embeddings",
            )

        elif strategy == "hybrid":
            # Weighted fusion based on ratios
            code_ratio = query_composition.code_ratio
            concept_ratio = query_composition.concept_ratio

            # Normalize weights
            total = code_ratio + concept_ratio
            if total > 0:
                domain_weight = code_ratio / total
                general_weight = concept_ratio / total
            else:
                domain_weight = 0.5
                general_weight = 0.5

            return EmbeddingWeights(
                general_weight=general_weight,
                domain_weight=domain_weight,
                strategy="weighted_fusion",
                reasoning=f"Hybrid query: {general_weight:.1%} general, {domain_weight:.1%} domain",
            )

        else:  # ensemble
            # Use both embeddings separately and combine results
            return EmbeddingWeights(
                general_weight=0.5,
                domain_weight=0.5,
                strategy="ensemble",
                reasoning="Complex query, using ensemble approach",
            )

    async def fuse_embeddings(
        self,
        general_embedding: np.ndarray,
        domain_embedding: np.ndarray,
        weights: EmbeddingWeights,
    ) -> np.ndarray:
        """
        Fuse embeddings based on weights

        Args:
            general_embedding: General purpose embedding
            domain_embedding: Domain-specific embedding
            weights: Embedding weights

        Returns:
            Fused embedding vector
        """
        if weights.strategy == "general_only":
            return general_embedding

        elif weights.strategy == "domain_only":
            return domain_embedding

        elif weights.strategy == "weighted_fusion":
            # Weighted average
            fused = (
                weights.general_weight * general_embedding
                + weights.domain_weight * domain_embedding
            )

            # Normalize
            norm = np.linalg.norm(fused)
            if norm > 0:
                fused = fused / norm

            return fused

        else:  # ensemble
            # For ensemble, return both (handled differently in search)
            # Here we return weighted average as fallback
            fused = 0.5 * general_embedding + 0.5 * domain_embedding
            norm = np.linalg.norm(fused)
            if norm > 0:
                fused = fused / norm
            return fused

    async def generate_hybrid_embedding(
        self,
        query: str,
        embedding_manager,
        query_composition: Optional[QueryComposition] = None,
    ) -> Tuple[np.ndarray, EmbeddingWeights]:
        """
        Generate optimal embedding for query

        Args:
            query: Search query
            embedding_manager: Embedding manager instance
            query_composition: Optional pre-computed composition

        Returns:
            Tuple of (embedding vector, weights used)
        """
        # Select weights
        weights = await self.select_embedding_weights(query, query_composition)

        # Generate embeddings based on strategy
        if weights.strategy == "general_only":
            # Only generate general embedding
            result = await embedding_manager.generate_embeddings(
                query, include_general=True, include_domain=False
            )
            embedding = result.general_embedding

        elif weights.strategy == "domain_only":
            # Only generate domain embedding
            result = await embedding_manager.generate_embeddings(
                query, include_general=False, include_domain=True
            )
            embedding = result.domain_embedding

        else:  # weighted_fusion or ensemble
            # Generate both embeddings
            result = await embedding_manager.generate_embeddings(
                query, include_general=True, include_domain=True
            )

            # Fuse embeddings
            embedding = await self.fuse_embeddings(
                result.general_embedding, result.domain_embedding, weights
            )

        logger.debug(
            "Generated hybrid embedding",
            strategy=weights.strategy,
            reasoning=weights.reasoning,
        )

        return embedding, weights

    async def ensemble_search(
        self,
        query: str,
        embedding_manager,
        vector_store,
        filters: Optional[Dict[str, Any]],
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """
        Perform ensemble search using both embeddings

        Args:
            query: Search query
            embedding_manager: Embedding manager
            vector_store: Vector store for search
            filters: Optional filters
            max_results: Maximum results

        Returns:
            Combined and re-ranked results
        """
        logger.info("Performing ensemble search")

        # Generate both embeddings
        result = await embedding_manager.generate_embeddings(
            query, include_general=True, include_domain=True
        )

        # Search with general embedding
        general_results = await self._search_with_embedding(
            result.general_embedding, vector_store, filters, max_results, "general"
        )

        # Search with domain embedding
        domain_results = await self._search_with_embedding(
            result.domain_embedding, vector_store, filters, max_results, "domain"
        )

        # Combine results with RRF (Reciprocal Rank Fusion)
        combined_results = self._combine_with_rrf(
            general_results, domain_results, max_results
        )

        logger.info(
            f"Ensemble search: {len(general_results)} + {len(domain_results)} -> {len(combined_results)} results"
        )

        return combined_results

    async def _search_with_embedding(
        self,
        embedding: np.ndarray,
        vector_store,
        filters: Optional[Dict[str, Any]],
        max_results: int,
        embedding_type: str,
    ) -> List[Dict[str, Any]]:
        """Search with a specific embedding"""
        # This would call vector store's search method
        # Placeholder implementation
        logger.debug(f"Searching with {embedding_type} embedding")
        return []

    def _combine_with_rrf(
        self,
        results1: List[Dict[str, Any]],
        results2: List[Dict[str, Any]],
        max_results: int,
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion

        Args:
            results1: First set of results
            results2: Second set of results
            max_results: Maximum results to return
            k: RRF parameter

        Returns:
            Combined and re-ranked results
        """
        # Collect all unique results
        all_results = {}  # chunk_id -> {result, rrf_score}

        # Process first result set
        for rank, result in enumerate(results1, 1):
            chunk_id = (
                result.chunk_id
                if hasattr(result, "chunk_id")
                else result.get("chunk_id")
            )
            if chunk_id:
                if chunk_id not in all_results:
                    all_results[chunk_id] = {"result": result, "rrf_score": 0.0}
                all_results[chunk_id]["rrf_score"] += 1.0 / (k + rank)

        # Process second result set
        for rank, result in enumerate(results2, 1):
            chunk_id = (
                result.chunk_id
                if hasattr(result, "chunk_id")
                else result.get("chunk_id")
            )
            if chunk_id:
                if chunk_id not in all_results:
                    all_results[chunk_id] = {"result": result, "rrf_score": 0.0}
                all_results[chunk_id]["rrf_score"] += 1.0 / (k + rank)

        # Sort by RRF score
        sorted_results = sorted(
            all_results.values(), key=lambda x: x["rrf_score"], reverse=True
        )

        # Extract results and update scores
        final_results = []
        for item in sorted_results[:max_results]:
            result = item["result"]
            result["rrf_score"] = item["rrf_score"]
            result["score"] = item["rrf_score"]  # Update score
            final_results.append(result)

        return final_results

    async def health_check(self) -> Dict[str, Any]:
        """Health check for hybrid embedding selector"""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "config": {
                "fusion_enabled": self.enable_fusion,
                "ensemble_enabled": self.enable_ensemble,
                "fusion_threshold": self.fusion_threshold,
            },
            "strategies": {
                "general_only": "Pure general embeddings",
                "domain_only": "Pure domain embeddings",
                "weighted_fusion": "Weighted combination",
                "ensemble": "Separate search + RRF fusion",
            },
        }
