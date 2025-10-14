"""
BGE Reranker
Cross-encoder reranking for improved search relevance
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from sentence_transformers import CrossEncoder
import torch
import numpy as np

from .vector_store import SearchResult
from ..shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RerankerConfig:
    """Configuration for BGE reranker"""

    model_name: str = "BAAI/bge-reranker-base"
    device: str = "auto"  # auto, cpu, cuda
    batch_size: int = 16
    max_length: int = 512
    enable_caching: bool = True
    score_threshold: float = 0.0  # Minimum score to keep results

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RerankedResult:
    """Reranked search result"""

    original_result: SearchResult
    rerank_score: float
    original_rank: int
    new_rank: int
    score_improvement: float  # Difference from original score


class BGEReranker:
    """BGE-based cross-encoder reranker for search results"""

    def __init__(self, config: RerankerConfig = None):
        self.config = config or RerankerConfig()
        self.model: Optional[CrossEncoder] = None
        self._initialized = False

        # Caching for query-document pairs
        self.score_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Performance stats
        self.reranking_stats = {
            "total_reranking_requests": 0,
            "total_documents_reranked": 0,
            "avg_reranking_time": 0.0,
            "avg_score_improvement": 0.0,
            "cache_hit_rate": 0.0,
        }

    async def initialize(self):
        """Initialize BGE reranker model"""
        if self._initialized:
            return

        logger.info("Initializing BGE Reranker")

        try:
            # Load BGE reranker model
            logger.info(f"Loading reranker model: {self.config.model_name}")

            self.model = CrossEncoder(
                self.config.model_name,
                max_length=self.config.max_length,
                device=self.config.device,
            )

            logger.info(f"BGE Reranker loaded on device: {self.config.device}")

            self._initialized = True
            logger.info("BGE Reranker initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize BGE Reranker: {e}")
            # Fall back to basic similarity reranking
            logger.warning("Falling back to basic similarity reranking")
            self.model = None
            self._initialized = True

    async def rerank(
        self, query: str, results: List[SearchResult], top_k: Optional[int] = None
    ) -> List[RerankedResult]:
        """Rerank search results using BGE cross-encoder"""
        if not self._initialized:
            await self.initialize()

        if not results:
            return []

        start_time = datetime.utcnow()

        logger.info(f"Reranking {len(results)} results for query: {query[:50]}...")

        try:
            if self.model is not None:
                reranked_results = await self._rerank_with_bge(query, results, top_k)
            else:
                # Fallback to basic reranking
                reranked_results = await self._rerank_basic(query, results, top_k)

            # Update statistics
            reranking_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_stats(len(results), reranking_time, reranked_results)

            logger.info(
                f"Reranking completed in {reranking_time:.3f}s. "
                f"Returned {len(reranked_results)} results"
            )

            return reranked_results

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original results as fallback
            return [
                RerankedResult(
                    original_result=result,
                    rerank_score=result.score,
                    original_rank=i,
                    new_rank=i,
                    score_improvement=0.0,
                )
                for i, result in enumerate(results)
            ]

    async def _rerank_with_bge(
        self, query: str, results: List[SearchResult], top_k: Optional[int]
    ) -> List[RerankedResult]:
        """Rerank using BGE cross-encoder model"""

        # Prepare query-document pairs
        query_doc_pairs = []
        cached_scores = {}
        uncached_indices = []

        for i, result in enumerate(results):
            cache_key = self._get_cache_key(query, result.content)

            if self.config.enable_caching and cache_key in self.score_cache:
                cached_scores[i] = self.score_cache[cache_key]
                self.cache_hits += 1
            else:
                query_doc_pairs.append([query, result.content])
                uncached_indices.append(i)
                self.cache_misses += 1

        # Get scores for uncached pairs
        uncached_scores = []
        if query_doc_pairs:
            uncached_scores = await self._get_batch_scores(query_doc_pairs)

            # Cache the scores
            if self.config.enable_caching:
                for idx, score in zip(uncached_indices, uncached_scores):
                    cache_key = self._get_cache_key(query, results[idx].content)
                    self.score_cache[cache_key] = score

        # Combine cached and uncached scores
        all_scores = {}
        all_scores.update(cached_scores)

        for idx, score in zip(uncached_indices, uncached_scores):
            all_scores[idx] = score

        # Create reranked results
        reranked_results = []
        for i, result in enumerate(results):
            rerank_score = all_scores[i]

            # Skip results below threshold
            if rerank_score < self.config.score_threshold:
                continue

            reranked_result = RerankedResult(
                original_result=result,
                rerank_score=rerank_score,
                original_rank=i,
                new_rank=0,  # Will be set after sorting
                score_improvement=rerank_score - result.score,
            )
            reranked_results.append(reranked_result)

        # Sort by rerank score (descending)
        reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)

        # Update new ranks and apply top_k limit
        if top_k:
            reranked_results = reranked_results[:top_k]

        for i, result in enumerate(reranked_results):
            result.new_rank = i

        return reranked_results

    async def _rerank_basic(
        self, query: str, results: List[SearchResult], top_k: Optional[int]
    ) -> List[RerankedResult]:
        """Basic reranking fallback using simple text matching"""

        logger.info("Using basic reranking fallback")

        query_terms = set(query.lower().split())

        reranked_results = []
        for i, result in enumerate(results):
            # Simple term overlap scoring
            content_terms = set(result.content.lower().split())
            overlap = len(query_terms.intersection(content_terms))
            total_terms = len(query_terms.union(content_terms))

            # Jaccard similarity as rerank score
            jaccard_score = overlap / total_terms if total_terms > 0 else 0.0

            # Combine with original score
            combined_score = 0.7 * result.score + 0.3 * jaccard_score

            reranked_result = RerankedResult(
                original_result=result,
                rerank_score=combined_score,
                original_rank=i,
                new_rank=0,  # Will be set after sorting
                score_improvement=combined_score - result.score,
            )
            reranked_results.append(reranked_result)

        # Sort by combined score
        reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)

        # Apply top_k limit and update ranks
        if top_k:
            reranked_results = reranked_results[:top_k]

        for i, result in enumerate(reranked_results):
            result.new_rank = i

        return reranked_results

    async def _get_batch_scores(self, query_doc_pairs: List[List[str]]) -> List[float]:
        """Get reranking scores for batch of query-document pairs"""

        if not query_doc_pairs:
            return []

        try:
            # Process in batches to avoid memory issues
            all_scores = []
            batch_size = self.config.batch_size

            for i in range(0, len(query_doc_pairs), batch_size):
                batch_pairs = query_doc_pairs[i : i + batch_size]

                # Get scores from cross-encoder
                batch_scores = self.model.predict(batch_pairs)

                # Convert to list if numpy array
                if hasattr(batch_scores, "tolist"):
                    batch_scores = batch_scores.tolist()
                elif not isinstance(batch_scores, list):
                    batch_scores = [float(batch_scores)]

                all_scores.extend(batch_scores)

            return all_scores

        except Exception as e:
            logger.error(f"Failed to get batch scores: {e}")
            # Return neutral scores as fallback
            return [0.5] * len(query_doc_pairs)

    def _get_cache_key(self, query: str, document: str) -> str:
        """Generate cache key for query-document pair"""
        import hashlib

        key_data = f"{query}|||{document[:500]}"  # Use first 500 chars of document
        return hashlib.md5(key_data.encode()).hexdigest()

    def _update_stats(
        self,
        num_results: int,
        reranking_time: float,
        reranked_results: List[RerankedResult],
    ):
        """Update reranking statistics"""
        self.reranking_stats["total_reranking_requests"] += 1
        self.reranking_stats["total_documents_reranked"] += num_results

        # Update average reranking time
        total_requests = self.reranking_stats["total_reranking_requests"]
        current_avg = self.reranking_stats["avg_reranking_time"]
        self.reranking_stats["avg_reranking_time"] = (
            current_avg * (total_requests - 1) + reranking_time
        ) / total_requests

        # Update average score improvement
        if reranked_results:
            avg_improvement = sum(r.score_improvement for r in reranked_results) / len(
                reranked_results
            )
            current_avg_improvement = self.reranking_stats["avg_score_improvement"]
            self.reranking_stats["avg_score_improvement"] = (
                current_avg_improvement * (total_requests - 1) + avg_improvement
            ) / total_requests

        # Update cache hit rate
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests > 0:
            self.reranking_stats["cache_hit_rate"] = (
                self.cache_hits / total_cache_requests
            )

    async def evaluate_reranking_quality(
        self,
        query: str,
        original_results: List[SearchResult],
        ground_truth_relevant: List[str],
    ) -> Dict[str, float]:
        """Evaluate reranking quality against ground truth"""

        # Get reranked results
        reranked_results = await self.rerank(query, original_results)

        # Extract document IDs or content for comparison
        original_order = [r.chunk_id for r in original_results]
        reranked_order = [r.original_result.chunk_id for r in reranked_results]

        # Calculate metrics
        metrics = {}

        # NDCG calculation
        def calculate_dcg(relevance_scores: List[float]) -> float:
            dcg = 0.0
            for i, score in enumerate(relevance_scores):
                dcg += score / np.log2(i + 2)  # i+2 because log2(1) = 0
            return dcg

        # Create relevance scores (1 if relevant, 0 if not)
        original_relevance = [
            1.0 if chunk_id in ground_truth_relevant else 0.0
            for chunk_id in original_order
        ]
        reranked_relevance = [
            1.0 if r.original_result.chunk_id in ground_truth_relevant else 0.0
            for r in reranked_results
        ]

        # Calculate NDCG
        if ground_truth_relevant:
            ideal_relevance = sorted(original_relevance, reverse=True)

            original_dcg = calculate_dcg(original_relevance)
            reranked_dcg = calculate_dcg(reranked_relevance)
            ideal_dcg = calculate_dcg(ideal_relevance)

            metrics["original_ndcg"] = (
                original_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            )
            metrics["reranked_ndcg"] = (
                reranked_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            )
            metrics["ndcg_improvement"] = (
                metrics["reranked_ndcg"] - metrics["original_ndcg"]
            )

        # Calculate precision at different k values
        for k in [1, 3, 5, 10]:
            if len(reranked_results) >= k:
                relevant_in_top_k = sum(
                    1
                    for r in reranked_results[:k]
                    if r.original_result.chunk_id in ground_truth_relevant
                )
                metrics[f"precision_at_{k}"] = relevant_in_top_k / k

        # Calculate rank correlation (Spearman)
        try:
            from scipy.stats import spearmanr

            original_ranks = list(range(len(original_results)))
            reranked_ranks = [
                original_results.index(r.original_result) for r in reranked_results
            ]
            correlation, p_value = spearmanr(original_ranks, reranked_ranks)
            metrics["rank_correlation"] = correlation
        except ImportError:
            logger.warning("scipy not available for rank correlation calculation")

        return metrics

    async def get_stats(self) -> Dict[str, Any]:
        """Get reranking statistics"""
        return {
            **self.reranking_stats,
            "cache_size": len(self.score_cache),
            "model_loaded": self.model is not None,
            "config": {
                "model_name": self.config.model_name,
                "device": self.config.device,
                "batch_size": self.config.batch_size,
                "score_threshold": self.config.score_threshold,
            },
        }

    async def clear_cache(self):
        """Clear reranking cache"""
        self.score_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.reranking_stats["cache_hit_rate"] = 0.0
        logger.info("Reranking cache cleared")

    async def health_check(self) -> Dict[str, Any]:
        """Check health of reranker"""
        try:
            if not self._initialized:
                return {"status": "not_initialized"}

            # Test reranking with dummy data
            dummy_results = [
                SearchResult(
                    chunk_id="test_1",
                    content="This is a test document about software engineering",
                    score=0.8,
                    metadata={},
                    document_id="doc_1",
                    chunk_type="text",
                ),
                SearchResult(
                    chunk_id="test_2",
                    content="Another test document about programming",
                    score=0.6,
                    metadata={},
                    document_id="doc_2",
                    chunk_type="text",
                ),
            ]

            reranked = await self.rerank(
                "software engineering test", dummy_results, top_k=2
            )

            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "model_name": self.config.model_name,
                "device": self.config.device,
                "test_reranking_successful": len(reranked) == 2,
                "cache_stats": {
                    "cache_size": len(self.score_cache),
                    "cache_hit_rate": self.reranking_stats["cache_hit_rate"],
                },
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def shutdown(self):
        """Shutdown reranker"""
        logger.info("Shutting down BGE Reranker")

        # Clear model from memory
        self.model = None

        logger.info("BGE Reranker shutdown complete")
