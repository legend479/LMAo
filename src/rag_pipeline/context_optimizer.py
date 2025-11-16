"""
Context Optimizer
Intelligent filtering, ranking, and compression of retrieved chunks for optimal LLM generation
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from src.shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizedContext:
    """Result of context optimization"""

    chunks: List[Dict[str, Any]]  # Filtered and ranked chunks
    total_tokens: int  # Estimated token count
    compression_ratio: float  # How much was compressed
    diversity_score: float  # How diverse the selected chunks are
    relevance_score: float  # Average relevance of selected chunks
    metadata: Dict[str, Any]  # Additional metadata


class ContextOptimizer:
    """Optimize retrieved context for LLM generation"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Configuration parameters
        self.max_tokens = self.config.get("max_tokens", 4000)  # Max context tokens
        self.relevance_threshold = self.config.get("relevance_threshold", 0.3)
        self.diversity_weight = self.config.get("diversity_weight", 0.3)
        self.mmr_lambda = self.config.get(
            "mmr_lambda", 0.7
        )  # Balance relevance vs diversity
        self.enable_deduplication = self.config.get("enable_deduplication", True)
        self.enable_compression = self.config.get("enable_compression", True)

        self._initialized = False

    async def initialize(self):
        """Initialize context optimizer"""
        if self._initialized:
            return

        logger.info("Initializing Context Optimizer")
        self._initialized = True
        logger.info("Context Optimizer initialized")

    async def optimize_context(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        max_tokens: Optional[int] = None,
        strategy: str = "mmr",  # "mmr", "relevance", "diversity"
    ) -> OptimizedContext:
        """
        Optimize retrieved chunks for LLM context

        Args:
            chunks: Retrieved chunks with scores
            query: Original query for relevance calculation
            max_tokens: Maximum tokens to include (overrides config)
            strategy: Selection strategy ("mmr", "relevance", "diversity")

        Returns:
            OptimizedContext with filtered and ranked chunks
        """
        if not self._initialized:
            await self.initialize()

        logger.info(
            "Optimizing context",
            input_chunks=len(chunks),
            strategy=strategy,
            max_tokens=max_tokens or self.max_tokens,
        )

        if not chunks:
            return OptimizedContext(
                chunks=[],
                total_tokens=0,
                compression_ratio=0.0,
                diversity_score=0.0,
                relevance_score=0.0,
                metadata={"strategy": strategy, "input_chunks": 0},
            )

        # Step 1: Filter by relevance threshold
        filtered_chunks = self._filter_by_relevance(chunks)

        # Step 2: Deduplicate similar chunks
        if self.enable_deduplication:
            filtered_chunks = self._deduplicate_chunks(filtered_chunks)

        # Step 3: Select chunks based on strategy
        if strategy == "mmr":
            selected_chunks = self._select_with_mmr(filtered_chunks, query, max_tokens)
        elif strategy == "diversity":
            selected_chunks = self._select_by_diversity(filtered_chunks, max_tokens)
        else:  # relevance
            selected_chunks = self._select_by_relevance(filtered_chunks, max_tokens)

        # Step 4: Compress chunks if needed
        if self.enable_compression:
            selected_chunks = self._compress_chunks(selected_chunks, max_tokens)

        # Step 5: Calculate metrics
        total_tokens = self._estimate_tokens(selected_chunks)
        compression_ratio = len(selected_chunks) / len(chunks) if chunks else 0.0
        diversity_score = self._calculate_diversity(selected_chunks)
        relevance_score = self._calculate_average_relevance(selected_chunks)

        logger.info(
            "Context optimized",
            input_chunks=len(chunks),
            output_chunks=len(selected_chunks),
            compression_ratio=f"{compression_ratio:.2%}",
            total_tokens=total_tokens,
            diversity_score=f"{diversity_score:.2f}",
            relevance_score=f"{relevance_score:.2f}",
        )

        return OptimizedContext(
            chunks=selected_chunks,
            total_tokens=total_tokens,
            compression_ratio=compression_ratio,
            diversity_score=diversity_score,
            relevance_score=relevance_score,
            metadata={
                "strategy": strategy,
                "input_chunks": len(chunks),
                "filtered_chunks": len(filtered_chunks),
                "deduplication_enabled": self.enable_deduplication,
                "compression_enabled": self.enable_compression,
            },
        )

    def _filter_by_relevance(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter chunks below relevance threshold"""
        filtered = [
            chunk
            for chunk in chunks
            if chunk.get("score", 0.0) >= self.relevance_threshold
        ]

        logger.debug(
            f"Relevance filtering: {len(chunks)} -> {len(filtered)} chunks "
            f"(threshold: {self.relevance_threshold})"
        )

        return filtered

    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate or highly similar chunks"""
        if not chunks:
            return chunks

        deduplicated = []
        seen_content = set()

        for chunk in chunks:
            content = chunk.get("content", "")

            # Simple deduplication based on content hash
            content_hash = hash(content[:200])  # Use first 200 chars for hash

            if content_hash not in seen_content:
                seen_content.add(content_hash)
                deduplicated.append(chunk)

        logger.debug(
            f"Deduplication: {len(chunks)} -> {len(deduplicated)} chunks "
            f"({len(chunks) - len(deduplicated)} duplicates removed)"
        )

        return deduplicated

    def _select_with_mmr(
        self, chunks: List[Dict[str, Any]], query: str, max_tokens: Optional[int]
    ) -> List[Dict[str, Any]]:
        """
        Select chunks using Maximal Marginal Relevance (MMR)
        Balances relevance and diversity
        """
        max_tokens = max_tokens or self.max_tokens

        if not chunks:
            return []

        selected = []
        remaining = chunks.copy()
        current_tokens = 0

        # Start with the most relevant chunk
        remaining.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        first_chunk = remaining.pop(0)
        selected.append(first_chunk)
        current_tokens += self._estimate_chunk_tokens(first_chunk)

        # Iteratively select chunks that maximize MMR score
        while remaining and current_tokens < max_tokens:
            mmr_scores = []

            for chunk in remaining:
                # Relevance score (already computed)
                relevance = chunk.get("score", 0.0)

                # Diversity score (minimum similarity to selected chunks)
                diversity = self._calculate_chunk_diversity(chunk, selected)

                # MMR score: balance relevance and diversity
                mmr_score = (
                    self.mmr_lambda * relevance + (1 - self.mmr_lambda) * diversity
                )

                mmr_scores.append((chunk, mmr_score))

            # Select chunk with highest MMR score
            if mmr_scores:
                mmr_scores.sort(key=lambda x: x[1], reverse=True)
                next_chunk, _ = mmr_scores[0]

                chunk_tokens = self._estimate_chunk_tokens(next_chunk)
                if current_tokens + chunk_tokens <= max_tokens:
                    selected.append(next_chunk)
                    current_tokens += chunk_tokens
                    remaining.remove(next_chunk)
                else:
                    break  # Can't fit more chunks

        logger.debug(
            f"MMR selection: selected {len(selected)} chunks ({current_tokens} tokens)"
        )

        return selected

    def _select_by_diversity(
        self, chunks: List[Dict[str, Any]], max_tokens: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Select chunks prioritizing diversity"""
        max_tokens = max_tokens or self.max_tokens

        if not chunks:
            return []

        selected = []
        remaining = chunks.copy()
        current_tokens = 0

        # Start with highest scoring chunk
        remaining.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        first_chunk = remaining.pop(0)
        selected.append(first_chunk)
        current_tokens += self._estimate_chunk_tokens(first_chunk)

        # Select most diverse chunks
        while remaining and current_tokens < max_tokens:
            diversity_scores = [
                (chunk, self._calculate_chunk_diversity(chunk, selected))
                for chunk in remaining
            ]

            diversity_scores.sort(key=lambda x: x[1], reverse=True)
            next_chunk, _ = diversity_scores[0]

            chunk_tokens = self._estimate_chunk_tokens(next_chunk)
            if current_tokens + chunk_tokens <= max_tokens:
                selected.append(next_chunk)
                current_tokens += chunk_tokens
                remaining.remove(next_chunk)
            else:
                break

        return selected

    def _select_by_relevance(
        self, chunks: List[Dict[str, Any]], max_tokens: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Select chunks by relevance score only"""
        max_tokens = max_tokens or self.max_tokens

        # Sort by relevance
        sorted_chunks = sorted(chunks, key=lambda x: x.get("score", 0.0), reverse=True)

        selected = []
        current_tokens = 0

        for chunk in sorted_chunks:
            chunk_tokens = self._estimate_chunk_tokens(chunk)
            if current_tokens + chunk_tokens <= max_tokens:
                selected.append(chunk)
                current_tokens += chunk_tokens
            else:
                break

        return selected

    def _compress_chunks(
        self, chunks: List[Dict[str, Any]], max_tokens: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Compress chunks if they exceed token limit"""
        max_tokens = max_tokens or self.max_tokens
        total_tokens = self._estimate_tokens(chunks)

        if total_tokens <= max_tokens:
            return chunks  # No compression needed

        # Calculate compression ratio needed
        compression_ratio = max_tokens / total_tokens

        compressed_chunks = []
        for chunk in chunks:
            content = chunk.get("content", "")

            # Compress content by truncating
            target_length = int(len(content) * compression_ratio)
            compressed_content = content[:target_length]

            # Add ellipsis if truncated
            if len(compressed_content) < len(content):
                compressed_content += "..."

            compressed_chunk = chunk.copy()
            compressed_chunk["content"] = compressed_content
            compressed_chunk["compressed"] = True
            compressed_chunks.append(compressed_chunk)

        logger.debug(
            f"Compressed {len(chunks)} chunks from {total_tokens} to ~{max_tokens} tokens"
        )

        return compressed_chunks

    def _calculate_chunk_diversity(
        self, chunk: Dict[str, Any], selected_chunks: List[Dict[str, Any]]
    ) -> float:
        """Calculate diversity score for a chunk relative to selected chunks"""
        if not selected_chunks:
            return 1.0

        chunk_content = chunk.get("content", "")

        # Calculate minimum similarity to selected chunks
        min_similarity = 1.0

        for selected in selected_chunks:
            selected_content = selected.get("content", "")
            similarity = self._calculate_text_similarity(
                chunk_content, selected_content
            )
            min_similarity = min(min_similarity, similarity)

        # Diversity is inverse of similarity
        diversity = 1.0 - min_similarity

        return diversity

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (Jaccard similarity)"""
        if not text1 or not text2:
            return 0.0

        # Tokenize
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        # Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))

        return intersection / union if union > 0 else 0.0

    def _calculate_diversity(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate overall diversity score for selected chunks"""
        if len(chunks) < 2:
            return 1.0

        similarities = []
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                content1 = chunks[i].get("content", "")
                content2 = chunks[j].get("content", "")
                similarity = self._calculate_text_similarity(content1, content2)
                similarities.append(similarity)

        # Average similarity
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # Diversity is inverse of similarity
        return 1.0 - avg_similarity

    def _calculate_average_relevance(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate average relevance score"""
        if not chunks:
            return 0.0

        scores = [chunk.get("score", 0.0) for chunk in chunks]
        return sum(scores) / len(scores)

    def _estimate_chunk_tokens(self, chunk: Dict[str, Any]) -> int:
        """Estimate token count for a single chunk"""
        content = chunk.get("content", "")
        # Rough estimate: 1 token ≈ 4 characters
        return len(content) // 4

    def _estimate_tokens(self, chunks: List[Dict[str, Any]]) -> int:
        """Estimate total token count for chunks"""
        return sum(self._estimate_chunk_tokens(chunk) for chunk in chunks)

    async def health_check(self) -> Dict[str, Any]:
        """Health check for context optimizer"""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "config": {
                "max_tokens": self.max_tokens,
                "relevance_threshold": self.relevance_threshold,
                "mmr_lambda": self.mmr_lambda,
                "deduplication_enabled": self.enable_deduplication,
                "compression_enabled": self.enable_compression,
            },
        }
