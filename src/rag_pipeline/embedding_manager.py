"""
Embedding Manager
Dual embedding system with general-purpose and domain-specific models
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
import pickle
import os

from sentence_transformers import SentenceTransformer
import torch

from ..shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""

    general_model_name: str = "all-mpnet-base-v2"
    domain_model_name: str = "microsoft/graphcodebert-base"
    cache_embeddings: bool = True
    cache_dir: str = "data/embeddings"
    batch_size: int = 32
    max_sequence_length: int = 512
    device: str = "auto"  # auto, cpu, cuda
    normalize_embeddings: bool = True

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""

    general_embedding: np.ndarray
    domain_embedding: np.ndarray
    text: str
    model_versions: Dict[str, str]
    generation_time: float
    cached: bool = False


@dataclass
class ComparisonResult:
    """Result of model performance comparison"""

    query: str
    general_model_results: List[Tuple[str, float]]  # (text, similarity_score)
    domain_model_results: List[Tuple[str, float]]
    ground_truth: List[str]
    general_model_metrics: Dict[str, float]
    domain_model_metrics: Dict[str, float]
    recommendation: str  # Which model performed better


class EmbeddingManager:
    """Manager for dual embedding system with caching and comparison"""

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.general_model: Optional[SentenceTransformer] = None
        self.domain_model: Optional[SentenceTransformer] = None
        self._initialized = False

        # Model metadata
        self.model_info = {
            "general": {
                "name": self.config.general_model_name,
                "version": None,
                "dimensions": None,
            },
            "domain": {
                "name": self.config.domain_model_name,
                "version": None,
                "dimensions": None,
            },
        }

        # Embedding cache
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Performance tracking
        self.embedding_stats = {
            "total_embeddings_generated": 0,
            "general_model_usage": 0,
            "domain_model_usage": 0,
            "avg_generation_time": 0.0,
            "cache_hit_rate": 0.0,
        }

    async def initialize(self):
        """Initialize embedding models"""
        if self._initialized:
            return

        logger.info("Initializing Embedding Manager")

        try:
            # Load general-purpose model
            logger.info(f"Loading general model: {self.config.general_model_name}")
            self.general_model = SentenceTransformer(
                self.config.general_model_name, device=self.config.device
            )

            # Set max sequence length
            self.general_model.max_seq_length = self.config.max_sequence_length

            # Get model info
            self.model_info["general"][
                "dimensions"
            ] = self.general_model.get_sentence_embedding_dimension()
            self.model_info["general"]["version"] = getattr(
                self.general_model, "version", "unknown"
            )

            logger.info(
                f"General model loaded: {self.model_info['general']['dimensions']} dimensions"
            )

            # Load domain-specific model (GraphCodeBERT)
            logger.info(f"Loading domain model: {self.config.domain_model_name}")
            try:
                self.domain_model = SentenceTransformer(
                    self.config.domain_model_name, device=self.config.device
                )
                self.domain_model.max_seq_length = self.config.max_sequence_length

                self.model_info["domain"][
                    "dimensions"
                ] = self.domain_model.get_sentence_embedding_dimension()
                self.model_info["domain"]["version"] = getattr(
                    self.domain_model, "version", "unknown"
                )

                logger.info(
                    f"Domain model loaded: {self.model_info['domain']['dimensions']} dimensions"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to load domain model {self.config.domain_model_name}: {e}"
                )
                logger.info("Falling back to general model for domain embeddings")
                self.domain_model = self.general_model
                self.model_info["domain"] = self.model_info["general"].copy()

            # Load embedding cache if it exists
            await self._load_embedding_cache()

            self._initialized = True
            logger.info("Embedding Manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Embedding Manager: {e}")
            raise

    async def generate_embeddings(
        self, text: str, include_general: bool = True, include_domain: bool = True
    ) -> EmbeddingResult:
        """Generate embeddings using both models"""
        if not self._initialized:
            await self.initialize()

        start_time = datetime.utcnow()

        # Check cache first
        cache_key = self._get_cache_key(text, include_general, include_domain)
        if self.config.cache_embeddings and cache_key in self.embedding_cache:
            self.cache_hits += 1
            cached_result = self.embedding_cache[cache_key]
            cached_result.cached = True
            return cached_result

        self.cache_misses += 1

        # Generate embeddings
        general_embedding = None
        domain_embedding = None

        if include_general:
            general_embedding = await self._generate_single_embedding(
                text, self.general_model, "general"
            )
            self.embedding_stats["general_model_usage"] += 1

        if include_domain and self.domain_model != self.general_model:
            domain_embedding = await self._generate_single_embedding(
                text, self.domain_model, "domain"
            )
            self.embedding_stats["domain_model_usage"] += 1
        elif include_domain:
            # Use general model embedding for domain if they're the same
            domain_embedding = general_embedding
            self.embedding_stats["domain_model_usage"] += 1

        # Create result
        generation_time = (datetime.utcnow() - start_time).total_seconds()

        result = EmbeddingResult(
            general_embedding=general_embedding,
            domain_embedding=domain_embedding,
            text=text,
            model_versions={
                "general": self.model_info["general"]["version"],
                "domain": self.model_info["domain"]["version"],
            },
            generation_time=generation_time,
            cached=False,
        )

        # Cache result
        if self.config.cache_embeddings:
            self.embedding_cache[cache_key] = result

            # Periodically save cache
            if len(self.embedding_cache) % 100 == 0:
                await self._save_embedding_cache()

        # Update stats
        self.embedding_stats["total_embeddings_generated"] += 1
        total_time = self.embedding_stats["avg_generation_time"] * (
            self.embedding_stats["total_embeddings_generated"] - 1
        )
        self.embedding_stats["avg_generation_time"] = (
            total_time + generation_time
        ) / self.embedding_stats["total_embeddings_generated"]
        self.embedding_stats["cache_hit_rate"] = self.cache_hits / (
            self.cache_hits + self.cache_misses
        )

        return result

    async def generate_batch_embeddings(
        self,
        texts: List[str],
        include_general: bool = True,
        include_domain: bool = True,
    ) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts efficiently"""
        if not self._initialized:
            await self.initialize()

        logger.info(f"Generating batch embeddings for {len(texts)} texts")

        results = []
        batch_size = self.config.batch_size

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Check cache for each text
            batch_results = []
            uncached_texts = []
            uncached_indices = []

            for j, text in enumerate(batch_texts):
                cache_key = self._get_cache_key(text, include_general, include_domain)
                if self.config.cache_embeddings and cache_key in self.embedding_cache:
                    cached_result = self.embedding_cache[cache_key]
                    cached_result.cached = True
                    batch_results.append(cached_result)
                    self.cache_hits += 1
                else:
                    batch_results.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(j)
                    self.cache_misses += 1

            # Generate embeddings for uncached texts
            if uncached_texts:
                start_time = datetime.utcnow()

                general_embeddings = None
                domain_embeddings = None

                if include_general:
                    general_embeddings = await self._generate_batch_single_embedding(
                        uncached_texts, self.general_model, "general"
                    )

                if include_domain and self.domain_model != self.general_model:
                    domain_embeddings = await self._generate_batch_single_embedding(
                        uncached_texts, self.domain_model, "domain"
                    )
                elif include_domain:
                    domain_embeddings = general_embeddings

                generation_time = (datetime.utcnow() - start_time).total_seconds()

                # Create results for uncached texts
                for k, text_idx in enumerate(uncached_indices):
                    text = uncached_texts[k]

                    result = EmbeddingResult(
                        general_embedding=(
                            general_embeddings[k]
                            if general_embeddings is not None
                            else None
                        ),
                        domain_embedding=(
                            domain_embeddings[k]
                            if domain_embeddings is not None
                            else None
                        ),
                        text=text,
                        model_versions={
                            "general": self.model_info["general"]["version"],
                            "domain": self.model_info["domain"]["version"],
                        },
                        generation_time=generation_time
                        / len(uncached_texts),  # Average per text
                        cached=False,
                    )

                    batch_results[text_idx] = result

                    # Cache result
                    if self.config.cache_embeddings:
                        cache_key = self._get_cache_key(
                            text, include_general, include_domain
                        )
                        self.embedding_cache[cache_key] = result

            results.extend(batch_results)

        # Update stats
        self.embedding_stats["total_embeddings_generated"] += len(texts)
        if include_general:
            self.embedding_stats["general_model_usage"] += len(texts)
        if include_domain:
            self.embedding_stats["domain_model_usage"] += len(texts)

        self.embedding_stats["cache_hit_rate"] = self.cache_hits / (
            self.cache_hits + self.cache_misses
        )

        # Save cache periodically
        if self.config.cache_embeddings and len(self.embedding_cache) % 100 == 0:
            await self._save_embedding_cache()

        logger.info(
            f"Batch embedding generation completed. Cache hit rate: {self.embedding_stats['cache_hit_rate']:.2%}"
        )

        return results

    async def _generate_single_embedding(
        self, text: str, model: SentenceTransformer, model_type: str
    ) -> np.ndarray:
        """Generate embedding using a single model"""
        try:
            # Truncate text if too long
            if (
                len(text) > self.config.max_sequence_length * 4
            ):  # Rough character estimate
                text = text[: self.config.max_sequence_length * 4]

            # Generate embedding
            embedding = model.encode(
                text,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=True,
            )

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate {model_type} embedding: {e}")
            # Return zero vector as fallback
            dimensions = self.model_info[model_type]["dimensions"]
            return np.zeros(dimensions, dtype=np.float32)

    async def _generate_batch_single_embedding(
        self, texts: List[str], model: SentenceTransformer, model_type: str
    ) -> List[np.ndarray]:
        """Generate embeddings for multiple texts using a single model"""
        try:
            # Truncate texts if too long
            processed_texts = []
            for text in texts:
                if len(text) > self.config.max_sequence_length * 4:
                    text = text[: self.config.max_sequence_length * 4]
                processed_texts.append(text)

            # Generate embeddings
            embeddings = model.encode(
                processed_texts,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=True,
                batch_size=self.config.batch_size,
            )

            return [emb for emb in embeddings]

        except Exception as e:
            logger.error(f"Failed to generate batch {model_type} embeddings: {e}")
            # Return zero vectors as fallback
            dimensions = self.model_info[model_type]["dimensions"]
            return [np.zeros(dimensions, dtype=np.float32) for _ in texts]

    def calculate_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray, metric: str = "cosine"
    ) -> float:
        """Calculate similarity between two embeddings"""
        if metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        elif metric == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(embedding1 - embedding2)
            return 1.0 / (1.0 + distance)

        elif metric == "dot_product":
            # Dot product similarity
            return np.dot(embedding1, embedding2)

        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")

    async def compare_model_performance(
        self,
        query: str,
        candidate_texts: List[str],
        ground_truth: List[str],
        top_k: int = 10,
    ) -> ComparisonResult:
        """Compare performance of general vs domain-specific models"""
        if not self._initialized:
            await self.initialize()

        logger.info(f"Comparing model performance for query: {query[:50]}...")

        # Generate query embeddings
        query_result = await self.generate_embeddings(query)

        # Generate embeddings for candidate texts
        candidate_results = await self.generate_batch_embeddings(candidate_texts)

        # Calculate similarities for both models
        general_similarities = []
        domain_similarities = []

        for i, candidate_result in enumerate(candidate_results):
            # General model similarity
            if (
                query_result.general_embedding is not None
                and candidate_result.general_embedding is not None
            ):
                general_sim = self.calculate_similarity(
                    query_result.general_embedding, candidate_result.general_embedding
                )
                general_similarities.append((candidate_texts[i], general_sim))

            # Domain model similarity
            if (
                query_result.domain_embedding is not None
                and candidate_result.domain_embedding is not None
            ):
                domain_sim = self.calculate_similarity(
                    query_result.domain_embedding, candidate_result.domain_embedding
                )
                domain_similarities.append((candidate_texts[i], domain_sim))

        # Sort by similarity (descending)
        general_similarities.sort(key=lambda x: x[1], reverse=True)
        domain_similarities.sort(key=lambda x: x[1], reverse=True)

        # Take top-k results
        general_top_k = general_similarities[:top_k]
        domain_top_k = domain_similarities[:top_k]

        # Calculate metrics
        general_metrics = self._calculate_retrieval_metrics(
            [text for text, _ in general_top_k], ground_truth
        )
        domain_metrics = self._calculate_retrieval_metrics(
            [text for text, _ in domain_top_k], ground_truth
        )

        # Determine recommendation
        general_score = general_metrics.get("f1_score", 0.0)
        domain_score = domain_metrics.get("f1_score", 0.0)

        if domain_score > general_score * 1.1:  # 10% improvement threshold
            recommendation = "domain"
        elif general_score > domain_score * 1.1:
            recommendation = "general"
        else:
            recommendation = "similar"

        return ComparisonResult(
            query=query,
            general_model_results=general_top_k,
            domain_model_results=domain_top_k,
            ground_truth=ground_truth,
            general_model_metrics=general_metrics,
            domain_model_metrics=domain_metrics,
            recommendation=recommendation,
        )

    def _calculate_retrieval_metrics(
        self, retrieved: List[str], ground_truth: List[str]
    ) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score"""
        if not ground_truth:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        retrieved_set = set(retrieved)
        ground_truth_set = set(ground_truth)

        true_positives = len(retrieved_set.intersection(ground_truth_set))

        precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
        recall = true_positives / len(ground_truth_set) if ground_truth_set else 0.0

        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": true_positives,
        }

    def _get_cache_key(
        self, text: str, include_general: bool, include_domain: bool
    ) -> str:
        """Generate cache key for embedding"""
        key_data = f"{text}_{include_general}_{include_domain}_{self.config.general_model_name}_{self.config.domain_model_name}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def _load_embedding_cache(self):
        """Load embedding cache from disk"""
        if not self.config.cache_embeddings:
            return

        cache_file = os.path.join(self.config.cache_dir, "embedding_cache.pkl")

        try:
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
            self.embedding_cache = {}

    async def _save_embedding_cache(self):
        """Save embedding cache to disk"""
        if not self.config.cache_embeddings:
            return

        try:
            os.makedirs(self.config.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.config.cache_dir, "embedding_cache.pkl")

            with open(cache_file, "wb") as f:
                pickle.dump(self.embedding_cache, f)

            logger.debug(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "general_model": self.model_info["general"],
            "domain_model": self.model_info["domain"],
            "config": {
                "cache_embeddings": self.config.cache_embeddings,
                "batch_size": self.config.batch_size,
                "max_sequence_length": self.config.max_sequence_length,
                "device": self.config.device,
                "normalize_embeddings": self.config.normalize_embeddings,
            },
            "stats": self.embedding_stats,
        }

    async def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.embedding_stats["cache_hit_rate"] = 0.0
        logger.info("Embedding cache cleared")

    async def health_check(self) -> Dict[str, Any]:
        """Check health of embedding manager"""
        try:
            if not self._initialized:
                return {"status": "not_initialized"}

            # Test embedding generation
            test_result = await self.generate_embeddings(
                "test embedding", include_general=True, include_domain=True
            )

            return {
                "status": "healthy",
                "models_loaded": {
                    "general": self.general_model is not None,
                    "domain": self.domain_model is not None,
                },
                "model_info": self.model_info,
                "cache_stats": {
                    "cache_size": len(self.embedding_cache),
                    "cache_hit_rate": self.embedding_stats["cache_hit_rate"],
                },
                "test_embedding_shape": {
                    "general": (
                        test_result.general_embedding.shape
                        if test_result.general_embedding is not None
                        else None
                    ),
                    "domain": (
                        test_result.domain_embedding.shape
                        if test_result.domain_embedding is not None
                        else None
                    ),
                },
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def shutdown(self):
        """Shutdown embedding manager and save cache"""
        logger.info("Shutting down Embedding Manager")

        if self.config.cache_embeddings:
            await self._save_embedding_cache()

        # Clear models from memory
        self.general_model = None
        self.domain_model = None

        logger.info("Embedding Manager shutdown complete")
