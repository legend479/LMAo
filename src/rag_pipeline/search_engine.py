"""
Hybrid Search Engine
Advanced search engine combining vector similarity and keyword search with RRF
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from .vector_store import ElasticsearchStore, SearchResult, SearchResponse
from .reranker import BGEReranker, RerankerConfig
from ..shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RRFConfig:
    """Configuration for Reciprocal Rank Fusion"""

    k: int = 60  # RRF parameter (typically 60)
    weights: Dict[str, float] = None  # Weights for different search types

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "vector": 0.6,  # Vector search weight
                "keyword": 0.4,  # Keyword search weight
            }


@dataclass
class SearchConfig:
    """Configuration for search engine"""

    default_max_results: int = 10
    enable_reranking: bool = True
    rrf_config: RRFConfig = None
    search_timeout_seconds: int = 30
    enable_query_expansion: bool = True
    enable_spell_correction: bool = True

    def __post_init__(self):
        if self.rrf_config is None:
            self.rrf_config = RRFConfig()


@dataclass
class FusedSearchResult:
    """Search result after RRF fusion"""

    chunk_id: str
    content: str
    fused_score: float
    individual_scores: Dict[str, float]  # Scores from different search methods
    individual_ranks: Dict[str, int]  # Ranks from different search methods
    metadata: Dict[str, Any]
    document_id: str
    chunk_type: str
    parent_chunk_id: Optional[str] = None
    highlights: Optional[Dict[str, List[str]]] = None


class HybridSearchEngine:
    """Hybrid search engine with RRF score fusion"""

    def __init__(
        self,
        vector_store: ElasticsearchStore,
        embedding_manager=None,
        config: SearchConfig = None,
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager  # Will be set by RAG pipeline
        self.config = config or SearchConfig()
        self._initialized = False

        # Query processing components
        self.query_expander = QueryExpander()
        self.spell_corrector = SpellCorrector()

        # Reranker
        self.reranker = (
            BGEReranker(RerankerConfig()) if self.config.enable_reranking else None
        )

        # Search statistics
        self.search_stats = {
            "total_searches": 0,
            "avg_response_time": 0.0,
            "search_types_used": {"hybrid": 0, "vector": 0, "keyword": 0},
        }

    async def initialize(self):
        """Initialize search engine"""
        if self._initialized:
            return

        logger.info("Initializing Hybrid Search Engine")

        # Ensure vector store is initialized
        if not self.vector_store._initialized:
            await self.vector_store.initialize()

        # Initialize query processing components
        await self.query_expander.initialize()
        await self.spell_corrector.initialize()

        # Initialize reranker if enabled
        if self.reranker:
            await self.reranker.initialize()

        self._initialized = True
        logger.info("Hybrid Search Engine initialized")

    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = None,
        search_type: str = "hybrid",
    ) -> SearchResponse:
        """Execute hybrid search with RRF fusion"""

        if not self._initialized:
            await self.initialize()

        max_results = max_results or self.config.default_max_results
        start_time = datetime.utcnow()

        logger.info(
            "Executing hybrid search",
            query=query[:100],
            search_type=search_type,
            max_results=max_results,
        )

        try:
            # Preprocess query
            processed_query = await self._preprocess_query(query)

            if search_type == "hybrid":
                response = await self._hybrid_search_with_rrf(
                    processed_query, filters, max_results
                )
            elif search_type == "vector":
                response = await self.vector_store.search_chunks(
                    processed_query, filters, max_results, "vector"
                )
            elif search_type == "keyword":
                response = await self.vector_store.search_chunks(
                    processed_query, filters, max_results, "keyword"
                )
            else:
                raise ValueError(f"Unsupported search type: {search_type}")

            # Update statistics
            search_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_search_stats(search_type, search_time)

            logger.info(
                "Search completed",
                query=query[:100],
                results_count=len(response.results),
                search_time=search_time,
            )

            return response

        except Exception as e:
            logger.error("Search failed", query=query[:100], error=str(e))
            raise

    async def _hybrid_search_with_rrf(
        self, query: str, filters: Optional[Dict[str, Any]], max_results: int
    ) -> SearchResponse:
        """Execute hybrid search with Reciprocal Rank Fusion"""

        # Execute multiple search strategies in parallel
        search_tasks = []

        # Keyword search
        search_tasks.append(
            self.vector_store.search_chunks(query, filters, max_results * 2, "keyword")
        )

        # Vector search (placeholder - will be enhanced in task 2.4)
        # For now, we'll use a different keyword search with different parameters
        search_tasks.append(
            self._alternative_keyword_search(query, filters, max_results * 2)
        )

        # Execute searches concurrently
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Handle any exceptions
        valid_results = []
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                logger.warning(f"Search method {i} failed: {result}")
            else:
                valid_results.append(result)

        if not valid_results:
            # Return empty response if all searches failed
            return SearchResponse(
                results=[],
                total_hits=0,
                max_score=0.0,
                took_ms=0,
                query=query,
                search_type="hybrid",
            )

        # Apply RRF fusion
        fused_results = self._apply_rrf_fusion(valid_results, query)

        # Limit to requested number of results
        fused_results = fused_results[:max_results]

        # Convert back to SearchResult format
        final_results = []
        for fused_result in fused_results:
            search_result = SearchResult(
                chunk_id=fused_result.chunk_id,
                content=fused_result.content,
                score=fused_result.fused_score,
                metadata=fused_result.metadata,
                document_id=fused_result.document_id,
                chunk_type=fused_result.chunk_type,
                parent_chunk_id=fused_result.parent_chunk_id,
                highlights=fused_result.highlights,
            )
            final_results.append(search_result)

        # Apply reranking if enabled
        if self.reranker and final_results:
            try:
                reranked_results = await self.reranker.rerank(
                    query, final_results, max_results
                )
                # Convert reranked results back to SearchResult format
                final_results = [rr.original_result for rr in reranked_results]
                # Update scores with rerank scores
                for i, rr in enumerate(reranked_results):
                    final_results[i].score = rr.rerank_score

                logger.info(f"Applied reranking to {len(final_results)} results")
            except Exception as e:
                logger.warning(f"Reranking failed, using original results: {e}")

        return SearchResponse(
            results=final_results,
            total_hits=sum(r.total_hits for r in valid_results),
            max_score=final_results[0].score if final_results else 0.0,
            took_ms=max(r.took_ms for r in valid_results),
            query=query,
            search_type="hybrid",
        )

    async def _alternative_keyword_search(
        self, query: str, filters: Optional[Dict[str, Any]], max_results: int
    ) -> SearchResponse:
        """Alternative keyword search with different parameters"""

        # This is a placeholder for vector search
        # For now, we'll do a phrase-based keyword search as an alternative
        search_body = {
            "size": max_results,
            "query": {
                "bool": {
                    "should": [
                        {"match_phrase": {"content": {"query": query, "boost": 2.0}}},
                        {
                            "match": {
                                "content": {
                                    "query": query,
                                    "operator": "and",
                                    "boost": 1.5,
                                }
                            }
                        },
                        {
                            "wildcard": {
                                "content": {
                                    "value": f"*{query.split()[0] if query.split() else query}*",
                                    "boost": 0.5,
                                }
                            }
                        },
                    ]
                }
            },
        }

        start_time = datetime.utcnow()
        response = await self.vector_store.client.search(
            index=self.vector_store.chunk_index_name, body=search_body
        )
        search_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Parse results
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            result = SearchResult(
                chunk_id=source["chunk_id"],
                content=source["content"],
                score=hit["_score"],
                metadata=source,
                document_id=source["document_id"],
                chunk_type=source["chunk_type"],
                parent_chunk_id=source.get("parent_chunk_id"),
                highlights=hit.get("highlight", {}),
            )
            results.append(result)

        return SearchResponse(
            results=results,
            total_hits=response["hits"]["total"]["value"],
            max_score=response["hits"]["max_score"] or 0.0,
            took_ms=int(search_time),
            query=query,
            search_type="alternative_keyword",
        )

    async def _vector_search(
        self, query: str, filters: Optional[Dict[str, Any]], max_results: int
    ) -> SearchResponse:
        """Vector-based semantic search using embeddings"""

        if not self.embedding_manager:
            logger.warning(
                "Embedding manager not available - falling back to keyword search"
            )
            return await self._keyword_search(query, filters, max_results)

        try:
            # Generate query embedding
            query_embedding_result = await self.embedding_manager.generate_embeddings(
                query
            )

            # Use general embedding for vector search (can be made configurable)
            query_vector = query_embedding_result.general_embedding
            if query_vector is None:
                logger.warning(
                    "Failed to generate query embedding - falling back to keyword search"
                )
                return await self._keyword_search(query, filters, max_results)

            # Construct vector search query
            search_body = {
                "size": max_results,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding_general') + 1.0",
                            "params": {"query_vector": query_vector.tolist()},
                        },
                    }
                },
                "_source": {"excludes": ["embedding_general", "embedding_domain"]},
            }

            # Add filters if provided
            if filters:
                filter_clauses = []

                if "document_category" in filters:
                    filter_clauses.append(
                        {"term": {"document_category": filters["document_category"]}}
                    )

                if "chunk_type" in filters:
                    filter_clauses.append(
                        {"term": {"chunk_type": filters["chunk_type"]}}
                    )

                if filter_clauses:
                    search_body["query"]["script_score"]["query"] = {
                        "bool": {"filter": filter_clauses}
                    }

            # Execute search
            start_time = datetime.utcnow()
            response = await self.vector_store.client.search(
                index=self.vector_store.chunk_index_name, body=search_body
            )
            search_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Parse results
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                result = SearchResult(
                    chunk_id=source["chunk_id"],
                    content=source["content"],
                    score=hit["_score"],
                    metadata=source,
                    document_id=source["document_id"],
                    chunk_type=source["chunk_type"],
                    parent_chunk_id=source.get("parent_chunk_id"),
                    highlights={},
                )
                results.append(result)

            return SearchResponse(
                results=results,
                total_hits=response["hits"]["total"]["value"],
                max_score=response["hits"]["max_score"] or 0.0,
                took_ms=int(search_time),
                query=query,
                search_type="vector",
            )

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            logger.warning("Falling back to keyword search")
            return await self._keyword_search(query, filters, max_results)

    def _apply_rrf_fusion(
        self, search_responses: List[SearchResponse], query: str
    ) -> List[FusedSearchResult]:
        """Apply Reciprocal Rank Fusion to combine search results"""

        # Collect all unique results
        all_results = {}  # chunk_id -> {result, scores, ranks}

        for i, response in enumerate(search_responses):
            search_method = f"method_{i}"

            for rank, result in enumerate(response.results, 1):
                chunk_id = result.chunk_id

                if chunk_id not in all_results:
                    all_results[chunk_id] = {
                        "result": result,
                        "scores": {},
                        "ranks": {},
                    }

                all_results[chunk_id]["scores"][search_method] = result.score
                all_results[chunk_id]["ranks"][search_method] = rank

        # Calculate RRF scores
        fused_results = []
        k = self.config.rrf_config.k

        for chunk_id, data in all_results.items():
            rrf_score = 0.0

            # Calculate RRF score: sum of 1/(k + rank) for each method
            for method, rank in data["ranks"].items():
                method_weight = self.config.rrf_config.weights.get(method, 1.0)
                rrf_score += method_weight / (k + rank)

            fused_result = FusedSearchResult(
                chunk_id=chunk_id,
                content=data["result"].content,
                fused_score=rrf_score,
                individual_scores=data["scores"],
                individual_ranks=data["ranks"],
                metadata=data["result"].metadata,
                document_id=data["result"].document_id,
                chunk_type=data["result"].chunk_type,
                parent_chunk_id=data["result"].parent_chunk_id,
                highlights=data["result"].highlights,
            )

            fused_results.append(fused_result)

        # Sort by fused score (descending)
        fused_results.sort(key=lambda x: x.fused_score, reverse=True)

        return fused_results

    async def _preprocess_query(self, query: str) -> str:
        """Preprocess query with expansion and spell correction"""

        processed_query = query

        # Apply spell correction if enabled
        if self.config.enable_spell_correction:
            corrected = await self.spell_corrector.correct(processed_query)
            if corrected != processed_query:
                logger.info(
                    "Applied spell correction",
                    original=processed_query,
                    corrected=corrected,
                )
                processed_query = corrected

        # Apply query expansion if enabled
        if self.config.enable_query_expansion:
            expanded = await self.query_expander.expand(processed_query)
            if expanded != processed_query:
                logger.info(
                    "Applied query expansion",
                    original=processed_query,
                    expanded=expanded,
                )
                processed_query = expanded

        return processed_query

    def _update_search_stats(self, search_type: str, search_time: float):
        """Update search statistics"""
        self.search_stats["total_searches"] += 1

        # Update average response time
        total = self.search_stats["total_searches"]
        current_avg = self.search_stats["avg_response_time"]
        self.search_stats["avg_response_time"] = (
            current_avg * (total - 1) + search_time
        ) / total

        # Update search type counts
        if search_type in self.search_stats["search_types_used"]:
            self.search_stats["search_types_used"][search_type] += 1

    async def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return {
            **self.search_stats,
            "config": {
                "default_max_results": self.config.default_max_results,
                "enable_reranking": self.config.enable_reranking,
                "rrf_k": self.config.rrf_config.k,
                "rrf_weights": self.config.rrf_config.weights,
            },
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check health of search engine"""
        try:
            # Check vector store health
            vector_store_health = await self.vector_store.health_check()

            return {
                "status": (
                    "healthy"
                    if vector_store_health["status"] == "healthy"
                    else "degraded"
                ),
                "vector_store": vector_store_health,
                "search_stats": self.search_stats,
                "initialized": self._initialized,
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def shutdown(self):
        """Shutdown search engine"""
        logger.info("Shutting down Hybrid Search Engine")
        await self.vector_store.shutdown()


class QueryExpander:
    """Query expansion for better search results"""

    def __init__(self):
        self._initialized = False

        # Technical synonyms and expansions
        self.expansions = {
            "function": ["method", "procedure", "routine"],
            "class": ["object", "type", "structure"],
            "variable": ["var", "field", "attribute"],
            "algorithm": ["method", "approach", "technique"],
            "database": ["db", "datastore", "repository"],
            "api": ["interface", "endpoint", "service"],
            "bug": ["error", "issue", "defect", "problem"],
            "test": ["testing", "validation", "verification"],
            "code": ["implementation", "source", "program"],
            "documentation": ["docs", "manual", "guide"],
        }

    async def initialize(self):
        """Initialize query expander"""
        self._initialized = True
        logger.info("Query expander initialized")

    async def expand(self, query: str) -> str:
        """Expand query with synonyms and related terms"""
        if not self._initialized:
            await self.initialize()

        words = query.lower().split()
        expanded_terms = []

        for word in words:
            expanded_terms.append(word)

            # Add synonyms if available
            if word in self.expansions:
                # Add one most relevant synonym to avoid query bloat
                expanded_terms.append(self.expansions[word][0])

        # Return expanded query (limit expansion to avoid over-expansion)
        if len(expanded_terms) > len(words) * 1.5:  # Max 50% expansion
            return query  # Return original if too much expansion

        return " ".join(expanded_terms)


class SpellCorrector:
    """Simple spell correction for technical terms"""

    def __init__(self):
        self._initialized = False

        # Common technical term corrections
        self.corrections = {
            "algoritm": "algorithm",
            "fucntion": "function",
            "varible": "variable",
            "databse": "database",
            "implmentation": "implementation",
            "documention": "documentation",
            "authentification": "authentication",
            "authorisation": "authorization",
            "optimisation": "optimization",
            "initialisation": "initialization",
        }

    async def initialize(self):
        """Initialize spell corrector"""
        self._initialized = True
        logger.info("Spell corrector initialized")

    async def correct(self, query: str) -> str:
        """Apply spell corrections to query"""
        if not self._initialized:
            await self.initialize()

        words = query.split()
        corrected_words = []

        for word in words:
            word_lower = word.lower()
            if word_lower in self.corrections:
                # Preserve original case
                if word.isupper():
                    corrected_words.append(self.corrections[word_lower].upper())
                elif word.istitle():
                    corrected_words.append(self.corrections[word_lower].title())
                else:
                    corrected_words.append(self.corrections[word_lower])
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)
