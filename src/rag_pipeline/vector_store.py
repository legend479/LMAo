"""
Elasticsearch Vector Store
Vector database implementation using Elasticsearch with hybrid search support
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError

from .models import ProcessedDocument
from ..shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Single search result"""

    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    document_id: str
    chunk_type: str
    parent_chunk_id: Optional[str] = None
    highlights: Optional[Dict[str, List[str]]] = None


@dataclass
class SearchResponse:
    """Search response with results and metadata"""

    results: List[SearchResult]
    total_hits: int
    max_score: float
    took_ms: int
    query: str
    search_type: str
    aggregations: Optional[Dict[str, Any]] = None


@dataclass
class ElasticsearchConfig:
    """Configuration for Elasticsearch connection"""

    hosts: List[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    use_ssl: bool = False
    verify_certs: bool = True
    ca_certs: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3

    def __post_init__(self):
        if self.hosts is None:
            # Default to http://localhost:9200 with proper scheme
            self.hosts = ["http://localhost:9200"]


class ElasticsearchStore:
    """Elasticsearch-based vector store with hybrid search capabilities"""

    def __init__(self, config: ElasticsearchConfig = None, embedding_manager=None):
        self.config = config or ElasticsearchConfig()
        self.client: Optional[AsyncElasticsearch] = None
        self.embedding_manager = (
            embedding_manager  # Set by RAG pipeline during initialization
        )
        self.index_name = "se_sme_documents"
        self.chunk_index_name = "se_sme_chunks"
        self._initialized = False

        # Index mappings
        self.document_mapping = {
            "mappings": {
                "properties": {
                    "document_id": {"type": "keyword"},
                    "title": {"type": "text", "analyzer": "standard"},
                    "content": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                    },
                    "author": {"type": "keyword"},
                    "category": {"type": "keyword"},
                    "file_path": {"type": "keyword"},
                    "file_size": {"type": "long"},
                    "mime_type": {"type": "keyword"},
                    "creation_date": {"type": "date"},
                    "modification_date": {"type": "date"},
                    "processing_date": {"type": "date"},
                    "content_hash": {"type": "keyword"},
                    "chunk_count": {"type": "integer"},
                    "word_count": {"type": "integer"},
                    "processing_time": {"type": "float"},
                    "tags": {"type": "keyword"},
                    "keywords": {"type": "keyword"},
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "code_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "snowball"],
                        },
                        "technical_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "technical_synonyms"],
                        },
                    },
                    "filter": {
                        "technical_synonyms": {
                            "type": "synonym",
                            "synonyms": [
                                "function,method,procedure",
                                "class,object,type",
                                "variable,var,field",
                                "parameter,param,argument,arg",
                            ],
                        }
                    },
                },
            },
        }

        self.chunk_mapping = {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "code": {"type": "text", "analyzer": "code_analyzer"},
                            "technical": {
                                "type": "text",
                                "analyzer": "technical_analyzer",
                            },
                        },
                    },
                    "chunk_type": {"type": "keyword"},
                    "size_category": {"type": "keyword"},
                    "parent_chunk_id": {"type": "keyword"},
                    "child_chunk_ids": {"type": "keyword"},
                    "start_index": {"type": "integer"},
                    "end_index": {"type": "integer"},
                    "word_count": {"type": "integer"},
                    "char_count": {"type": "integer"},
                    "line_count": {"type": "integer"},
                    "sentence_count": {"type": "integer"},
                    "text_type": {"type": "keyword"},
                    "code_type": {"type": "keyword"},
                    # Document metadata (denormalized for search)
                    "document_title": {"type": "text"},
                    "document_author": {"type": "keyword"},
                    "document_category": {"type": "keyword"},
                    "document_creation_date": {"type": "date"},
                    "mime_type": {"type": "keyword"},
                    # Vector embeddings for semantic search
                    "embedding_general": {
                        "type": "dense_vector",
                        "dims": 768,  # all-mpnet-base-v2 dimensions
                        "index": True,
                        "similarity": "cosine",
                    },
                    "embedding_domain": {
                        "type": "dense_vector",
                        "dims": 768,  # GraphCodeBERT dimensions
                        "index": True,
                        "similarity": "cosine",
                    },
                    # Search metadata
                    "indexed_date": {"type": "date"},
                    "last_updated": {"type": "date"},
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "code_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop"],
                        },
                        "technical_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "technical_synonyms"],
                        },
                    },
                    "filter": {
                        "technical_synonyms": {
                            "type": "synonym",
                            "synonyms": [
                                "function,method,procedure",
                                "class,object,type",
                                "variable,var,field",
                                "parameter,param,argument,arg",
                                "algorithm,algo,method",
                                "database,db,datastore",
                                "api,interface,endpoint",
                            ],
                        }
                    },
                },
            },
        }

    async def initialize(self):
        """Initialize Elasticsearch connection and indices"""
        if self._initialized:
            return

        logger.info("Initializing Elasticsearch connection")

        # Create Elasticsearch client
        client_config = {
            "hosts": self.config.hosts,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
        }

        if self.config.username and self.config.password:
            client_config["basic_auth"] = (self.config.username, self.config.password)

        if self.config.use_ssl:
            client_config["use_ssl"] = True
            client_config["verify_certs"] = self.config.verify_certs
            if self.config.ca_certs:
                client_config["ca_certs"] = self.config.ca_certs

        self.client = AsyncElasticsearch(**client_config)

        try:
            # Test connection
            info = await self.client.info()
            logger.info("Connected to Elasticsearch", version=info["version"]["number"])

            # Create indices if they don't exist
            await self._create_indices()

            self._initialized = True
            logger.info("Elasticsearch store initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize Elasticsearch", error=str(e))
            raise

    async def _create_indices(self):
        """Create Elasticsearch indices with proper mappings"""

        # Create document index
        if not await self.client.indices.exists(index=self.index_name):
            await self.client.indices.create(
                index=self.index_name, body=self.document_mapping
            )
            logger.info(f"Created document index: {self.index_name}")
        else:
            logger.info(f"Document index already exists: {self.index_name}")

        # Create chunk index
        if not await self.client.indices.exists(index=self.chunk_index_name):
            await self.client.indices.create(
                index=self.chunk_index_name, body=self.chunk_mapping
            )
            logger.info(f"Created chunk index: {self.chunk_index_name}")
        else:
            logger.info(f"Chunk index already exists: {self.chunk_index_name}")

    async def store_document(self, processed_doc: ProcessedDocument) -> str:
        """Store a processed document and its chunks"""
        if not self._initialized:
            await self.initialize()

        logger.info(
            "Storing document",
            doc_id=processed_doc.document_id,
            chunk_count=len(processed_doc.chunks),
        )

        # Debug: Log chunk details
        for i, chunk in enumerate(processed_doc.chunks[:3]):  # Log first 3 chunks
            logger.info(
                f"Chunk {i}: id={chunk.chunk_id}, type={chunk.chunk_type}, content_length={len(chunk.content)}, content_preview='{chunk.content[:50]}...'"
            )

        if len(processed_doc.chunks) == 0:
            logger.warning("No chunks to store! Document processing may have failed.")

        try:
            # Store document metadata
            doc_data = {
                "document_id": processed_doc.document_id,
                "title": processed_doc.metadata.title,
                "content": processed_doc.content[
                    :10000
                ],  # Store first 10k chars for search
                "author": processed_doc.metadata.author,
                "category": processed_doc.metadata.category,
                "file_path": processed_doc.original_path,
                "file_size": processed_doc.metadata.file_size,
                "mime_type": processed_doc.metadata.mime_type,
                "creation_date": processed_doc.metadata.created_at,
                "modification_date": processed_doc.metadata.modified_at,
                "processing_date": datetime.utcnow(),
                "content_hash": processed_doc.content_hash,
                "chunk_count": len(processed_doc.chunks),
                "word_count": processed_doc.metadata.word_count,
                "processing_time": processed_doc.processing_time,
                "tags": processed_doc.metadata.tags or [],
                "keywords": processed_doc.metadata.keywords or [],
            }

            # Store document
            await self.client.index(
                index=self.index_name, id=processed_doc.document_id, body=doc_data
            )

            # Store chunks with embeddings
            chunk_operations = []
            for chunk in processed_doc.chunks:
                chunk_data = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": processed_doc.document_id,
                    "content": chunk.content,
                    "chunk_type": chunk.chunk_type,
                    "size_category": chunk.metadata.get("size_category"),
                    "parent_chunk_id": chunk.parent_chunk_id,
                    "child_chunk_ids": chunk.child_chunk_ids,
                    "start_index": chunk.start_char,
                    "end_index": chunk.end_char,
                    "word_count": chunk.metadata.get("word_count", 0),
                    "char_count": chunk.metadata.get("char_count", 0),
                    "line_count": chunk.metadata.get("line_count", 0),
                    "sentence_count": chunk.metadata.get("sentence_count", 0),
                    "text_type": chunk.metadata.get("text_type"),
                    "code_type": chunk.metadata.get("code_type"),
                    # Document metadata (denormalized)
                    "document_title": processed_doc.metadata.title,
                    "document_author": processed_doc.metadata.author,
                    "document_category": processed_doc.metadata.category,
                    "document_creation_date": processed_doc.metadata.created_at,
                    "mime_type": processed_doc.metadata.mime_type,
                    # Search metadata
                    "indexed_date": datetime.utcnow(),
                    "last_updated": datetime.utcnow(),
                }

                # Add embeddings if embedding manager is available
                if (
                    self.embedding_manager
                    and hasattr(chunk, "embeddings")
                    and chunk.embeddings
                ):
                    if "general" in chunk.embeddings:
                        chunk_data["embedding_general"] = chunk.embeddings[
                            "general"
                        ].tolist()
                    if "domain" in chunk.embeddings:
                        chunk_data["embedding_domain"] = chunk.embeddings[
                            "domain"
                        ].tolist()

                # Add to bulk operations
                chunk_operations.extend(
                    [
                        {
                            "index": {
                                "_index": self.chunk_index_name,
                                "_id": chunk.chunk_id,
                            }
                        },
                        chunk_data,
                    ]
                )

            # Bulk index chunks
            if chunk_operations:
                logger.info(
                    f"Bulk indexing {len(chunk_operations)//2} chunks to index '{self.chunk_index_name}'"
                )
                response = await self.client.bulk(body=chunk_operations)

                # Check for errors
                if response.get("errors"):
                    errors = [
                        item
                        for item in response["items"]
                        if "error" in item.get("index", {})
                    ]
                    logger.warning(f"Bulk indexing errors: {len(errors)} chunks failed")
                    for error in errors[:5]:  # Log first 5 errors
                        logger.error(
                            "Chunk indexing error", error=error["index"]["error"]
                        )
                else:
                    logger.info(
                        f"Successfully indexed {len(chunk_operations)//2} chunks"
                    )
            else:
                logger.warning("No chunk operations to perform - no chunks to index!")

            # Refresh indices to make documents searchable immediately
            await self.client.indices.refresh(
                index=[self.index_name, self.chunk_index_name]
            )

            logger.info(
                "Document stored successfully",
                doc_id=processed_doc.document_id,
                chunks_stored=len(processed_doc.chunks),
            )

            return processed_doc.document_id

        except Exception as e:
            logger.error(
                "Failed to store document",
                doc_id=processed_doc.document_id,
                error=str(e),
            )
            raise

    async def search_chunks(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = 10,
        search_type: str = "hybrid",
    ) -> SearchResponse:
        """Search chunks using various search strategies"""
        if not self._initialized:
            await self.initialize()

        logger.info("Searching chunks", query=query[:100], search_type=search_type)

        try:
            if search_type == "hybrid":
                return await self._hybrid_search(query, filters, max_results)
            elif search_type == "vector":
                return await self._vector_search(query, filters, max_results)
            elif search_type == "keyword":
                return await self._keyword_search(query, filters, max_results)
            else:
                raise ValueError(f"Unsupported search type: {search_type}")

        except Exception as e:
            logger.error("Search failed", query=query[:100], error=str(e))
            raise

    async def _hybrid_search(
        self, query: str, filters: Optional[Dict[str, Any]], max_results: int
    ) -> SearchResponse:
        """Hybrid search combining keyword and vector search"""

        # Keyword-based search with multi-field matching and technical content support
        search_body = {
            "size": max_results,
            "query": {
                "bool": {
                    "should": [
                        # Multi-match query for general content
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "content^2",
                                    "document_title^1.5",
                                    "content.technical",
                                ],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                            }
                        },
                        # Match query for code content
                        {"match": {"content.code": {"query": query, "boost": 1.2}}},
                        # Term queries for exact matches
                        {
                            "term": {
                                "chunk_type": {
                                    "value": (
                                        "code"
                                        if self._looks_like_code_query(query)
                                        else "text"
                                    ),
                                    "boost": 0.5,
                                }
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            },
            "highlight": {
                "fields": {"content": {"fragment_size": 150, "number_of_fragments": 3}}
            },
            "_source": {
                "excludes": [
                    "embedding_general",
                    "embedding_domain",
                ]  # Exclude large embedding fields
            },
        }

        # Add filters
        if filters:
            filter_clauses = []

            if "document_category" in filters:
                filter_clauses.append(
                    {"term": {"document_category": filters["document_category"]}}
                )

            if "chunk_type" in filters:
                filter_clauses.append({"term": {"chunk_type": filters["chunk_type"]}})

            if "size_category" in filters:
                filter_clauses.append(
                    {"term": {"size_category": filters["size_category"]}}
                )

            if "date_range" in filters:
                date_filter = {"range": {"document_creation_date": {}}}
                if "gte" in filters["date_range"]:
                    date_filter["range"]["document_creation_date"]["gte"] = filters[
                        "date_range"
                    ]["gte"]
                if "lte" in filters["date_range"]:
                    date_filter["range"]["document_creation_date"]["lte"] = filters[
                        "date_range"
                    ]["lte"]
                filter_clauses.append(date_filter)

            if filter_clauses:
                search_body["query"]["bool"]["filter"] = filter_clauses

        # Execute search
        start_time = datetime.utcnow()

        # Debug: Log the search query for hybrid search
        logger.info(
            f"Hybrid search on index '{self.chunk_index_name}' with query: {search_body}"
        )

        response = await self.client.search(
            index=self.chunk_index_name, body=search_body
        )
        search_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Debug: Log the raw response for hybrid search
        logger.info(
            f"Hybrid search response: total_hits={response['hits']['total']['value']}, max_score={response['hits'].get('max_score')}"
        )

        # Parse results
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            highlights = hit.get("highlight", {})

            result = SearchResult(
                chunk_id=source["chunk_id"],
                content=source["content"],
                score=hit["_score"],
                metadata=source,
                document_id=source["document_id"],
                chunk_type=source["chunk_type"],
                parent_chunk_id=source.get("parent_chunk_id"),
                highlights=highlights,
            )
            results.append(result)

        return SearchResponse(
            results=results,
            total_hits=response["hits"]["total"]["value"],
            max_score=response["hits"]["max_score"] or 0.0,
            took_ms=int(search_time),
            query=query,
            search_type="hybrid",
        )

    async def _keyword_search(
        self, query: str, filters: Optional[Dict[str, Any]], max_results: int
    ) -> SearchResponse:
        """Pure keyword-based search using BM25"""

        search_body = {
            "size": max_results,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content^2", "document_title^1.5"],
                    "type": "best_fields",
                }
            },
            "highlight": {
                "fields": {"content": {"fragment_size": 150, "number_of_fragments": 3}}
            },
        }

        # Add filters (similar to hybrid search)
        if filters:
            filter_clauses = []
            for key, value in filters.items():
                if isinstance(value, list):
                    filter_clauses.append({"terms": {f"metadata.{key}": value}})
                else:
                    filter_clauses.append({"term": {f"metadata.{key}": value}})

            if filter_clauses:
                search_body["query"] = {
                    "bool": {"must": [search_body["query"]], "filter": filter_clauses}
                }

        start_time = datetime.utcnow()

        # Debug: Check total document count in index
        try:
            count_response = await self.client.count(index=self.chunk_index_name)
            logger.info(
                f"Total documents in index '{self.chunk_index_name}': {count_response['count']}"
            )
        except Exception as e:
            logger.error(f"Failed to count documents in index: {e}")

        # Debug: Log the search query for keyword search
        logger.info(
            f"Keyword search on index '{self.chunk_index_name}' with query: {search_body}"
        )

        response = await self.client.search(
            index=self.chunk_index_name, body=search_body
        )
        search_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Debug: Log the raw response for keyword search
        logger.info(
            f"Keyword search response: total_hits={response['hits']['total']['value']}, max_score={response['hits'].get('max_score')}"
        )

        # Parse results (similar to hybrid search)
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
            search_type="keyword",
        )

    async def _vector_search(
        self, query: str, filters: Optional[Dict[str, Any]], max_results: int
    ) -> SearchResponse:
        """Vector-based semantic search using embeddings"""

        # Vector search requires embedding manager to be available
        if not self.embedding_manager:
            logger.warning(
                "Embedding manager not available - falling back to keyword search"
            )
            return await self._keyword_search(query, filters, max_results)

        logger.info("Performing vector similarity search", query=query[:100])
        start_time = datetime.utcnow()

        try:
            # Determine if we should use domain embeddings for code queries
            use_domain_embedding = self._looks_like_code_query(query)

            # Generate appropriate embeddings for the query
            embedding_result = await self.embedding_manager.generate_embeddings(
                query,
                include_general=not use_domain_embedding,  # Only include general if not using domain
                include_domain=use_domain_embedding,  # Only include domain if needed
            )

            # Select the appropriate embedding based on query type
            if use_domain_embedding:
                logger.debug("Using domain-specific embeddings for code-like query")
                query_embedding = embedding_result.domain_embedding.tolist()
                embedding_field = "embedding_domain"
            else:
                query_embedding = embedding_result.general_embedding.tolist()
                embedding_field = "embedding_general"

            # Build the k-NN query
            knn_query = {
                "field": embedding_field,
                "query_vector": query_embedding,
                "k": max_results,
                "num_candidates": max(
                    100, max_results * 10
                ),  # Consider more candidates for better recall
            }

            # Build the search request
            search_body = {
                "knn": knn_query,
                "size": max_results,
                "_source": {
                    "excludes": [
                        "embedding_general",
                        "embedding_domain",  # Exclude large embedding fields
                    ]
                },
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

                if "size_category" in filters:
                    filter_clauses.append(
                        {"term": {"size_category": filters["size_category"]}}
                    )

                if "date_range" in filters:
                    date_filter = {"range": {"document_creation_date": {}}}
                    if "gte" in filters["date_range"]:
                        date_filter["range"]["document_creation_date"]["gte"] = filters[
                            "date_range"
                        ]["gte"]
                    if "lte" in filters["date_range"]:
                        date_filter["range"]["document_creation_date"]["lte"] = filters[
                            "date_range"
                        ]["lte"]
                    filter_clauses.append(date_filter)

                if filter_clauses:
                    search_body["query"] = {"bool": {"filter": filter_clauses}}

            # Execute the search
            logger.debug("Executing vector search", search_body=search_body)
            response = await self.client.search(
                index=self.chunk_index_name, body=search_body
            )

            search_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(
                f"Vector search completed in {search_time:.2f}ms with {len(response['hits']['hits'])} results"
            )

            # Process and return results
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
            logger.error("Vector search failed", query=query[:100], error=str(e))
            # Fall back to keyword search if vector search fails
            logger.warning("Falling back to keyword search due to vector search error")
            return await self._keyword_search(query, filters, max_results)

    def _looks_like_code_query(self, query: str) -> bool:
        """Heuristic to determine if query is looking for code"""
        code_keywords = [
            "function",
            "def",
            "class",
            "method",
            "variable",
            "import",
            "algorithm",
            "implementation",
            "code",
            "syntax",
            "programming",
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in code_keywords)

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        if not self._initialized:
            await self.initialize()

        try:
            response = await self.client.get(index=self.index_name, id=document_id)
            return response["_source"]
        except NotFoundError:
            return None

    async def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk by ID"""
        if not self._initialized:
            await self.initialize()

        try:
            response = await self.client.get(index=self.chunk_index_name, id=chunk_id)
            return response["_source"]
        except NotFoundError:
            return None

    async def delete_document(self, document_id: str) -> bool:
        """Delete document and all its chunks"""
        if not self._initialized:
            await self.initialize()

        try:
            # Delete document
            await self.client.delete(index=self.index_name, id=document_id)

            # Delete all chunks for this document
            await self.client.delete_by_query(
                index=self.chunk_index_name,
                body={"query": {"term": {"document_id": document_id}}},
            )

            logger.info("Document deleted", document_id=document_id)
            return True

        except NotFoundError:
            logger.warning("Document not found for deletion", document_id=document_id)
            return False
        except Exception as e:
            logger.error(
                "Failed to delete document", document_id=document_id, error=str(e)
            )
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored documents and chunks"""
        if not self._initialized:
            await self.initialize()

        try:
            # Check if indices exist
            doc_index_exists = await self.client.indices.exists(index=self.index_name)
            chunk_index_exists = await self.client.indices.exists(
                index=self.chunk_index_name
            )

            logger.info(
                f"Index existence: documents={doc_index_exists}, chunks={chunk_index_exists}"
            )

            # Get document count
            doc_count = (
                await self.client.count(index=self.index_name)
                if doc_index_exists
                else {"count": 0}
            )

            # Get chunk count
            chunk_count = (
                await self.client.count(index=self.chunk_index_name)
                if chunk_index_exists
                else {"count": 0}
            )

            logger.info(
                f"Document counts: documents={doc_count['count']}, chunks={chunk_count['count']}"
            )

            # Get chunk type distribution
            chunk_agg = await self.client.search(
                index=self.chunk_index_name,
                body={
                    "size": 0,
                    "aggs": {
                        "chunk_types": {"terms": {"field": "chunk_type"}},
                        "size_categories": {"terms": {"field": "size_category"}},
                    },
                },
            )

            return {
                "total_documents": doc_count["count"],
                "total_chunks": chunk_count["count"],
                "chunk_types": {
                    bucket["key"]: bucket["doc_count"]
                    for bucket in chunk_agg["aggregations"]["chunk_types"]["buckets"]
                },
                "size_categories": {
                    bucket["key"]: bucket["doc_count"]
                    for bucket in chunk_agg["aggregations"]["size_categories"][
                        "buckets"
                    ]
                },
            }

        except Exception as e:
            logger.error("Failed to get stats", error=str(e))
            return {}

    async def health_check(self) -> Dict[str, Any]:
        """Check health of Elasticsearch connection"""
        try:
            if not self.client:
                return {"status": "unhealthy", "error": "Client not initialized"}

            # Check cluster health
            health = await self.client.cluster.health()

            # Check if indices exist
            doc_exists = await self.client.indices.exists(index=self.index_name)
            chunk_exists = await self.client.indices.exists(index=self.chunk_index_name)

            return {
                "status": (
                    "healthy"
                    if health["status"] in ["green", "yellow"]
                    else "unhealthy"
                ),
                "cluster_status": health["status"],
                "cluster_name": health["cluster_name"],
                "number_of_nodes": health["number_of_nodes"],
                "indices": {"documents": doc_exists, "chunks": chunk_exists},
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def shutdown(self):
        """Close Elasticsearch connection"""
        if self.client:
            await self.client.close()
            logger.info("Elasticsearch connection closed")
