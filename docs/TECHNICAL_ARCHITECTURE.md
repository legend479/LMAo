# LMA-o: Technical Architecture Documentation

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Component Details](#component-details)
3. [Data Flow](#data-flow)
4. [API Specifications](#api-specifications)
5. [Database Schema](#database-schema)
6. [Integration Patterns](#integration-patterns)

## Architecture Overview

### System Design Principles

The LMA-o system follows these core architectural principles:

1. **Microservices Architecture**: Loosely coupled services with clear boundaries
2. **Event-Driven Communication**: Asynchronous messaging for scalability
3. **Stateless Services**: State managed externally for horizontal scaling
4. **API-First Design**: Well-defined interfaces between components
5. **Fault Tolerance**: Graceful degradation and error recovery
6. **Observability**: Comprehensive logging, metrics, and tracing

### Component Interaction Diagram

```
┌─────────────┐
│   Client    │
│  (Browser)  │
└──────┬──────┘
       │ HTTP/WebSocket
       ▼
┌─────────────────────────────────────────┐
│         API Server (FastAPI)            │
│  ┌────────────┐  ┌──────────────────┐  │
│  │ Auth       │  │ Rate Limiter     │  │
│  │ Middleware │  │ & Validation     │  │
│  └────────────┘  └──────────────────┘  │
└──────┬──────────────────┬───────────────┘
       │                  │
       │ REST/gRPC        │ WebSocket
       ▼                  ▼
┌──────────────────┐  ┌──────────────────┐
│  Agent Server    │  │  RAG Pipeline    │
│  (LangGraph)     │◄─┤  (Elasticsearch) │
│                  │  │                  │
│  ┌────────────┐  │  │  ┌────────────┐ │
│  │ Orchestr.  │  │  │  │ Search     │ │
│  │ Planning   │  │  │  │ Embedding  │ │
│  │ Memory     │  │  │  │ Reranking  │ │
│  └────────────┘  │  │  └────────────┘ │
└──────┬───────────┘  └──────┬───────────┘
       │                     │
       │ Tool Execution      │ Document Storage
       ▼                     ▼
┌──────────────────┐  ┌──────────────────┐
│  Tool Registry   │  │  Elasticsearch   │
│  - Knowledge     │  │  - Vectors       │
│  - Documents     │  │  - Metadata      │
│  - Code Exec     │  │  - Full-text     │
│  - Email         │  │                  │
└──────────────────┘  └──────────────────┘
       │                     │
       └──────────┬──────────┘
                  ▼
       ┌──────────────────────┐
       │  Data Layer          │
       │  ┌────────────────┐  │
       │  │  PostgreSQL    │  │
       │  │  Redis Cache   │  │
       │  └────────────────┘  │
       └──────────────────────┘
```

## Component Details

### 1. API Server Architecture

**Technology**: FastAPI with Uvicorn ASGI server

**Layers**:

```
┌─────────────────────────────────────────┐
│         Middleware Stack                │
│  1. Compression (Brotli/Gzip)          │
│  2. Security Headers (CORS, CSP)       │
│  3. Validation (Pydantic)              │
│  4. Rate Limiting (Token Bucket)       │
│  5. Logging (Structured)               │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│         Router Layer                    │
│  /health    - Health checks            │
│  /auth      - Authentication           │
│  /api/v1/chat - Chat endpoints         │
│  /api/v1/documents - Document mgmt     │
│  /api/v1/tools - Tool execution        │
│  /admin     - Admin functions          │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│         Service Layer                   │
│  - Agent Service Client                │
│  - RAG Service Client                  │
│  - Cache Manager                       │
│  - Session Manager                     │
└─────────────────────────────────────────┘
```

**Key Components**:

1. **Authentication System** (`auth/`)
   - JWT token generation and validation
   - Role-based access control (RBAC)
   - User session management
   - API key authentication

2. **Rate Limiting** (`middleware/rate_limiting.py`)
   - Token bucket algorithm
   - Per-user and global limits
   - Redis-backed state
   - Configurable thresholds

3. **WebSocket Manager** (`main.py`)
   - Real-time bidirectional communication
   - Connection pooling
   - Message queuing
   - Heartbeat monitoring

4. **Cache Manager** (`cache/cache_manager.py`)
   - Redis-based caching
   - TTL management
   - Cache invalidation strategies
   - Hit/miss metrics

### 2. Agent Server Architecture

**Technology**: LangGraph for stateful workflows

**Core Components**:

```
┌─────────────────────────────────────────┐
│      LangGraph Orchestrator             │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │  Workflow State Machine           │ │
│  │  - State: WorkflowState           │ │
│  │  - Nodes: Task executors          │ │
│  │  - Edges: Dependencies            │ │
│  │  - Checkpoints: State persistence │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│      Planning Module                    │
│  - Query Analysis                       │
│  - Task Decomposition                   │
│  - Tool Selection                       │
│  - Execution Planning                   │
│  - Adaptation Engine                    │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│      Memory Manager                     │
│  - Conversation History                 │
│  - User Preferences                     │
│  - Tool State                           │
│  - Context Window Management            │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│      Tool Registry                      │
│  - Tool Discovery                       │
│  - Execution Pool                       │
│  - Resource Management                  │
│  - Performance Monitoring               │
└─────────────────────────────────────────┘
```

**Workflow Execution**:

1. **Query Reception**
   - Receive user query
   - Load conversation context
   - Initialize workflow state

2. **Planning Phase**
   - Analyze query complexity
   - Decompose into subtasks
   - Identify dependencies
   - Select appropriate tools

3. **Execution Phase**
   - Create LangGraph workflow
   - Execute tasks in order
   - Handle parallel execution
   - Manage state transitions

4. **Synthesis Phase**
   - Aggregate tool results
   - Generate final response
   - Update conversation memory
   - Store execution history

**State Management**:

```python
@dataclass
class WorkflowState:
    session_id: str
    plan_id: str
    current_task: Optional[str]
    completed_tasks: List[str]
    failed_tasks: List[str]
    task_results: Dict[str, Any]
    context: Dict[str, Any]
    error_count: int
    last_error: Optional[str]
    execution_path: List[str]
    original_query: Optional[str]
    conversation_history: List[Dict[str, Any]]
```

### 3. RAG Pipeline Architecture

**Technology**: Elasticsearch + sentence-transformers

**Pipeline Stages**:

```
┌─────────────────────────────────────────┐
│      Document Ingestion                 │
│  1. File Discovery                      │
│  2. Format Detection                    │
│  3. Content Extraction                  │
│  4. Metadata Enrichment                 │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│      Document Processing                │
│  1. Content Cleaning                    │
│  2. Chunking Strategy Selection         │
│  3. Hierarchical Chunking               │
│  4. Chunk Metadata Generation           │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│      Embedding Generation               │
│  1. Model Selection (Hybrid)            │
│  2. Batch Processing                    │
│  3. Embedding Caching                   │
│  4. Quality Validation                  │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│      Vector Storage                     │
│  1. Elasticsearch Indexing              │
│  2. Dense Vector Storage                │
│  3. Sparse Keyword Indexing             │
│  4. Metadata Storage                    │
└─────────────────────────────────────────┘
```

**Search Pipeline**:

```
┌─────────────────────────────────────────┐
│      Query Processing                   │
│  1. Query Reformulation                 │
│  2. Query Expansion                     │
│  3. Sub-query Generation                │
│  4. Intent Classification               │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│      Hybrid Search                      │
│  1. Vector Search (Cosine Similarity)   │
│  2. Keyword Search (BM25)               │
│  3. Score Fusion (RRF)                  │
│  4. Initial Ranking                     │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│      Reranking                          │
│  1. Cross-encoder Scoring               │
│  2. Relevance Assessment                │
│  3. Diversity Optimization (MMR)        │
│  4. Final Ranking                       │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│      Context Optimization               │
│  1. Token Budget Management             │
│  2. Redundancy Removal                  │
│  3. Hierarchical Context Assembly       │
│  4. Citation Generation                 │
└─────────────────────────────────────────┘
```

**Embedding Models**:

1. **General Purpose**: `sentence-transformers/all-mpnet-base-v2`
   - 768-dimensional embeddings
   - Optimized for semantic similarity
   - Fast inference time

2. **Domain-Specific**: `microsoft/graphcodebert-base`
   - Code-aware embeddings
   - Understands programming constructs
   - Better for technical content

**Hybrid Embedding Selection**:
- Automatic model selection based on content type
- Code snippets → GraphCodeBERT
- Natural language → all-mpnet-base-v2
- Mixed content → Ensemble approach

### 4. Tool Registry Architecture

**Tool Lifecycle**:

```
┌─────────────────────────────────────────┐
│      Tool Registration                  │
│  1. Schema Definition                   │
│  2. Capability Declaration              │
│  3. Resource Requirements               │
│  4. Security Validation                 │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│      Tool Discovery                     │
│  1. Capability Matching                 │
│  2. Multi-criteria Scoring              │
│  3. Preference Alignment                │
│  4. Tool Selection                      │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│      Tool Execution                     │
│  1. Parameter Validation                │
│  2. Resource Allocation                 │
│  3. Sandboxed Execution                 │
│  4. Result Collection                   │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│      Performance Monitoring             │
│  1. Execution Time Tracking             │
│  2. Success Rate Calculation            │
│  3. Resource Usage Monitoring           │
│  4. Quality Assessment                  │
└─────────────────────────────────────────┘
```

**Tool Selection Algorithm**:

```python
def select_best_tool(criteria, available_tools, context):
    # 1. Filter by capabilities
    capable_tools = filter_by_capability(available_tools, criteria)
    
    # 2. Multi-criteria scoring
    scores = {}
    for tool in capable_tools:
        scores[tool] = {
            'capability': score_capability(tool, criteria),
            'performance': score_performance(tool),
            'resource': score_resource_efficiency(tool, context),
            'preference': score_user_preference(tool, context)
        }
    
    # 3. Weighted aggregation
    weights = {
        'capability': 0.35,
        'performance': 0.25,
        'resource': 0.20,
        'preference': 0.20
    }
    
    overall_scores = calculate_weighted_scores(scores, weights)
    
    # 4. Select best tool with fallbacks
    primary_tool = max(overall_scores, key=overall_scores.get)
    fallback_tools = sorted(overall_scores, key=overall_scores.get, reverse=True)[1:3]
    
    return ToolSelectionResult(
        primary=primary_tool,
        fallbacks=fallback_tools,
        confidence=overall_scores[primary_tool]
    )
```

## Data Flow

### 1. Chat Message Flow

```
User Input
    │
    ▼
┌─────────────────┐
│  API Server     │
│  - Validate     │
│  - Authenticate │
│  - Rate Limit   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Agent Server   │
│  - Load Context │
│  - Plan Tasks   │
│  - Execute      │
└────────┬────────┘
         │
         ├──────────────┐
         │              │
         ▼              ▼
┌─────────────┐  ┌─────────────┐
│ RAG Pipeline│  │Tool Registry│
│ - Search    │  │ - Execute   │
│ - Retrieve  │  │ - Monitor   │
└──────┬──────┘  └──────┬──────┘
       │                │
       └────────┬───────┘
                │
                ▼
       ┌─────────────────┐
       │  LLM Provider   │
       │  - Generate     │
       │  - Stream       │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │  Agent Server   │
       │  - Synthesize   │
       │  - Store Memory │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │  API Server     │
       │  - Format       │
       │  - Stream Back  │
       └────────┬────────┘
                │
                ▼
           User Response
```

### 2. Document Ingestion Flow

```
Document Upload
    │
    ▼
┌─────────────────┐
│  API Server     │
│  - Validate     │
│  - Store Temp   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Pipeline   │
│  - Detect Format│
│  - Extract Text │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Chunking       │
│  - Strategy     │
│  - Hierarchy    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embedding      │
│  - Model Select │
│  - Generate     │
│  - Cache        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Elasticsearch  │
│  - Index Vector │
│  - Store Meta   │
│  - Update Stats │
└────────┬────────┘
         │
         ▼
    Ingestion Complete
```

### 3. Tool Execution Flow

```
Tool Request
    │
    ▼
┌─────────────────┐
│  Agent Server   │
│  - Validate     │
│  - Select Tool  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tool Registry  │
│  - Load Tool    │
│  - Check Res    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Execution Pool │
│  - Allocate     │
│  - Execute      │
│  - Monitor      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tool Instance  │
│  - Validate In  │
│  - Execute Logic│
│  - Return Result│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Agent Server   │
│  - Process Res  │
│  - Update State │
└────────┬────────┘
         │
         ▼
    Tool Result
```

## API Specifications

### REST API Endpoints

**Base URL**: `http://localhost:8000`

#### Authentication

```
POST /auth/login
Request:
{
  "username": "string",
  "password": "string"
}

Response:
{
  "access_token": "string",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### Chat Endpoints

```
POST /api/v1/chat/message
Headers:
  Authorization: Bearer <token>
Request:
{
  "message": "string",
  "session_id": "string",
  "context": {}
}

Response:
{
  "response": "string",
  "session_id": "string",
  "timestamp": "ISO8601",
  "metadata": {}
}
```

```
WebSocket /api/v1/chat/ws/{session_id}
Headers:
  Authorization: Bearer <token>

Messages:
{
  "type": "message|status|error",
  "content": "string",
  "metadata": {}
}
```

#### Document Endpoints

```
POST /api/v1/documents/upload
Headers:
  Authorization: Bearer <token>
  Content-Type: multipart/form-data
Request:
  file: <binary>
  metadata: <json>

Response:
{
  "document_id": "string",
  "status": "success",
  "chunks_processed": 42,
  "processing_time": 1.23
}
```

```
POST /api/v1/documents/search
Headers:
  Authorization: Bearer <token>
Request:
{
  "query": "string",
  "filters": {},
  "max_results": 10,
  "search_type": "hybrid"
}

Response:
{
  "query": "string",
  "results": [
    {
      "chunk_id": "string",
      "content": "string",
      "score": 0.95,
      "metadata": {}
    }
  ],
  "total_results": 42,
  "processing_time": 0.15
}
```

#### Tool Endpoints

```
GET /api/v1/tools
Headers:
  Authorization: Bearer <token>

Response:
{
  "tools": [
    {
      "name": "string",
      "description": "string",
      "capabilities": [],
      "parameters": {}
    }
  ]
}
```

```
POST /api/v1/tools/{tool_name}/execute
Headers:
  Authorization: Bearer <token>
Request:
{
  "parameters": {},
  "session_id": "string"
}

Response:
{
  "tool_name": "string",
  "result": {},
  "status": "success",
  "execution_time": 0.5
}
```

### Internal Service APIs

#### Agent Server API

```
POST /process
Request:
{
  "message": "string",
  "session_id": "string",
  "user_id": "string"
}

Response:
{
  "response": "string",
  "session_id": "string",
  "timestamp": "ISO8601",
  "metadata": {
    "plan_id": "string",
    "tasks_completed": 3,
    "execution_time": 2.5
  }
}
```

#### RAG Pipeline API

```
POST /ingest
Request:
  file: <binary>
  metadata: <json>

Response:
{
  "document_id": "string",
  "chunks_processed": 42,
  "status": "success"
}
```

```
POST /search
Request:
{
  "query": "string",
  "filters": {},
  "max_results": 10
}

Response:
{
  "results": [],
  "total_results": 42,
  "processing_time": 0.15
}
```

## Database Schema

### PostgreSQL Schema

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Sessions table
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    session_token VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Conversations table
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id),
    user_id UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    title VARCHAR(500),
    metadata JSONB
);

-- Messages table
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id),
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    tokens_used INTEGER,
    model_used VARCHAR(100)
);

-- Tool executions table
CREATE TABLE tool_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id),
    tool_name VARCHAR(255) NOT NULL,
    parameters JSONB,
    result JSONB,
    status VARCHAR(50) NOT NULL,
    execution_time FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT
);

-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    filename VARCHAR(500) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size INTEGER,
    content_hash VARCHAR(64) UNIQUE,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    status VARCHAR(50) NOT NULL,
    metadata JSONB,
    chunks_count INTEGER
);

-- Feedback table
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    session_id UUID REFERENCES sessions(id),
    message_id UUID REFERENCES messages(id),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    feedback_text TEXT,
    category VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System metrics table
CREATE TABLE system_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_name VARCHAR(255) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Indexes
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_token ON sessions(session_token);
CREATE INDEX idx_conversations_session_id ON conversations(session_id);
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);
CREATE INDEX idx_tool_executions_session_id ON tool_executions(session_id);
CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_documents_content_hash ON documents(content_hash);
CREATE INDEX idx_feedback_user_id ON feedback(user_id);
CREATE INDEX idx_system_metrics_name_timestamp ON system_metrics(metric_name, timestamp);
```

### Elasticsearch Schema

```json
{
  "mappings": {
    "properties": {
      "document_id": {
        "type": "keyword"
      },
      "chunk_id": {
        "type": "keyword"
      },
      "content": {
        "type": "text",
        "analyzer": "standard"
      },
      "content_vector_general": {
        "type": "dense_vector",
        "dims": 768,
        "index": true,
        "similarity": "cosine"
      },
      "content_vector_domain": {
        "type": "dense_vector",
        "dims": 768,
        "index": true,
        "similarity": "cosine"
      },
      "chunk_type": {
        "type": "keyword"
      },
      "parent_chunk_id": {
        "type": "keyword"
      },
      "metadata": {
        "type": "object",
        "properties": {
          "filename": {"type": "keyword"},
          "file_type": {"type": "keyword"},
          "page_number": {"type": "integer"},
          "section": {"type": "text"},
          "language": {"type": "keyword"},
          "created_at": {"type": "date"},
          "tags": {"type": "keyword"}
        }
      },
      "created_at": {
        "type": "date"
      },
      "updated_at": {
        "type": "date"
      }
    }
  },
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "index": {
      "similarity": {
        "default": {
          "type": "BM25"
        }
      }
    }
  }
}
```

### Redis Data Structures

```
# Session data
session:{session_id} -> Hash
  - user_id
  - created_at
  - last_activity
  - metadata (JSON)

# Rate limiting
rate_limit:{user_id}:{endpoint} -> String (counter)
  TTL: 60 seconds

# Cache
cache:{key} -> String (JSON)
  TTL: configurable

# Conversation context
context:{session_id} -> List (messages)
  TTL: 24 hours

# Tool execution state
tool_state:{execution_id} -> Hash
  - status
  - progress
  - result
  TTL: 1 hour
```

## Integration Patterns

### 1. Service-to-Service Communication

**Pattern**: REST API with Circuit Breaker

```python
class ServiceClient:
    def __init__(self, base_url, timeout=30, max_retries=3):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.circuit_breaker = CircuitBreaker()
    
    async def call_service(self, endpoint, method="GET", data=None):
        if self.circuit_breaker.is_open():
            raise ServiceUnavailableError("Circuit breaker is open")
        
        for attempt in range(self.max_retries):
            try:
                response = await self._make_request(endpoint, method, data)
                self.circuit_breaker.record_success()
                return response
            except Exception as e:
                self.circuit_breaker.record_failure()
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### 2. Event-Driven Communication

**Pattern**: Pub/Sub with Redis

```python
class EventBus:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.subscribers = {}
    
    async def publish(self, event_type, data):
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.redis.publish(event_type, json.dumps(event))
    
    async def subscribe(self, event_type, handler):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
        
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(event_type)
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                event = json.loads(message["data"])
                for handler in self.subscribers[event_type]:
                    await handler(event)
```

### 3. Caching Strategy

**Pattern**: Cache-Aside with TTL

```python
class CacheManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour
    
    async def get_or_compute(self, key, compute_fn, ttl=None):
        # Try cache first
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)
        
        # Compute if not in cache
        result = await compute_fn()
        
        # Store in cache
        await self.redis.setex(
            key,
            ttl or self.default_ttl,
            json.dumps(result)
        )
        
        return result
    
    async def invalidate(self, pattern):
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)
```

### 4. Error Handling Pattern

**Pattern**: Structured Error Responses

```python
class APIError(Exception):
    def __init__(self, message, code, details=None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

@app.exception_handler(APIError)
async def api_error_handler(request, exc):
    return JSONResponse(
        status_code=exc.code,
        content={
            "error": {
                "message": exc.message,
                "code": exc.code,
                "details": exc.details,
                "request_id": request.state.request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )
```

### 5. Authentication Flow

**Pattern**: JWT with Refresh Tokens

```python
class AuthManager:
    def __init__(self, secret_key, algorithm="HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire = 30  # minutes
        self.refresh_token_expire = 7  # days
    
    def create_access_token(self, user_id, role):
        payload = {
            "sub": user_id,
            "role": role,
            "type": "access",
            "exp": datetime.utcnow() + timedelta(minutes=self.access_token_expire)
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id):
        payload = {
            "sub": user_id,
            "type": "refresh",
            "exp": datetime.utcnow() + timedelta(days=self.refresh_token_expire)
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token):
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
```

---

**Document Version**: 1.0  
**Last Updated**: November 2024  
**Authors**: Raveesh Vyas, Prakhar Singhal
