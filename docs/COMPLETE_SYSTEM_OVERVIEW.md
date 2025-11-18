# LMA-o: Complete System Overview

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Technology Stack](#technology-stack)
5. [Key Features](#key-features)
6. [Deployment Architecture](#deployment-architecture)

## Executive Summary

**LMA-o (Learning-Material Agent - omni)** is a sophisticated AI-powered software engineering assistant designed to provide comprehensive support for software development tasks, educational content generation, and knowledge retrieval. The system combines advanced Retrieval-Augmented Generation (RAG) capabilities with agentic workflows powered by LangGraph, offering a scalable, modular architecture that supports multiple LLM providers.

### Authors
- Raveesh Vyas
- Prakhar Singhal

### Project Vision
To create an intelligent, context-aware assistant that can understand complex software engineering queries, retrieve relevant information from a knowledge base, execute tools dynamically, and generate high-quality responses tailored to both technical accuracy and pedagogical clarity.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Web UI      │  │  CLI Tools   │  │  API Clients │         │
│  │  (React)     │  │  (Python)    │  │  (REST/WS)   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway Layer                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  API Server (FastAPI)                                     │  │
│  │  - Authentication & Authorization                         │  │
│  │  - Rate Limiting & Validation                            │  │
│  │  - WebSocket Management                                   │  │
│  │  - Request Routing                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestration Layer                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Agent Server (LangGraph)                                 │  │
│  │  - Task Planning & Decomposition                         │  │
│  │  - Workflow Orchestration                                │  │
│  │  - Tool Selection & Execution                            │  │
│  │  - Memory Management                                      │  │
│  │  - Error Handling & Recovery                             │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   RAG Pipeline Service   │  │   Tool Registry Service  │
│  - Document Ingestion    │  │  - Tool Management       │
│  - Embedding Generation  │  │  - Execution Pool        │
│  - Hybrid Search         │  │  - Performance Monitor   │
│  - Context Optimization  │  │  - Security Scanner      │
└──────────────────────────┘  └──────────────────────────┘
                │                           │
                ▼                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    Data & Storage Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Elasticsearch│  │  PostgreSQL  │  │    Redis     │      │
│  │ (Vector DB)  │  │  (Metadata)  │  │   (Cache)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    External Services                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   OpenAI     │  │  Anthropic   │  │  Google AI   │      │
│  │   (GPT-4)    │  │   (Claude)   │  │  (Gemini)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    Ollama    │  │  Prometheus  │  │   Grafana    │      │
│  │   (Local)    │  │  (Metrics)   │  │ (Monitoring) │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

### Microservices Architecture

The system is composed of six primary microservices:

1. **API Server** (Port 8000)
   - FastAPI-based REST API and WebSocket gateway
   - Handles authentication, rate limiting, and request validation
   - Routes requests to appropriate backend services

2. **Agent Server** (Port 8001)
   - LangGraph-powered orchestration engine
   - Manages conversation state and workflow execution
   - Coordinates tool execution and LLM interactions

3. **RAG Pipeline** (Port 8002)
   - Document processing and ingestion
   - Hybrid search with Elasticsearch
   - Context optimization and reranking

4. **Web UI** (Port 3000)
   - React-based user interface
   - Real-time WebSocket communication
   - Progressive Web App (PWA) support

5. **PostgreSQL Database** (Port 5432)
   - Stores user data, session information
   - Maintains conversation history
   - Tracks system metrics

6. **Elasticsearch** (Port 9200)
   - Vector database for document embeddings
   - Full-text search capabilities
   - Hybrid search with BM25 and dense vectors

## Core Components

### 1. API Server (`src/api_server/`)

**Purpose**: Gateway for all external requests with comprehensive validation and security.

**Key Modules**:
- `main.py`: FastAPI application setup and lifecycle management
- `routers/`: API endpoint definitions
  - `chat.py`: Chat and conversation endpoints
  - `documents.py`: Document management
  - `tools.py`: Tool execution endpoints
  - `auth.py`: Authentication and authorization
  - `admin.py`: Administrative functions
  - `health.py`: Health check endpoints
- `middleware/`: Request/response processing
  - `security.py`: Security headers and CORS
  - `rate_limiting.py`: Rate limiting implementation
  - `logging.py`: Request logging
  - `validation.py`: Input validation
  - `compression.py`: Response compression
- `cache/`: Caching layer with Redis
- `auth/`: JWT-based authentication
- `performance/`: Performance monitoring

**Key Features**:
- JWT-based authentication with role-based access control (RBAC)
- Token bucket rate limiting (configurable per user/global)
- Comprehensive input validation and sanitization
- WebSocket support for real-time communication
- Response compression (Brotli/Gzip)
- Structured logging with request IDs
- Health monitoring and metrics collection

### 2. Agent Server (`src/agent_server/`)

**Purpose**: Core orchestration engine managing conversations and tool execution.

**Key Modules**:
- `main.py`: Agent server initialization and FastAPI endpoints
- `orchestrator.py`: LangGraph-based workflow orchestration
- `planning.py`: Task planning and decomposition
- `memory.py`: Conversation and user memory management
- `tools/`: Tool registry and execution
  - `registry.py`: Tool registration and discovery
  - `tool_registry.py`: Tool lifecycle management
  - `knowledge_retrieval.py`: RAG integration
  - `document_generation.py`: Document creation (DOCX, PDF, PPT)
  - `code_tool_generator.py`: Dynamic code tool generation
  - `compiler_runtime.py`: Code execution sandbox
  - `email_automation.py`: Email sending capabilities
  - `readability_scoring.py`: Content quality assessment
- `code_generation.py`: Code generation capabilities
- `content_generation.py`: Content creation
- `educational_content.py`: Educational material generation
- `feedback_system.py`: User feedback collection
- `feedback_learning.py`: Learning from feedback
- `prompt_engineering.py`: Prompt optimization

**Key Features**:
- LangGraph-powered stateful workflows
- Hierarchical task decomposition
- Dynamic tool selection with multi-criteria optimization
- Conversation memory with context management
- Error handling and recovery strategies
- Feedback collection and learning system
- Checkpoint-based state persistence
- Parallel task execution

### 3. RAG Pipeline (`src/rag_pipeline/`)

**Purpose**: Specialized retrieval system optimized for software engineering content.

**Key Modules**:
- `main.py`: RAG pipeline orchestration
- `document_processor.py`: Multi-format document parsing
- `document_ingestion.py`: Document ingestion service
- `optimized_ingestion.py`: High-performance batch ingestion (5-10x faster)
- `embedding_manager.py`: Dual embedding model management
- `hybrid_embeddings.py`: Intelligent embedding model selection
- `vector_store.py`: Elasticsearch integration
- `search_engine.py`: Hybrid search implementation
- `query_processor.py`: Query reformulation and expansion
- `adaptive_retrieval.py`: Adaptive retrieval strategies
- `context_optimizer.py`: Context compression and optimization
- `reranker.py`: Result reranking with cross-encoders
- `chunking_strategies.py`: Content-aware chunking
- `models.py`: Data models for RAG pipeline

**Key Features**:
- Multi-format document support (PDF, DOCX, PPTX, MD, TXT)
- Dual embedding models:
  - General: `sentence-transformers/all-mpnet-base-v2`
  - Domain-specific: `microsoft/graphcodebert-base`
- Hybrid search combining:
  - Dense vector similarity (cosine)
  - Sparse keyword matching (BM25)
  - Reciprocal Rank Fusion (RRF) for score combination
- Query reformulation and expansion
- Adaptive retrieval with quality assessment
- Context optimization with MMR (Maximal Marginal Relevance)
- BGE reranker for final relevance scoring
- Hierarchical chunking with parent-child relationships
- Optimized batch ingestion (5-10x performance improvement)

### 4. Shared Components (`src/shared/`)

**Purpose**: Common utilities and services used across all components.

**Key Modules**:
- `config.py`: Centralized configuration management
- `logging.py`: Structured logging with structlog
- `metrics.py`: Prometheus metrics collection
- `health.py`: Health check utilities
- `models.py`: Shared data models
- `services.py`: Service registry for inter-service communication
- `session_manager.py`: Session management
- `validation.py`: Input validation utilities
- `startup.py`: Common startup procedures
- `llm/`: LLM integration layer
  - `integration.py`: High-level LLM integration
  - `client.py`: Unified LLM client
  - `factory.py`: LLM client factory
  - `models.py`: LLM data models
  - `providers/`: Provider implementations
    - `openai.py`: OpenAI integration
    - `anthropic.py`: Anthropic (Claude) integration
    - `google.py`: Google AI (Gemini) integration
    - `ollama.py`: Ollama (local) integration
- `database/`: Database utilities
  - `connection.py`: Database connection management
  - `models.py`: Database models
  - `operations.py`: Common database operations

**Key Features**:
- Multi-provider LLM support with unified interface
- Environment-based configuration with validation
- Structured logging with context
- Prometheus metrics integration
- Health check framework
- Service discovery and communication
- Database connection pooling
- Session management with Redis

### 5. Web UI (`ui/`)

**Purpose**: Comprehensive user interface for all system features.

**Technology Stack**:
- React 18 with TypeScript
- Material-UI (MUI) for components
- Redux for state management
- Redux Persist for state persistence
- React Router for navigation
- Axios for HTTP requests
- Socket.IO for WebSocket communication
- React Markdown for message rendering
- React Syntax Highlighter for code display
- Tailwind CSS for styling

**Key Features**:
- Real-time chat interface with markdown support
- Code syntax highlighting
- File upload with drag-and-drop
- Document generation interface
- Tool execution dashboard
- Admin panel for system monitoring
- Responsive design for mobile/desktop
- Progressive Web App (PWA) capabilities
- Dark/light theme support
- Notification system

## Technology Stack

### Backend
- **Python 3.9+**: Primary programming language
- **FastAPI**: Modern async web framework
- **LangChain**: LLM application framework
- **LangGraph**: Stateful workflow orchestration
- **Pydantic**: Data validation and settings management
- **SQLAlchemy**: ORM for database operations
- **Alembic**: Database migrations

### Data Storage
- **PostgreSQL 15**: Relational database for structured data
- **Elasticsearch 8.11**: Vector database and search engine
- **Redis 7**: Caching and session storage

### LLM Providers
- **OpenAI**: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
- **Anthropic**: Claude 3 (Opus, Sonnet, Haiku), Claude 2.1
- **Google AI**: Gemini Pro, Gemini 1.5 Pro/Flash
- **Ollama**: Local deployment (Llama 2, Mistral, Code Llama, etc.)

### Machine Learning
- **sentence-transformers**: Text embeddings
- **transformers**: Hugging Face models
- **torch**: PyTorch for model inference

### Document Processing
- **pypdf**: PDF parsing
- **python-docx**: Word document handling
- **python-pptx**: PowerPoint generation
- **markdown**: Markdown processing
- **reportlab**: PDF generation

### Frontend
- **React 18**: UI framework
- **TypeScript**: Type-safe JavaScript
- **Material-UI**: Component library
- **Redux**: State management
- **Socket.IO**: Real-time communication
- **Axios**: HTTP client

### DevOps & Monitoring
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization
- **structlog**: Structured logging

### Testing
- **pytest**: Python testing framework
- **pytest-asyncio**: Async test support
- **pytest-cov**: Code coverage
- **Jest**: JavaScript testing
- **React Testing Library**: React component testing

## Key Features

### 1. Multi-Provider LLM Support

The system provides a unified interface for multiple LLM providers, allowing seamless switching and fallback strategies:

**Supported Providers**:
- **OpenAI**: Industry-leading models with function calling
- **Anthropic**: Large context windows and safety features
- **Google AI**: Multimodal capabilities and cost-effective options
- **Ollama**: Privacy-focused local deployment

**Key Capabilities**:
- Provider-agnostic API
- Automatic fallback on errors
- Provider-specific parameter support
- Usage tracking and metrics
- Cost optimization strategies

### 2. Advanced RAG Pipeline

Sophisticated document retrieval system optimized for software engineering content:

**Document Processing**:
- Multi-format support (PDF, DOCX, PPTX, MD, TXT)
- Intelligent chunking strategies
- Metadata extraction and enrichment
- Hierarchical document structure preservation

**Search Capabilities**:
- Hybrid search (vector + keyword)
- Query reformulation and expansion
- Adaptive retrieval strategies
- Context optimization with MMR
- Cross-encoder reranking

**Performance**:
- Optimized batch ingestion (5-10x faster)
- Embedding caching
- Incremental updates
- Parallel processing

### 3. Agentic Workflows

LangGraph-powered orchestration for complex multi-step tasks:

**Planning**:
- Hierarchical task decomposition
- Dependency analysis
- Resource estimation
- Parallel execution optimization

**Execution**:
- Stateful workflow management
- Checkpoint-based recovery
- Dynamic tool selection
- Error handling and fallback strategies

**Memory**:
- Conversation history
- User preferences
- Tool execution state
- Long-term memory

### 4. Comprehensive Tool Integration

Extensible tool system with dynamic registration and execution:

**Built-in Tools**:
- Knowledge Retrieval: RAG-based information retrieval
- Document Generation: DOCX, PDF, PPT creation
- Code Execution: Sandboxed code running
- Email Automation: Automated email sending
- Readability Scoring: Content quality assessment
- Code Tool Generator: Dynamic tool creation

**Tool Features**:
- Dynamic registration and discovery
- Schema-based validation
- Resource management
- Performance monitoring
- Security scanning
- Caching and optimization

### 5. Real-time Communication

WebSocket-based real-time features:

**Chat Interface**:
- Streaming responses
- Typing indicators
- Message history
- File attachments

**Notifications**:
- Task progress updates
- System alerts
- Tool execution status
- Error notifications

### 6. Monitoring and Observability

Comprehensive system monitoring and logging:

**Metrics**:
- Request/response metrics
- LLM usage and costs
- Tool execution performance
- System resource utilization

**Logging**:
- Structured logging with context
- Request tracing
- Error tracking
- Audit logs

**Dashboards**:
- Grafana dashboards for visualization
- Real-time system health
- Performance analytics
- Usage statistics

## Deployment Architecture

### Docker Compose Deployment

The system uses Docker Compose for orchestration of all services:

**Services**:
1. `api-server`: API Gateway (Port 8000)
2. `agent-server`: Agent Orchestration (Port 8001)
3. `rag-pipeline`: RAG Service (Port 8002)
4. `web-ui`: Frontend (Port 3000)
5. `postgres`: Database (Port 5432)
6. `redis`: Cache (Port 6379)
7. `elasticsearch`: Search Engine (Port 9200)
8. `prometheus`: Metrics (Port 9090)
9. `grafana`: Monitoring (Port 3001)

**Network**:
- Custom bridge network (`se-sme-network`)
- Subnet: 172.28.0.0/16
- Internal service communication
- External access through API Gateway

**Volumes**:
- `postgres_data`: Database persistence
- `redis_data`: Cache persistence
- `elasticsearch_data`: Search index persistence
- `prometheus_data`: Metrics storage
- `grafana_data`: Dashboard configuration
- `./uploads`: File uploads
- `./logs`: Application logs
- `./data`: RAG pipeline data

**Resource Limits**:
- API Server: 2 CPU, 2GB RAM
- Agent Server: 2 CPU, 3GB RAM
- RAG Pipeline: 4 CPU, 4GB RAM
- Elasticsearch: 2 CPU, 2GB RAM
- PostgreSQL: 2 CPU, 1GB RAM

### Environment Configuration

**Required Environment Variables**:
```bash
# LLM Configuration
LLM_PROVIDER=openai|anthropic|google|ollama
LLM_MODEL=model-name
LLM_API_KEY=your-api-key

# Provider-specific Keys
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key

# Local Deployment
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Database
DATABASE_URL=postgresql://user:pass@host:port/db
REDIS_URL=redis://host:port/db

# Elasticsearch
ELASTICSEARCH_HOST=elasticsearch
ELASTICSEARCH_PORT=9200

# Security
SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Application
ENVIRONMENT=development|production
LOG_LEVEL=INFO|DEBUG
```

### Scaling Considerations

**Horizontal Scaling**:
- API Server: Multiple instances behind load balancer
- Agent Server: Stateless with Redis-backed state
- RAG Pipeline: Parallel document processing

**Vertical Scaling**:
- Elasticsearch: Increase heap size for large datasets
- PostgreSQL: Tune connection pool and cache sizes
- Redis: Increase memory for larger caches

**Performance Optimization**:
- Enable response caching
- Use connection pooling
- Implement request batching
- Optimize database queries
- Use CDN for static assets

### Security

**Authentication**:
- JWT-based authentication
- Role-based access control (RBAC)
- API key management
- Session management

**Network Security**:
- HTTPS/TLS encryption
- CORS configuration
- Rate limiting
- Input validation and sanitization

**Data Security**:
- Encrypted database connections
- Secure credential storage
- Audit logging
- Data encryption at rest

### Monitoring and Maintenance

**Health Checks**:
- Service health endpoints
- Database connectivity checks
- External service validation
- Resource utilization monitoring

**Backup Strategy**:
- Automated database backups
- Elasticsearch snapshot management
- Configuration backups
- Disaster recovery procedures

**Logging**:
- Centralized log aggregation
- Log rotation and retention
- Error tracking and alerting
- Performance profiling

---

**Document Version**: 1.0  
**Last Updated**: November 2024  
**Authors**: Raveesh Vyas, Prakhar Singhal
