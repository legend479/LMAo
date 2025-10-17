# SE SME Agent - System Documentation

## Overview

The Software Engineering Subject Matter Expert (SE SME) Agent is a sophisticated, multi-component AI system that combines advanced Retrieval-Augmented Generation (RAG) capabilities with agentic workflows and comprehensive tool integration. The system is designed to provide expert-level assistance for software engineering tasks.

## Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web UI        │────│   API Server     │────│  Agent Server   │
│   (React)       │    │   (FastAPI)      │    │  (LangGraph)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                │                        │
                       ┌──────────────────┐    ┌─────────────────┐
                       │  RAG Pipeline    │────│  Tool Registry  │
                       │ (Elasticsearch)  │    │   (Dynamic)     │
                       └──────────────────┘    └─────────────────┘
                                │
                       ┌──────────────────┐
                       │   Data Layer     │
                       │ (PostgreSQL +    │
                       │  Redis + ES)     │
                       └──────────────────┘
```

### Core Components

#### 1. API Server (FastAPI)
- **Purpose**: Gateway for all external requests with robust validation and security
- **Key Features**:
  - Advanced security middleware with prompt injection detection
  - WebSocket support for real-time communication
  - Comprehensive rate limiting and CORS handling
  - Health checks and monitoring endpoints

#### 2. Agent Server (LangGraph)
- **Purpose**: Core orchestration engine managing conversations and tool execution
- **Key Features**:
  - Stateful workflow management with Redis persistence
  - Advanced planning with task decomposition
  - Multi-step reasoning and error recovery
  - Resource management and execution monitoring

#### 3. RAG Pipeline (Elasticsearch)
- **Purpose**: Specialized retrieval system optimized for software engineering content
- **Key Features**:
  - Multi-format document processing (PDF, DOCX, PPTX, MD, TXT)
  - Hybrid search with RRF score fusion
  - Advanced chunking strategies
  - Query expansion and spell correction

#### 4. Tool Registry
- **Purpose**: Dynamic tool registration, discovery, and lifecycle management
- **Key Features**:
  - Multi-criteria tool selection
  - Performance monitoring and analytics
  - Version management
  - Resource allocation and execution pooling

## Key Features

### 1. Advanced Security
- **Prompt Injection Detection**: 20+ patterns for detecting injection attempts
- **Output Moderation**: Content filtering for harmful or sensitive information
- **Traditional Security**: XSS, SQL injection, and other attack prevention
- **Authentication**: JWT-based authentication with role-based access control

### 2. Intelligent Planning
- **Intent Classification**: Automatic detection of user intent and task complexity
- **Entity Extraction**: Recognition of programming languages, frameworks, and concepts
- **Task Decomposition**: Hierarchical breakdown of complex requests
- **Dependency Analysis**: Automatic detection and resolution of task dependencies

### 3. Hybrid Search
- **Vector Search**: Semantic similarity using sentence transformers
- **Keyword Search**: Traditional BM25-based text matching
- **RRF Fusion**: Reciprocal Rank Fusion for optimal result combination
- **Reranking**: BGE cross-encoder for final relevance scoring

### 4. Tool Ecosystem
- **Knowledge Retrieval**: Advanced RAG-based information retrieval
- **Document Generation**: DOCX, PDF, and PowerPoint generation
- **Code Generation**: Programming assistance and code creation
- **Analysis Tools**: Code review and quality assessment

### 5. Real-time Communication
- **WebSocket Support**: Bidirectional real-time communication
- **Connection Management**: Automatic reconnection and session handling
- **Typing Indicators**: Real-time typing status
- **Message History**: Persistent conversation storage

## Technology Stack

### Backend
- **Python 3.9+**: Core programming language
- **FastAPI**: Modern, fast web framework
- **LangGraph**: Stateful workflow orchestration
- **LangChain**: LLM integration and tooling
- **Pydantic**: Data validation and settings management

### Data Storage
- **PostgreSQL**: Primary relational database
- **Redis**: Caching and session storage
- **Elasticsearch**: Vector search and document storage

### Document Processing
- **PyPDF2**: PDF text extraction
- **python-docx**: Word document processing
- **python-pptx**: PowerPoint processing
- **ReportLab**: PDF generation

### AI/ML
- **Sentence Transformers**: Text embeddings
- **Transformers**: NLP model integration
- **BGE Reranker**: Result reranking

### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Multi-service orchestration
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards

## Configuration

### Environment Variables

```bash
# Application
ENVIRONMENT=development
APP_NAME=SE SME Agent
VERSION=1.0.0
DEBUG=true

# API Server
API_HOST=0.0.0.0
API_PORT=8000
ALLOWED_ORIGINS=["http://localhost:3000"]

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/se_sme_agent
REDIS_URL=redis://localhost:6379/0

# Elasticsearch
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_INDEX=se_sme_documents

# Security
SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
LLM_API_KEY=your-openai-api-key-here
```

### Configuration Management
- Environment-specific settings (development, staging, production)
- Comprehensive validation with detailed error messages
- Automatic configuration discovery and validation
- Support for both environment variables and config files

## API Endpoints

### Health and Status
- `GET /health/` - Overall system health
- `GET /health/ready` - Readiness check
- `GET /health/live` - Liveness check

### Authentication
- `POST /auth/login` - User authentication
- `POST /auth/refresh` - Token refresh
- `POST /auth/logout` - User logout

### Chat and Conversation
- `POST /api/v1/chat/message` - Send message to agent
- `WebSocket /api/v1/chat/ws/{session_id}` - Real-time chat
- `GET /api/v1/chat/sessions` - List user sessions
- `GET /api/v1/chat/sessions/{session_id}/history` - Chat history

### Document Management
- `POST /api/v1/documents/upload` - Upload documents
- `GET /api/v1/documents/` - List documents
- `DELETE /api/v1/documents/{doc_id}` - Delete document

### Tool Management
- `GET /api/v1/tools/` - List available tools
- `POST /api/v1/tools/{tool_name}/execute` - Execute tool
- `GET /api/v1/tools/{tool_name}/schema` - Get tool schema

### Administration
- `GET /admin/stats` - System statistics
- `GET /admin/users` - User management
- `POST /admin/tools/register` - Register new tool

## Data Models

### Core Models
- **User**: User account and preferences
- **Session**: Chat session management
- **Message**: Individual chat messages
- **Document**: Uploaded document metadata
- **Tool**: Tool registration and metadata

### Agent Models
- **ExecutionPlan**: Task execution planning
- **WorkflowState**: LangGraph state management
- **ToolResult**: Tool execution results
- **ExecutionContext**: Execution environment

### RAG Models
- **ProcessedDocument**: Document processing results
- **Chunk**: Document chunk with embeddings
- **SearchResult**: Search result with relevance scores
- **SearchResponse**: Complete search response

## Security Features

### Input Validation
- Comprehensive parameter validation using Pydantic
- SQL injection prevention
- XSS protection
- File upload validation

### Prompt Injection Protection
- Pattern-based detection (20+ patterns)
- Context manipulation detection
- Encoding/obfuscation detection
- Social engineering attempt detection

### Output Moderation
- Harmful content filtering
- Sensitive information detection
- Personal data protection
- System information leakage prevention

### Authentication & Authorization
- JWT-based authentication
- Role-based access control
- Session management
- Rate limiting per user

## Performance Features

### Caching Strategy
- Redis-based session caching
- Query result caching with TTL
- Document processing cache
- Tool execution result cache

### Resource Management
- Concurrent execution limits
- Memory usage monitoring
- CPU usage tracking
- Automatic resource cleanup

### Optimization
- Async/await throughout the codebase
- Connection pooling for databases
- Lazy loading of heavy components
- Efficient chunking strategies

## Monitoring and Observability

### Metrics Collection
- Request/response metrics
- Tool execution metrics
- System resource usage
- Error rates and patterns

### Logging
- Structured logging with JSON format
- Multiple log levels and categories
- Request tracing and correlation IDs
- Security event logging

### Health Checks
- Component-level health monitoring
- Dependency health verification
- Automatic failover capabilities
- Performance threshold monitoring

## Error Handling

### Graceful Degradation
- Fallback mechanisms for failed components
- Partial functionality when services are down
- User-friendly error messages
- Automatic retry with exponential backoff

### Error Recovery
- Checkpoint-based workflow recovery
- Transaction rollback capabilities
- State restoration mechanisms
- Comprehensive error logging

## Deployment Architecture

### Development Environment
```bash
# Start development services
make dev

# Run tests
make test

# Format and lint code
make format lint
```

### Production Deployment
```bash
# Build and deploy
make docker-build
make docker-up

# Monitor services
make monitor

# View logs
make docker-logs
```

### Service Dependencies
1. **PostgreSQL**: Primary database
2. **Redis**: Caching and sessions
3. **Elasticsearch**: Search and vectors
4. **Prometheus**: Metrics collection
5. **Grafana**: Monitoring dashboards

## Scalability Considerations

### Horizontal Scaling
- Stateless API servers
- Load balancer support
- Database connection pooling
- Distributed caching

### Vertical Scaling
- Resource monitoring and alerting
- Automatic scaling triggers
- Performance optimization
- Memory management

### Data Scaling
- Elasticsearch cluster support
- Database partitioning strategies
- Efficient indexing
- Data archival policies

## Maintenance and Operations

### Regular Maintenance
- Database optimization and cleanup
- Index maintenance and rebuilding
- Log rotation and archival
- Security updates and patches

### Backup and Recovery
- Automated database backups
- Document storage backups
- Configuration backups
- Disaster recovery procedures

### Monitoring and Alerting
- System health monitoring
- Performance threshold alerts
- Error rate monitoring
- Security incident detection

## Development Guidelines

### Code Standards
- Python PEP 8 compliance
- Type hints throughout
- Comprehensive docstrings
- Unit test coverage > 80%

### Git Workflow
- Feature branch development
- Pull request reviews
- Automated testing on CI/CD
- Semantic versioning

### Testing Strategy
- Unit tests for all components
- Integration tests for workflows
- End-to-end testing
- Performance testing

## Troubleshooting

### Common Issues
1. **Service Connection Failures**: Check network connectivity and service health
2. **Authentication Errors**: Verify JWT configuration and token expiration
3. **Search Performance**: Monitor Elasticsearch cluster health and indexing
4. **Memory Issues**: Check resource usage and optimize chunking strategies

### Debug Mode
- Enable debug logging: `DEBUG=true`
- Access debug endpoints: `/debug/`
- Monitor resource usage: `/admin/stats`
- Check service health: `/health/`

## Future Enhancements

### Planned Features
- Multi-language support
- Advanced analytics dashboard
- Custom tool development SDK
- Enterprise SSO integration

### Scalability Improvements
- Microservices architecture
- Kubernetes deployment
- Advanced caching strategies
- Performance optimization

### AI/ML Enhancements
- Custom model fine-tuning
- Advanced reasoning capabilities
- Multi-modal support
- Improved context understanding