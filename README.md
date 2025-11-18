# LMA-o : Learning-Material Agent (omni)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![Documentation](https://img.shields.io/badge/docs-complete-success.svg)](./docs/)

## Authors
* **Raveesh Vyas**
* **Prakhar Singhal**

---

A sophisticated AI-powered software engineering assistant combining multi-provider LLM support, advanced RAG capabilities, and comprehensive tool integration. Built with modern microservices architecture for scalability and reliability.

## 🌟 Key Highlights

- **🤖 Multi-Provider LLM Support**: Seamlessly switch between OpenAI, Anthropic (Claude), Google AI (Gemini), and Ollama (local deployment)
- **📚 Advanced RAG Pipeline**: Hybrid search with dual embedding models, query reformulation, and intelligent reranking
- **🔄 Agentic Workflows**: LangGraph-powered stateful orchestration with dynamic tool selection
- **🛠️ Comprehensive Tool Integration**: 10+ built-in tools including code execution, document generation, and email automation
- **💬 Real-time Chat Interface**: WebSocket-based communication with streaming responses
- **🏗️ Scalable Architecture**: Microservices design with Docker Compose orchestration
- **📊 Production-Ready**: Complete monitoring, logging, and observability stack
- **🔒 Enterprise Security**: JWT authentication, RBAC, rate limiting, and input validation

## 🤖 Supported LLM Providers

### Cloud Providers
- **OpenAI**: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
- **Anthropic**: Claude 3 (Opus, Sonnet, Haiku), Claude 2.1
- **Google AI**: Gemini Pro, Gemini Pro Vision, Gemini 1.5 Pro/Flash

### Local Deployment
- **Ollama**: Llama 2, Code Llama, Mistral, Mixtral, Phi, and more

## 📋 Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd LMAo
cp .env.example .env
# Edit .env with your configuration
```

### 2. Configure LLM Providers
```bash
# Choose your primary provider
LLM_PROVIDER=openai  # or anthropic, google, ollama

# Add API keys
OPENAI_API_KEY=your-openai-key
GOOGLE_API_KEY=your-google-key
ANTHROPIC_API_KEY=your-anthropic-key

# For local deployment with Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

### 3. Start with Docker
```bash
docker-compose up -d
```

### 4. Test LLM Integration
```bash
python test_llm_integration.py
```

## 🔧 Local Development with Ollama

For privacy-focused local deployment:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve

# Pull models
ollama pull llama2
ollama pull codellama

# Configure for local use
LLM_PROVIDER=ollama
LLM_MODEL=llama2
```

## 📚 Documentation

### 📑 [Documentation Index](docs/INDEX.md) - Complete documentation navigation

### Core Documentation
- **[Complete System Overview](docs/COMPLETE_SYSTEM_OVERVIEW.md)** - Comprehensive system architecture and features
- **[Technical Architecture](docs/TECHNICAL_ARCHITECTURE.md)** - Detailed technical design and data flows
- **[Features and Capabilities](docs/FEATURES_AND_CAPABILITIES.md)** - Complete feature documentation
- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation with examples
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - Production deployment and configuration
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Development setup and contribution guidelines

### Specialized Guides
- **[LLM Integration Guide](docs/LLM_INTEGRATION.md)** - Multi-provider LLM usage and configuration
- **[Architecture Design](docs/Architecture.md)** - System design philosophy and patterns

### Quick Links
- [Installation Guide](docs/DEPLOYMENT_GUIDE.md#quick-start)
- [API Authentication](docs/API_REFERENCE.md#authentication)
- [Development Setup](docs/DEVELOPER_GUIDE.md#development-setup)
- [Troubleshooting](docs/DEPLOYMENT_GUIDE.md#troubleshooting)

## 🛠️ Usage Examples

### Simple Text Generation
```python
from src.shared.llm.integration import generate_text

response = await generate_text(
    prompt="Explain microservices architecture",
    system_prompt="You are a software engineering expert.",
    temperature=0.7
)
```

### Provider-Specific Usage
```python
from src.shared.llm.integration import get_llm_integration
from src.shared.llm.models import LLMProvider

integration = await get_llm_integration()

# Use Google AI specifically
async with integration.use_provider(LLMProvider.GOOGLE):
    response = await integration.generate_response(
        prompt="Analyze this code for security issues",
        model="gemini-2.5-flash"
    )
```

### Streaming Responses
```python
from src.shared.llm.integration import stream_text

async for chunk in stream_text(
    prompt="Write a Python function for binary search",
    temperature=0.3
):
    print(chunk, end="", flush=True)
```

## 🏗️ System Architecture

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
│                      API Gateway (FastAPI)                       │
│  Authentication • Rate Limiting • Validation • WebSocket         │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   Agent Server           │  │   RAG Pipeline           │
│   (LangGraph)            │  │   (Elasticsearch)        │
│  • Task Planning         │  │  • Document Processing   │
│  • Tool Orchestration    │  │  • Hybrid Search         │
│  • Memory Management     │  │  • Context Optimization  │
│  • Error Recovery        │  │  • Reranking             │
└──────────────────────────┘  └──────────────────────────┘
                │                           │
                └─────────────┬─────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data & Storage Layer                          │
│  PostgreSQL • Redis • Elasticsearch • Prometheus • Grafana       │
└─────────────────────────────────────────────────────────────────┘
```

### Microservices
- **API Server** (Port 8000): REST API and WebSocket gateway
- **Agent Server** (Port 8001): LangGraph orchestration engine
- **RAG Pipeline** (Port 8002): Document processing and retrieval
- **Web UI** (Port 3000): React-based user interface
- **PostgreSQL** (Port 5432): Relational database
- **Redis** (Port 6379): Caching and session storage
- **Elasticsearch** (Port 9200): Vector database and search
- **Prometheus** (Port 9090): Metrics collection
- **Grafana** (Port 3001): Monitoring dashboards

## 🔍 Core Components

### Backend Services
- **API Server** (`src/api_server/`): FastAPI gateway with authentication, rate limiting, and WebSocket support
- **Agent Server** (`src/agent_server/`): LangGraph orchestration with task planning and tool execution
- **RAG Pipeline** (`src/rag_pipeline/`): Elasticsearch-powered hybrid search with dual embeddings
- **LLM Integration** (`src/shared/llm/`): Unified interface for multiple LLM providers

### Frontend
- **Web UI** (`ui/`): React + TypeScript with Material-UI, Redux, and real-time WebSocket

### Data Layer
- **PostgreSQL**: User data, sessions, conversations, and system metrics
- **Redis**: Caching, rate limiting, and session storage
- **Elasticsearch**: Vector database with hybrid search capabilities

### Monitoring
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization dashboards and analytics

## 🚦 Health Check

```bash
# Check system status
curl http://localhost:8000/health

# Test LLM providers
curl -X POST http://localhost:8000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "session_id": "test"}'
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support and Resources

### Documentation
- **[Documentation Index](docs/INDEX.md)** - Navigate all documentation
- **[Troubleshooting Guide](docs/DEPLOYMENT_GUIDE.md#troubleshooting)** - Common issues and solutions
- **[LLM Integration Guide](docs/LLM_INTEGRATION.md)** - LLM-specific issues
- **[API Reference](docs/API_REFERENCE.md)** - API usage and examples

### Getting Help
- **Issues**: [GitHub Issues](https://github.com/yourusername/LMAo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/LMAo/discussions)
- **Diagnostics**: Run `python test_llm_integration.py` to diagnose LLM issues
- **Logs**: Check `logs/` directory for detailed error information

### Community
- Star the project on GitHub
- Share your use cases and feedback
- Contribute to documentation
- Report bugs and suggest features
