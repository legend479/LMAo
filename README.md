# LMA-o : Learning-Material Agent (omni)
## Authors
* Raveesh Vyas
* Prakhar Singhal


A sophisticated AI-powered software engineering assistant with multi-provider LLM support, advanced RAG capabilities, and comprehensive tool integration.

## 🚀 Features

- **Multi-Provider LLM Support**: OpenAI, Anthropic (Claude), Google AI (Gemini), and Ollama (local deployment)
- **Advanced RAG Pipeline**: Intelligent document processing and retrieval
- **Agentic Workflows**: LangGraph-powered task orchestration
- **Comprehensive Tool Integration**: Code execution, analysis, and automation
- **Real-time Chat Interface**: WebSocket-based communication
- **Scalable Architecture**: Microservices with Docker deployment

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

- [LLM Integration Guide](docs/LLM_INTEGRATION.md) - Comprehensive guide for LLM usage
- [Quick Setup Guide](QUICK_SETUP_GUIDE.md) - Fast deployment instructions
- [System Documentation](SYSTEM_DOCUMENTATION.md) - Architecture and design
- [Design and Organization](DESIGN_AND_ORGANIZATION.md) - System design philosophy

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
        model="gemini-pro"
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

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web UI        │────│   API Server     │────│  Agent Server   │
│   (React)       │    │   (FastAPI)      │    │  (LangGraph)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                       ┌──────────────────┐    ┌─────────────────┐
                       │  RAG Pipeline    │    │  LLM Providers  │
                       │ (Elasticsearch)  │    │ Multi-Provider  │
                       └──────────────────┘    └─────────────────┘
```

## 🔍 Key Components

- **API Server**: FastAPI-based REST API and WebSocket gateway
- **Agent Server**: LangGraph orchestration with multi-provider LLM support
- **RAG Pipeline**: Elasticsearch-powered document processing and retrieval
- **LLM Integration**: Unified interface for OpenAI, Anthropic, Google AI, and Ollama
- **Web UI**: React-based user interface
- **Tool Registry**: Extensible tool system for various tasks

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

## 🆘 Support

- Check [LLM Integration Guide](docs/LLM_INTEGRATION.md) for LLM-specific issues
- Run `python test_llm_integration.py` to diagnose problems
- Review logs for detailed error information
- See [Troubleshooting](docs/LLM_INTEGRATION.md#troubleshooting) section
