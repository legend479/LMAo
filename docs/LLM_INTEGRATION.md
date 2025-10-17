# LLM Integration Guide

This document explains how to use the integrated LLM system in the SE SME Agent, which now supports OpenAI, Anthropic, Google AI (Gemini), and Ollama (local deployment).

## Overview

The LLM integration provides a unified interface for multiple LLM providers, allowing you to:

- Switch between different providers seamlessly
- Use provider-specific features and parameters
- Handle errors and fallbacks gracefully
- Monitor usage and performance metrics
- Deploy locally with Ollama or use cloud APIs

## Supported Providers

### 1. OpenAI
- **Models**: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, etc.
- **Features**: Function calling, vision (GPT-4V), streaming
- **Configuration**: Requires API key

### 2. Anthropic (Claude)
- **Models**: Claude 3 (Opus, Sonnet, Haiku), Claude 2.1, Claude 2.0
- **Features**: Large context windows, streaming, safety features
- **Configuration**: Requires API key

### 3. Google AI (Gemini)
- **Models**: Gemini Pro, Gemini Pro Vision, Gemini 1.5 Pro/Flash
- **Features**: Multimodal capabilities, large context windows, safety settings
- **Configuration**: Requires API key

### 4. Ollama (Local)
- **Models**: Llama 2, Code Llama, Mistral, Mixtral, Phi, etc.
- **Features**: Local deployment, privacy, no API costs
- **Configuration**: Requires Ollama server running locally

## Quick Start

### 1. Environment Setup

Copy `.env.example` to `.env` and configure your providers:

```bash
# Primary LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
LLM_API_KEY=your-primary-api-key

# Provider-specific API Keys
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-ai-api-key

# Local/Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Test the Integration

```bash
python test_llm_integration.py
```

## Usage Examples

### Simple Text Generation

```python
from src.shared.llm.integration import generate_text

# Generate text with default provider
response = await generate_text(
    prompt="Explain microservices architecture",
    system_prompt="You are a software engineering expert.",
    temperature=0.7
)
print(response)
```

### Using Specific Providers

```python
from src.shared.llm.integration import get_llm_integration
from src.shared.llm.models import LLMProvider

integration = await get_llm_integration()

# Use Google AI specifically
async with integration.use_provider(LLMProvider.GOOGLE):
    response = await integration.generate_response(
        prompt="What are the benefits of containerization?",
        model="gemini-pro",
        temperature=0.5
    )
```

### Streaming Responses

```python
from src.shared.llm.integration import stream_text

async for chunk in stream_text(
    prompt="Write a Python function for binary search",
    system_prompt="You are a coding assistant.",
    temperature=0.3
):
    print(chunk, end="", flush=True)
```

### Multi-turn Conversations

```python
from src.shared.llm.integration import chat_with_llm

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is REST API?"},
]

response = await chat_with_llm(conversation)
print(response['content'])

# Continue conversation
conversation.append({"role": "assistant", "content": response['content']})
conversation.append({"role": "user", "content": "How is it different from GraphQL?"})

response = await chat_with_llm(conversation)
print(response['content'])
```

## Provider-Specific Features

### Google AI (Gemini)

```python
from src.shared.llm import create_google_client

client = await create_google_client(
    api_key="your-google-api-key",
    model="gemini-pro"
)

response = await client.simple_chat(
    message="Analyze this code for security issues",
    extra_params={
        "top_k": 40,
        "safety_settings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
    }
)
```

### Ollama (Local)

```python
from src.shared.llm import create_ollama_client

client = await create_ollama_client(
    model="llama2",
    base_url="http://localhost:11434"
)

# Check available models
models = await client.get_available_models()
print(f"Available models: {models}")

# Use Ollama-specific parameters
response = await client.simple_chat(
    message="Explain design patterns",
    extra_params={
        "top_k": 40,
        "repeat_penalty": 1.1,
        "num_ctx": 4096,  # Context window
        "temperature": 0.7
    }
)
```

## Local Deployment with Ollama

### 1. Install Ollama

```bash
# On macOS
brew install ollama

# On Linux
curl -fsSL https://ollama.ai/install.sh | sh

# On Windows
# Download from https://ollama.ai/download
```

### 2. Start Ollama Server

```bash
ollama serve
```

### 3. Pull Models

```bash
# Pull Llama 2 (7B)
ollama pull llama2

# Pull Code Llama for coding tasks
ollama pull codellama

# Pull Mistral for general tasks
ollama pull mistral

# Pull smaller model for testing
ollama pull phi
```

### 4. Configure for Ollama

```python
# Set environment or update config
LLM_PROVIDER=ollama
LLM_MODEL=llama2
OLLAMA_BASE_URL=http://localhost:11434
```

## Integration with Existing System

### In Agent Server

```python
# src/agent_server/orchestrator.py
from src.shared.llm.integration import get_llm_integration

class LangGraphOrchestrator:
    async def _execute_content_generation_task(self, task, state):
        integration = await get_llm_integration()
        
        response = await integration.generate_response(
            prompt=task.get("prompt", ""),
            system_prompt="You are an expert software engineering assistant.",
            temperature=0.7,
            max_tokens=2000
        )
        
        return {
            "type": "content_generation",
            "result": response,
            "execution_time": 0.2,
        }
```

### In Content Generation

```python
# src/agent_server/content_generation.py
from src.shared.llm.integration import generate_text

async def generate_documentation(self, request):
    prompt = f"""
    Generate documentation for the following code:
    
    {request.code}
    
    Include:
    - Purpose and functionality
    - Parameters and return values
    - Usage examples
    - Best practices
    """
    
    response = await generate_text(
        prompt=prompt,
        system_prompt="You are a technical documentation expert.",
        temperature=0.3,
        max_tokens=1500
    )
    
    return response
```

## Error Handling

```python
from src.shared.llm.models import (
    LLMError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMConnectionError
)

try:
    response = await generate_text("Your prompt here")
except LLMRateLimitError:
    # Handle rate limiting
    await asyncio.sleep(60)  # Wait and retry
except LLMAuthenticationError:
    # Handle invalid API key
    logger.error("Invalid API key")
except LLMConnectionError:
    # Handle connection issues
    logger.error("Connection failed, trying fallback provider")
except LLMError as e:
    # Handle general LLM errors
    logger.error(f"LLM error: {str(e)}")
```

## Monitoring and Metrics

```python
from src.shared.llm.integration import get_llm_integration

integration = await get_llm_integration()

# Get usage metrics
metrics = integration.get_metrics()
print(f"Total requests: {metrics['total_requests']}")
print(f"Total tokens: {metrics['total_tokens']}")
print(f"Available providers: {metrics['available_providers']}")

# Validate providers
validation = await integration.validate_providers()
for provider, is_valid in validation.items():
    print(f"{provider}: {'✓' if is_valid else '✗'}")
```

## Best Practices

### 1. Provider Selection
- **OpenAI**: Best for general tasks, function calling
- **Anthropic**: Best for long-form content, safety-critical applications
- **Google AI**: Best for multimodal tasks, large context needs
- **Ollama**: Best for privacy, cost control, offline usage

### 2. Model Selection
- **Code generation**: Use specialized models (GPT-4, Code Llama)
- **Analysis**: Use reasoning-focused models (Claude, GPT-4)
- **Simple tasks**: Use efficient models (GPT-3.5, Gemini Pro)
- **Local deployment**: Start with smaller models (Llama 2 7B, Phi)

### 3. Performance Optimization
- Use streaming for long responses
- Set appropriate temperature (0.1-0.3 for factual, 0.7-0.9 for creative)
- Limit max_tokens to control costs and latency
- Implement caching for repeated queries

### 4. Error Handling
- Always implement fallback providers
- Handle rate limits with exponential backoff
- Log errors for monitoring and debugging
- Validate API keys before deployment

## Troubleshooting

### Common Issues

1. **"Provider not available"**
   - Check API keys in environment variables
   - Verify provider is properly configured
   - Test connection with validation methods

2. **"Model not found"**
   - Check model name spelling
   - Verify model is available for your provider
   - For Ollama: ensure model is pulled locally

3. **"Connection failed"**
   - For cloud providers: check internet connection
   - For Ollama: ensure server is running (`ollama serve`)
   - Check firewall and proxy settings

4. **"Rate limit exceeded"**
   - Implement exponential backoff
   - Consider upgrading API plan
   - Switch to alternative provider temporarily

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger("src.shared.llm").setLevel(logging.DEBUG)
```

## Migration Guide

### From Direct OpenAI Usage

**Before:**
```python
import openai
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)
```

**After:**
```python
from src.shared.llm.integration import generate_text
response = await generate_text("Hello")
```

### Adding New Providers

To add a new provider:

1. Create provider class in `src/shared/llm/providers/`
2. Implement `BaseLLMProvider` interface
3. Add to factory in `src/shared/llm/factory.py`
4. Update configuration in `src/shared/config.py`
5. Add tests and documentation

## API Reference

See the individual module documentation:
- `src/shared/llm/models.py` - Data models and types
- `src/shared/llm/client.py` - Core client interface
- `src/shared/llm/providers/` - Provider implementations
- `src/shared/llm/factory.py` - Client factory functions
- `src/shared/llm/integration.py` - High-level integration helpers

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Run `python test_llm_integration.py` to diagnose issues
3. Review logs for detailed error information
4. Check provider documentation for API-specific issues