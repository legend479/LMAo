# LMA-o: Quick Reference Guide

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/LMAo.git
cd LMAo

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start services
docker-compose up -d

# Verify installation
curl http://localhost:8000/health
```

## Configuration

### Essential Environment Variables

```bash
# LLM Provider
LLM_PROVIDER=openai|anthropic|google|ollama
LLM_API_KEY=your-api-key
LLM_MODEL=model-name

# Security
SECRET_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')

# Database
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379/0
ELASTICSEARCH_HOST=elasticsearch
```

## API Quick Start

### Authentication

```bash
# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user@example.com", "password": "password"}'

# Response
{
  "access_token": "eyJhbGc...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Send Chat Message

```bash
curl -X POST http://localhost:8000/api/v1/chat/message \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain microservices",
    "session_id": "my-session"
  }'
```

### Upload Document

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@document.pdf" \
  -F 'metadata={"title": "My Document"}'
```

### Search Documents

```bash
curl -X POST http://localhost:8000/api/v1/documents/search \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication in FastAPI",
    "max_results": 10
  }'
```

## Python SDK Usage

### Basic Chat

```python
from src.shared.llm.integration import generate_text

response = await generate_text(
    prompt="Explain microservices architecture",
    system_prompt="You are a software engineering expert.",
    temperature=0.7
)
print(response)
```

### Streaming Response

```python
from src.shared.llm.integration import stream_text

async for chunk in stream_text(
    prompt="Write a Python function for binary search",
    temperature=0.3
):
    print(chunk, end="", flush=True)
```

### Provider-Specific Usage

```python
from src.shared.llm.integration import get_llm_integration
from src.shared.llm.models import LLMProvider

integration = await get_llm_integration()

# Use specific provider
async with integration.use_provider(LLMProvider.GOOGLE):
    response = await integration.generate_response(
        prompt="Analyze this code",
        model="gemini-1.5-pro"
    )
```

## Docker Commands

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f [service-name]

# Restart service
docker-compose restart [service-name]

# Scale service
docker-compose up -d --scale api-server=3

# Check status
docker-compose ps

# Execute command in container
docker-compose exec api-server bash
```

## Common Tasks

### Create Admin User

```bash
docker-compose exec api-server python -m src.scripts.create_admin \
  --username admin@example.com \
  --password secure_password \
  --role admin
```

### Database Backup

```bash
docker-compose exec -T postgres pg_dump -U postgres lmao_prod | \
  gzip > backup_$(date +%Y%m%d).sql.gz
```

### Database Restore

```bash
gunzip < backup_20241119.sql.gz | \
  docker-compose exec -T postgres psql -U postgres lmao_prod
```

### View Metrics

```bash
# Prometheus
open http://localhost:9090

# Grafana
open http://localhost:3001
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs [service-name]

# Check port usage
sudo netstat -tulpn | grep [port]

# Verify configuration
docker-compose config

# Check resources
docker stats
```

### Database Connection Error

```bash
# Check database status
docker-compose exec postgres pg_isready

# Verify credentials
docker-compose exec postgres psql -U postgres -l

# Reset database
docker-compose down -v
docker-compose up -d
```

### LLM API Error

```bash
# Test API connection
python test_llm_integration.py

# Verify API key
echo $OPENAI_API_KEY

# Check provider status
curl https://status.openai.com/api/v2/status.json
```

## Performance Tuning

### Database Optimization

```sql
-- Connect to database
docker-compose exec postgres psql -U postgres lmao_prod

-- Analyze tables
ANALYZE;

-- Vacuum
VACUUM ANALYZE;

-- Reindex
REINDEX DATABASE lmao_prod;
```

### Elasticsearch Optimization

```bash
# Force merge
curl -X POST "localhost:9200/lmao_documents/_forcemerge?max_num_segments=1"

# Clear cache
curl -X POST "localhost:9200/_cache/clear"

# Check cluster health
curl http://localhost:9200/_cluster/health
```

### Redis Optimization

```bash
# Check memory usage
docker-compose exec redis redis-cli INFO memory

# Clear cache
docker-compose exec redis redis-cli FLUSHDB

# Monitor commands
docker-compose exec redis redis-cli MONITOR
```

## Monitoring

### Health Checks

```bash
# API Server
curl http://localhost:8000/health

# Agent Server
curl http://localhost:8001/health

# RAG Pipeline
curl http://localhost:8002/health
```

### Metrics

```bash
# System stats
curl http://localhost:8000/admin/stats

# Prometheus metrics
curl http://localhost:9090/api/v1/query?query=up

# Service metrics
curl http://localhost:8000/metrics
```

## Development

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest

# Format code
black src/
isort src/

# Lint
flake8 src/
mypy src/
```

### Frontend Development

```bash
cd ui

# Install dependencies
npm install

# Start development server
npm start

# Run tests
npm test

# Build for production
npm run build
```

## Useful Commands

### System Information

```bash
# Check versions
docker --version
docker-compose --version
python --version
node --version

# Check disk space
df -h

# Check memory
free -h

# Check CPU
top
```

### Log Management

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api-server

# Save logs to file
docker-compose logs > logs/combined.log

# Clear logs
docker-compose down
docker system prune -a
```

## Quick Links

- [Full Documentation](INDEX.md)
- [API Reference](API_REFERENCE.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Developer Guide](DEVELOPER_GUIDE.md)
- [Troubleshooting](DEPLOYMENT_GUIDE.md#troubleshooting)

---

**Document Version**: 1.0  
**Last Updated**: November 2024  
**Authors**: Raveesh Vyas, Prakhar Singhal
