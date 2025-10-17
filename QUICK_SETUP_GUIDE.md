# SE SME Agent - Quick Setup Guide

## Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **Node.js**: 16 or higher (for UI development)
- **Docker**: 20.10 or higher
- **Docker Compose**: 2.0 or higher
- **Git**: Latest version

### Hardware Requirements
- **RAM**: Minimum 8GB, Recommended 16GB
- **Storage**: Minimum 10GB free space
- **CPU**: Multi-core processor recommended

## Quick Start (Docker - Recommended)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd se-sme-agent
```

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables (optional for development)
nano .env
```

### 3. Start All Services
```bash
# Build and start all services
make docker-build
make docker-up

# Or use Docker Compose directly
docker-compose up -d
```

### 4. Verify Installation
```bash
# Check service health
curl http://localhost:8000/health

# View logs
make docker-logs

# Access services:
# - API: http://localhost:8000
# - UI: http://localhost:3000
# - Grafana: http://localhost:3001 (admin/admin)
# - Prometheus: http://localhost:9090
```

## Development Setup

### 1. Python Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install
# Or manually:
pip install -r requirements-dev.txt
```

### 2. Database Setup
```bash
# Start only database services
docker-compose up -d postgres redis elasticsearch

# Wait for services to be ready
sleep 30

# Run database migrations (if any)
make db-migrate
```

### 3. Start Development Servers
```bash
# Start API server
python -m uvicorn src.api_server.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, start UI (if developing UI)
cd ui
npm install
npm start
```

## Configuration

### Essential Environment Variables
```bash
# .env file
ENVIRONMENT=development
DEBUG=true

# Database connections
DATABASE_URL=postgresql://postgres:password@localhost:5432/se_sme_agent
REDIS_URL=redis://localhost:6379/0
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200

# Security (change in production!)
SECRET_KEY=your-secret-key-change-in-production

# LLM Configuration (required for full functionality)
LLM_PROVIDER=openai
LLM_API_KEY=your-openai-api-key-here
```

### Optional Configuration
```bash
# Rate limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60

# File uploads
MAX_FILE_SIZE=10485760  # 10MB
UPLOAD_DIR=./uploads

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

## Service Architecture

### Core Services
1. **API Server** (Port 8000): Main application gateway
2. **Agent Server** (Port 8001): AI agent orchestration
3. **RAG Pipeline** (Port 8002): Document processing and search
4. **Web UI** (Port 3000): User interface

### Supporting Services
1. **PostgreSQL** (Port 5432): Primary database
2. **Redis** (Port 6379): Caching and sessions
3. **Elasticsearch** (Port 9200): Search and vectors
4. **Prometheus** (Port 9090): Metrics collection
5. **Grafana** (Port 3001): Monitoring dashboards

## Testing

### Run All Tests
```bash
make test

# Or run specific test types
pytest tests/ -v
pytest tests/test_api_server.py -v
pytest tests/test_tool_registry.py -v
```

### Test Coverage
```bash
pytest tests/ --cov=src --cov-report=html
# View coverage report: htmlcov/index.html
```

## Common Commands

### Development
```bash
make dev          # Start development environment
make test         # Run tests
make format       # Format code
make lint         # Lint code
make clean        # Clean up generated files
```

### Docker Operations
```bash
make docker-build # Build Docker images
make docker-up    # Start services
make docker-down  # Stop services
make docker-logs  # View logs
```

### Database Operations
```bash
make db-migrate   # Run migrations
make db-reset     # Reset database
make backup       # Create backup
```

## API Usage Examples

### Health Check
```bash
curl http://localhost:8000/health
```

### Send Chat Message
```bash
curl -X POST http://localhost:8000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, SE SME Agent!", "session_id": "test_session"}'
```

### List Available Tools
```bash
curl http://localhost:8000/api/v1/tools/
```

### Upload Document
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@document.pdf" \
  -F "metadata={\"category\": \"documentation\"}"
```

## WebSocket Connection

### JavaScript Example
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/chat/ws/session_123');

ws.onopen = function(event) {
    console.log('Connected to SE SME Agent');
    ws.send(JSON.stringify({
        type: 'message',
        data: { message: 'Hello from WebSocket!' }
    }));
};

ws.onmessage = function(event) {
    const response = JSON.parse(event.data);
    console.log('Agent response:', response);
};
```

## Monitoring and Debugging

### Access Monitoring Dashboards
- **Grafana**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Elasticsearch**: http://localhost:9200

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api-server

# With timestamps
docker-compose logs -f -t api-server
```

### Debug Mode
```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Access debug endpoints
curl http://localhost:8000/debug/
```

## Troubleshooting

### Common Issues

#### 1. Services Won't Start
```bash
# Check Docker status
docker --version
docker-compose --version

# Check port conflicts
netstat -tulpn | grep :8000

# Restart services
make docker-down
make docker-up
```

#### 2. Database Connection Issues
```bash
# Check PostgreSQL status
docker-compose exec postgres pg_isready -U postgres

# Reset database
make db-reset

# Check connection string
echo $DATABASE_URL
```

#### 3. Elasticsearch Issues
```bash
# Check Elasticsearch health
curl http://localhost:9200/_cluster/health

# Check disk space (Elasticsearch needs space)
df -h

# Restart Elasticsearch
docker-compose restart elasticsearch
```

#### 4. Memory Issues
```bash
# Check memory usage
docker stats

# Reduce Elasticsearch memory
# Edit docker-compose.yml: ES_JAVA_OPTS=-Xms256m -Xmx256m
```

### Performance Optimization

#### 1. For Development
```bash
# Use fewer services
docker-compose up -d postgres redis elasticsearch

# Reduce log verbosity
export LOG_LEVEL=INFO
```

#### 2. For Production
```bash
# Use production environment
export ENVIRONMENT=production
export DEBUG=false

# Optimize database connections
export DATABASE_POOL_SIZE=20
export DATABASE_MAX_OVERFLOW=30
```

## Security Considerations

### Development Security
- Never commit real API keys or secrets
- Use strong passwords for databases
- Keep dependencies updated

### Production Security
- Change all default passwords
- Use environment-specific secrets
- Enable HTTPS/TLS
- Configure proper firewall rules
- Regular security updates

## Next Steps

### After Setup
1. **Test the API**: Use the provided examples to test functionality
2. **Upload Documents**: Add some documents to test RAG functionality
3. **Explore Tools**: Try different tools through the API
4. **Monitor Performance**: Check Grafana dashboards

### Development
1. **Read the Code**: Explore the codebase structure
2. **Run Tests**: Ensure everything works correctly
3. **Add Features**: Follow the development guidelines
4. **Contribute**: Submit pull requests for improvements

### Production Deployment
1. **Security Review**: Audit security configurations
2. **Performance Testing**: Load test the system
3. **Monitoring Setup**: Configure alerts and monitoring
4. **Backup Strategy**: Implement backup and recovery procedures

## Getting Help

### Documentation
- **System Documentation**: See SYSTEM_DOCUMENTATION.md
- **API Documentation**: http://localhost:8000/docs (when running)
- **Code Comments**: Comprehensive inline documentation

### Logs and Debugging
- **Application Logs**: Check Docker logs
- **Health Endpoints**: Use /health endpoints
- **Debug Mode**: Enable debug logging

### Community
- **Issues**: Report bugs and feature requests
- **Discussions**: Join community discussions
- **Contributing**: See contribution guidelines

## Quick Reference

### Service URLs (Development)
- **API Server**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Web UI**: http://localhost:3000
- **Grafana**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Elasticsearch**: http://localhost:9200

### Important Files
- **Configuration**: `.env`, `src/shared/config.py`
- **Docker**: `docker-compose.yml`, `docker/*/Dockerfile`
- **Tests**: `tests/`
- **Documentation**: `docs/`

### Key Commands
```bash
make help         # Show all available commands
make dev          # Start development environment
make test         # Run tests
make docker-up    # Start all services
make monitor      # Open monitoring dashboards
```