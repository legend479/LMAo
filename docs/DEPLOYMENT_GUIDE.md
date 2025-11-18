# LMA-o: Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Production Deployment](#production-deployment)
4. [Configuration](#configuration)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)
7. [Maintenance](#maintenance)

## Prerequisites

### System Requirements

**Minimum Requirements**:
- CPU: 4 cores
- RAM: 8 GB
- Storage: 50 GB SSD
- OS: Linux (Ubuntu 20.04+), macOS, Windows with WSL2

**Recommended Requirements**:
- CPU: 8+ cores
- RAM: 16+ GB
- Storage: 100+ GB SSD
- OS: Linux (Ubuntu 22.04+)

### Software Dependencies

**Required**:
- Docker 24.0+
- Docker Compose 2.20+
- Python 3.9+ (for local development)
- Node.js 18+ (for UI development)

**Optional**:
- Ollama (for local LLM deployment)
- Nginx (for reverse proxy)
- Let's Encrypt (for SSL certificates)

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/LMAo.git
cd LMAo
```

### 2. Environment Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit configuration
nano .env
```

**Minimum Required Configuration**:
```bash
# LLM Provider (choose one)
LLM_PROVIDER=openai
LLM_API_KEY=your-api-key-here
LLM_MODEL=gpt-3.5-turbo

# Security
SECRET_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')

# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO
```

### 3. Start Services

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4. Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Check Agent Server
curl http://localhost:8001/health

# Check RAG Pipeline
curl http://localhost:8002/health

# Access Web UI
open http://localhost:3000
```

### 5. Create Admin User

```bash
# Access API container
docker-compose exec api-server bash

# Create admin user
python -m src.scripts.create_admin \
  --username admin@example.com \
  --password secure_password \
  --role admin
```

## Production Deployment

### Architecture Overview

```
Internet
    │
    ▼
┌─────────────────┐
│  Load Balancer  │
│  (Nginx/HAProxy)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  API Gateway    │
│  (Port 8000)    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│ Agent  │ │  RAG   │
│ Server │ │Pipeline│
└────────┘ └────────┘
    │         │
    └────┬────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│Postgres│ │Elastic │
└────────┘ └────────┘
```

### 1. Production Environment Setup

**Create production environment file**:
```bash
cp .env.example .env.production
```

**Configure production settings**:
```bash
# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Security
SECRET_KEY=<generate-secure-key>
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
DATABASE_URL=postgresql://user:password@postgres:5432/lmao_prod
REDIS_URL=redis://redis:6379/0

# LLM Provider
LLM_PROVIDER=openai
LLM_API_KEY=<your-production-api-key>
LLM_MODEL=gpt-4

# CORS
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

### 2. SSL/TLS Configuration

**Using Let's Encrypt with Nginx**:

```nginx
# /etc/nginx/sites-available/lmao
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # API Gateway
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket
    location /api/v1/chat/ws/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }

    # Web UI
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**Obtain SSL Certificate**:
```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### 3. Database Setup

**PostgreSQL Configuration**:

```bash
# Create production database
docker-compose exec postgres psql -U postgres

CREATE DATABASE lmao_prod;
CREATE USER lmao_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE lmao_prod TO lmao_user;
\q

# Run migrations
docker-compose exec api-server alembic upgrade head
```

**Database Backup**:
```bash
# Create backup script
cat > backup_db.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backups/postgres"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/lmao_backup_$TIMESTAMP.sql"

mkdir -p $BACKUP_DIR
docker-compose exec -T postgres pg_dump -U postgres lmao_prod > $BACKUP_FILE
gzip $BACKUP_FILE

# Keep only last 7 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete
EOF

chmod +x backup_db.sh

# Add to crontab (daily at 2 AM)
echo "0 2 * * * /path/to/backup_db.sh" | crontab -
```

### 4. Elasticsearch Configuration

**Production Settings**:

```yaml
# docker-compose.prod.yml
elasticsearch:
  image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
  environment:
    - discovery.type=single-node
    - xpack.security.enabled=true
    - ELASTIC_PASSWORD=secure_password
    - "ES_JAVA_OPTS=-Xms2g -Xmx2g"
  volumes:
    - elasticsearch_data:/usr/share/elasticsearch/data
  ulimits:
    memlock:
      soft: -1
      hard: -1
    nofile:
      soft: 65536
      hard: 65536
```

**Index Management**:
```bash
# Create index template
curl -X PUT "localhost:9200/_index_template/lmao_documents" \
  -H 'Content-Type: application/json' \
  -d @elasticsearch_template.json

# Optimize indices
curl -X POST "localhost:9200/lmao_documents/_forcemerge?max_num_segments=1"
```

### 5. Docker Compose Production

**docker-compose.prod.yml**:
```yaml
version: '3.8'

services:
  api-server:
    image: lmao/api-server:latest
    restart: always
    env_file: .env.production
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # ... other services
```

**Deploy**:
```bash
# Build images
docker-compose -f docker-compose.prod.yml build

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale api-server=3
```

## Configuration

### Environment Variables

**Core Settings**:
```bash
# Application
APP_NAME=LMA-o
VERSION=1.0.0
ENVIRONMENT=production|development|testing
DEBUG=false

# API Server
API_HOST=0.0.0.0
API_PORT=8000
ALLOWED_ORIGINS=https://yourdomain.com

# Agent Server
AGENT_HOST=agent-server
AGENT_PORT=8001

# RAG Pipeline
RAG_HOST=rag-pipeline
RAG_PORT=8002
```

**Database Configuration**:
```bash
# PostgreSQL
DATABASE_URL=postgresql://user:pass@host:5432/db
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Redis
REDIS_URL=redis://host:6379/0
REDIS_MAX_CONNECTIONS=50

# Elasticsearch
ELASTICSEARCH_HOST=elasticsearch
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_INDEX=lmao_documents
```

**LLM Configuration**:
```bash
# Provider Selection
LLM_PROVIDER=openai|anthropic|google|ollama

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4
OPENAI_ORGANIZATION=org-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Google AI
GOOGLE_API_KEY=AIza...
GOOGLE_MODEL=gemini-1.5-pro

# Ollama (Local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

**Security Configuration**:
```bash
# JWT
SECRET_KEY=<generate-secure-key>
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_BURST=10

# CORS
ALLOWED_ORIGINS=https://yourdomain.com
ALLOWED_METHODS=GET,POST,PUT,DELETE
ALLOWED_HEADERS=*
```

**Performance Configuration**:
```bash
# Concurrency
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=300
WORKER_PROCESSES=4

# Caching
CACHE_TTL=3600
CACHE_MAX_SIZE=1000

# File Upload
MAX_FILE_SIZE=10485760  # 10MB
UPLOAD_DIR=./uploads
```

### Feature Flags

```bash
# Tools
ENABLE_CODE_EXECUTION=true
ENABLE_EMAIL_TOOLS=true
ENABLE_DOCUMENT_GENERATION=true

# RAG Features
RAG_ENABLE_RERANKING=true
RAG_ENABLE_QUERY_REFORMULATION=true
RAG_ENABLE_ADAPTIVE_RETRIEVAL=true

# Monitoring
ENABLE_METRICS=true
ENABLE_TRACING=true
METRICS_PORT=9090
```

## Monitoring

### Prometheus Configuration

**prometheus.yml**:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'api-server'
    static_configs:
      - targets: ['api-server:9091']
  
  - job_name: 'agent-server'
    static_configs:
      - targets: ['agent-server:9092']
  
  - job_name: 'rag-pipeline'
    static_configs:
      - targets: ['rag-pipeline:9093']
  
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
  
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

### Grafana Dashboards

**Import Dashboards**:
1. Access Grafana: http://localhost:3001
2. Login (admin/admin)
3. Import dashboards from `docker/grafana/dashboards/`

**Key Metrics**:
- Request rate and latency
- Error rates
- LLM token usage
- Database connections
- Cache hit rates
- System resources (CPU, memory, disk)

### Logging

**Centralized Logging**:
```bash
# Configure log aggregation
docker-compose logs -f > logs/combined.log

# Use ELK stack for production
# See docker-compose.elk.yml
```

**Log Levels**:
- `DEBUG`: Detailed debugging information
- `INFO`: General informational messages
- `WARNING`: Warning messages
- `ERROR`: Error messages
- `CRITICAL`: Critical errors

### Alerting

**Alert Rules** (Prometheus):
```yaml
groups:
  - name: lmao_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
      
      - alert: HighLatency
        expr: http_request_duration_seconds{quantile="0.95"} > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
      
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
```

## Troubleshooting

### Common Issues

#### 1. Service Won't Start

**Check logs**:
```bash
docker-compose logs <service-name>
```

**Common causes**:
- Port already in use
- Missing environment variables
- Database connection failed
- Insufficient resources

**Solutions**:
```bash
# Check port usage
sudo netstat -tulpn | grep <port>

# Verify environment
docker-compose config

# Check resources
docker stats
```

#### 2. Database Connection Errors

**Symptoms**:
- "Connection refused"
- "Authentication failed"
- "Database does not exist"

**Solutions**:
```bash
# Check database status
docker-compose exec postgres pg_isready

# Verify credentials
docker-compose exec postgres psql -U postgres -l

# Reset database
docker-compose down -v
docker-compose up -d postgres
# Wait for initialization
docker-compose up -d
```

#### 3. Elasticsearch Issues

**Symptoms**:
- "Connection timeout"
- "Index not found"
- "Out of memory"

**Solutions**:
```bash
# Check Elasticsearch health
curl http://localhost:9200/_cluster/health

# Increase heap size
# Edit docker-compose.yml:
# ES_JAVA_OPTS=-Xms2g -Xmx2g

# Recreate indices
curl -X DELETE http://localhost:9200/lmao_documents
docker-compose restart rag-pipeline
```

#### 4. LLM API Errors

**Symptoms**:
- "Invalid API key"
- "Rate limit exceeded"
- "Model not found"

**Solutions**:
```bash
# Verify API key
echo $OPENAI_API_KEY

# Test API connection
python test_llm_integration.py

# Switch provider
# Edit .env:
LLM_PROVIDER=anthropic
docker-compose restart agent-server
```

### Debug Mode

**Enable debug logging**:
```bash
# Edit .env
LOG_LEVEL=DEBUG
DEBUG=true

# Restart services
docker-compose restart
```

**Access container shell**:
```bash
# API Server
docker-compose exec api-server bash

# Agent Server
docker-compose exec agent-server bash

# Check Python environment
python -c "import sys; print(sys.path)"
pip list
```

## Maintenance

### Regular Maintenance Tasks

**Daily**:
- Monitor system health
- Check error logs
- Verify backups

**Weekly**:
- Review performance metrics
- Update dependencies
- Clean up old logs

**Monthly**:
- Database optimization
- Security updates
- Capacity planning

### Backup and Recovery

**Automated Backups**:
```bash
#!/bin/bash
# backup_all.sh

# Database backup
docker-compose exec -T postgres pg_dump -U postgres lmao_prod | \
  gzip > backups/db_$(date +%Y%m%d).sql.gz

# Elasticsearch snapshot
curl -X PUT "localhost:9200/_snapshot/backup/snapshot_$(date +%Y%m%d)" \
  -H 'Content-Type: application/json' \
  -d '{"indices": "lmao_documents"}'

# Configuration backup
tar -czf backups/config_$(date +%Y%m%d).tar.gz .env docker-compose.yml

# Upload to S3 (optional)
aws s3 sync backups/ s3://your-bucket/lmao-backups/
```

**Recovery**:
```bash
# Restore database
gunzip < backups/db_20241119.sql.gz | \
  docker-compose exec -T postgres psql -U postgres lmao_prod

# Restore Elasticsearch
curl -X POST "localhost:9200/_snapshot/backup/snapshot_20241119/_restore"

# Restore configuration
tar -xzf backups/config_20241119.tar.gz
```

### Updates and Upgrades

**Update Docker Images**:
```bash
# Pull latest images
docker-compose pull

# Restart services
docker-compose up -d

# Verify versions
docker-compose exec api-server python --version
```

**Database Migrations**:
```bash
# Create migration
docker-compose exec api-server alembic revision --autogenerate -m "description"

# Apply migration
docker-compose exec api-server alembic upgrade head

# Rollback if needed
docker-compose exec api-server alembic downgrade -1
```

### Performance Tuning

**Database Optimization**:
```sql
-- Analyze tables
ANALYZE;

-- Vacuum
VACUUM ANALYZE;

-- Reindex
REINDEX DATABASE lmao_prod;
```

**Elasticsearch Optimization**:
```bash
# Force merge
curl -X POST "localhost:9200/lmao_documents/_forcemerge?max_num_segments=1"

# Clear cache
curl -X POST "localhost:9200/_cache/clear"
```

**Redis Optimization**:
```bash
# Check memory usage
docker-compose exec redis redis-cli INFO memory

# Clear cache if needed
docker-compose exec redis redis-cli FLUSHDB
```

---

**Document Version**: 1.0  
**Last Updated**: November 2024  
**Authors**: Raveesh Vyas, Prakhar Singhal
