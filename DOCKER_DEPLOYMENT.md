# Docker Deployment Guide for SE SME Agent

## Overview

This guide provides comprehensive instructions for deploying the SE SME Agent system using Docker and Docker Compose.

## System Architecture

The system consists of the following services:

1. **API Server** (Port 8000) - Main REST API and WebSocket server
2. **Agent Server** (Port 8001) - AI agent orchestration and tool execution
3. **RAG Pipeline** (Port 8002) - Document processing and retrieval
4. **Web UI** (Port 3000) - React-based user interface
5. **PostgreSQL** (Port 5432) - Primary database
6. **Redis** (Port 6379) - Caching and session management
7. **Elasticsearch** (Port 9200) - Vector store and search engine
8. **Prometheus** (Port 9090) - Metrics collection
9. **Grafana** (Port 3001) - Monitoring dashboards

## Prerequisites

### System Requirements

- **CPU**: Minimum 4 cores, recommended 8+ cores
- **RAM**: Minimum 8GB, recommended 16GB+
- **Disk**: Minimum 20GB free space, recommended 50GB+ for data
- **OS**: Linux, macOS, or Windows with WSL2

### Software Requirements

- Docker Engine 20.10+
- Docker Compose 2.0+
- Git

### API Keys (Required)

You need at least one LLM provider API key:
- OpenAI API key (recommended)
- Anthropic API key (alternative)
- Google AI API key (alternative)

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
cd /path/to/project

# Copy environment file
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

### 2. Configure Environment Variables

Edit `.env` file and set the following required variables:

```bash
# LLM Configuration (REQUIRED)
LLM_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Other providers
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key

# Database (use strong passwords in production)
POSTGRES_PASSWORD=your-secure-password

# Grafana
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=your-secure-password

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### 3. Build and Start Services

```bash
# Build all Docker images (first time only)
docker-compose build

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4. Verify Deployment

Wait 1-2 minutes for all services to start, then check:

```bash
# Check health endpoints
curl http://localhost:8000/health  # API Server
curl http://localhost:8001/health  # Agent Server
curl http://localhost:8002/health  # RAG Pipeline
curl http://localhost:3000/health  # Web UI

# Or use the Makefile
make test-system
```

### 5. Access the System

- **Web UI**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Grafana Dashboard**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090

## Development Mode

For development with hot-reload:

```bash
# Start with development overrides
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Or use Makefile
make docker-up-dev
```

## Production Deployment

### Security Hardening

1. **Change Default Passwords**:
   ```bash
   # Generate strong passwords
   openssl rand -base64 32
   ```

2. **Use Secrets Management**:
   - Store API keys in Docker secrets or external vault
   - Never commit `.env` file to version control

3. **Enable HTTPS**:
   - Use a reverse proxy (nginx, Traefik, Caddy)
   - Configure SSL certificates

4. **Network Security**:
   - Restrict port exposure
   - Use firewall rules
   - Enable Docker network isolation

### Performance Optimization

1. **Resource Limits**: Already configured in docker-compose.yml
   - API Server: 2 CPU, 2GB RAM
   - Agent Server: 2 CPU, 3GB RAM
   - RAG Pipeline: 4 CPU, 4GB RAM
   - Elasticsearch: 2 CPU, 2GB RAM

2. **Scaling Services**:
   ```bash
   # Scale API server to 3 instances
   docker-compose up -d --scale api-server=3
   
   # Scale agent server to 2 instances
   docker-compose up -d --scale agent-server=2
   ```

3. **Database Tuning**: PostgreSQL is pre-configured with optimized settings

### Monitoring

1. **Prometheus Metrics**:
   - Available at http://localhost:9090
   - Scrapes metrics from all services

2. **Grafana Dashboards**:
   - Pre-configured dashboards in `docker/grafana/dashboards/`
   - Access at http://localhost:3001

3. **Log Aggregation**:
   ```bash
   # View all logs
   docker-compose logs -f
   
   # View specific service
   docker-compose logs -f api-server
   
   # Export logs
   docker-compose logs > system-logs.txt
   ```

## Data Management

### Backup

```bash
# Backup PostgreSQL database
docker-compose exec postgres pg_dump -U postgres se_sme_agent > backup.sql

# Backup Elasticsearch data
docker-compose exec elasticsearch curl -X PUT "localhost:9200/_snapshot/backup" \
  -H 'Content-Type: application/json' -d'{"type": "fs", "settings": {"location": "/backup"}}'

# Or use Makefile
make backup
```

### Restore

```bash
# Restore PostgreSQL database
cat backup.sql | docker-compose exec -T postgres psql -U postgres se_sme_agent

# Or use Makefile
make restore
```

### Volume Management

```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect se-sme-agent_postgres_data

# Backup volume
docker run --rm -v se-sme-agent_postgres_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/postgres-backup.tar.gz -C /data .

# Restore volume
docker run --rm -v se-sme-agent_postgres_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/postgres-backup.tar.gz -C /data
```

## Troubleshooting

### Common Issues

1. **Services Not Starting**:
   ```bash
   # Check logs
   docker-compose logs
   
   # Check resource usage
   docker stats
   
   # Restart services
   docker-compose restart
   ```

2. **Out of Memory**:
   ```bash
   # Check memory usage
   docker stats
   
   # Reduce Elasticsearch heap size in docker-compose.yml
   ES_JAVA_OPTS=-Xms512m -Xmx512m
   ```

3. **Port Conflicts**:
   ```bash
   # Check what's using the port
   netstat -tulpn | grep :8000
   
   # Change port in docker-compose.yml
   ports:
     - "8080:8000"  # Map to different host port
   ```

4. **Database Connection Issues**:
   ```bash
   # Check PostgreSQL is healthy
   docker-compose exec postgres pg_isready -U postgres
   
   # Reset database
   make db-reset
   ```

5. **Elasticsearch Yellow/Red Status**:
   ```bash
   # Check cluster health
   curl http://localhost:9200/_cluster/health?pretty
   
   # Increase disk space or adjust watermark settings
   ```

### Health Checks

```bash
# Check all service health
docker-compose ps

# Individual health checks
curl http://localhost:8000/health | jq
curl http://localhost:8001/health | jq
curl http://localhost:8002/health | jq

# Database health
docker-compose exec postgres pg_isready -U postgres

# Redis health
docker-compose exec redis redis-cli ping

# Elasticsearch health
curl http://localhost:9200/_cluster/health?pretty
```

### Performance Issues

1. **Slow Response Times**:
   - Check resource usage: `docker stats`
   - Review logs for errors
   - Increase resource limits in docker-compose.yml

2. **High Memory Usage**:
   - Reduce worker counts
   - Adjust Elasticsearch heap size
   - Enable memory limits

3. **Disk Space Issues**:
   ```bash
   # Check disk usage
   df -h
   
   # Clean up Docker
   docker system prune -a --volumes
   
   # Remove old logs
   docker-compose exec api-server find /app/logs -type f -mtime +7 -delete
   ```

## Maintenance

### Updates

```bash
# Pull latest images
docker-compose pull

# Rebuild services
docker-compose build --no-cache

# Restart with new images
docker-compose up -d
```

### Cleanup

```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v

# Clean up Docker system
docker system prune -a --volumes

# Or use Makefile
make clean
```

## Advanced Configuration

### Custom Network

```yaml
networks:
  se-sme-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16
          gateway: 172.28.0.1
```

### External Services

To use external PostgreSQL, Redis, or Elasticsearch:

1. Comment out the service in docker-compose.yml
2. Update connection strings in `.env`
3. Ensure network connectivity

### Load Balancing

For production with multiple instances:

```yaml
# Add nginx load balancer
nginx:
  image: nginx:alpine
  ports:
    - "80:80"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
  depends_on:
    - api-server
```

## Support

For issues and questions:
- Check logs: `docker-compose logs -f`
- Review health endpoints
- Check resource usage: `docker stats`
- Consult documentation in `/docs`

## Performance Benchmarks

Expected performance on recommended hardware:

- **API Response Time**: < 100ms (without LLM)
- **Document Ingestion**: 5-10 files/second
- **Search Queries**: < 500ms
- **Concurrent Users**: 100+
- **Memory Usage**: 6-8GB total
- **CPU Usage**: 30-50% average

## Security Checklist

- [ ] Changed default passwords
- [ ] API keys stored securely
- [ ] HTTPS enabled (production)
- [ ] Firewall configured
- [ ] Regular backups scheduled
- [ ] Monitoring alerts configured
- [ ] Log rotation enabled
- [ ] Security updates applied
- [ ] Network isolation configured
- [ ] Access controls implemented

## Next Steps

1. Configure your LLM provider API keys
2. Ingest your documents via the RAG pipeline
3. Test the system with sample queries
4. Set up monitoring and alerts
5. Configure backups
6. Review security settings
7. Scale services as needed

## Additional Resources

- API Documentation: http://localhost:8000/docs
- Grafana Dashboards: http://localhost:3001
- Prometheus Metrics: http://localhost:9090
- Project README: ./README.md
- Architecture Docs: ./docs/Architecture.md
