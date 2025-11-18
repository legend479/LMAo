#!/bin/bash

# Docker Health Check Script for SE SME Agent
# This script checks the health of all Docker services

set -e

echo "=========================================="
echo "SE SME Agent - Docker Health Check"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check service health
check_service() {
    local service_name=$1
    local url=$2
    local max_retries=30
    local retry_count=0
    
    echo -n "Checking $service_name... "
    
    while [ $retry_count -lt $max_retries ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Healthy${NC}"
            return 0
        fi
        retry_count=$((retry_count + 1))
        sleep 2
    done
    
    echo -e "${RED}✗ Unhealthy${NC}"
    return 1
}

# Function to check Docker service status
check_docker_service() {
    local service=$1
    echo -n "Checking Docker service $service... "
    
    if docker-compose ps | grep -q "$service.*Up"; then
        echo -e "${GREEN}✓ Running${NC}"
        return 0
    else
        echo -e "${RED}✗ Not Running${NC}"
        return 1
    fi
}

# Check if Docker Compose is running
if ! docker-compose ps > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker Compose services are not running${NC}"
    echo "Run 'docker-compose up -d' to start services"
    exit 1
fi

echo "1. Checking Docker Services Status"
echo "-----------------------------------"
check_docker_service "postgres" || true
check_docker_service "redis" || true
check_docker_service "elasticsearch" || true
check_docker_service "api-server" || true
check_docker_service "agent-server" || true
check_docker_service "rag-pipeline" || true
check_docker_service "web-ui" || true
check_docker_service "prometheus" || true
check_docker_service "grafana" || true
echo ""

echo "2. Checking Service Health Endpoints"
echo "-------------------------------------"
check_service "PostgreSQL" "http://localhost:5432" || true
check_service "Redis" "http://localhost:6379" || true
check_service "Elasticsearch" "http://localhost:9200/_cluster/health" || true
check_service "API Server" "http://localhost:8000/health" || true
check_service "Agent Server" "http://localhost:8001/health" || true
check_service "RAG Pipeline" "http://localhost:8002/health" || true
check_service "Web UI" "http://localhost:3000/health" || true
check_service "Prometheus" "http://localhost:9090/-/healthy" || true
check_service "Grafana" "http://localhost:3001/api/health" || true
echo ""

echo "3. Checking Resource Usage"
echo "--------------------------"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
echo ""

echo "4. Checking Disk Usage"
echo "----------------------"
df -h | grep -E "Filesystem|/dev/" | head -5
echo ""

echo "5. Checking Docker Volumes"
echo "--------------------------"
docker volume ls | grep se-sme-agent || echo "No volumes found"
echo ""

echo "=========================================="
echo "Health Check Complete"
echo "=========================================="
echo ""
echo "Access Points:"
echo "  - Web UI:        http://localhost:3000"
echo "  - API Docs:      http://localhost:8000/docs"
echo "  - Grafana:       http://localhost:3001 (admin/admin)"
echo "  - Prometheus:    http://localhost:9090"
echo "  - Elasticsearch: http://localhost:9200"
echo ""
echo "Logs: docker-compose logs -f [service-name]"
echo "Status: docker-compose ps"
echo ""
