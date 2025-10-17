#!/bin/bash

# SE SME Agent Startup Script
# This script helps start the system in different modes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  SE SME Agent Startup Script${NC}"
    echo -e "${BLUE}================================${NC}"
    echo
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    local missing_deps=()
    
    if ! command_exists python3; then
        missing_deps+=("python3")
    fi
    
    if ! command_exists node; then
        missing_deps+=("node")
    fi
    
    if ! command_exists docker; then
        missing_deps+=("docker")
    fi
    
    if ! command_exists docker-compose; then
        missing_deps+=("docker-compose")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        print_error "Please install the missing dependencies and try again."
        exit 1
    fi
    
    print_status "All requirements satisfied ✓"
}

# Function to wait for service to be ready
wait_for_service() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            print_status "$service_name is ready ✓"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start within expected time"
    return 1
}

# Function to start development environment
start_development() {
    print_status "Starting development environment..."
    
    # Check if .env exists
    if [ ! -f ".env" ]; then
        print_warning ".env file not found, creating from template..."
        cp .env.example .env
        print_warning "Please edit .env file with your configuration"
    fi
    
    # Start infrastructure services
    print_status "Starting infrastructure services..."
    docker-compose up -d postgres redis elasticsearch
    
    # Wait for services to be ready
    wait_for_service "PostgreSQL" "http://localhost:5432" || true
    wait_for_service "Redis" "http://localhost:6379" || true
    wait_for_service "Elasticsearch" "http://localhost:9200"
    
    print_status "Infrastructure services are ready!"
    print_status "You can now start the application services:"
    echo "  - API Server: make dev-api"
    echo "  - UI Server: make dev-ui"
    echo "  - Or both: make dev-all"
}

# Function to start production environment
start_production() {
    print_status "Starting production environment..."
    
    # Build images
    print_status "Building Docker images..."
    docker-compose build --no-cache
    
    # Start all services
    print_status "Starting all services..."
    docker-compose up -d
    
    # Wait for services
    print_status "Waiting for services to be ready..."
    sleep 10
    
    wait_for_service "API Server" "http://localhost:8000/health"
    wait_for_service "Agent Server" "http://localhost:8001/health"
    wait_for_service "RAG Pipeline" "http://localhost:8002/health"
    wait_for_service "Web UI" "http://localhost:3000"
    
    print_status "All services are ready!"
    print_status "Access points:"
    echo "  - Web UI: http://localhost:3000"
    echo "  - API Docs: http://localhost:8000/docs"
    echo "  - Grafana: http://localhost:3001 (admin/admin)"
    echo "  - Prometheus: http://localhost:9090"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    # Start test infrastructure
    docker-compose up -d postgres redis elasticsearch
    sleep 10
    
    # Run Python tests
    print_status "Running Python tests..."
    python -m pytest tests/ -v
    
    # Run UI tests if available
    if [ -d "ui" ] && [ -f "ui/package.json" ]; then
        print_status "Running UI tests..."
        cd ui && npm test -- --run --passWithNoTests
        cd ..
    fi
    
    print_status "Tests completed ✓"
}

# Function to show system status
show_status() {
    print_status "System Status:"
    echo
    
    # Check Docker services
    if docker-compose ps | grep -q "Up"; then
        print_status "Docker services:"
        docker-compose ps
    else
        print_warning "No Docker services running"
    fi
    
    echo
    
    # Check individual services
    services=(
        "API Server:http://localhost:8000/health"
        "Agent Server:http://localhost:8001/health"
        "RAG Pipeline:http://localhost:8002/health"
        "Web UI:http://localhost:3000"
        "Grafana:http://localhost:3001"
        "Prometheus:http://localhost:9090"
    )
    
    for service in "${services[@]}"; do
        IFS=':' read -r name url <<< "$service"
        if curl -f -s "$url" > /dev/null 2>&1; then
            echo -e "  ✓ $name: ${GREEN}Running${NC}"
        else
            echo -e "  ✗ $name: ${RED}Not responding${NC}"
        fi
    done
}

# Function to stop all services
stop_services() {
    print_status "Stopping all services..."
    docker-compose down -v
    print_status "All services stopped ✓"
}

# Function to show help
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  dev         Start development environment (infrastructure only)"
    echo "  prod        Start production environment (all services)"
    echo "  test        Run tests"
    echo "  status      Show system status"
    echo "  stop        Stop all services"
    echo "  help        Show this help message"
    echo
    echo "Examples:"
    echo "  $0 dev      # Start development environment"
    echo "  $0 prod     # Start production environment"
    echo "  $0 status   # Check system status"
}

# Main script logic
main() {
    print_header
    
    case "${1:-help}" in
        "dev"|"development")
            check_requirements
            start_development
            ;;
        "prod"|"production")
            check_requirements
            start_production
            ;;
        "test")
            check_requirements
            run_tests
            ;;
        "status")
            show_status
            ;;
        "stop")
            stop_services
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"