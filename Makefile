# SE SME Agent Makefile

.PHONY: help install dev build test clean docker-build docker-up docker-down logs

# Default target
help:
	@echo "SE SME Agent Development Commands"
	@echo "================================="
	@echo "install     - Install dependencies"
	@echo "dev         - Start development environment"
	@echo "build       - Build all services"
	@echo "test        - Run tests"
	@echo "clean       - Clean up generated files"
	@echo "docker-build - Build Docker images"
	@echo "docker-up   - Start Docker services"
	@echo "docker-down - Stop Docker services"
	@echo "logs        - View Docker logs"

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements-dev.txt
	@echo "Installing UI dependencies..."
	cd ui && npm install
	@echo "Setting up pre-commit hooks..."
	pre-commit install

# Start development environment
dev:
	@echo "Starting development environment..."
	docker-compose up -d postgres redis elasticsearch
	@echo "Waiting for services to be ready..."
	sleep 15
	@echo "Services started. You can now run:"
	@echo "  make dev-api    - Start API server"
	@echo "  make dev-ui     - Start UI server"
	@echo "  make dev-all    - Start all development servers"

# Start API server in development
dev-api:
	@echo "Starting API server..."
	python -m uvicorn src.api_server.main:app --reload --host 0.0.0.0 --port 8000

# Start UI in development
dev-ui:
	@echo "Starting UI development server..."
	cd ui && REACT_APP_SIMPLE_MODE=true npm start

# Start UI in full mode
dev-ui-full:
	@echo "Starting UI development server (full mode)..."
	cd ui && REACT_APP_SIMPLE_MODE=false npm start

# Start all development servers
dev-all:
	@echo "Starting all development servers..."
	make dev
	@echo "Starting API server in background..."
	python -m uvicorn src.api_server.main:app --reload --host 0.0.0.0 --port 8000 &
	@echo "Starting UI server..."
	cd ui && npm start

# Build all services
build:
	@echo "Building Python package..."
	pip install -e .
	@echo "Building UI..."
	cd ui && npm run build

# Run tests
test:
	@echo "Running Python tests..."
	pytest tests/ -v --cov=src --cov-report=html
	@echo "Running UI tests..."
	cd ui && npm test -- --run

# Clean up generated files
clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	cd ui && rm -rf build/ node_modules/.cache/

# Docker commands
docker-build:
	@echo "Building Docker images..."
	docker-compose build --parallel

docker-build-no-cache:
	@echo "Building Docker images (no cache)..."
	docker-compose build --no-cache --parallel

docker-up:
	@echo "Starting Docker services..."
	docker-compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 10
	@echo "Services started successfully!"
	@echo "API Server: http://localhost:8000"
	@echo "Agent Server: http://localhost:8001"
	@echo "RAG Pipeline: http://localhost:8002"
	@echo "Web UI: http://localhost:3000"
	@echo "Grafana: http://localhost:3001"

docker-up-dev:
	@echo "Starting Docker services in development mode..."
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
	@echo "Development mode enabled with hot-reload"

docker-down:
	@echo "Stopping Docker services..."
	docker-compose down

docker-down-volumes:
	@echo "Stopping Docker services and removing volumes..."
	docker-compose down -v

docker-restart:
	@echo "Restarting Docker services..."
	docker-compose restart

docker-logs:
	@echo "Viewing Docker logs..."
	docker-compose logs -f

docker-logs-api:
	@echo "Viewing API server logs..."
	docker-compose logs -f api-server

docker-logs-agent:
	@echo "Viewing Agent server logs..."
	docker-compose logs -f agent-server

docker-logs-rag:
	@echo "Viewing RAG pipeline logs..."
	docker-compose logs -f rag-pipeline

docker-logs-ui:
	@echo "Viewing Web UI logs..."
	docker-compose logs -f web-ui

docker-ps:
	@echo "Docker services status:"
	@docker-compose ps

docker-stats:
	@echo "Docker resource usage:"
	@docker stats --no-stream

docker-clean:
	@echo "Cleaning up Docker resources..."
	docker-compose down -v
	docker system prune -f
	@echo "Docker cleanup complete"

docker-clean-all:
	@echo "WARNING: This will remove ALL Docker resources including images!"
	@printf "Are you sure? [y/N] "; \
		read REPLY; \
		case "$$REPLY" in \
		  [Yy]*) docker-compose down -v; \
		         docker system prune -a -f --volumes; \
		         echo "Complete Docker cleanup done" ;; \
		  *)     echo "Aborted" ;; \
		esac

# Development utilities
format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/

lint:
	@echo "Linting code..."
	flake8 src/ tests/
	mypy src/

security-check:
	@echo "Running security checks..."
	bandit -r src/

# Database operations
db-migrate:
	@echo "Running database migrations..."
	alembic upgrade head

db-reset:
	@echo "Resetting database..."
	docker-compose exec postgres psql -U postgres -c "DROP DATABASE IF EXISTS se_sme_agent;"
	docker-compose exec postgres psql -U postgres -c "CREATE DATABASE se_sme_agent;"
	docker-compose exec postgres psql -U postgres -d se_sme_agent -f /docker-entrypoint-initdb.d/init.sql

# Monitoring
monitor:
	@echo "Opening monitoring dashboards..."
	@echo "Grafana: http://localhost:3001 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	@echo "Elasticsearch: http://localhost:9200"

# Production deployment
deploy-prod:
	@echo "Deploying to production..."
	@echo "This would typically involve:"
	@echo "1. Building production images"
	@echo "2. Pushing to container registry"
	@echo "3. Updating production environment"
	@echo "4. Running health checks"

# Backup
backup:
	@echo "Creating backup..."
	docker-compose exec postgres pg_dump -U postgres se_sme_agent > backup_$(shell date +%Y%m%d_%H%M%S).sql

# Restore
restore:
	@echo "Restoring from backup..."
	@printf "Enter backup file path: "; \
		read backup_file; \
		docker-compose exec -T postgres psql -U postgres se_sme_agent < "$$backup_file"

# Quick setup for new developers
setup:
	@echo "Setting up SE SME Agent for development..."
	@echo "1. Installing Python dependencies..."
	pip install -r requirements-dev.txt
	@echo "2. Installing UI dependencies..."
	cd ui && npm install
	@echo "3. Setting up environment file..."
	cp .env.example .env
	@echo "4. Setting up pre-commit hooks..."
	pre-commit install
	@echo "5. Building Docker images..."
	docker-compose build
	@echo ""
	@echo "Setup complete! Next steps:"
	@echo "  1. Edit .env file with your configuration"
	@echo "  2. Run 'make docker-up' to start all services"
	@echo "  3. Run 'make test' to verify everything works"
	@echo "  4. Visit http://localhost:8000/docs for API documentation"
	@echo "  5. Visit http://localhost:3000 for the web interface"

# Check system requirements
check-requirements:
	@echo "Checking system requirements..."
	@python --version || echo "❌ Python not found"
	@node --version || echo "❌ Node.js not found"
	@docker --version || echo "❌ Docker not found"
	@docker-compose --version || echo "❌ Docker Compose not found"
	@echo "✅ Requirements check complete"

# Full system test
test-system:
	@echo "Running full system test..."
	@echo "1. Testing API server..."
	curl -f http://localhost:8000/health || echo "❌ API server not responding"
	@echo "2. Testing Agent server..."
	curl -f http://localhost:8001/health || echo "❌ Agent server not responding"
	@echo "3. Testing RAG pipeline..."
	curl -f http://localhost:8002/health || echo "❌ RAG pipeline not responding"
	@echo "4. Testing UI..."
	curl -f http://localhost:3000 || echo "❌ UI not responding"
	@echo "✅ System test complete"