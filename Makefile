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
	sleep 10
	@echo "Starting API server..."
	python -m uvicorn src.api_server.main:app --reload --host 0.0.0.0 --port 8000 &
	@echo "Starting UI development server..."
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
	docker-compose build

docker-up:
	@echo "Starting Docker services..."
	docker-compose up -d

docker-down:
	@echo "Stopping Docker services..."
	docker-compose down

docker-logs:
	@echo "Viewing Docker logs..."
	docker-compose logs -f

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
	@read -p "Enter backup file path: " backup_file; \
	docker-compose exec -T postgres psql -U postgres se_sme_agent < $$backup_file