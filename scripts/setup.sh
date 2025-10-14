#!/bin/bash

# SE SME Agent Setup Script

# set -e

echo "🚀 Setting up SE SME Agent Development Environment"
echo "=================================================="

# Check if Python 3.9+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.9+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed"
    exit 1
fi

node_version=$(node --version)
echo "✅ Node.js version check passed: $node_version"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed"
    exit 1
fi

docker_version=$(docker --version)
echo "✅ Docker version check passed: $docker_version"

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is required but not installed"
    exit 1
fi

compose_version=$(docker-compose --version)
echo "✅ Docker Compose version check passed: $compose_version"

# Create virtual environment
echo "📦 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements-dev.txt

# Install UI dependencies
echo "📦 Installing UI dependencies..."
cd ui
npm install
cd ..

# Create environment file
if [ ! -f .env ]; then
    echo "📝 Creating environment file..."
    cp .env.example .env
    echo "⚠️  Please update .env file with your configuration"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads
mkdir -p data
mkdir -p logs

# Set up pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Build Docker images
echo "🐳 Building Docker images..."
docker-compose build

# Start infrastructure services
echo "🚀 Starting infrastructure services..."
docker-compose up -d postgres redis elasticsearch

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Run database initialization
echo "🗄️  Initializing database..."
docker-compose exec -T postgres psql -U postgres -d se_sme_agent -f /docker-entrypoint-initdb.d/init.sql

# Run tests to verify setup
echo "🧪 Running tests to verify setup..."
pytest tests/ -v

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Update .env file with your configuration"
echo "2. Start development with: make dev"
echo "3. Access the application at: http://localhost:3000"
echo "4. Access the API at: http://localhost:8000"
echo "5. View API docs at: http://localhost:8000/docs"
echo ""
echo "Useful commands:"
echo "- make help          # Show all available commands"
echo "- make docker-up     # Start all services"
echo "- make docker-down   # Stop all services"
echo "- make logs          # View service logs"
echo "- make test          # Run tests"
echo ""