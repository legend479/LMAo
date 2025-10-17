# SE SME Agent Test Suite

This directory contains a comprehensive test suite for the SE SME Agent system, covering all aspects from unit tests to end-to-end workflows.

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── shared/             # Shared utility tests (config, logging, validation)
│   ├── api_server/         # API server component tests
│   ├── agent_server/       # Agent server component tests
│   └── rag_pipeline/       # RAG pipeline component tests
├── integration/            # Integration tests between components
│   ├── test_api_agent_integration.py
│   └── test_rag_integration.py
├── e2e/                   # End-to-end workflow tests
│   ├── test_complete_workflows.py
│   └── test_user_workflows.py
├── performance/           # Performance and load tests
│   ├── test_load_testing.py
│   └── test_stress_testing.py
├── security/              # Security and vulnerability tests
│   ├── test_injection_prevention.py
│   └── test_authentication.py
├── monitoring/            # Monitoring and metrics tests
│   └── test_metrics_collection.py
├── conftest.py           # Shared test fixtures and configuration
├── pytest.ini           # Pytest configuration
├── requirements-test.txt # Test dependencies
├── run_all_tests.py     # Comprehensive test runner
└── README.md            # This file
```

## Test Categories

### 1. Unit Tests
- Test individual functions and classes in isolation
- Mock external dependencies
- Fast execution (< 1 second per test)
- High coverage of business logic

### 2. Integration Tests
- Test component interactions
- Use real databases and services where appropriate
- Medium execution time (1-10 seconds per test)
- Focus on interface contracts

### 3. End-to-End Tests
- Test complete user workflows
- Use full system stack
- Longer execution time (10+ seconds per test)
- Validate user experience

### 4. Performance Tests
- Load testing with multiple concurrent users
- Stress testing with high resource usage
- Benchmark testing for performance regression
- Memory and resource usage validation

### 5. Security Tests
- Authentication and authorization testing
- Input validation and injection prevention
- Access control verification
- Security vulnerability scanning

## Running Tests

### All Tests
```bash
make test
```

### Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# End-to-end tests only
pytest tests/e2e/ -v

# Performance tests
pytest tests/performance/ -v

# Security tests
pytest tests/security/ -v
```

### Test Coverage
```bash
# Generate coverage report
pytest --cov=src --cov-report=html tests/

# View coverage report
open htmlcov/index.html
```

### Test Markers
Tests are marked with pytest markers for easy filtering:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.security` - Security tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.requires_db` - Tests requiring database
- `@pytest.mark.requires_es` - Tests requiring Elasticsearch
- `@pytest.mark.requires_redis` - Tests requiring Redis

### Example Usage
```bash
# Run only fast unit tests
pytest -m "unit and not slow" tests/

# Run tests that don't require external services
pytest -m "not (requires_db or requires_es or requires_redis)" tests/

# Run integration tests with database
pytest -m "integration and requires_db" tests/
```

## Test Data Management

### Fixtures
- Use pytest fixtures for reusable test data
- Database fixtures with automatic cleanup
- Mock service fixtures for external dependencies

### Test Databases
- Separate test databases for isolation
- Automatic schema creation and cleanup
- Transaction rollback for test isolation

### Mock Services
- Mock external API calls
- Mock LLM responses for consistent testing
- Mock file system operations

## Continuous Integration

Tests are automatically run on:
- Pull requests
- Main branch commits
- Nightly builds (full test suite including performance)

### Test Requirements
- All new code must have unit tests
- Integration tests for new features
- Performance tests for critical paths
- Security tests for authentication/authorization changes

## Test Configuration

### Environment Variables
```bash
# Test environment
TEST_DATABASE_URL=postgresql://test:test@localhost:5432/test_se_sme_agent
TEST_REDIS_URL=redis://localhost:6379/1
TEST_ELASTICSEARCH_HOST=localhost
TEST_ELASTICSEARCH_PORT=9200

# Test settings
PYTEST_TIMEOUT=300
PYTEST_WORKERS=4
```

### Docker Test Environment
```bash
# Start test services
docker-compose -f docker-compose.test.yml up -d

# Run tests
make test

# Cleanup
docker-compose -f docker-compose.test.yml down -v
```