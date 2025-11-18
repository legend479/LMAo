# LMA-o: Developer Guide

## Table of Contents
1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
4. [Testing](#testing)
5. [Code Style](#code-style)
6. [Contributing](#contributing)

## Development Setup

### Local Development Environment

**1. Clone Repository**:
```bash
git clone https://github.com/yourusername/LMAo.git
cd LMAo
```

**2. Python Environment**:
```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**3. Node.js Environment** (for UI development):
```bash
cd ui
npm install
```

**4. Environment Configuration**:
```bash
cp .env.example .env
# Edit .env with development settings
```

**5. Start Development Services**:
```bash
# Start infrastructure (PostgreSQL, Redis, Elasticsearch)
docker-compose up -d postgres redis elasticsearch

# Start API Server
cd src/api_server
uvicorn main:app --reload --port 8000

# Start Agent Server
cd src/agent_server
uvicorn main:app --reload --port 8001

# Start RAG Pipeline
cd src/rag_pipeline
uvicorn main:app --reload --port 8002

# Start Web UI
cd ui
npm start
```

### IDE Setup

**VS Code Configuration** (`.vscode/settings.json`):
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "88"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "[python]": {
    "editor.rulers": [88]
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
```

**Recommended Extensions**:
- Python
- Pylance
- Black Formatter
- ESLint
- Prettier
- Docker
- GitLens

## Project Structure

```
LMAo/
├── src/                          # Source code
│   ├── api_server/              # API Gateway
│   │   ├── routers/             # API endpoints
│   │   ├── middleware/          # Request/response middleware
│   │   ├── auth/                # Authentication
│   │   ├── cache/               # Caching layer
│   │   └── main.py              # FastAPI application
│   ├── agent_server/            # Agent orchestration
│   │   ├── tools/               # Tool registry
│   │   ├── orchestrator.py     # LangGraph workflows
│   │   ├── planning.py          # Task planning
│   │   ├── memory.py            # Memory management
│   │   └── main.py              # Agent server
│   ├── rag_pipeline/            # RAG system
│   │   ├── document_processor.py
│   │   ├── embedding_manager.py
│   │   ├── search_engine.py
│   │   └── main.py
│   └── shared/                  # Shared utilities
│       ├── llm/                 # LLM integration
│       ├── database/            # Database utilities
│       ├── config.py            # Configuration
│       └── logging.py           # Logging setup
├── ui/                          # React frontend
│   ├── src/
│   │   ├── components/          # React components
│   │   ├── pages/               # Page components
│   │   ├── services/            # API services
│   │   ├── store/               # Redux store
│   │   └── App.tsx              # Main app
│   └── package.json
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── e2e/                     # End-to-end tests
├── docs/                        # Documentation
├── docker/                      # Docker configurations
├── scripts/                     # Utility scripts
├── .env.example                 # Example environment
├── docker-compose.yml           # Docker Compose config
├── requirements.txt             # Python dependencies
└── README.md                    # Project README
```

## Development Workflow

### Creating a New Feature

**1. Create Feature Branch**:
```bash
git checkout -b feature/your-feature-name
```

**2. Implement Feature**:
```python
# Example: Adding a new tool

# src/agent_server/tools/my_new_tool.py
from .registry import BaseTool, ToolResult, ExecutionContext

class MyNewTool(BaseTool):
    def __init__(self, config=None):
        super().__init__(config)
        self.metadata = self._create_metadata()
    
    def _create_metadata(self):
        return ToolMetadata(
            name="my_new_tool",
            description="Description of what the tool does",
            version="1.0.0",
            author="Your Name",
            category="utility",
            capabilities=self.get_capabilities(),
            resource_requirements=self.get_resource_requirements(),
            performance_metrics=PerformanceMetrics(),
            parameters={
                "param1": {
                    "type": "string",
                    "required": True,
                    "description": "Parameter description"
                }
            },
            required_params=["param1"],
            created_at=datetime.utcnow()
        )
    
    async def execute(self, parameters, context):
        # Validate parameters
        is_valid, errors = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(
                data=None,
                metadata={"errors": errors},
                execution_time=0.0,
                success=False,
                error_message=f"Validation failed: {errors}"
            )
        
        # Execute tool logic
        start_time = time.time()
        try:
            result = await self._execute_logic(parameters)
            execution_time = time.time() - start_time
            
            return ToolResult(
                data=result,
                metadata={"param1": parameters["param1"]},
                execution_time=execution_time,
                success=True
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                data=None,
                metadata={},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_logic(self, parameters):
        # Implement your tool logic here
        pass
    
    def get_schema(self):
        return {
            "name": "my_new_tool",
            "description": self.metadata.description,
            "parameters": self.metadata.parameters,
            "required_params": self.metadata.required_params
        }
    
    def get_capabilities(self):
        return ToolCapabilities(
            primary_capability=ToolCapability.TRANSFORMATION,
            secondary_capabilities=[],
            input_types=["string"],
            output_types=["string"]
        )
    
    def get_resource_requirements(self):
        return ResourceRequirements(
            cpu_cores=1.0,
            memory_mb=256,
            max_execution_time=30
        )
```

**3. Register Tool**:
```python
# src/agent_server/tools/auto_register.py
from .my_new_tool import MyNewTool

async def register_default_tools(registry):
    # ... existing tools ...
    
    # Register new tool
    my_tool = MyNewTool()
    await my_tool.initialize()
    await registry.register_tool(my_tool)
```

**4. Write Tests**:
```python
# tests/unit/test_my_new_tool.py
import pytest
from src.agent_server.tools.my_new_tool import MyNewTool
from src.agent_server.tools.registry import ExecutionContext

@pytest.mark.asyncio
async def test_my_new_tool_success():
    tool = MyNewTool()
    await tool.initialize()
    
    context = ExecutionContext(session_id="test")
    parameters = {"param1": "test_value"}
    
    result = await tool.execute(parameters, context)
    
    assert result.success is True
    assert result.data is not None
    assert result.execution_time > 0

@pytest.mark.asyncio
async def test_my_new_tool_validation_error():
    tool = MyNewTool()
    await tool.initialize()
    
    context = ExecutionContext(session_id="test")
    parameters = {}  # Missing required param
    
    result = await tool.execute(parameters, context)
    
    assert result.success is False
    assert "param1" in result.error_message
```

**5. Add Documentation**:
```python
# Add docstrings
class MyNewTool(BaseTool):
    """
    My New Tool
    
    This tool performs X operation on Y input.
    
    Parameters:
        param1 (str): Description of parameter
    
    Returns:
        ToolResult: Result containing processed data
    
    Example:
        >>> tool = MyNewTool()
        >>> result = await tool.execute({"param1": "value"}, context)
        >>> print(result.data)
    """
```

**6. Commit Changes**:
```bash
git add .
git commit -m "feat: add MyNewTool for X functionality"
git push origin feature/your-feature-name
```

### Adding API Endpoints

**1. Create Router**:
```python
# src/api_server/routers/my_endpoint.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from ..auth import get_current_user

router = APIRouter(prefix="/api/v1/myendpoint", tags=["My Endpoint"])

class MyRequest(BaseModel):
    param1: str
    param2: int = 10

class MyResponse(BaseModel):
    result: str
    metadata: dict

@router.post("/process", response_model=MyResponse)
async def process_data(
    request: MyRequest,
    current_user = Depends(get_current_user)
):
    """
    Process data with my endpoint
    
    - **param1**: Description
    - **param2**: Description (optional, default: 10)
    """
    try:
        # Process request
        result = await process_logic(request)
        
        return MyResponse(
            result=result,
            metadata={"user_id": current_user.id}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_logic(request: MyRequest):
    # Implement logic
    return f"Processed: {request.param1}"
```

**2. Register Router**:
```python
# src/api_server/main.py
from .routers import my_endpoint

app.include_router(my_endpoint.router)
```

**3. Test Endpoint**:
```python
# tests/integration/test_my_endpoint.py
from fastapi.testclient import TestClient
from src.api_server.main import app

client = TestClient(app)

def test_process_data():
    response = client.post(
        "/api/v1/myendpoint/process",
        json={"param1": "test", "param2": 20},
        headers={"Authorization": "Bearer test_token"}
    )
    
    assert response.status_code == 200
    assert "result" in response.json()
```

## Testing

### Running Tests

**All Tests**:
```bash
pytest
```

**Specific Test Suite**:
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Specific file
pytest tests/unit/test_my_tool.py

# Specific test
pytest tests/unit/test_my_tool.py::test_my_tool_success
```

**With Coverage**:
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Writing Tests

**Unit Test Example**:
```python
import pytest
from unittest.mock import Mock, patch
from src.agent_server.planning import PlanningModule

@pytest.fixture
def planning_module():
    return PlanningModule()

@pytest.mark.asyncio
async def test_create_plan(planning_module):
    query = "Test query"
    context = Mock()
    
    plan = await planning_module.create_plan(query, context)
    
    assert plan is not None
    assert len(plan.tasks) > 0
    assert plan.plan_id is not None

@pytest.mark.asyncio
async def test_create_plan_with_mock():
    with patch('src.agent_server.planning.LLMIntegration') as mock_llm:
        mock_llm.return_value.generate_response.return_value = "Mocked response"
        
        planning_module = PlanningModule()
        plan = await planning_module.create_plan("Test", Mock())
        
        assert plan is not None
        mock_llm.return_value.generate_response.assert_called_once()
```

**Integration Test Example**:
```python
import pytest
from src.agent_server.main import agent_server

@pytest.mark.asyncio
async def test_full_workflow():
    await agent_server.initialize()
    
    result = await agent_server.process_message(
        message="Test message",
        session_id="test_session"
    )
    
    assert result["response"] is not None
    assert result["session_id"] == "test_session"
    assert "metadata" in result
```

## Code Style

### Python Style Guide

**Follow PEP 8** with these specifics:
- Line length: 88 characters (Black default)
- Use type hints
- Docstrings for all public functions/classes
- Use async/await for I/O operations

**Example**:
```python
from typing import Dict, List, Optional
from datetime import datetime

async def process_documents(
    documents: List[str],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process a list of documents.
    
    Args:
        documents: List of document paths
        options: Optional processing options
    
    Returns:
        Dictionary containing processing results
    
    Raises:
        ValueError: If documents list is empty
        ProcessingError: If processing fails
    """
    if not documents:
        raise ValueError("Documents list cannot be empty")
    
    results = []
    for doc in documents:
        result = await process_single_document(doc, options)
        results.append(result)
    
    return {
        "processed": len(results),
        "timestamp": datetime.utcnow().isoformat(),
        "results": results
    }
```

### TypeScript Style Guide

**Follow Airbnb Style Guide** with:
- Use functional components with hooks
- Use TypeScript interfaces for props
- Use async/await for promises
- Use meaningful variable names

**Example**:
```typescript
import React, { useState, useEffect } from 'react';
import { Box, Typography } from '@mui/material';

interface MessageProps {
  content: string;
  timestamp: Date;
  sender: 'user' | 'assistant';
}

const Message: React.FC<MessageProps> = ({ content, timestamp, sender }) => {
  const [isVisible, setIsVisible] = useState(false);
  
  useEffect(() => {
    setIsVisible(true);
  }, []);
  
  return (
    <Box
      sx={{
        opacity: isVisible ? 1 : 0,
        transition: 'opacity 0.3s',
        padding: 2,
        backgroundColor: sender === 'user' ? '#e3f2fd' : '#f5f5f5'
      }}
    >
      <Typography variant="body1">{content}</Typography>
      <Typography variant="caption" color="textSecondary">
        {timestamp.toLocaleTimeString()}
      </Typography>
    </Box>
  );
};

export default Message;
```

### Code Formatting

**Python**:
```bash
# Format code
black src/

# Check formatting
black --check src/

# Sort imports
isort src/

# Lint
flake8 src/
mypy src/
```

**TypeScript**:
```bash
# Format code
npm run format

# Lint
npm run lint

# Type check
npm run type-check
```

## Contributing

### Contribution Guidelines

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Write/update tests**
5. **Update documentation**
6. **Submit pull request**

### Pull Request Process

**1. Before Submitting**:
- [ ] All tests pass
- [ ] Code is formatted
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] No merge conflicts

**2. PR Template**:
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
```

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Tests
- `chore`: Maintenance

**Examples**:
```bash
feat(agent): add new planning algorithm
fix(api): resolve rate limiting issue
docs(readme): update installation instructions
refactor(rag): optimize search performance
test(tools): add unit tests for email tool
```

### Code Review Guidelines

**For Reviewers**:
- Check code quality and style
- Verify tests are adequate
- Ensure documentation is clear
- Test functionality locally
- Provide constructive feedback

**For Contributors**:
- Respond to feedback promptly
- Make requested changes
- Keep PR scope focused
- Be open to suggestions

---

**Document Version**: 1.0  
**Last Updated**: November 2024  
**Authors**: Raveesh Vyas, Prakhar Singhal
