FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY pyproject.toml ./

# Create data directory
RUN mkdir -p data

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8002

# Run the RAG pipeline
CMD ["python", "-c", "import asyncio; from src.rag_pipeline.main import rag_pipeline; asyncio.run(rag_pipeline.initialize()); print('RAG Pipeline running...'); asyncio.get_event_loop().run_forever()"]