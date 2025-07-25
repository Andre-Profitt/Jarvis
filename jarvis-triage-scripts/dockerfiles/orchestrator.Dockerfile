# Jarvis Orchestrator Service
# Agent: Docker Captain
# Multi-stage build for optimal size

# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.2

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Export requirements
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Install dependencies
RUN pip install --prefix=/install -r requirements.txt

# Runtime stage
FROM python:3.12-slim

# Security: Run as non-root user
RUN useradd -m -u 1000 jarvis && \
    mkdir -p /app && \
    chown -R jarvis:jarvis /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application
WORKDIR /app
COPY --chown=jarvis:jarvis src/ ./src/
COPY --chown=jarvis:jarvis config/ ./config/

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

USER jarvis

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Start service
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]