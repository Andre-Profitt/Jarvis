# Consciousness Service Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy consciousness tool code
COPY tools/consciousness ./tools/consciousness
COPY core ./core
COPY utils ./utils

# Create non-root user
RUN useradd -m -u 1000 jarvis && chown -R jarvis:jarvis /app
USER jarvis

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV SERVICE_NAME=consciousness
ENV PORT=8001

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the service
CMD ["python", "-m", "tools.consciousness.service", "--host", "0.0.0.0", "--port", "8001"]