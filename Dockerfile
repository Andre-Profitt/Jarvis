FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libpq-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support (CPU version for now)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs models training_data storage mcp_servers

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create non-root user
RUN useradd -m -u 1000 jarvis && chown -R jarvis:jarvis /app
USER jarvis

# Expose ports
EXPOSE 8765 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Default command
CMD ["python", "LAUNCH-JARVIS-REAL.py"]