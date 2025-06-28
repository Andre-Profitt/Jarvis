# JARVIS Ecosystem Development Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Go 1.19+ (for Gemini CLI)
- Redis server
- Docker (optional, for containerized deployment)

### Initial Setup

1. **Install Poetry** (modern dependency management):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Clone and setup the project**:
   ```bash
   git clone <repository-url>
   cd JARVIS-ECOSYSTEM
   make dev-setup
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Launch JARVIS**:
   ```bash
   make run
   ```

## 📦 Dependency Management

### Why Poetry?

We use Poetry instead of pip + requirements.txt because:
- **Deterministic builds**: `poetry.lock` ensures everyone has exact same dependencies
- **Better dependency resolution**: Prevents conflicts automatically
- **Integrated virtual environment**: No need to manually manage venv
- **Separation of concerns**: Dev dependencies are clearly separated
- **Easy updates**: `poetry update` handles everything safely

### Common Commands

```bash
# Install all dependencies
poetry install

# Add a new dependency
poetry add package-name

# Add a dev dependency
poetry add --group dev package-name

# Update dependencies
poetry update

# Show dependency tree
poetry show --tree

# Export to requirements.txt (if needed)
poetry export -f requirements.txt --output requirements.txt
```

### Dependency Groups

- **Core**: Essential dependencies always installed
- **dev**: Development tools (testing, linting, etc.)
- **distributed**: Optional distributed computing (Ray, Dask)
- **mlops**: Optional ML operations (Wandb, MLflow)
- **kubernetes**: Optional K8s support

Install specific groups:
```bash
poetry install --with distributed,mlops
```

## 🛠️ Development Workflow

### 1. Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

Hooks include:
- Black (formatting)
- isort (import sorting)
- Flake8 (linting)
- MyPy (type checking)
- Bandit (security)
- Safety (vulnerability scanning)

### 2. Testing

```bash
# Run all tests
make test

# Run specific test
poetry run pytest tests/test_consciousness.py

# Run with coverage
poetry run pytest --cov=core --cov-report=html
```

### 3. Code Quality

```bash
# Format code
make format

# Run all linters
make lint

# Security scan
make security-scan
```

## 🏗️ Project Structure

```
JARVIS-ECOSYSTEM/
├── core/                    # Core JARVIS modules
│   ├── consciousness_*.py   # Consciousness simulation
│   ├── neural_*.py         # Neural resource management
│   ├── emotional_*.py      # Emotional intelligence
│   └── websocket_*.py      # WebSocket communication
├── plugins/                # Optional plugins
├── tests/                  # Test suite
├── docs/                   # Documentation
├── scripts/                # Utility scripts
├── pyproject.toml         # Poetry configuration
├── poetry.lock            # Locked dependencies
├── Makefile               # Development commands
└── .pre-commit-config.yaml # Pre-commit hooks
```

## 🔧 Configuration

### Environment Variables

Create `.env` file with:
```env
# API Keys
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
ELEVENLABS_API_KEY=your-key-here
GEMINI_API_KEY=your-key-here

# Configuration
JARVIS_PORT=8765
JARVIS_LOG_LEVEL=INFO
REDIS_URL=redis://localhost:6379
```

### pyproject.toml

Key sections:
- `[tool.poetry.dependencies]`: Production dependencies
- `[tool.poetry.group.dev.dependencies]`: Development tools
- `[tool.poetry.extras]`: Optional feature sets
- `[tool.black]`, `[tool.isort]`, etc.: Tool configurations

## 🐛 Troubleshooting

### Dependency Conflicts

```bash
# Check for conflicts
poetry check

# Show why a package is needed
poetry show --tree package-name

# Clear cache and reinstall
poetry cache clear pypi --all
poetry install --no-cache
```

### Virtual Environment Issues

```bash
# Show env info
poetry env info

# Remove and recreate
poetry env remove python
poetry install
```

### Performance Issues

```bash
# Profile the application
make profile

# Memory profiling
poetry run python -m memory_profiler LAUNCH-JARVIS-REAL.py
```

## 🚢 Deployment

### Docker

```bash
# Build image
make docker-build

# Run container
make docker-run
```

### Production

1. Use `poetry install --no-dev` for production
2. Set appropriate environment variables
3. Use a process manager (systemd, supervisor)
4. Configure proper logging
5. Set up monitoring (Prometheus metrics included)

## 📊 Monitoring

JARVIS includes built-in observability:
- Prometheus metrics at `/metrics`
- OpenTelemetry tracing
- Structured logging with structlog

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `make lint test`
5. Commit with conventional commits
6. Create a pull request

## 📚 Additional Resources

- [Poetry Documentation](https://python-poetry.org/docs/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Black Code Style](https://black.readthedocs.io/)
- [MyPy Type Checking](https://mypy.readthedocs.io/)

## 🆘 Getting Help

- Check the logs: `tail -f jarvis.log`
- Run diagnostics: `python pre-deployment-check.py`
- Enable debug mode: Set `JARVIS_LOG_LEVEL=DEBUG`
- Open an issue on GitHub