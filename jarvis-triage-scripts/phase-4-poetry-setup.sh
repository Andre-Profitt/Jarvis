#!/bin/bash
# Poetry Setup Script for Jarvis Services
# Agent: Python Modernizer
# Phase: 4 - Standardize Python Packaging

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
MAGENTA='\033[0;35m'
NC='\033[0m'

echo -e "${MAGENTA}🐍 Python Modernizer: Setting up Poetry for all services...${NC}"

# Install Poetry if not present
if ! command -v poetry &> /dev/null; then
    echo -e "${YELLOW}📦 Installing Poetry...${NC}"
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Function to setup Poetry for a service
setup_poetry_service() {
    local service_name=$1
    local service_path=$2
    local python_version=${3:-">=3.12,<3.13"}
    
    echo -e "${YELLOW}🔧 Setting up Poetry for ${service_name}...${NC}"
    
    cd "${service_path}"
    
    # Initialize Poetry project
    poetry init --name "jarvis-${service_name}" \
                --python "${python_version}" \
                --no-interaction
    
    # Add common dependencies based on service type
    case $service_name in
        "orchestrator")
            poetry add fastapi uvicorn pydantic pydantic-settings redis asyncpg
            poetry add --group dev pytest pytest-asyncio pytest-cov black ruff mypy
            ;;
        "core")
            poetry add numpy pandas scikit-learn
            poetry add --group dev pytest pytest-cov black ruff mypy
            ;;
        "plugins")
            poetry add pluggy pydantic
            poetry add --group dev pytest black ruff
            ;;
    esac
    
    # Create initial source structure
    mkdir -p src tests
    
    # Create __init__.py files
    touch src/__init__.py tests/__init__.py
    
    # Create pyproject.toml additions
    cat >> pyproject.toml << EOF

[tool.poetry.scripts]
jarvis-${service_name} = "src.main:main"

[tool.black]
line-length = 88
target-version = ['py312']

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "N", "UP", "S", "B", "A", "C90", "T20"]

[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
addopts = "-v --cov=src --cov-report=term-missing"
EOF
    
    # Lock dependencies
    poetry lock
    
    echo -e "${GREEN}✓ Poetry setup complete for ${service_name}${NC}"
    cd - > /dev/null
}

# Setup Poetry for each service
echo -e "${YELLOW}🚀 Setting up Python services...${NC}"

# Orchestrator
setup_poetry_service "orchestrator" "services/orchestrator"

# Core library
setup_poetry_service "core" "services/core"

# Plugin base
setup_poetry_service "plugins" "services/plugins"

# Create a workspace pyproject.toml at root
echo -e "${YELLOW}📋 Creating workspace configuration...${NC}"
cat > pyproject.toml << 'EOF'
# Jarvis Workspace Configuration
[tool.poetry]
name = "jarvis-workspace"
version = "0.1.0"
description = "Jarvis AI Ecosystem Workspace"
authors = ["Jarvis Team"]

[tool.poetry.dependencies]
python = ">=3.12,<3.13"

[tool.poetry.group.dev.dependencies]
pre-commit = "^3.7.0"
black = "^24.4.0"
ruff = "^0.4.4"
mypy = "^1.10.0"

[tool.black]
line-length = 88
target-version = ['py312']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
  | services/.*/\.venv
)/
'''

[tool.ruff]
line-length = 88
target-version = "py312"
extend-exclude = ["services/*/build", "services/*/dist"]

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
EOF

# Create shared development tools configuration
echo -e "${YELLOW}🛠️ Setting up shared development tools...${NC}"

# Pre-commit configuration
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: check-toml
      - id: check-json
      - id: detect-private-key

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/psf/black
    rev: 24.4.0
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        exclude: ^(docs/|tests/)
EOF

echo -e "${GREEN}✅ Poetry setup complete for all services!${NC}"
echo -e "${MAGENTA}📝 Next steps:${NC}"
echo "  1. cd into each service directory"
echo "  2. Run: poetry install"
echo "  3. Activate environment: poetry shell"
echo "  4. Start developing!"