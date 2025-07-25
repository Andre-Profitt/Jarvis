#!/bin/bash
# Documentation Organization Script
# Agent: Documentation Curator
# Phase: 7 - Curate Documentation

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}📚 Documentation Curator: Organizing documentation...${NC}"

# Create documentation structure
echo -e "${YELLOW}📁 Creating MkDocs structure...${NC}"
mkdir -p docs/{getting-started,api,architecture,guides,reference,archive}

# Create MkDocs configuration
cat > mkdocs.yml << 'EOF'
site_name: Jarvis AI Documentation
site_description: Advanced AI Ecosystem Documentation
site_author: Jarvis Team
site_url: https://jarvis-ai.github.io/jarvis

repo_name: Andre-Profitt/Jarvis
repo_url: https://github.com/Andre-Profitt/Jarvis
edit_uri: edit/main/docs/

theme:
  name: material
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.instant
    - navigation.tracking
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - navigation.top
    - search.suggest
    - search.highlight
    - content.tabs.link
    - content.code.annotation
    - content.code.copy

plugins:
  - search
  - mermaid2
  - git-revision-date-localized:
      enable_creation_date: true

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - attr_list
  - md_in_html
  - toc:
      permalink: true

nav:
  - Home: index.md
  - Getting Started:
    - Quick Start: getting-started/quickstart.md
    - Installation: getting-started/installation.md
    - Configuration: getting-started/configuration.md
    - First Steps: getting-started/first-steps.md
  - Architecture:
    - Overview: architecture/overview.md
    - Services: architecture/services.md
    - Plugins: architecture/plugins.md
    - Data Flow: architecture/data-flow.md
  - API Reference:
    - REST API: api/rest.md
    - WebSocket: api/websocket.md
    - Python SDK: api/python-sdk.md
    - JavaScript SDK: api/js-sdk.md
  - Guides:
    - Development: guides/development.md
    - Deployment: guides/deployment.md
    - Contributing: guides/contributing.md
    - Security: guides/security.md
  - Reference:
    - CLI: reference/cli.md
    - Environment: reference/environment.md
    - Troubleshooting: reference/troubleshooting.md
    - FAQ: reference/faq.md

extra:
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/Andre-Profitt/Jarvis
    - icon: fontawesome/brands/docker
      link: https://hub.docker.com/r/jarvis-ai/jarvis
  version:
    provider: mike
EOF

# Create main documentation files
echo -e "${YELLOW}📝 Creating documentation content...${NC}"

# Index page
cat > docs/index.md << 'EOF'
# Jarvis AI Documentation

Welcome to the official documentation for **Jarvis AI** - an advanced AI ecosystem designed to provide seamless, intelligent assistance.

## What is Jarvis?

Jarvis is a comprehensive AI assistant platform that combines:

- 🤖 **Natural Language Processing** - Understand and respond to human language
- 🧠 **Machine Learning** - Continuously improve through interaction
- 🔌 **Plugin Architecture** - Extend functionality with custom modules
- 🌐 **Multi-Platform** - Web, mobile, and API access
- 🔒 **Privacy-First** - Your data stays yours

## Quick Links

<div class="grid cards" markdown>

- :rocket: **[Quick Start](getting-started/quickstart.md)**
  
    Get up and running in 5 minutes

- :wrench: **[API Reference](api/rest.md)**
  
    Complete API documentation

- :building_construction: **[Architecture](architecture/overview.md)**
  
    System design and components

- :handshake: **[Contributing](guides/contributing.md)**
  
    Help improve Jarvis

</div>

## Features

### Intelligent Conversation
Jarvis understands context and maintains conversation history for natural interactions.

### Multi-Modal Input
Support for text, voice, and image inputs across all platforms.

### Extensible Design
Build custom plugins to add new capabilities and integrations.

### Real-Time Processing
WebSocket support for instant responses and live updates.

## Getting Help

- 📖 Browse the [documentation](getting-started/quickstart.md)
- 💬 Join our [Discord community](https://discord.gg/jarvis-ai)
- 🐛 Report issues on [GitHub](https://github.com/Andre-Profitt/Jarvis/issues)
- 📧 Contact support at support@jarvis-ai.dev
EOF

# Quick Start guide
cat > docs/getting-started/quickstart.md << 'EOF'
# Quick Start Guide

Get Jarvis up and running in minutes!

## Prerequisites

- Python 3.12 or higher
- Node.js 20 or higher
- Docker (optional, for containerized deployment)
- 8GB RAM minimum

## Installation

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/Andre-Profitt/Jarvis.git
cd Jarvis

# Start with Docker Compose
docker-compose up -d

# Check status
docker-compose ps
```

### Manual Installation

=== "Orchestrator Service"

    ```bash
    cd services/orchestrator
    poetry install
    poetry run uvicorn src.main:app --reload
    ```

=== "UI Service"

    ```bash
    cd services/ui
    npm install
    npm run dev
    ```

=== "Mobile App"

    ```bash
    cd services/mobile_app
    npm install
    npm run ios  # or npm run android
    ```

## First Request

Test the API with curl:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, Jarvis!"}'
```

## Next Steps

- [Configure your environment](configuration.md)
- [Explore the API](../api/rest.md)
- [Build your first plugin](../guides/development.md)
EOF

# Architecture overview
cat > docs/architecture/overview.md << 'EOF'
# Architecture Overview

Jarvis follows a microservices architecture with clear separation of concerns.

## System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        UI[Web UI]
        Mobile[Mobile App]
        API[API Clients]
    end
    
    subgraph "Service Layer"
        Orchestrator[Orchestrator Service]
        Core[Core Library]
        Plugins[Plugin System]
    end
    
    subgraph "Data Layer"
        PostgreSQL[(PostgreSQL)]
        Redis[(Redis)]
        S3[Object Storage]
    end
    
    UI --> Orchestrator
    Mobile --> Orchestrator
    API --> Orchestrator
    
    Orchestrator --> Core
    Orchestrator --> Plugins
    Orchestrator --> PostgreSQL
    Orchestrator --> Redis
    
    Core --> S3
```

## Key Components

### Orchestrator Service
- FastAPI-based REST API
- WebSocket support for real-time communication
- Request routing and load balancing
- Authentication and authorization

### Core Library
- Natural language processing
- Machine learning models
- Business logic implementation
- No direct I/O operations

### Plugin System
- Dynamic plugin loading
- Standardized plugin interface
- Isolated execution environment
- Version management

## Design Principles

1. **Separation of Concerns** - Each service has a single responsibility
2. **Scalability** - Horizontal scaling through containerization
3. **Resilience** - Fault tolerance and graceful degradation
4. **Security** - Defense in depth, least privilege access
5. **Observability** - Comprehensive logging and monitoring
EOF

# Move and organize existing docs
echo -e "${YELLOW}🚚 Organizing existing documentation...${NC}"

# Archive old vision documents
for doc in CLAUDE*.md ELITE*.md WORLD_CLASS*.md; do
    if [ -f "$doc" ]; then
        mv "$doc" docs/archive/ 2>/dev/null || true
        echo "  Archived: $doc"
    fi
done

# Keep essential docs in root
KEEP_DOCS=("README.md" "LICENSE" "CONTRIBUTING.md" "SECURITY.md")
for doc in "${KEEP_DOCS[@]}"; do
    if [ ! -f "$doc" ]; then
        echo "  Creating: $doc"
        case "$doc" in
            "CONTRIBUTING.md")
                cat > "$doc" << 'EOF'
# Contributing to Jarvis

We love contributions! Please read our guidelines before submitting PRs.

## Code of Conduct
Be respectful and inclusive. We're all here to build something amazing together.

## How to Contribute
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a PR

## Development Setup
See our [development guide](docs/guides/development.md) for setup instructions.
EOF
                ;;
            "SECURITY.md")
                cat > "$doc" << 'EOF'
# Security Policy

## Reporting Security Vulnerabilities

Please report security vulnerabilities to security@jarvis-ai.dev

Do NOT create public issues for security problems.

## Supported Versions
- Current release: Full support
- Previous release: Security fixes only
- Older versions: No support
EOF
                ;;
        esac
    fi
done

# Create documentation requirements
cat > docs/requirements.txt << 'EOF'
mkdocs==1.5.3
mkdocs-material==9.5.3
mkdocs-mermaid2-plugin==1.1.1
mkdocs-git-revision-date-localized-plugin==1.2.2
pymdown-extensions==10.5
EOF

echo -e "${GREEN}✅ Documentation structure created!${NC}"
echo -e "${PURPLE}📋 Next steps:${NC}"
echo "  1. Install MkDocs: pip install -r docs/requirements.txt"
echo "  2. Preview docs: mkdocs serve"
echo "  3. Build docs: mkdocs build"
echo "  4. Deploy to GitHub Pages: mkdocs gh-deploy"