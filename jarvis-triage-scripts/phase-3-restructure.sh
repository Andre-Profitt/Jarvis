#!/bin/bash
# Jarvis Repository Restructuring Script
# Agent: Structure Architect
# Phase: 3 - Create Three-Layer Architecture

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🏗️ Structure Architect: Beginning repository restructure...${NC}"

# Create new structure
echo -e "${YELLOW}📁 Creating three-layer architecture...${NC}"

# Main directories
mkdir -p docs/{guides,api,architecture,archive}
mkdir -p services/{orchestrator,core,plugins,mobile_app,ui}
mkdir -p infra/{terraform,k8s,docker}
mkdir -p tools/{scripts,migrations}

# Service subdirectories
echo -e "${YELLOW}🔧 Setting up service structures...${NC}"

# Orchestrator service
mkdir -p services/orchestrator/{src,tests,config}
cat > services/orchestrator/README.md << 'EOF'
# Jarvis Orchestrator Service

FastAPI-based orchestration layer for the Jarvis AI ecosystem.

## Setup
```bash
poetry install
poetry run uvicorn src.main:app --reload
```

## Architecture
- FastAPI for HTTP API
- Redis for caching
- PostgreSQL for persistence
- WebSocket support for real-time features
EOF

# Core library
mkdir -p services/core/{jarvis_core,tests}
cat > services/core/README.md << 'EOF'
# Jarvis Core Library

Pure Python implementation of core Jarvis functionality.
No I/O operations - designed for maximum reusability.

## Installation
```bash
poetry add jarvis-core
```
EOF

# Plugin system
mkdir -p services/plugins/{jarvis_plugin_base,examples}
cat > services/plugins/README.md << 'EOF'
# Jarvis Plugin System

Extensible plugin architecture for Jarvis.

## Creating a Plugin
1. Inherit from `JarvisPlugin`
2. Implement required methods
3. Register with orchestrator
EOF

# UI service
mkdir -p services/ui/{components,pages,public,styles}
cat > services/ui/README.md << 'EOF'
# Jarvis UI

Next.js 14 application with TypeScript and Tailwind CSS.

## Development
```bash
npm install
npm run dev
```
EOF

# Mobile app
mkdir -p services/mobile_app/{src,ios,android,assets}
cat > services/mobile_app/README.md << 'EOF'
# Jarvis Mobile App

React Native application for iOS and Android.

## Setup
```bash
npm install
npx pod-install
npm run ios
npm run android
```
EOF

# Move existing files to new structure
echo -e "${YELLOW}🚚 Migrating existing components...${NC}"

# Map old structure to new
declare -A migrations=(
    ["orchestrator"]="services/orchestrator/src"
    ["core"]="services/core/jarvis_core"
    ["plugins"]="services/plugins"
    ["jarvis-ui"]="services/ui"
    ["mobile_app"]="services/mobile_app"
    ["docs"]="docs/archive"
)

# Create migration script
cat > tools/scripts/migrate-structure.py << 'EOF'
#!/usr/bin/env python3
"""
Migrate Jarvis repository to new structure
Agent: Structure Architect
"""
import os
import shutil
from pathlib import Path
import json

def migrate_component(src, dst):
    """Move component preserving git history"""
    if not Path(src).exists():
        print(f"  ⚠️  Source not found: {src}")
        return False
    
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    
    # Use git mv to preserve history
    os.system(f"git mv {src} {dst} 2>/dev/null || mv {src} {dst}")
    print(f"  ✓ Migrated: {src} → {dst}")
    return True

def main():
    migrations = {
        "orchestrator": "services/orchestrator/src",
        "core": "services/core/jarvis_core",
        "plugins": "services/plugins",
        "jarvis-ui": "services/ui",
        "jarvis-world-class-ui": "services/ui/legacy",
        "web_dashboard": "services/ui/dashboard",
        "mobile_app": "services/mobile_app",
    }
    
    print("🚀 Starting migration...")
    
    for src, dst in migrations.items():
        if Path(src).exists():
            migrate_component(src, dst)
    
    # Archive old docs
    old_docs = ["CLAUDE*.md", "ELITE_*.md", "*.md"]
    for pattern in old_docs:
        os.system(f"find . -maxdepth 1 -name '{pattern}' -exec mv {{}} docs/archive/ \; 2>/dev/null")
    
    print("✅ Migration complete!")

if __name__ == "__main__":
    main()
EOF

chmod +x tools/scripts/migrate-structure.py

echo -e "${GREEN}✅ Repository structure prepared!${NC}"
echo -e "${BLUE}📋 Next steps:${NC}"
echo "  1. Review the new structure"
echo "  2. Run migration script: python tools/scripts/migrate-structure.py"
echo "  3. Update import paths in Python files"
echo "  4. Commit the restructure"