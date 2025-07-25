#!/bin/bash
# Jarvis Repository Analysis Script
# Agent: Repository Analyzer
# Phase: 1 - Identify Artifacts

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 Repository Analyzer: Starting artifact identification...${NC}"

# Work in a fresh clone
WORK_DIR="jarvis-analysis"
rm -rf "${WORK_DIR}"
git clone https://github.com/Andre-Profitt/Jarvis "${WORK_DIR}"
cd "${WORK_DIR}"

# Analysis results file
REPORT="artifact-analysis-report.json"

echo -e "${YELLOW}📊 Analyzing repository structure...${NC}"

# Function to get size in bytes
get_size_bytes() {
    if [[ -d "$1" ]]; then
        du -sb "$1" 2>/dev/null | cut -f1 || echo "0"
    else
        stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null || echo "0"
    fi
}

# Initialize report
cat > "${REPORT}" << 'EOF'
{
  "analysis_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "artifacts_to_remove": {
    "large_binaries": [],
    "build_artifacts": [],
    "cache_files": [],
    "node_modules": [],
    "secrets": []
  },
  "size_analysis": {},
  "recommendations": []
}
EOF

echo -e "${YELLOW}🔎 Scanning for large binaries and data files...${NC}"
find . -type f \( -name "*.pt" -o -name "*.h5" -o -name "*.onnx" -o -name "*.pkl" -o -name "*.zip" -o -name "*.tar.gz" \) -size +1M | while read -r file; do
    size=$(get_size_bytes "$file")
    echo "  Found: $file ($(numfmt --to=iec-i --suffix=B $size))"
done

echo -e "${YELLOW}🗑️ Identifying build and cache artifacts...${NC}"
find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name "build" -o -name "dist" -o -name "*.egg-info" \) | while read -r dir; do
    size=$(get_size_bytes "$dir")
    echo "  Found: $dir ($(numfmt --to=iec-i --suffix=B $size))"
done

echo -e "${YELLOW}📦 Checking for node_modules...${NC}"
find . -type d -name "node_modules" | while read -r dir; do
    size=$(get_size_bytes "$dir")
    echo "  Found: $dir ($(numfmt --to=iec-i --suffix=B $size))"
done

echo -e "${YELLOW}🔐 Scanning for potential secrets...${NC}"
find . -type f \( -name ".env" -o -name "*.pem" -o -name "*.key" -o -name "*.crt" \) | while read -r file; do
    echo "  Found: $file"
done

# Special directories from the playbook
echo -e "${YELLOW}📁 Checking special directories...${NC}"
for dir in "artifacts" "training_data" "JARVIS-KNOWLEDGE" ".ruv-swarm"; do
    if [[ -d "$dir" ]]; then
        size=$(get_size_bytes "$dir")
        echo -e "  ${RED}Heavy directory: $dir ($(numfmt --to=iec-i --suffix=B $size))${NC}"
    fi
done

# Generate .gitignore recommendations
echo -e "${GREEN}📝 Generating .gitignore recommendations...${NC}"
cat > recommended.gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST
.pytest_cache/
.coverage
.tox/
.nox/
.hypothesis/
*.cover
.env
.venv
env/
venv/
ENV/

# ML/AI artifacts
artifacts/
training_data/
JARVIS-KNOWLEDGE/
.ruv-swarm/
checkpoints/
*.pt
*.h5
*.onnx
*.pkl
*.model
*.weights

# Node
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.npm

# IDE
.idea/
.vscode/
*.swp
*.swo
*~
.project
.classpath
.c9/
*.launch
.settings/

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Mobile
*.apk
*.ipa
*.dex
mobile_app/build/
mobile_app/.gradle/

# Secrets
*.pem
*.key
*.crt
.env*
!.env.example
EOF

echo -e "${GREEN}✅ Analysis complete! Check artifact-analysis-report.json for details${NC}"