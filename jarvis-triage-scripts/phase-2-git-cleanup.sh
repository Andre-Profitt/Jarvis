#!/bin/bash
# Git History Cleanup Script
# Agent: Git Surgeon
# Phase: 2 - Purge Heavy History

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${RED}🔧 Git Surgeon: Beginning repository surgery...${NC}"

# Safety check
if [ ! -f "jarvis-backup-*.tgz" ]; then
    echo -e "${RED}❌ ERROR: No backup found! Run phase-0-backup.sh first${NC}"
    exit 1
fi

# Install git-filter-repo if needed
if ! command -v git-filter-repo &> /dev/null; then
    echo -e "${YELLOW}📦 Installing git-filter-repo...${NC}"
    pip install git-filter-repo
fi

# Paths to remove from history
REMOVE_PATHS=(
    "artifacts/"
    "training_data/"
    "JARVIS-KNOWLEDGE/"
    ".ruv-swarm/"
    "checkpoints/"
    "__pycache__/"
    "node_modules/"
    "build/"
    "dist/"
    ".idea/"
    ".vscode/"
)

# File patterns to remove
REMOVE_PATTERNS=(
    "*.pt"
    "*.h5"
    "*.onnx"
    "*.pkl"
    "*.zip"
    "*.tar.gz"
    "*.pyc"
    "*.pyo"
    "*.pyd"
    "*.so"
    "*.dylib"
    "*.dll"
    "*.egg"
    "*.egg-info"
    ".DS_Store"
    "Thumbs.db"
)

echo -e "${YELLOW}📊 Repository size before cleanup:${NC}"
du -sh .git

# Remove paths
echo -e "${YELLOW}🗑️ Removing artifact directories from history...${NC}"
for path in "${REMOVE_PATHS[@]}"; do
    if git ls-files --error-unmatch "$path" &>/dev/null || git ls-tree -r HEAD --name-only | grep -q "^$path"; then
        echo "  Removing: $path"
        git filter-repo --path "$path" --invert-paths --force
    fi
done

# Remove file patterns
echo -e "${YELLOW}🗑️ Removing artifact files from history...${NC}"
for pattern in "${REMOVE_PATTERNS[@]}"; do
    echo "  Removing pattern: $pattern"
    git filter-repo --path-glob "$pattern" --invert-paths --force
done

# Remove empty commits
echo -e "${YELLOW}🧹 Removing empty commits...${NC}"
git filter-repo --empty=drop --force

# Aggressive garbage collection
echo -e "${YELLOW}🗜️ Running aggressive garbage collection...${NC}"
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo -e "${GREEN}📊 Repository size after cleanup:${NC}"
du -sh .git

# Size comparison
BEFORE_SIZE=$(git count-objects -vH | grep "size-pack" | cut -d' ' -f2)
echo -e "${GREEN}✅ Git surgery complete!${NC}"
echo -e "Final repository size: ${BEFORE_SIZE}"

# Create cleanup report
cat > git-cleanup-report.json << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "removed_paths": $(printf '%s\n' "${REMOVE_PATHS[@]}" | jq -R . | jq -s .),
  "removed_patterns": $(printf '%s\n' "${REMOVE_PATTERNS[@]}" | jq -R . | jq -s .),
  "final_size": "${BEFORE_SIZE}",
  "agent": "Git Surgeon"
}
EOF

echo -e "${GREEN}📄 Cleanup report saved to git-cleanup-report.json${NC}"