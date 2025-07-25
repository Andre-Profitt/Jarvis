#!/bin/bash
# Jarvis Repository Backup Script
# Agent: Git Surgeon
# Phase: 0 - Safe Snapshot Creation

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🔧 Git Surgeon: Starting Jarvis repository backup...${NC}"

# Configuration
REPO_URL="https://github.com/Andre-Profitt/Jarvis"
BACKUP_DIR="jarvis-backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="jarvis-backup-${TIMESTAMP}"

# Create backup directory
mkdir -p "${BACKUP_DIR}"
cd "${BACKUP_DIR}"

echo -e "${YELLOW}📦 Creating mirror clone...${NC}"
git clone --mirror "${REPO_URL}" "${BACKUP_NAME}.git"

echo -e "${YELLOW}🗜️ Compressing backup...${NC}"
tar -czf "${BACKUP_NAME}.tgz" "${BACKUP_NAME}.git"

# Calculate sizes
REPO_SIZE=$(du -sh "${BACKUP_NAME}.git" | cut -f1)
ARCHIVE_SIZE=$(du -sh "${BACKUP_NAME}.tgz" | cut -f1)

echo -e "${GREEN}✅ Backup complete!${NC}"
echo -e "Repository size: ${REPO_SIZE}"
echo -e "Archive size: ${ARCHIVE_SIZE}"
echo -e "Location: ${PWD}/${BACKUP_NAME}.tgz"

# Store backup metadata
cat > "${BACKUP_NAME}.metadata.json" << EOF
{
  "timestamp": "${TIMESTAMP}",
  "repo_url": "${REPO_URL}",
  "repo_size": "${REPO_SIZE}",
  "archive_size": "${ARCHIVE_SIZE}",
  "archive_path": "${PWD}/${BACKUP_NAME}.tgz",
  "agent": "Git Surgeon",
  "phase": 0
}
EOF

# Cleanup uncompressed backup
rm -rf "${BACKUP_NAME}.git"

echo -e "${GREEN}🔒 Backup secured and metadata saved${NC}"