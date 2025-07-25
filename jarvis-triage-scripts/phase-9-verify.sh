#!/bin/bash
# Final Verification Script
# Agent: Quality Guardian
# Phase: 9 - Comprehensive Verification

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🛡️ Quality Guardian: Running final verification...${NC}"

# Initialize report
REPORT_FILE="verification-report.json"
PASSED_CHECKS=0
FAILED_CHECKS=0

# Function to check condition
check() {
    local description=$1
    local command=$2
    local expected=${3:-0}
    
    echo -n "Checking: $description... "
    
    if eval "$command" &>/dev/null; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((PASSED_CHECKS++))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        ((FAILED_CHECKS++))
        return 1
    fi
}

# Function to measure size
get_size() {
    if [ -d "$1" ]; then
        du -sh "$1" 2>/dev/null | cut -f1 || echo "N/A"
    else
        echo "Not found"
    fi
}

echo -e "${YELLOW}📊 Repository Verification${NC}"
echo "================================"

# Structure checks
echo -e "\n${YELLOW}📁 Directory Structure${NC}"
check "docs/ directory exists" "[ -d docs ]"
check "services/ directory exists" "[ -d services ]"
check "infra/ directory exists" "[ -d infra ]"
check "tools/ directory exists" "[ -d tools ]"
check "Orchestrator service exists" "[ -d services/orchestrator ]"
check "Core library exists" "[ -d services/core ]"
check "UI service exists" "[ -d services/ui ]"

# Git checks
echo -e "\n${YELLOW}🔧 Git Repository${NC}"
REPO_SIZE=$(get_size .git)
echo "Repository size: $REPO_SIZE"
check "Repository size < 500MB" "[ $(du -sm .git | cut -f1) -lt 500 ]"
check ".gitignore exists" "[ -f .gitignore ]"
check "No __pycache__ in git" "! git ls-files | grep -q __pycache__"
check "No node_modules in git" "! git ls-files | grep -q node_modules"

# Configuration checks
echo -e "\n${YELLOW}⚙️ Configuration${NC}"
check ".env.example exists" "[ -f .env.example ]"
check "Docker compose file exists" "[ -f docker-compose.yml ]"
check "MkDocs configuration exists" "[ -f mkdocs.yml ]"
check "Pre-commit config exists" "[ -f .pre-commit-config.yaml ]"

# Python checks
echo -e "\n${YELLOW}🐍 Python Services${NC}"
for service in orchestrator core plugins; do
    if [ -d "services/$service" ]; then
        check "$service has pyproject.toml" "[ -f services/$service/pyproject.toml ]"
        check "$service has poetry.lock" "[ -f services/$service/poetry.lock ]"
    fi
done

# Documentation checks
echo -e "\n${YELLOW}📚 Documentation${NC}"
check "README.md exists" "[ -f README.md ]"
check "CONTRIBUTING.md exists" "[ -f CONTRIBUTING.md ]"
check "SECURITY.md exists" "[ -f SECURITY.md ]"
check "Docs index exists" "[ -f docs/index.md ]"
check "Archive directory exists" "[ -d docs/archive ]"

# CI/CD checks
echo -e "\n${YELLOW}🚀 CI/CD${NC}"
check "GitHub Actions directory" "[ -d .github/workflows ]"
check "CI workflow exists" "[ -f .github/workflows/ci.yml ] || [ -f jarvis-triage-scripts/phase-5-cicd.yml ]"

# Docker checks
echo -e "\n${YELLOW}🐳 Docker${NC}"
check "Orchestrator Dockerfile exists" "[ -f jarvis-triage-scripts/dockerfiles/orchestrator.Dockerfile ]"
check "UI Dockerfile exists" "[ -f jarvis-triage-scripts/dockerfiles/ui.Dockerfile ]"
check "docker-compose.yml is valid" "docker-compose config -q"

# Scripts checks
echo -e "\n${YELLOW}📜 Triage Scripts${NC}"
check "All phase scripts created" "ls jarvis-triage-scripts/phase-*.sh 2>/dev/null | wc -l | grep -q 6"
check "Scripts are executable" "[ -x jarvis-triage-scripts/phase-0-backup.sh ]"

# Generate summary
echo -e "\n${CYAN}📊 Verification Summary${NC}"
echo "================================"
TOTAL_CHECKS=$((PASSED_CHECKS + FAILED_CHECKS))
SUCCESS_RATE=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))

echo -e "Total checks: $TOTAL_CHECKS"
echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"
echo -e "Success rate: $SUCCESS_RATE%"

# Generate JSON report
cat > "$REPORT_FILE" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "agent": "Quality Guardian",
  "total_checks": $TOTAL_CHECKS,
  "passed": $PASSED_CHECKS,
  "failed": $FAILED_CHECKS,
  "success_rate": $SUCCESS_RATE,
  "repository_size": "$REPO_SIZE",
  "status": $([ $FAILED_CHECKS -eq 0 ] && echo '"READY"' || echo '"NEEDS_ATTENTION"')
}
EOF

# Final verdict
echo ""
if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed! Repository is ready for production.${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️ Some checks failed. Review and fix issues before proceeding.${NC}"
    exit 1
fi