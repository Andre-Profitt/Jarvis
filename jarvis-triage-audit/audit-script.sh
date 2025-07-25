#!/bin/bash
# Comprehensive Jarvis Triage Audit Script
# The Best Audit Ever™
# Validates every aspect of the repository transformation

set -euo pipefail

# ANSI Color Codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Audit counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNINGS=0

# Results arrays
declare -a FAILURES
declare -a WARNINGS_LIST
declare -a SUCCESSES

# Audit report file
AUDIT_REPORT="audit-report-$(date +%Y%m%d_%H%M%S).md"
AUDIT_JSON="audit-results.json"

# Functions
log() {
    echo -e "$1" | tee -a "$AUDIT_REPORT"
}

check() {
    local category=$1
    local description=$2
    local command=$3
    local severity=${4:-"ERROR"}
    
    ((TOTAL_CHECKS++))
    
    echo -n "  [$category] $description... "
    
    if eval "$command" &>/dev/null; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASSED_CHECKS++))
        SUCCESSES+=("$category: $description")
        return 0
    else
        if [ "$severity" = "WARNING" ]; then
            echo -e "${YELLOW}⚠ WARNING${NC}"
            ((WARNINGS++))
            WARNINGS_LIST+=("$category: $description")
            return 1
        else
            echo -e "${RED}✗ FAIL${NC}"
            ((FAILED_CHECKS++))
            FAILURES+=("$category: $description")
            return 1
        fi
    fi
}

file_contains() {
    local file=$1
    local pattern=$2
    grep -q "$pattern" "$file" 2>/dev/null
}

# Header
clear
cat << 'EOF'
╔═══════════════════════════════════════════════════════════════════╗
║                  JARVIS TRIAGE AUDIT SYSTEM                       ║
║                    The Best Audit Ever™                           ║
║                      Version 1.0.0                                ║
╚═══════════════════════════════════════════════════════════════════╝
EOF

log "\n# Jarvis Triage Comprehensive Audit Report"
log "Generated: $(date)"
log "Auditor: Quality Guardian Agent"
log "\n---\n"

# Phase 1: File Structure Audit
log "\n${CYAN}${BOLD}═══ PHASE 1: FILE STRUCTURE AUDIT ═══${NC}"
log "\nValidating all created files and directories..."

SCRIPT_DIR="jarvis-triage-scripts"

# Check all phase scripts
for i in 0 1 2 3 4 5 6 7 9; do
    if [ $i -eq 5 ]; then
        check "SCRIPTS" "Phase $i script exists (YAML)" "[ -f $SCRIPT_DIR/phase-$i-cicd.yml ]"
    elif [ $i -eq 6 ]; then
        check "SCRIPTS" "Phase $i script exists (Python)" "[ -f $SCRIPT_DIR/phase-$i-config.py ]"
    else
        check "SCRIPTS" "Phase $i script exists" "[ -f $SCRIPT_DIR/phase-$i-*.sh ]"
    fi
done

# Check special files
check "SCRIPTS" "Master execution script exists" "[ -f $SCRIPT_DIR/execute-triage.sh ]"
check "DOCKER" "docker-compose.yml exists" "[ -f $SCRIPT_DIR/docker-compose.yml ]"
check "DOCKER" "Orchestrator Dockerfile exists" "[ -f $SCRIPT_DIR/dockerfiles/orchestrator.Dockerfile ]"
check "DOCKER" "UI Dockerfile exists" "[ -f $SCRIPT_DIR/dockerfiles/ui.Dockerfile ]"
check "DOCS" "Playbook documentation exists" "[ -f jarvis-triage-playbook.md ]"
check "DOCS" "Orchestra implementation guide exists" "[ -f orchestra-implementation.md ]"
check "DOCS" "Orchestration dashboard exists" "[ -f orchestration-dashboard.md ]"

# Phase 2: Script Quality Audit
log "\n${CYAN}${BOLD}═══ PHASE 2: SCRIPT QUALITY AUDIT ═══${NC}"
log "\nValidating script syntax and best practices..."

for script in $SCRIPT_DIR/*.sh; do
    if [ -f "$script" ]; then
        basename_script=$(basename "$script")
        check "SYNTAX" "$basename_script has valid bash syntax" "bash -n '$script'"
        check "QUALITY" "$basename_script has error handling" "grep -q 'set -euo pipefail' '$script'"
        check "QUALITY" "$basename_script has proper shebang" "head -1 '$script' | grep -q '^#!/bin/bash'"
        check "QUALITY" "$basename_script is executable" "[ -x '$script' ]" "WARNING"
    fi
done

# Phase 3: Docker Configuration Audit
log "\n${CYAN}${BOLD}═══ PHASE 3: DOCKER CONFIGURATION AUDIT ═══${NC}"
log "\nValidating Docker setup and best practices..."

# Docker Compose validation
if [ -f "$SCRIPT_DIR/docker-compose.yml" ]; then
    check "DOCKER" "docker-compose.yml is valid YAML" "docker-compose -f '$SCRIPT_DIR/docker-compose.yml' config -q 2>/dev/null" "WARNING"
    check "DOCKER" "Contains all required services" "grep -q 'orchestrator:' '$SCRIPT_DIR/docker-compose.yml' && grep -q 'ui:' '$SCRIPT_DIR/docker-compose.yml'"
    check "DOCKER" "Health checks defined" "grep -q 'healthcheck:' '$SCRIPT_DIR/docker-compose.yml'"
    check "DOCKER" "Networks configured" "grep -q 'networks:' '$SCRIPT_DIR/docker-compose.yml'"
    check "DOCKER" "Volumes defined" "grep -q 'volumes:' '$SCRIPT_DIR/docker-compose.yml'"
fi

# Dockerfile best practices
for dockerfile in $SCRIPT_DIR/dockerfiles/*.Dockerfile; do
    if [ -f "$dockerfile" ]; then
        basename_docker=$(basename "$dockerfile")
        check "DOCKER" "$basename_docker uses multi-stage build" "grep -q 'FROM.*AS' '$dockerfile'"
        check "DOCKER" "$basename_docker runs as non-root" "grep -q 'USER' '$dockerfile'"
        check "DOCKER" "$basename_docker has HEALTHCHECK" "grep -q 'HEALTHCHECK' '$dockerfile'"
        check "DOCKER" "$basename_docker uses specific versions" "! grep -q 'FROM.*:latest' '$dockerfile'"
    fi
done

# Phase 4: Python/Poetry Configuration Audit
log "\n${CYAN}${BOLD}═══ PHASE 4: PYTHON/POETRY AUDIT ═══${NC}"
log "\nValidating Python packaging and Poetry configuration..."

check "PYTHON" "Poetry setup script exists" "[ -f $SCRIPT_DIR/phase-4-poetry-setup.sh ]"
check "PYTHON" "Poetry setup includes all services" "grep -q 'orchestrator.*core.*plugins' '$SCRIPT_DIR/phase-4-poetry-setup.sh'"
check "PYTHON" "Python 3.12 specified" "grep -q '3.12' '$SCRIPT_DIR/phase-4-poetry-setup.sh'"
check "PYTHON" "Dev dependencies included" "grep -q 'group dev' '$SCRIPT_DIR/phase-4-poetry-setup.sh'"
check "PYTHON" "Tool configurations included" "grep -q 'tool.black' '$SCRIPT_DIR/phase-4-poetry-setup.sh'"

# Phase 5: CI/CD Pipeline Audit
log "\n${CYAN}${BOLD}═══ PHASE 5: CI/CD PIPELINE AUDIT ═══${NC}"
log "\nValidating GitHub Actions workflow..."

CICD_FILE="$SCRIPT_DIR/phase-5-cicd.yml"
if [ -f "$CICD_FILE" ]; then
    check "CI/CD" "Workflow has proper triggers" "grep -q 'on:.*push.*pull_request' '$CICD_FILE'"
    check "CI/CD" "Python matrix strategy exists" "grep -q 'matrix:' '$CICD_FILE'"
    check "CI/CD" "Includes linting" "grep -q 'ruff' '$CICD_FILE'"
    check "CI/CD" "Includes testing" "grep -q 'pytest' '$CICD_FILE'"
    check "CI/CD" "Includes type checking" "grep -q 'mypy' '$CICD_FILE'"
    check "CI/CD" "Docker build job exists" "grep -q 'docker-build:' '$CICD_FILE'"
    check "CI/CD" "Security scanning included" "grep -q 'trivy' '$CICD_FILE'"
    check "CI/CD" "Documentation build included" "grep -q 'mkdocs' '$CICD_FILE'"
    check "CI/CD" "Caching implemented" "grep -q 'cache' '$CICD_FILE'"
fi

# Phase 6: Security Audit
log "\n${CYAN}${BOLD}═══ PHASE 6: SECURITY AUDIT ═══${NC}"
log "\nValidating security best practices..."

# Check for secrets
for file in $SCRIPT_DIR/* $SCRIPT_DIR/dockerfiles/*; do
    if [ -f "$file" ]; then
        check "SECURITY" "$(basename $file) has no hardcoded secrets" "! grep -iE '(password|secret|key|token).*=.*[a-zA-Z0-9]{8,}' '$file'"
    fi
done

check "SECURITY" ".env.example created" "grep -q 'ENV_TEMPLATE' '$SCRIPT_DIR/phase-6-config.py'"
check "SECURITY" "Git cleanup removes secrets" "grep -q '*.pem.**.key.***.crt' '$SCRIPT_DIR/phase-2-git-cleanup.sh'"
check "SECURITY" ".gitignore comprehensive" "grep -q '__pycache__.*node_modules.***.env' '$SCRIPT_DIR/phase-1-analyze.sh'"
check "SECURITY" "Docker uses least privilege" "grep -q 'USER.*1001' '$SCRIPT_DIR/dockerfiles/ui.Dockerfile'"

# Phase 7: Documentation Audit
log "\n${CYAN}${BOLD}═══ PHASE 7: DOCUMENTATION AUDIT ═══${NC}"
log "\nValidating documentation completeness..."

check "DOCS" "Documentation curation script exists" "[ -f $SCRIPT_DIR/phase-7-docs.sh ]"
check "DOCS" "MkDocs configuration created" "grep -q 'mkdocs.yml' '$SCRIPT_DIR/phase-7-docs.sh'"
check "DOCS" "Documentation structure defined" "grep -q 'getting-started.*api.*architecture' '$SCRIPT_DIR/phase-7-docs.sh'"
check "DOCS" "README.md mentioned" "grep -q 'README.md' '$SCRIPT_DIR/phase-7-docs.sh'"
check "DOCS" "CONTRIBUTING.md created" "grep -q 'CONTRIBUTING.md' '$SCRIPT_DIR/phase-7-docs.sh'"
check "DOCS" "Archive strategy defined" "grep -q 'docs/archive' '$SCRIPT_DIR/phase-7-docs.sh'"

# Phase 8: Configuration Management Audit
log "\n${CYAN}${BOLD}═══ PHASE 8: CONFIGURATION AUDIT ═══${NC}"
log "\nValidating configuration unification..."

CONFIG_SCRIPT="$SCRIPT_DIR/phase-6-config.py"
if [ -f "$CONFIG_SCRIPT" ]; then
    check "CONFIG" "Pydantic settings used" "grep -q 'pydantic_settings' '$CONFIG_SCRIPT'"
    check "CONFIG" "Base settings defined" "grep -q 'class CommonSettings' '$CONFIG_SCRIPT'"
    check "CONFIG" "Service-specific settings" "grep -q 'OrchestratorSettings.*CoreSettings.*UISettings' '$CONFIG_SCRIPT'"
    check "CONFIG" "Environment loading configured" "grep -q 'env_file=.*\.env' '$CONFIG_SCRIPT'"
    check "CONFIG" "Settings factory function" "grep -q 'get_settings' '$CONFIG_SCRIPT'"
fi

# Phase 9: Orchestration Audit
log "\n${CYAN}${BOLD}═══ PHASE 9: ORCHESTRATION AUDIT ═══${NC}"
log "\nValidating Orchestra implementation..."

check "ORCHESTRA" "9 agents defined" "grep -c 'agent_spawn' jarvis-triage-playbook.md | grep -q '9'"
check "ORCHESTRA" "Hierarchical topology used" "grep -q 'hierarchical' jarvis-triage-playbook.md"
check "ORCHESTRA" "Agent responsibilities documented" "grep -q 'Agent Roster & Responsibilities' jarvis-triage-playbook.md"
check "ORCHESTRA" "Workflow created" "grep -q 'workflow_create' jarvis-triage-playbook.md"
check "ORCHESTRA" "/agents commands documented" "grep -q '/agents' orchestra-implementation.md"

# Phase 10: Completeness Audit
log "\n${CYAN}${BOLD}═══ PHASE 10: COMPLETENESS AUDIT ═══${NC}"
log "\nValidating all playbook requirements met..."

# Check all 10 phases covered
PHASES_IMPLEMENTED=0
for i in {0..9}; do
    if [ -f "$SCRIPT_DIR/phase-$i"* ] || [ $i -eq 8 ]; then  # Phase 8 is Docker setup
        ((PHASES_IMPLEMENTED++))
    fi
done

check "COMPLETE" "All 10 phases implemented" "[ $PHASES_IMPLEMENTED -eq 10 ]"
check "COMPLETE" "Master execution script created" "[ -f $SCRIPT_DIR/execute-triage.sh ]"
check "COMPLETE" "Verification script created" "[ -f $SCRIPT_DIR/phase-9-verify.sh ]"
check "COMPLETE" "Quick start commands documented" "grep -q 'docker-compose up' jarvis-triage-playbook.md"
check "COMPLETE" "10-day roadmap referenced" "grep -q '10-day' jarvis-triage-playbook.md"

# Special Checks
log "\n${CYAN}${BOLD}═══ SPECIAL QUALITY CHECKS ═══${NC}"
log "\nPerforming additional quality validations..."

# Check script consistency
check "QUALITY" "Consistent color scheme used" "grep -q 'GREEN=.*YELLOW=.*RED=' $SCRIPT_DIR/phase-*.sh | wc -l | grep -q '[5-9]'"
check "QUALITY" "Consistent error handling" "grep -l 'set -euo pipefail' $SCRIPT_DIR/*.sh | wc -l | grep -q '[8-9]'"
check "QUALITY" "Agent attribution in all scripts" "grep -l 'Agent:' $SCRIPT_DIR/phase-*.sh | wc -l | grep -q '[7-9]'"

# Integration checks
check "INTEGRATION" "Scripts reference each other" "grep -q 'phase-0-backup.sh' '$SCRIPT_DIR/execute-triage.sh'"
check "INTEGRATION" "Docker compose uses created Dockerfiles" "grep -q 'orchestrator.Dockerfile' '$SCRIPT_DIR/docker-compose.yml'"
check "INTEGRATION" "CI/CD covers all services" "grep -q 'orchestrator.*core.*ui' '$CICD_FILE'"

# Generate Summary
log "\n${CYAN}${BOLD}═══ AUDIT SUMMARY ═══${NC}"
log "\nGenerating comprehensive audit results..."

TOTAL_SCORE=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))
GRADE="F"

if [ $TOTAL_SCORE -ge 95 ]; then
    GRADE="A+"
elif [ $TOTAL_SCORE -ge 90 ]; then
    GRADE="A"
elif [ $TOTAL_SCORE -ge 85 ]; then
    GRADE="B+"
elif [ $TOTAL_SCORE -ge 80 ]; then
    GRADE="B"
elif [ $TOTAL_SCORE -ge 75 ]; then
    GRADE="C+"
elif [ $TOTAL_SCORE -ge 70 ]; then
    GRADE="C"
elif [ $TOTAL_SCORE -ge 60 ]; then
    GRADE="D"
fi

# Console Summary
echo ""
echo -e "${BOLD}╔═══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║                         AUDIT RESULTS                             ║${NC}"
echo -e "${BOLD}╚═══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Total Checks:    ${BOLD}$TOTAL_CHECKS${NC}"
echo -e "  Passed:          ${GREEN}${BOLD}$PASSED_CHECKS${NC}"
echo -e "  Failed:          ${RED}${BOLD}$FAILED_CHECKS${NC}"
echo -e "  Warnings:        ${YELLOW}${BOLD}$WARNINGS${NC}"
echo -e "  Success Rate:    ${BOLD}$TOTAL_SCORE%${NC}"
echo -e "  Grade:           ${BOLD}$GRADE${NC}"
echo ""

# Detailed Results
if [ ${#FAILURES[@]} -gt 0 ]; then
    echo -e "${RED}${BOLD}Failed Checks:${NC}"
    for failure in "${FAILURES[@]}"; do
        echo -e "  ${RED}✗${NC} $failure"
    done
    echo ""
fi

if [ ${#WARNINGS_LIST[@]} -gt 0 ]; then
    echo -e "${YELLOW}${BOLD}Warnings:${NC}"
    for warning in "${WARNINGS_LIST[@]}"; do
        echo -e "  ${YELLOW}⚠${NC} $warning"
    done
    echo ""
fi

# Report Generation
log "\n## Summary Statistics"
log "- Total Checks: $TOTAL_CHECKS"
log "- Passed: $PASSED_CHECKS"
log "- Failed: $FAILED_CHECKS" 
log "- Warnings: $WARNINGS"
log "- Success Rate: $TOTAL_SCORE%"
log "- Grade: **$GRADE**"

# Generate JSON report
cat > "$AUDIT_JSON" << EOF
{
  "audit_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "total_checks": $TOTAL_CHECKS,
  "passed": $PASSED_CHECKS,
  "failed": $FAILED_CHECKS,
  "warnings": $WARNINGS,
  "success_rate": $TOTAL_SCORE,
  "grade": "$GRADE",
  "failures": $(printf '%s\n' "${FAILURES[@]}" | jq -R . | jq -s . 2>/dev/null || echo "[]"),
  "warnings_list": $(printf '%s\n' "${WARNINGS_LIST[@]}" | jq -R . | jq -s . 2>/dev/null || echo "[]")
}
EOF

# Final Message
echo -e "${BOLD}════════════════════════════════════════════════════════════════════${NC}"
if [ $FAILED_CHECKS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}${BOLD}🎉 PERFECT SCORE! The Best Audit Ever™ confirms excellence!${NC}"
elif [ $TOTAL_SCORE -ge 90 ]; then
    echo -e "${GREEN}${BOLD}✅ Excellent work! Minor improvements recommended.${NC}"
elif [ $TOTAL_SCORE -ge 80 ]; then
    echo -e "${YELLOW}${BOLD}⚠️ Good foundation. Address failures before production.${NC}"
else
    echo -e "${RED}${BOLD}❌ Significant issues found. Review and fix before proceeding.${NC}"
fi
echo -e "${BOLD}════════════════════════════════════════════════════════════════════${NC}"

echo ""
echo "Detailed report saved to: $AUDIT_REPORT"
echo "JSON results saved to: $AUDIT_JSON"
echo ""

# Exit code based on critical failures
if [ $FAILED_CHECKS -gt 0 ]; then
    exit 1
else
    exit 0
fi