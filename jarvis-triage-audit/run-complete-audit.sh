#!/bin/bash
# Master Audit Runner
# Executes the complete audit suite

set -euo pipefail

# Colors
BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BOLD}${CYAN}🏆 Running The Best Audit Ever™${NC}"
echo "===================================="

# Step 1: Fix any issues first
echo -e "\n${YELLOW}Step 1: Applying fixes...${NC}"
bash fix-issues.sh

# Step 2: Run comprehensive audit
echo -e "\n${YELLOW}Step 2: Running comprehensive audit...${NC}"
bash audit-script.sh

# Step 3: Generate visualizations
echo -e "\n${YELLOW}Step 3: Generating visualizations...${NC}"
if command -v python3 &> /dev/null; then
    python3 audit-visualization.py || echo "Visualization requires matplotlib"
else
    echo "Python not available for visualization"
fi

# Step 4: Display summary
echo -e "\n${GREEN}${BOLD}═══ AUDIT COMPLETE ═══${NC}"
echo -e "\nAudit artifacts created:"
echo "  ✓ audit-report-*.md - Detailed audit report"
echo "  ✓ audit-results.json - Machine-readable results"
echo "  ✓ executive-summary.md - Executive overview"
echo "  ✓ audit-dashboard.png - Visual dashboard (if matplotlib available)"
echo "  ✓ audit-report.html - Interactive HTML report"

# Display executive summary
echo -e "\n${CYAN}${BOLD}Executive Summary:${NC}"
head -n 30 executive-summary.md | tail -n 20

echo -e "\n${GREEN}✨ The Best Audit Ever™ is complete!${NC}"