#!/bin/bash
# JARVIS Test Coverage Baseline Script
# ====================================
# This script establishes a baseline for test coverage and tracks progress

echo "🚀 JARVIS Test Coverage Baseline Tool"
echo "===================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create coverage directory
mkdir -p coverage_reports

# Get current date for report naming
DATE=$(date +%Y%m%d_%H%M%S)

# Function to run coverage and save report
run_coverage() {
    local report_name=$1
    local extra_args=$2
    
    echo -e "${YELLOW}Running coverage analysis: ${report_name}${NC}"
    
    # Run pytest with coverage
    pytest --cov=core --cov=plugins --cov=tools \
           --cov-report=term-missing \
           --cov-report=html:htmlcov_${report_name} \
           --cov-report=json:coverage_reports/${report_name}_${DATE}.json \
           ${extra_args} > coverage_reports/${report_name}_${DATE}.txt 2>&1
    
    # Extract coverage percentage
    COVERAGE=$(grep "TOTAL" coverage_reports/${report_name}_${DATE}.txt | awk '{print $4}' | sed 's/%//')
    
    if [ -z "$COVERAGE" ]; then
        echo -e "${RED}❌ Failed to run coverage analysis${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Coverage: ${COVERAGE}%${NC}"
    return 0
}

# Function to compare coverage
compare_coverage() {
    local baseline=$1
    local current=$2
    
    if [ -f "coverage_reports/${baseline}.json" ] && [ -f "coverage_reports/${current}.json" ]; then
        python3 - <<EOF
import json
import sys

with open('coverage_reports/${baseline}.json', 'r') as f:
    baseline_data = json.load(f)
    
with open('coverage_reports/${current}.json', 'r') as f:
    current_data = json.load(f)

baseline_pct = baseline_data['totals']['percent_covered']
current_pct = current_data['totals']['percent_covered']
improvement = current_pct - baseline_pct

print(f"\n📊 Coverage Comparison:")
print(f"   Baseline: {baseline_pct:.2f}%")
print(f"   Current:  {current_pct:.2f}%")
print(f"   Change:   {improvement:+.2f}%")

if improvement > 0:
    print(f"\n✅ Coverage improved by {improvement:.2f}%!")
elif improvement < 0:
    print(f"\n⚠️  Coverage decreased by {abs(improvement):.2f}%")
else:
    print(f"\n➡️  Coverage unchanged")
EOF
    fi
}

# Main execution
echo "1️⃣  Establishing baseline coverage..."
echo ""

# Run baseline coverage
run_coverage "baseline" "-x"

# Save baseline coverage percentage
if [ $? -eq 0 ]; then
    echo $COVERAGE > coverage_reports/baseline_coverage.txt
    echo ""
    echo -e "${GREEN}✅ Baseline established: ${COVERAGE}%${NC}"
    echo ""
    
    # Generate summary report
    echo "📋 Coverage Summary Report" > coverage_reports/baseline_summary_${DATE}.md
    echo "=========================" >> coverage_reports/baseline_summary_${DATE}.md
    echo "" >> coverage_reports/baseline_summary_${DATE}.md
    echo "Date: $(date)" >> coverage_reports/baseline_summary_${DATE}.md
    echo "Overall Coverage: ${COVERAGE}%" >> coverage_reports/baseline_summary_${DATE}.md
    echo "" >> coverage_reports/baseline_summary_${DATE}.md
    echo "## Module Coverage" >> coverage_reports/baseline_summary_${DATE}.md
    echo "" >> coverage_reports/baseline_summary_${DATE}.md
    
    # Extract per-module coverage
    grep -E "^(core|plugins|tools)" coverage_reports/baseline_${DATE}.txt | \
        awk '{printf "- %-40s %s\n", $1, $4}' >> coverage_reports/baseline_summary_${DATE}.md
    
    echo ""
    echo "📊 Detailed reports available:"
    echo "   - HTML Report: htmlcov_baseline/index.html"
    echo "   - JSON Report: coverage_reports/baseline_${DATE}.json"
    echo "   - Text Report: coverage_reports/baseline_${DATE}.txt"
    echo "   - Summary:     coverage_reports/baseline_summary_${DATE}.md"
    
    # Check if we meet the 80% threshold
    if (( $(echo "$COVERAGE < 80" | bc -l) )); then
        echo ""
        echo -e "${YELLOW}⚠️  Warning: Coverage is below 80% threshold${NC}"
        echo "   Target: 80%"
        echo "   Needed: $(echo "80 - $COVERAGE" | bc)% more coverage"
    fi
else
    echo -e "${RED}❌ Failed to establish baseline${NC}"
    exit 1
fi

# Create quick run scripts
echo ""
echo "Creating convenience scripts..."

# Create daily progress tracker
cat > run_daily_coverage.sh << 'EOF'
#!/bin/bash
# Run daily coverage check
DATE=$(date +%Y%m%d)
pytest --cov=core --cov=plugins --cov=tools \
       --cov-report=term-missing \
       --cov-report=html \
       --cov-report=json:coverage_reports/daily_${DATE}.json \
       > coverage_reports/daily_${DATE}.txt 2>&1

COVERAGE=$(grep "TOTAL" coverage_reports/daily_${DATE}.txt | awk '{print $4}')
echo "Today's coverage: $COVERAGE"

# Compare with baseline
if [ -f coverage_reports/baseline_coverage.txt ]; then
    BASELINE=$(cat coverage_reports/baseline_coverage.txt)
    echo "Baseline coverage: ${BASELINE}%"
fi
EOF

chmod +x run_daily_coverage.sh

# Create component-specific test runner
cat > test_component.sh << 'EOF'
#!/bin/bash
# Test specific component
COMPONENT=$1
if [ -z "$COMPONENT" ]; then
    echo "Usage: ./test_component.sh <component_name>"
    echo "Example: ./test_component.sh neural_resource_manager"
    exit 1
fi

pytest tests/test_${COMPONENT}.py -v \
       --cov=core.${COMPONENT} \
       --cov-report=term-missing
EOF

chmod +x test_component.sh

echo -e "${GREEN}✅ Created helper scripts:${NC}"
echo "   - ./run_daily_coverage.sh  - Track daily progress"
echo "   - ./test_component.sh      - Test specific component"

echo ""
echo "🎯 Next Steps:"
echo "   1. Review the baseline coverage report"
echo "   2. Focus on modules with lowest coverage"
echo "   3. Run tests for specific components"
echo "   4. Track daily progress"
echo ""
echo "Happy testing! 🚀"