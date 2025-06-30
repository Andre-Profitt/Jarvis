#!/bin/bash
# JARVIS Test Suite Runner
# Run all tests and generate coverage reports

echo "🧪 JARVIS Test Suite Runner"
echo "=========================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -d "core" ] || [ ! -d "tests" ]; then
    echo -e "${RED}Error: Must run from JARVIS-ECOSYSTEM root directory${NC}"
    exit 1
fi

# Install test dependencies if needed
echo -e "${YELLOW}📦 Checking test dependencies...${NC}"
pip install -q pytest pytest-cov pytest-asyncio pytest-mock 2>/dev/null

# Run tests with coverage
echo -e "\n${YELLOW}🏃 Running full test suite...${NC}"
pytest tests/ \
    --cov=core \
    --cov-report=html \
    --cov-report=term \
    --cov-report=term-missing \
    -v \
    --tb=short \
    --maxfail=5

# Check exit code
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ All tests passed!${NC}"
    
    # Open coverage report
    if command -v open &> /dev/null; then
        echo -e "${YELLOW}📊 Opening coverage report...${NC}"
        open htmlcov/index.html
    else
        echo -e "${YELLOW}📊 Coverage report generated at: htmlcov/index.html${NC}"
    fi
else
    echo -e "\n${RED}❌ Some tests failed. Please check the output above.${NC}"
    
    # Show summary of failures
    echo -e "\n${YELLOW}📝 Quick failure summary:${NC}"
    pytest tests/ --tb=no -q | grep FAILED
fi

# Show coverage summary
echo -e "\n${YELLOW}📈 Coverage Summary:${NC}"
coverage report --skip-covered --skip-empty | grep -E "(TOTAL|core/)"

echo -e "\n${GREEN}Done!${NC}"