#!/bin/bash
# JARVIS Test Runner by Category
# ==============================
# Run specific categories of tests with appropriate settings

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧪 JARVIS Test Runner by Category${NC}"
echo "=================================="
echo ""

# Function to run tests with timing
run_tests() {
    local category=$1
    local pytest_args=$2
    local description=$3
    
    echo -e "${YELLOW}Running ${description}...${NC}"
    echo ""
    
    # Start timer
    start_time=$(date +%s)
    
    # Run tests
    pytest ${pytest_args} --tb=short
    test_exit_code=$?
    
    # End timer
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo ""
    if [ $test_exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ ${description} completed successfully in ${duration}s${NC}"
    else
        echo -e "${RED}❌ ${description} failed after ${duration}s${NC}"
    fi
    
    return $test_exit_code
}

# Function to run with coverage
run_with_coverage() {
    local category=$1
    local pytest_args=$2
    local description=$3
    
    echo -e "${YELLOW}Running ${description} with coverage...${NC}"
    echo ""
    
    pytest ${pytest_args} \
        --cov=core --cov=plugins --cov=tools \
        --cov-report=term-missing \
        --cov-report=html:htmlcov_${category} \
        --tb=short
}

# Parse command line arguments
CATEGORY=$1
WITH_COVERAGE=${2:-"no"}

case $CATEGORY in
    "unit")
        if [ "$WITH_COVERAGE" == "coverage" ]; then
            run_with_coverage "unit" "-m unit -v" "unit tests"
        else
            run_tests "unit" "-m unit -v" "unit tests"
        fi
        ;;
        
    "integration")
        echo -e "${BLUE}Note: Integration tests may take longer to run${NC}"
        echo ""
        if [ "$WITH_COVERAGE" == "coverage" ]; then
            run_with_coverage "integration" "-m integration -v" "integration tests"
        else
            run_tests "integration" "-m integration -v" "integration tests"
        fi
        ;;
        
    "fast")
        if [ "$WITH_COVERAGE" == "coverage" ]; then
            run_with_coverage "fast" "-m 'not slow' -v" "fast tests only"
        else
            run_tests "fast" "-m 'not slow' -v" "fast tests only"
        fi
        ;;
        
    "security")
        echo -e "${BLUE}🔒 Security tests check authentication, encryption, and input validation${NC}"
        echo ""
        if [ "$WITH_COVERAGE" == "coverage" ]; then
            run_with_coverage "security" "-m security -v" "security tests"
        else
            run_tests "security" "-m security -v" "security tests"
        fi
        ;;
        
    "performance")
        echo -e "${BLUE}⚡ Performance tests may take several minutes${NC}"
        echo ""
        run_tests "performance" "-m performance -v --benchmark-only" "performance tests"
        ;;
        
    "smoke")
        echo -e "${BLUE}💨 Running smoke tests (critical paths only)${NC}"
        echo ""
        run_tests "smoke" "-m 'not slow and not integration' -x --ff" "smoke tests"
        ;;
        
    "all")
        echo -e "${BLUE}Running all tests...${NC}"
        echo ""
        if [ "$WITH_COVERAGE" == "coverage" ]; then
            run_with_coverage "all" "-v" "all tests"
        else
            run_tests "all" "-v" "all tests"
        fi
        ;;
        
    "failed")
        echo -e "${BLUE}Re-running previously failed tests...${NC}"
        echo ""
        run_tests "failed" "--lf -v" "failed tests"
        ;;
        
    "modified")
        echo -e "${BLUE}Running tests for modified files...${NC}"
        echo ""
        # Use pytest-testmon if available
        if pip show pytest-testmon > /dev/null 2>&1; then
            run_tests "modified" "--testmon -v" "tests for modified code"
        else
            echo -e "${YELLOW}Install pytest-testmon for smarter test selection:${NC}"
            echo "  pip install pytest-testmon"
            echo ""
            echo "Falling back to running all tests..."
            run_tests "all" "-v" "all tests"
        fi
        ;;
        
    "parallel")
        echo -e "${BLUE}Running tests in parallel...${NC}"
        echo ""
        if pip show pytest-xdist > /dev/null 2>&1; then
            run_tests "parallel" "-n auto -v" "parallel tests"
        else
            echo -e "${RED}pytest-xdist not installed!${NC}"
            echo "Install with: pip install pytest-xdist"
            exit 1
        fi
        ;;
        
    "debug")
        echo -e "${BLUE}Running tests with debugging enabled...${NC}"
        echo ""
        run_tests "debug" "-vvs --pdb --pdbcls=IPython.terminal.debugger:Pdb" "debug mode"
        ;;
        
    "watch")
        echo -e "${BLUE}Starting test watcher...${NC}"
        echo "Tests will re-run automatically when files change"
        echo "Press Ctrl+C to stop"
        echo ""
        
        if command -v pytest-watch > /dev/null 2>&1; then
            pytest-watch -- -v
        else
            echo -e "${RED}pytest-watch not installed!${NC}"
            echo "Install with: pip install pytest-watch"
            exit 1
        fi
        ;;
        
    *)
        echo "Usage: $0 <category> [coverage]"
        echo ""
        echo "Categories:"
        echo "  unit        - Run unit tests only"
        echo "  integration - Run integration tests"
        echo "  fast        - Run fast tests (exclude slow)"
        echo "  security    - Run security tests"
        echo "  performance - Run performance benchmarks"
        echo "  smoke       - Run smoke tests (critical paths)"
        echo "  all         - Run all tests"
        echo "  failed      - Re-run failed tests"
        echo "  modified    - Run tests for modified code"
        echo "  parallel    - Run tests in parallel"
        echo "  debug       - Run with debugger enabled"
        echo "  watch       - Watch files and auto-run tests"
        echo ""
        echo "Options:"
        echo "  coverage    - Generate coverage report"
        echo ""
        echo "Examples:"
        echo "  $0 unit              # Run unit tests"
        echo "  $0 unit coverage     # Run unit tests with coverage"
        echo "  $0 fast              # Run only fast tests"
        echo "  $0 parallel coverage # Run all tests in parallel with coverage"
        exit 1
        ;;
esac

# Show summary
echo ""
echo "=================================="
if [ -f ".coverage" ]; then
    echo -e "${GREEN}Coverage data available in .coverage${NC}"
fi

if [ -d "htmlcov_${CATEGORY}" ] && [ "$WITH_COVERAGE" == "coverage" ]; then
    echo -e "${GREEN}HTML coverage report: htmlcov_${CATEGORY}/index.html${NC}"
    echo ""
    echo "To view: open htmlcov_${CATEGORY}/index.html"
fi

# Exit with test status
exit $test_exit_code