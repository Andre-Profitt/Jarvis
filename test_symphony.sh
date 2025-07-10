#!/bin/bash

# Test JARVIS Symphony Setup
echo "🎼 Testing JARVIS Symphony Setup"
echo "================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test 1: Check Python
echo -e "\n${YELLOW}Test 1: Python Installation${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✅ Python installed: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}❌ Python not found${NC}"
fi

# Test 2: Check npm/npx
echo -e "\n${YELLOW}Test 2: NPM/NPX Installation${NC}"
if command -v npx &> /dev/null; then
    NPX_VERSION=$(npx --version)
    echo -e "${GREEN}✅ NPX installed: v$NPX_VERSION${NC}"
else
    echo -e "${RED}❌ NPX not found${NC}"
fi

# Test 3: Check ruv-swarm
echo -e "\n${YELLOW}Test 3: ruv-swarm Availability${NC}"
if npx ruv-swarm --help &> /dev/null; then
    echo -e "${GREEN}✅ ruv-swarm is available${NC}"
    
    # Get version info
    echo -e "\n${YELLOW}ruv-swarm commands:${NC}"
    npx ruv-swarm --help | grep -E "init|spawn|orchestrate|status|monitor" | head -5
else
    echo -e "${RED}❌ ruv-swarm not found${NC}"
    echo "Install with: npm install -g ruv-swarm"
fi

# Test 4: Check symphony files
echo -e "\n${YELLOW}Test 4: Symphony Files${NC}"
FILES=(
    "jarvis_symphony_orchestrator.py"
    "JARVIS_SYMPHONY_PLAN.md"
    "start_symphony.sh"
    "symphony_monitor.py"
    "mcp_symphony_bridge.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $file${NC}"
    else
        echo -e "${RED}❌ $file missing${NC}"
    fi
done

# Test 5: Quick swarm test
echo -e "\n${YELLOW}Test 5: Quick Swarm Test${NC}"
echo "Attempting to initialize a test swarm..."

# Try to init and immediately cleanup
npx ruv-swarm init mesh 3 --no-interactive 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Swarm initialization successful${NC}"
    
    # Get status
    echo -e "\n${YELLOW}Swarm Status:${NC}"
    npx ruv-swarm status
    
    # Cleanup
    echo -e "\n${YELLOW}Cleaning up test swarm...${NC}"
    npx ruv-swarm cleanup 2>/dev/null || true
else
    echo -e "${YELLOW}⚠️  Could not initialize test swarm (may already exist)${NC}"
fi

echo -e "\n${GREEN}🎭 Symphony setup test complete!${NC}"
echo ""
echo "To start the full symphony, run:"
echo "  ./start_symphony.sh"
echo ""
echo "To monitor the symphony, run:"
echo "  python3 symphony_monitor.py"