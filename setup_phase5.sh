#!/bin/bash

# JARVIS Phase 5 Quick Setup Script
# Sets up and launches the Natural Interaction system

echo "🚀 JARVIS Phase 5 - Natural Interaction Setup"
echo "============================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -d "core" ]; then
    echo -e "${YELLOW}⚠️  Please run this script from the JARVIS-ECOSYSTEM directory${NC}"
    exit 1
fi

echo -e "\n${BLUE}📋 Checking Phase 5 components...${NC}"

# Check required files
required_files=(
    "core/conversational_memory.py"
    "core/emotional_continuity.py"
    "core/natural_language_flow.py"
    "core/natural_interaction_core.py"
    "test_phase5.py"
    "integrate_phase5.py"
)

all_present=true
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "❌ $file missing"
        all_present=false
    fi
done

if [ "$all_present" = false ]; then
    echo -e "\n${YELLOW}Some files are missing. Please ensure all Phase 5 files are present.${NC}"
    exit 1
fi

echo -e "\n${GREEN}✅ All Phase 5 components found!${NC}"

# Check Python
echo -e "\n${BLUE}🐍 Checking Python environment...${NC}"
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version)
    echo -e "${GREEN}✓${NC} $python_version"
else
    echo -e "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check dependencies
echo -e "\n${BLUE}📦 Checking dependencies...${NC}"
python3 -c "import asyncio" 2>/dev/null && echo -e "${GREEN}✓${NC} asyncio" || echo -e "${YELLOW}⚠️${NC} asyncio"
python3 -c "import numpy" 2>/dev/null && echo -e "${GREEN}✓${NC} numpy" || echo -e "${YELLOW}⚠️${NC} numpy (optional, install for full features)"

# Create logs directory if needed
mkdir -p logs

echo -e "\n${BLUE}🎯 Phase 5 Setup Complete!${NC}"
echo -e "\n${GREEN}Available commands:${NC}"
echo "1. Run tests:        python3 test_phase5.py"
echo "2. Interactive demo: python3 test_phase5.py (choose option 2)"
echo "3. Integration demo: python3 integrate_phase5.py"
echo "4. Open dashboard:   open jarvis-phase5-monitor.html"

echo -e "\n${BLUE}Would you like to:${NC}"
echo "1) Run the test suite"
echo "2) Start interactive demo"
echo "3) Run integration demo"
echo "4) Exit"

read -p "Choose an option (1-4): " choice

case $choice in
    1)
        echo -e "\n${GREEN}🧪 Running Phase 5 tests...${NC}"
        python3 test_phase5.py
        ;;
    2)
        echo -e "\n${GREEN}💬 Starting interactive demo...${NC}"
        echo "2" | python3 test_phase5.py
        ;;
    3)
        echo -e "\n${GREEN}🔗 Running integration demo...${NC}"
        python3 integrate_phase5.py
        ;;
    4)
        echo -e "\n${GREEN}👋 Setup complete. Ready to use Phase 5!${NC}"
        ;;
    *)
        echo -e "\n${YELLOW}Invalid option. Exiting.${NC}"
        ;;
esac
