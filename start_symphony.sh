#!/bin/bash

# JARVIS Symphony Launcher
# 🎼 Let the symphony begin!

echo "🎼 JARVIS Ultimate System Symphony"
echo "=================================="
echo ""

# Colors for beautiful output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "jarvis_symphony_orchestrator.py" ]; then
    echo -e "${RED}❌ Error: Not in JARVIS directory${NC}"
    echo "Please run from the Jarvis directory"
    exit 1
fi

# Check dependencies
echo -e "${CYAN}🔍 Checking dependencies...${NC}"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is required${NC}"
    exit 1
fi

# Check for npm/npx
if ! command -v npx &> /dev/null; then
    echo -e "${RED}❌ npx is required for ruv-swarm${NC}"
    exit 1
fi

# Check for ruv-swarm
echo -e "${CYAN}🐝 Checking ruv-swarm...${NC}"
if ! npx ruv-swarm --help &> /dev/null; then
    echo -e "${YELLOW}⚠️  ruv-swarm not found, attempting to install...${NC}"
    npm install -g ruv-swarm
fi

echo -e "${GREEN}✅ All dependencies satisfied!${NC}"
echo ""

# Create performance directory
mkdir -p artifacts/symphony_performance

# Set environment
export JARVIS_SYMPHONY_MODE=true
export JARVIS_HOME=$(pwd)

# Display the plan
echo -e "${PURPLE}📜 Symphony Plan:${NC}"
echo "  Movement I:    Foundation (Andante)"
echo "  Movement II:   Features (Allegro)"
echo "  Movement III:  Intelligence (Moderato)"
echo "  Movement IV:   Finale (Presto)"
echo ""

# Countdown
echo -e "${YELLOW}🎭 Raising the baton in...${NC}"
for i in 3 2 1; do
    echo "  $i..."
    sleep 1
done

echo ""
echo -e "${GREEN}🎼 Let the symphony begin!${NC}"
echo ""

# Start the orchestrator
python3 jarvis_symphony_orchestrator.py

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎭 Bravo! The symphony is complete!${NC}"
    echo -e "${CYAN}📊 Performance log saved to: jarvis_symphony_performance.json${NC}"
    
    # Show summary
    if [ -f "jarvis_symphony_performance.json" ]; then
        echo ""
        echo -e "${PURPLE}📈 Performance Summary:${NC}"
        python3 -c "
import json
with open('jarvis_symphony_performance.json', 'r') as f:
    data = json.load(f)
    print(f'  Movements completed: {len(data[\"score\"][\"movements\"])}')
    print(f'  Total tasks executed: {len(data[\"performance_log\"])}')
    print(f'  Completion time: {data[\"completed_at\"]}')
"
    fi
    
    echo ""
    echo -e "${YELLOW}🚀 JARVIS is ready to launch!${NC}"
    echo ""
    echo "To start JARVIS:"
    echo "  ./launch_jarvis_ultimate.sh"
    echo ""
    echo "To view the performance:"
    echo "  cat jarvis_symphony_performance.json | jq"
else
    echo ""
    echo -e "${RED}❌ Symphony encountered errors${NC}"
    echo "Check the logs for details"
    exit 1
fi