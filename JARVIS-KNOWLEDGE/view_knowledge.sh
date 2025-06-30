#!/bin/bash
# Quick viewer for JARVIS Knowledge Base

echo "🧠 JARVIS Knowledge Base Viewer"
echo "==============================="
echo ""

# Define colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display file with header
show_file() {
    local file=$1
    local title=$2
    
    echo -e "${BLUE}📄 $title${NC}"
    echo "----------------------------------------"
    head -20 "$file" | sed 's/^/  /'
    echo "  ..."
    echo ""
}

# Change to JARVIS directory
cd /Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM/JARVIS-KNOWLEDGE

# Show each knowledge file
show_file "JARVIS-MASTER-CONTEXT.md" "Master Context"
show_file "CURRENT-STATE.md" "Current State"
show_file "EVOLUTION-LOG.md" "Evolution Log"
show_file "COMPONENT-MAP.md" "Component Map"

echo -e "${GREEN}✅ Knowledge Base Status:${NC}"
echo "  - Total MD files: $(ls *.md 2>/dev/null | wc -l)"
echo "  - Last updated: $(stat -f "%Sm" CURRENT-STATE.md 2>/dev/null || date)"
echo ""

echo "📝 Quick Actions:"
echo "  - View full file: cat JARVIS-KNOWLEDGE/<filename>"
echo "  - Edit state: vim JARVIS-KNOWLEDGE/CURRENT-STATE.md"
echo "  - New session: cat JARVIS-KNOWLEDGE/SESSION-TEMPLATE.md"
echo ""

echo "🚀 Next conversation: Start with SESSION-TEMPLATE.md!"
