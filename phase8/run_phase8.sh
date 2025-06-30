#!/bin/bash
# JARVIS Phase 8 Quick Launch Script

echo "🚀 Launching JARVIS Phase 8: UX Enhancements"
echo "==========================================="

# Navigate to JARVIS directory
cd "$(dirname "$0")/.."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✓ Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  No virtual environment found"
fi

# Install any missing dependencies
echo "📦 Checking dependencies..."
pip install -q websockets 2>/dev/null

# Launch Phase 8
echo "🎨 Starting JARVIS with Phase 8 enhancements..."
python phase8/launch_phase8.py &
JARVIS_PID=$!

# Wait a moment for the server to start
sleep 3

# Open the dashboard
echo "📊 Opening UX Dashboard..."
if command -v open &> /dev/null; then
    # macOS
    open phase8/jarvis-ux-dashboard.html
elif command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open phase8/jarvis-ux-dashboard.html
else
    echo "⚠️  Please open phase8/jarvis-ux-dashboard.html in your browser"
fi

echo ""
echo "✅ Phase 8 is running!"
echo "   - JARVIS PID: $JARVIS_PID"
echo "   - WebSocket: ws://localhost:8890"
echo "   - Dashboard: file://$(pwd)/phase8/jarvis-ux-dashboard.html"
echo ""
echo "Press Ctrl+C to stop JARVIS"

# Wait for JARVIS process
wait $JARVIS_PID
