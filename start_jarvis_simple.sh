#!/bin/bash
# Simple JARVIS Startup Script

echo "🤖 Starting JARVIS..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed!"
    exit 1
fi

# Check if in correct directory
if [ ! -f "jarvis_minimal_working.py" ]; then
    echo "❌ Cannot find jarvis_minimal_working.py"
    echo "Make sure you're in the JARVIS directory"
    exit 1
fi

# Install minimal dependencies if needed
if ! python3 -c "import dotenv" 2>/dev/null; then
    echo "📦 Installing minimal dependencies..."
    pip3 install -r requirements-minimal.txt
fi

# Run JARVIS
echo "🚀 Launching JARVIS..."
echo ""
python3 launch_jarvis_minimal.py
