#!/bin/bash
# JARVIS Phase 3 Runner

echo "🚀 Starting JARVIS Phase 3..."
echo ""

# Change to JARVIS directory
cd "$(dirname "$0")"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not installed."
    exit 1
fi

# Check for virtual environment
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
else
    echo "ℹ️  No virtual environment found. Using system Python."
fi

# Run Phase 3
if [ "$1" == "test" ]; then
    echo "🧪 Running Phase 3 tests..."
    python3 test_jarvis_phase3.py
elif [ "$1" == "interactive-test" ]; then
    echo "🎮 Running interactive test..."
    python3 test_jarvis_phase3.py interactive
elif [ "$1" == "status" ]; then
    echo "📊 Checking Phase 3 status..."
    python3 launch_jarvis_phase3.py --status
else
    echo "🤖 Launching JARVIS with Phase 3 enhancements..."
    python3 launch_jarvis_phase3.py "$@"
fi
