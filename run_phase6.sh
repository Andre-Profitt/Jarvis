#!/bin/bash
# JARVIS Phase 6 Runner Script

echo "🚀 Starting JARVIS Phase 6: Natural Language Flow & Emotional Intelligence"
echo "=================================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found!"
    exit 1
fi

# Navigate to JARVIS directory
cd /Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM

# Run setup first if requested
if [ "$1" = "--setup" ]; then
    echo "📦 Running setup..."
    python3 setup_jarvis_phase6.py
    echo ""
fi

# Run tests if requested
if [ "$1" = "--test" ]; then
    echo "🧪 Running Phase 6 tests..."
    python3 test_jarvis_phase6.py
    exit 0
fi

# Show options if no argument
if [ -z "$1" ]; then
    echo ""
    echo "Options:"
    echo "  1) Interactive Demo"
    echo "  2) Full Demo Scenarios"
    echo "  3) Real-time Conversation"
    echo "  4) Emotional Analysis Test"
    echo "  5) Run Setup"
    echo "  6) Run Tests"
    echo ""
    read -p "Select option (1-6): " choice
    
    case $choice in
        5)
            python3 setup_jarvis_phase6.py
            ;;
        6)
            python3 test_jarvis_phase6.py
            ;;
        *)
            # Pass choice to launcher
            echo $choice | python3 launch_jarvis_phase6.py
            ;;
    esac
else
    # Run launcher directly
    python3 launch_jarvis_phase6.py
fi
