#!/bin/bash
# JARVIS Phase 2 Run Script

echo "🚀 JARVIS Phase 2 Launcher"
echo "========================="
echo ""
echo "Select an option:"
echo "1) Run Phase 2 Demo"
echo "2) Run Quick Test"
echo "3) Run Full Test Suite"
echo "4) Run Integrated JARVIS with Phase 2"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "Starting Phase 2 Demo..."
        python phase2/launch_phase2_demo.py
        ;;
    2)
        echo "Running Quick Test..."
        python phase2/quick_test_phase2.py
        ;;
    3)
        echo "Running Full Test Suite..."
        python phase2/test_phase2.py
        ;;
    4)
        echo "Starting Integrated JARVIS..."
        # First ensure Phase 1 is running
        python launch_jarvis_phase1.py &
        PHASE1_PID=$!
        sleep 2
        
        # Then start Phase 2 demo
        python phase2/launch_phase2_demo.py
        
        # Cleanup
        kill $PHASE1_PID 2>/dev/null
        ;;
    *)
        echo "Invalid choice. Please run again and select 1-4."
        exit 1
        ;;
esac
