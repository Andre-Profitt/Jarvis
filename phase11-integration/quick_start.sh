#!/bin/bash
# Quick start script for JARVIS Phase 11

echo "🚀 Starting JARVIS Phase 11 - Complete System Integration"
echo ""
echo "This will launch the fully integrated JARVIS system with all 10 phases working together!"
echo ""

# Navigate to the phase 11 directory
cd "$(dirname "$0")"

# Check if we should run tests first
read -p "Run integration tests first? (recommended) [Y/n]: " run_tests
if [[ $run_tests != "n" && $run_tests != "N" ]]; then
    echo "Running tests..."
    python3 launch_phase11.py --test-only
    if [ $? -ne 0 ]; then
        echo "❌ Tests failed! Please fix issues before launching."
        exit 1
    fi
    echo "✅ All tests passed!"
    echo ""
fi

# Check deployment type
echo "Select deployment configuration:"
echo "1) Production (default)"
echo "2) Development"
echo "3) AWS"
echo "4) GCP"
echo "5) On-Premise"
read -p "Enter choice [1-5]: " choice

case $choice in
    2) CONFIG="default" ;;
    3) CONFIG="aws" ;;
    4) CONFIG="gcp" ;;
    5) CONFIG="on_premise" ;;
    *) CONFIG="production" ;;
esac

echo ""
echo "Launching JARVIS Phase 11 with $CONFIG configuration..."
echo ""

# Launch JARVIS
python3 launch_phase11.py --config $CONFIG

# If you want to use the full deployment script instead, uncomment:
# ./deploy_phase11.sh $CONFIG deploy
