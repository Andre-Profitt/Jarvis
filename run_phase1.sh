#!/bin/bash
# Quick start script for JARVIS Phase 1

echo "🚀 Starting JARVIS Phase 1 Enhanced System..."
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Navigate to JARVIS directory
cd "$(dirname "$0")"

# Run the setup script
python3 setup_jarvis_phase1.py

echo ""
echo "👋 JARVIS Phase 1 shutdown complete"