#!/bin/bash

# JARVIS Phase 9 - Quick Setup Script
# ====================================

echo "🚀 JARVIS Phase 9 - Performance Optimization Setup"
echo "=================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Install required packages
echo ""
echo "📦 Installing required packages..."
pip3 install -q redis websockets psutil numpy lz4 pytest aiofiles

# Check Redis
echo ""
echo "🔍 Checking Redis..."
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo "✓ Redis is running"
    else
        echo "⚠️  Redis is installed but not running"
        echo "   Start with: redis-server"
    fi
else
    echo "⚠️  Redis not installed"
    echo "   Install with: brew install redis (macOS) or sudo apt install redis (Ubuntu)"
fi

# Create necessary directories
echo ""
echo "📁 Setting up directories..."
mkdir -p logs
mkdir -p cache

# Run tests
echo ""
echo "🧪 Running Phase 9 tests..."
python3 phase9/test_phase9.py

# Make launch script executable
chmod +x phase9/launch_phase9.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Quick Start Commands:"
echo "   1. Start with monitoring:    python3 phase9/launch_phase9.py --monitor"
echo "   2. Run demo:                 python3 phase9/launch_phase9.py --demo"
echo "   3. Open dashboard:           open phase9/performance_monitor.html"
echo "   4. Run batch test:           python3 phase9/launch_phase9.py --batch 100"
echo ""
echo "📚 For more information, see: phase9/README.md"
