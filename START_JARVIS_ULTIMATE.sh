#!/bin/bash
# Start JARVIS ULTIMATE

echo "🚀 STARTING JARVIS ULTIMATE..."

cd "$(dirname "$0")"

# Kill any existing processes
pkill -f "jarvis" 2>/dev/null
pkill -f "python" 2>/dev/null
sleep 1

# Start JARVIS in background with logging
nohup python3 JARVIS_ULTIMATE_FINAL.py > jarvis.log 2>&1 &

echo "✅ JARVIS starting..."
sleep 3

# Check if it's running
if lsof -i :8888 > /dev/null 2>&1; then
    echo "✅ JARVIS is running on http://localhost:8888"
    open http://localhost:8888
else
    echo "❌ JARVIS failed to start. Check jarvis.log"
    tail jarvis.log
fi