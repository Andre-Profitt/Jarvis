#!/bin/bash
# Launch Enhanced JARVIS

cd /Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM

# Kill old processes
echo "Stopping old JARVIS processes..."
pkill -f jarvis
sleep 2

# Launch enhanced version
echo "Launching JARVIS Enhanced..."
python3 jarvis_enhanced.py > logs/jarvis_enhanced_launch.log 2>&1 &
echo $! > jarvis.pid

echo "✅ JARVIS Enhanced launched with PID: $(cat jarvis.pid)"
echo "📝 Check logs at: logs/jarvis_enhanced_launch.log"
echo "📊 Monitor with: tail -f logs/jarvis_enhanced_*.log"
