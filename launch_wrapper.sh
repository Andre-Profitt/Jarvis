#!/bin/bash
cd /Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM
mkdir -p logs
export PYTHONUNBUFFERED=1
python3 LAUNCH-JARVIS-REAL.py 2>&1 | tee logs/jarvis_launch_$(date +%Y%m%d_%H%M%S).log &
echo $! > jarvis.pid
echo "✅ JARVIS launched with PID: $(cat jarvis.pid)"
