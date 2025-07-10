#!/bin/bash
cd /Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM
python3 jarvis_minimal.py > logs/jarvis_minimal_launch.log 2>&1 &
echo $! > jarvis.pid
echo "JARVIS launched with PID: $(cat jarvis.pid)"
echo "Check logs at: logs/jarvis_minimal_launch.log"
