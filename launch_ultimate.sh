#!/bin/bash
# Launch JARVIS Ultimate - Your Living AI Companion

echo "🧠 Awakening JARVIS Ultimate..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Kill old basic versions
pkill -f jarvis_minimal
pkill -f jarvis_enhanced

# Launch Ultimate
cd /Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM
python3 jarvis_ultimate.py
