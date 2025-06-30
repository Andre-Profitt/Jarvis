#!/bin/bash
# THE REAL JARVIS - CONVERSATION MODE

echo "🚀 STARTING THE REAL JARVIS WITH CONVERSATION MODE..."
echo ""

cd "$(dirname "$0")"

# Kill any existing JARVIS processes
pkill -f "jarvis" 2>/dev/null

# Quick dependency check
pip install -q SpeechRecognition pyaudio pyttsx3 elevenlabs openai flask flask-socketio flask-cors

# Launch JARVIS
python3 jarvis_conversation_mode.py