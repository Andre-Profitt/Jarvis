#!/bin/bash
# Launch JARVIS Ultimate with ElevenLabs Voice and Web Interface

echo "🚀 Launching JARVIS Ultimate..."
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           JARVIS Ultimate - Full Featured AI Assistant        ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║  🌐 Opening web interface in your browser...                  ║"
echo "║  🎤 ElevenLabs voice enabled                                 ║"
echo "║  🤖 GPT-4 AI integration active                              ║"
echo "║  ⚡ Real-time communication ready                            ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Open browser after a short delay
(sleep 2 && open http://localhost:8888) &

# Start JARVIS
python3 jarvis_ultimate_elevenlabs.py