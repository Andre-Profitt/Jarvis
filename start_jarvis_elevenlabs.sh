#!/bin/bash
# Quick launcher for JARVIS with ElevenLabs voice

echo "🚀 Starting JARVIS with ElevenLabs Voice..."
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              JARVIS ElevenLabs Voice Assistant                ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║  🎤 Say 'JARVIS' to activate                                 ║"
echo "║  🔊 Ultra-realistic ElevenLabs voice enabled                 ║"
echo "║  ⚡ Natural conversation mode                                 ║"
echo "║  ❌ Press Ctrl+C to exit                                     ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"
python3 jarvis_elevenlabs_voice_fixed.py