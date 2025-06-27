#!/bin/bash
# Setup API Keys for JARVIS

echo "🔐 Setting up API keys for JARVIS..."

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Configure Gemini CLI
echo "🌟 Configuring Google Gemini CLI..."
gemini config set api_key "$GEMINI_API_KEY"

echo "✅ API keys configured!"
echo ""
echo "🎯 Available AI Models:"
echo "  ✅ Claude Desktop (unlimited via x200 subscription)"
echo "  ✅ Claude Code (Cline in VS Code)"
echo "  ✅ Google Gemini (2M context + multimodal)"
echo "  ✅ GPT-4 (via OpenAI API)"
echo "  ✅ ElevenLabs (ultra-realistic voice)"
echo ""
echo "🚀 JARVIS is ready with full AI power!"