#!/bin/bash

# JARVIS v10.0 - Complete Setup Script
# This ensures all dependencies are installed

echo "🚀 JARVIS v10.0 Complete Setup"
echo "=============================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Install all required packages
echo ""
echo "📦 Installing dependencies..."

# Core packages
pip3 install --upgrade pip
pip3 install openai google-generativeai elevenlabs
pip3 install speechrecognition pyaudio sounddevice numpy
pip3 install rich

# Additional packages for full functionality
pip3 install requests python-dotenv
pip3 install scipy

# For macOS audio
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Detected macOS - Installing audio dependencies..."
    if command -v brew &> /dev/null; then
        brew install portaudio
    else
        echo "⚠️  Homebrew not found. Please install portaudio manually."
    fi
fi

echo ""
echo "✅ All dependencies installed!"
echo ""
echo "🎯 Quick Start Commands:"
echo "  1. cd /Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM"
echo "  2. python3 launch_jarvis_ultimate.py"
echo ""
echo "  Or use the shortcut: jarvis (after sourcing ~/.zshrc)"
echo ""
echo "🎉 Setup complete! Your JARVIS v10.0 is ready to launch!"
