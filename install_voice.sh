#!/bin/bash
# Quick install script for JARVIS voice recognition

echo "🎤 Installing JARVIS Voice Recognition..."
echo "========================================"

# Install Python packages
echo "📦 Installing Python packages..."
pip3 install SpeechRecognition pyaudio sounddevice numpy

# Check if on macOS and install portaudio
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 macOS detected. Installing portaudio..."
    
    # Check if brew is installed
    if command -v brew &> /dev/null; then
        brew install portaudio
    else
        echo "❌ Homebrew not found. Please install it first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "🚀 Now you can run:"
echo "   python3 jarvis_always_on.py"
echo ""
echo "Then just say 'Hey JARVIS'!"
