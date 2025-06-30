#!/bin/bash
# Quick voice dependencies installer

echo "📦 Installing JARVIS Voice Dependencies..."

# Core voice packages
pip install --upgrade SpeechRecognition
pip install --upgrade pyttsx3
pip install --upgrade sounddevice
pip install --upgrade numpy
pip install --upgrade pyaudio

# Enhanced voice features
pip install --upgrade openai
pip install --upgrade openai-whisper
pip install --upgrade elevenlabs

# macOS specific
if [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v brew &> /dev/null; then
        brew install portaudio
    else
        echo "⚠️  Homebrew not found. Install from https://brew.sh"
    fi
fi

echo "✅ Voice dependencies installed!"