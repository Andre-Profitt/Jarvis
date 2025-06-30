#!/bin/bash
# Install missing AI libraries for JARVIS

echo "📦 Installing AI libraries for JARVIS..."

cd /Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM

# Install OpenAI
echo "Installing OpenAI..."
pip3 install --user openai

# Install Google Generative AI
echo "Installing Google Generative AI..."
pip3 install --user google-generativeai

# Install other useful libraries
echo "Installing additional libraries..."
pip3 install --user elevenlabs websockets aiofiles

echo "✅ Installation complete!"
echo "🚀 Ready to test multi-AI integration!"
