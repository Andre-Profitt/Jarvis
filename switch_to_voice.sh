#!/bin/bash
#
# JARVIS Voice Mode Quick Switch
# One command to switch from web UI to voice-first interface
#

JARVIS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$JARVIS_ROOT"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           JARVIS Voice-First Mode Activator                   ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Check Python
echo "🐍 Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found! Please install Python 3.8+"
    exit 1
fi
echo "✅ Python 3 found: $(python3 --version)"

# Step 2: Install dependencies
echo ""
echo "📦 Installing voice dependencies..."
python3 -m pip install -q --upgrade pip
python3 -m pip install -q SpeechRecognition pyttsx3 sounddevice numpy pyaudio

# Additional dependencies for enhanced features
python3 -m pip install -q openai whisper elevenlabs

# macOS specific
if [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v brew &> /dev/null; then
        echo "🍎 Installing macOS audio dependencies..."
        brew install portaudio 2>/dev/null || echo "✅ Portaudio already installed"
    fi
fi

# Step 3: Configure JARVIS for voice mode
echo ""
echo "🔧 Configuring JARVIS for voice-first mode..."
python3 configure_voice_mode.py

# Step 4: Make scripts executable
chmod +x launch_voice_jarvis.py
chmod +x configure_voice_mode.py

# Step 5: Create desktop shortcut (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    DESKTOP="$HOME/Desktop"
    if [ -d "$DESKTOP" ]; then
        cat > "$DESKTOP/JARVIS Voice.command" << EOF
#!/bin/bash
cd "$JARVIS_ROOT"
python3 launch_voice_jarvis.py
EOF
        chmod +x "$DESKTOP/JARVIS Voice.command"
        echo "✅ Desktop shortcut created!"
    fi
fi

# Step 6: Launch JARVIS in voice mode
echo ""
echo "🚀 Launching JARVIS in voice-first mode..."
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  🎤 Say 'Hey JARVIS' to start talking                        ║"
echo "║  🔊 Natural conversation mode enabled                         ║"
echo "║  ⚡ Voice commands processed in real-time                     ║"
echo "║  ❌ Press Ctrl+C to exit                                     ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Launch JARVIS
python3 launch_voice_jarvis.py