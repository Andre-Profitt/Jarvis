#!/bin/bash

# JARVIS 10/10 - Ultimate Seamless Setup
# Complete setup in under 2 minutes!

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Banner
clear
echo -e "${BLUE}"
echo "     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗"
echo "     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝"
echo "     ██║███████║██████╔╝██║   ██║██║███████╗"
echo "██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║"
echo "╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║"
echo " ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝"
echo -e "${NC}"
echo -e "${GREEN}10/10 Ultimate Seamless AI Assistant${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Get directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Step counter
STEP=1
TOTAL_STEPS=7

step() {
    echo -e "\n${BLUE}[$STEP/$TOTAL_STEPS]${NC} $1"
    ((STEP++))
}

# Check Python
step "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found!${NC}"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"

# Create virtual environment
step "Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
step "Installing core dependencies..."
pip install --quiet --upgrade pip

# Core requirements
cat > requirements_core.txt << EOF
speechrecognition>=3.10.0
pyttsx3>=2.90
pyaudio>=0.2.11
sounddevice>=0.4.6
numpy>=1.24.0
python-daemon>=2.3.0
psutil>=5.9.0
EOF

pip install --quiet -r requirements_core.txt

# Optional AI providers (install silently, don't fail)
pip install --quiet openai anthropic google-generativeai 2>/dev/null || true

echo -e "${GREEN}✅ Dependencies installed${NC}"

# Make scripts executable
step "Setting up JARVIS commands..."
chmod +x jarvis_10_seamless.py
chmod +x jarvis_background_service.py

# Create command shortcuts
cat > jarvis << 'EOF'
#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
source venv/bin/activate 2>/dev/null
python3 jarvis_10_seamless.py "$@"
EOF

cat > jarvis-service << 'EOF'
#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
source venv/bin/activate 2>/dev/null
python3 jarvis_background_service.py "$@"
EOF

chmod +x jarvis jarvis-service

# Add to PATH
SHELL_RC="$HOME/.zshrc"
if [ -f "$HOME/.bash_profile" ]; then
    SHELL_RC="$HOME/.bash_profile"
fi

if ! grep -q "JARVIS 10/10" "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo "# JARVIS 10/10 AI Assistant" >> "$SHELL_RC"
    echo "export PATH=\"$DIR:\$PATH\"" >> "$SHELL_RC"
fi

echo -e "${GREEN}✅ Commands created${NC}"

# Setup configuration
step "Creating configuration..."
mkdir -p "$HOME/.jarvis"

if [ ! -f "$HOME/.jarvis/config.json" ]; then
    cat > "$HOME/.jarvis/config.json" << EOF
{
    "auto_start": true,
    "restart_on_failure": true,
    "features": {
        "voice_activation": true,
        "continuous_listening": true,
        "background_learning": true,
        "proactive_assistance": true
    }
}
EOF
fi

# Create .env for API keys
if [ ! -f ".env" ]; then
    cat > .env << EOF
# Optional API Keys (for enhanced features)
# Leave blank if you don't have them - JARVIS will still work!

OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
GOOGLE_AI_API_KEY=""

# Voice settings
JARVIS_VOICE_SPEED=180
JARVIS_VOICE_VOLUME=0.9
EOF
fi

echo -e "${GREEN}✅ Configuration created${NC}"

# Test microphone
step "Testing microphone access..."
python3 -c "
import speech_recognition as sr
r = sr.Recognizer()
m = sr.Microphone()
with m as source:
    r.adjust_for_ambient_noise(source, duration=0.5)
print('✅ Microphone access confirmed')
" 2>/dev/null || echo -e "${YELLOW}⚠️  Microphone test failed - check permissions${NC}"

# macOS specific setup
if [[ "$OSTYPE" == "darwin"* ]]; then
    step "Setting up macOS integration..."
    
    # Request microphone permission
    if ! plutil -p ~/Library/Preferences/com.apple.TCC/TCC.db &> /dev/null; then
        echo -e "${YELLOW}📱 Please grant microphone access when prompted${NC}"
    fi
    
    # Create app bundle
    APP_DIR="$HOME/Applications/JARVIS 10.app"
    mkdir -p "$APP_DIR/Contents/MacOS"
    mkdir -p "$APP_DIR/Contents/Resources"
    
    # App launcher
    cat > "$APP_DIR/Contents/MacOS/JARVIS" << EOF
#!/bin/bash
cd "$DIR"
source venv/bin/activate
python3 jarvis_10_seamless.py
EOF
    
    chmod +x "$APP_DIR/Contents/MacOS/JARVIS"
    
    # Info.plist
    cat > "$APP_DIR/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>JARVIS</string>
    <key>CFBundleName</key>
    <string>JARVIS 10</string>
    <key>CFBundleDisplayName</key>
    <string>JARVIS AI Assistant</string>
    <key>CFBundleIdentifier</key>
    <string>com.jarvis.assistant</string>
    <key>CFBundleVersion</key>
    <string>10.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>JARVIS needs microphone access to hear your commands</string>
</dict>
</plist>
EOF
    
    echo -e "${GREEN}✅ macOS app created${NC}"
else
    ((STEP--))
fi

# Quick start guide
step "Setup complete! 🎉"

echo
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✨ JARVIS 10/10 is ready to use!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

echo -e "${BLUE}🚀 Quick Start:${NC}"
echo
echo "  1. Start JARVIS now:"
echo -e "     ${YELLOW}./jarvis${NC}"
echo
echo "  2. Or install as always-running service:"
echo -e "     ${YELLOW}./jarvis-service install${NC}"
echo -e "     ${YELLOW}./jarvis-service start${NC}"
echo
echo "  3. After restarting terminal, just type:"
echo -e "     ${YELLOW}jarvis${NC}"
echo

echo -e "${BLUE}💬 How to use:${NC}"
echo "  • Just say 'Hey JARVIS' and start talking"
echo "  • No need to repeat 'Hey JARVIS' - just keep talking"
echo "  • JARVIS will learn your patterns and preferences"
echo "  • Try: 'Hey JARVIS, open Safari'"
echo

echo -e "${BLUE}⚡ Features:${NC}"
echo "  ✅ Always listening (no mode switching)"
echo "  ✅ Natural conversation (maintains context)"
echo "  ✅ System control (apps, volume, brightness)"
echo "  ✅ Learning system (gets smarter over time)"
echo "  ✅ Zero friction (just talk naturally)"
echo

# Auto-start option
echo -e "${YELLOW}Would you like to start JARVIS now? (y/n)${NC}"
read -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Starting JARVIS...${NC}"
    source venv/bin/activate
    python3 jarvis_10_seamless.py
fi