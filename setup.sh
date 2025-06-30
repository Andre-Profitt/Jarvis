#!/bin/bash

# JARVIS Quick Setup Script
# Run this once to set up everything

echo "🚀 JARVIS Quick Setup"
echo "===================="

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Make scripts executable
chmod +x jarvis.py jarvis_seamless.py jarvis_seamless_v2.py

# Create command line shortcut
echo "📝 Creating 'jarvis' command..."

# Add to shell profile
SHELL_PROFILE="$HOME/.zshrc"
if [ -f "$HOME/.bash_profile" ]; then
    SHELL_PROFILE="$HOME/.bash_profile"
fi

# Check if already added
if ! grep -q "alias jarvis=" "$SHELL_PROFILE"; then
    echo "" >> "$SHELL_PROFILE"
    echo "# JARVIS AI Assistant" >> "$SHELL_PROFILE"
    echo "alias jarvis='cd $DIR && python3 jarvis.py'" >> "$SHELL_PROFILE"
    echo "✅ Added 'jarvis' command to $SHELL_PROFILE"
else
    echo "✅ 'jarvis' command already exists"
fi

# Create desktop app (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🖥️  Creating desktop app..."
    
    APP_DIR="$HOME/Applications/JARVIS.app"
    mkdir -p "$APP_DIR/Contents/MacOS"
    
    # Create launch script
    cat > "$APP_DIR/Contents/MacOS/JARVIS" << EOF
#!/bin/bash
cd "$DIR"
osascript -e 'tell application "Terminal" to do script "cd $DIR && python3 jarvis.py"'
EOF
    
    chmod +x "$APP_DIR/Contents/MacOS/JARVIS"
    
    # Create Info.plist
    cat > "$APP_DIR/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>JARVIS</string>
    <key>CFBundleName</key>
    <string>JARVIS</string>
    <key>CFBundleIdentifier</key>
    <string>com.jarvis.assistant</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
</dict>
</plist>
EOF
    
    echo "✅ Created JARVIS.app in ~/Applications"
fi

echo ""
echo "✨ Setup Complete!"
echo ""
echo "🎯 How to use JARVIS:"
echo ""
echo "  Option 1: Command Line"
echo "  └─ Type 'jarvis' in Terminal (after restarting Terminal)"
echo ""
echo "  Option 2: Desktop App (macOS)"  
echo "  └─ Open JARVIS from ~/Applications"
echo ""
echo "  Option 3: Direct Launch"
echo "  └─ Run: python3 jarvis.py"
echo ""
echo "🎤 Once started:"
echo "  • Say 'Hey JARVIS' to activate"
echo "  • Or just start giving commands"
echo "  • JARVIS learns and adapts to you"
echo ""
echo "Ready to start? Run: source $SHELL_PROFILE && jarvis"
