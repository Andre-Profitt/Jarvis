#!/bin/bash

# JARVIS Mobile API Launcher

echo "📱 Starting JARVIS Mobile API Server..."
echo "======================================="

# Get directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "📦 Installing dependencies..."
pip install --quiet flask flask-socketio flask-cors python-socketio[client] redis pyjwt

# Create requirements file
cat > mobile_app/requirements.txt << EOF
flask>=2.3.0
flask-socketio>=5.3.0
flask-cors>=4.0.0
python-socketio[client]>=5.9.0
pyjwt>=2.8.0
redis>=4.6.0
eventlet>=0.33.0
EOF

# Get IP addresses
LOCAL_IP=$(hostname -I | awk '{print $1}')
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    LOCAL_IP=$(ipconfig getifaddr en0 || ipconfig getifaddr en1)
fi

# Start the mobile API
echo
echo "🚀 Launching JARVIS Mobile API..."
echo "=================================="
echo
echo "📱 Mobile API Endpoints:"
echo "   Local:    http://localhost:5001"
echo "   Network:  http://$LOCAL_IP:5001"
echo
echo "📲 Configure your mobile app with:"
echo "   API URL: http://$LOCAL_IP:5001"
echo
echo "🔒 For external access, use ngrok:"
echo "   ngrok http 5001"
echo
echo "Press Ctrl+C to stop the server"
echo

# Run the mobile API
cd mobile_app
python jarvis_mobile_api.py