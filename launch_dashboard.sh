#!/bin/bash

# JARVIS Web Dashboard Launcher

echo "🌐 Starting JARVIS Web Dashboard..."
echo "=================================="

# Get directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Running setup first..."
    ./setup_10_seamless.sh
fi

# Activate virtual environment
source venv/bin/activate

# Install dashboard requirements
echo "📦 Checking dashboard dependencies..."
pip install --quiet flask flask-socketio flask-cors python-socketio[client]

# Create requirements file for dashboard
cat > web_dashboard/requirements.txt << EOF
flask>=2.3.0
flask-socketio>=5.3.0
flask-cors>=4.0.0
python-socketio[client]>=5.9.0
eventlet>=0.33.0
EOF

# Start dashboard
echo
echo "🚀 Launching JARVIS Dashboard..."
echo "=================================="
echo
echo "📱 Access the dashboard at:"
echo "   Local:    http://localhost:5000"
echo "   Network:  http://$(hostname -I | awk '{print $1}'):5000"
echo
echo "🔒 Security Note: Dashboard is accessible from your network."
echo "   For internet access, use a secure tunnel or VPN."
echo
echo "Press Ctrl+C to stop the dashboard"
echo

# Run the dashboard
cd web_dashboard
python app.py