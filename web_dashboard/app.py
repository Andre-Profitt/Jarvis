#!/usr/bin/env python3
"""
JARVIS Web Dashboard
Real-time monitoring and control interface for JARVIS
"""

from flask import Flask, render_template, jsonify, request, session
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import json
import os
import sys
from datetime import datetime, timedelta
import logging
from pathlib import Path
import secrets
import asyncio
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import JARVIS components
try:
    from core.voice_first_engine import VoiceFirstEngine
    from core.anticipatory_ai_engine import AnticipatoryEngine
    from core.swarm_integration import JARVISSwarmBridge
except ImportError:
    print("Warning: Some JARVIS components not available")

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(32)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis.dashboard")

# Global state
dashboard_state = {
    "jarvis_status": "offline",
    "listening": False,
    "last_command": None,
    "last_response": None,
    "command_history": [],
    "system_metrics": {
        "cpu_usage": 0,
        "memory_usage": 0,
        "active_agents": 0,
        "response_time": 0
    },
    "voice_activity": False,
    "predictions": [],
    "smart_home_devices": [],
    "calendar_events": []
}

# Connected clients
connected_clients = set()


class JARVISDashboardInterface:
    """Interface between JARVIS and the web dashboard"""
    
    def __init__(self, jarvis_instance=None):
        self.jarvis = jarvis_instance
        self.socketio = socketio
        
    def update_status(self, status: str):
        """Update JARVIS status"""
        dashboard_state["jarvis_status"] = status
        self.broadcast_update("status_update", {"status": status})
        
    def update_listening(self, listening: bool):
        """Update listening state"""
        dashboard_state["listening"] = listening
        self.broadcast_update("listening_update", {"listening": listening})
        
    def add_command(self, command: str, response: str):
        """Add command to history"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "response": response
        }
        
        dashboard_state["last_command"] = command
        dashboard_state["last_response"] = response
        dashboard_state["command_history"].insert(0, entry)
        
        # Keep only last 100 commands
        if len(dashboard_state["command_history"]) > 100:
            dashboard_state["command_history"] = dashboard_state["command_history"][:100]
            
        self.broadcast_update("command_update", entry)
        
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update system metrics"""
        dashboard_state["system_metrics"].update(metrics)
        self.broadcast_update("metrics_update", metrics)
        
    def update_predictions(self, predictions: List[Dict[str, Any]]):
        """Update AI predictions"""
        dashboard_state["predictions"] = predictions[:5]  # Top 5
        self.broadcast_update("predictions_update", {"predictions": predictions})
        
    def update_voice_activity(self, active: bool):
        """Update voice activity indicator"""
        dashboard_state["voice_activity"] = active
        self.broadcast_update("voice_activity", {"active": active})
        
    def broadcast_update(self, event: str, data: Any):
        """Broadcast update to all connected clients"""
        socketio.emit(event, data, namespace='/', broadcast=True)


# Dashboard interface instance
dashboard_interface = JARVISDashboardInterface()


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/status')
def get_status():
    """Get current JARVIS status"""
    return jsonify({
        "status": dashboard_state["jarvis_status"],
        "listening": dashboard_state["listening"],
        "metrics": dashboard_state["system_metrics"],
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/history')
def get_history():
    """Get command history"""
    return jsonify({
        "history": dashboard_state["command_history"][:50],
        "total": len(dashboard_state["command_history"])
    })


@app.route('/api/command', methods=['POST'])
def send_command():
    """Send command to JARVIS"""
    data = request.json
    command = data.get('command', '')
    
    if not command:
        return jsonify({"error": "No command provided"}), 400
        
    # Send command to JARVIS (if connected)
    if dashboard_interface.jarvis:
        try:
            # Process command
            response = asyncio.run(
                dashboard_interface.jarvis.process_voice_command(command)
            )
            
            # Add to history
            dashboard_interface.add_command(command, response.get("response", ""))
            
            return jsonify({
                "success": True,
                "response": response.get("response", ""),
                "confidence": response.get("confidence", 0)
            })
            
        except Exception as e:
            logger.error(f"Command processing failed: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        # Demo mode - simulate response
        response = f"Processed: {command}"
        dashboard_interface.add_command(command, response)
        
        return jsonify({
            "success": True,
            "response": response,
            "confidence": 0.9
        })


@app.route('/api/metrics')
def get_metrics():
    """Get system metrics"""
    return jsonify(dashboard_state["system_metrics"])


@app.route('/api/predictions')
def get_predictions():
    """Get AI predictions"""
    return jsonify({
        "predictions": dashboard_state["predictions"]
    })


@app.route('/api/smart_home')
def get_smart_home():
    """Get smart home device status"""
    # This would integrate with smart home system
    return jsonify({
        "devices": dashboard_state["smart_home_devices"]
    })


@app.route('/api/calendar')
def get_calendar():
    """Get calendar events"""
    # This would integrate with calendar system
    return jsonify({
        "events": dashboard_state["calendar_events"]
    })


# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_id = request.sid
    connected_clients.add(client_id)
    
    logger.info(f"Client connected: {client_id}")
    
    # Send current state
    emit('initial_state', {
        "status": dashboard_state["jarvis_status"],
        "listening": dashboard_state["listening"],
        "last_command": dashboard_state["last_command"],
        "last_response": dashboard_state["last_response"],
        "metrics": dashboard_state["system_metrics"]
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    connected_clients.discard(client_id)
    logger.info(f"Client disconnected: {client_id}")


@socketio.on('toggle_listening')
def handle_toggle_listening():
    """Toggle JARVIS listening state"""
    if dashboard_interface.jarvis:
        # Toggle actual JARVIS listening
        new_state = not dashboard_state["listening"]
        # This would toggle the actual JARVIS listening state
        dashboard_interface.update_listening(new_state)
        
        emit('listening_update', {"listening": new_state}, broadcast=True)


@socketio.on('send_command')
def handle_websocket_command(data):
    """Handle command via WebSocket"""
    command = data.get('command', '')
    
    if command:
        # Process through regular API
        if dashboard_interface.jarvis:
            response = asyncio.run(
                dashboard_interface.jarvis.process_voice_command(command)
            )
            
            dashboard_interface.add_command(
                command,
                response.get("response", "")
            )
        else:
            # Demo mode
            response = {"response": f"Demo: {command}", "confidence": 0.9}
            dashboard_interface.add_command(command, response["response"])


@socketio.on('request_metrics')
def handle_metrics_request():
    """Send current metrics"""
    emit('metrics_update', dashboard_state["system_metrics"])


# Background tasks (when integrated with JARVIS)
def start_metric_updates():
    """Start periodic metric updates"""
    def update_loop():
        while True:
            # Update metrics (would get from actual JARVIS)
            import random
            metrics = {
                "cpu_usage": random.randint(10, 50),
                "memory_usage": random.randint(200, 800),
                "active_agents": random.randint(1, 8),
                "response_time": random.randint(50, 200)
            }
            
            dashboard_interface.update_metrics(metrics)
            socketio.sleep(5)  # Update every 5 seconds
            
    socketio.start_background_task(update_loop)


# Demo data generator (for testing without JARVIS)
def generate_demo_data():
    """Generate demo data for testing"""
    import random
    
    # Add some demo commands
    demo_commands = [
        ("What's the weather?", "It's 72°F and sunny today."),
        ("Open Safari", "Opening Safari for you."),
        ("Set a reminder", "What would you like me to remind you about?"),
        ("Turn on the lights", "Living room lights are now on.")
    ]
    
    for cmd, resp in demo_commands:
        dashboard_interface.add_command(cmd, resp)
        
    # Add demo predictions
    demo_predictions = [
        {"type": "routine", "description": "Time for your morning coffee", "confidence": 0.92},
        {"type": "suggestion", "description": "You have a meeting in 30 minutes", "confidence": 0.88},
        {"type": "automation", "description": "Adjusting temperature for evening", "confidence": 0.75}
    ]
    
    dashboard_interface.update_predictions(demo_predictions)
    
    # Set online status
    dashboard_interface.update_status("online")
    dashboard_interface.update_listening(True)


# Create required directories
template_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"
template_dir.mkdir(exist_ok=True)
static_dir.mkdir(exist_ok=True)
(static_dir / "css").mkdir(exist_ok=True)
(static_dir / "js").mkdir(exist_ok=True)


def run_dashboard(host='0.0.0.0', port=5000, debug=False):
    """Run the dashboard server"""
    logger.info(f"Starting JARVIS Dashboard on http://{host}:{port}")
    
    # Start background tasks
    start_metric_updates()
    
    # Generate demo data if not connected to JARVIS
    if not dashboard_interface.jarvis:
        generate_demo_data()
        
    # Run server
    socketio.run(app, host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_dashboard(debug=True)