#!/usr/bin/env python3
"""
JARVIS Mobile API Server
RESTful API for mobile app communication with JARVIS.
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import jwt
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import redis
from functools import wraps
import hashlib
import uuid

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("jarvis.mobile_api")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('JARVIS_SECRET_KEY', 'jarvis-mobile-secret-key-change-in-production')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Redis for session management
try:
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    redis_client.ping()
except:
    redis_client = None
    logger.warning("Redis not available, using in-memory storage")

# In-memory storage fallback
sessions = {}
devices = {}
jarvis_instance = None


class MobileSession:
    """Represents a mobile device session"""
    
    def __init__(self, device_id: str, device_info: Dict[str, Any]):
        self.device_id = device_id
        self.device_info = device_info
        self.user_id = device_info.get('user_id', 'default')
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        self.auth_token = None
        self.push_token = device_info.get('push_token')
        self.capabilities = device_info.get('capabilities', {})
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'device_id': self.device_id,
            'device_info': self.device_info,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'last_active': self.last_active.isoformat(),
            'capabilities': self.capabilities
        }


# Authentication decorator
def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({'error': 'No authorization header'}), 401
            
        try:
            # Extract token
            token = auth_header.split(' ')[1]
            
            # Verify token
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            
            # Get session
            device_id = payload['device_id']
            session = get_session(device_id)
            
            if not session:
                return jsonify({'error': 'Invalid session'}), 401
                
            # Update last active
            session.last_active = datetime.now()
            save_session(session)
            
            g.session = session
            g.device_id = device_id
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except Exception as e:
            return jsonify({'error': 'Invalid token'}), 401
            
        return f(*args, **kwargs)
        
    return decorated_function


def get_session(device_id: str) -> Optional[MobileSession]:
    """Get session from storage"""
    if redis_client:
        data = redis_client.get(f"session:{device_id}")
        if data:
            return MobileSession(**json.loads(data))
    else:
        return sessions.get(device_id)
        
        
def save_session(session: MobileSession):
    """Save session to storage"""
    if redis_client:
        redis_client.setex(
            f"session:{session.device_id}",
            3600 * 24 * 7,  # 7 days
            json.dumps(session.to_dict())
        )
    else:
        sessions[session.device_id] = session


# API Routes

@app.route('/api/v1/register', methods=['POST'])
def register_device():
    """Register a new mobile device"""
    try:
        data = request.json
        
        # Validate required fields
        required = ['device_id', 'platform', 'model', 'app_version']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
                
        device_id = data['device_id']
        
        # Create session
        session = MobileSession(device_id, data)
        
        # Generate auth token
        token_payload = {
            'device_id': device_id,
            'exp': datetime.utcnow() + timedelta(days=30)
        }
        session.auth_token = jwt.encode(token_payload, app.config['SECRET_KEY'], algorithm='HS256')
        
        # Save session
        save_session(session)
        
        # Store device info
        devices[device_id] = data
        
        logger.info(f"Registered device: {device_id} ({data['platform']} {data['model']})")
        
        return jsonify({
            'success': True,
            'auth_token': session.auth_token,
            'device_id': device_id,
            'features': get_available_features()
        })
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/command', methods=['POST'])
@require_auth
def send_command():
    """Send a voice command to JARVIS"""
    try:
        data = request.json
        command = data.get('command')
        
        if not command:
            return jsonify({'error': 'No command provided'}), 400
            
        # Log command
        logger.info(f"Mobile command from {g.device_id}: {command}")
        
        # Process through JARVIS
        if jarvis_instance:
            result = asyncio.run(jarvis_instance.process_voice_command(command))
            
            # Track usage
            track_command(g.session, command, result)
            
            return jsonify({
                'success': True,
                'response': result['response'],
                'confidence': result.get('confidence', 0.9),
                'suggestions': get_suggestions(command, result)
            })
        else:
            return jsonify({
                'success': False,
                'error': 'JARVIS not available'
            }), 503
            
    except Exception as e:
        logger.error(f"Command error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/status', methods=['GET'])
@require_auth
def get_status():
    """Get JARVIS status and system info"""
    try:
        if jarvis_instance:
            status = jarvis_instance.get_status_summary()
            
            # Add mobile-specific info
            status['mobile'] = {
                'connected_devices': len(devices),
                'your_device': g.session.device_info,
                'last_command': get_last_command(g.device_id)
            }
            
            return jsonify(status)
        else:
            return jsonify({
                'jarvis_status': 'offline',
                'message': 'JARVIS is not running'
            })
            
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/smart-home/devices', methods=['GET'])
@require_auth
def get_smart_devices():
    """Get list of smart home devices"""
    try:
        if jarvis_instance and jarvis_instance.smart_home:
            devices = []
            
            for device_id, device in jarvis_instance.smart_home.devices.items():
                devices.append({
                    'id': device_id,
                    'name': device.name,
                    'type': device.device_type,
                    'state': device.get_state(),
                    'room': device.room,
                    'capabilities': device.capabilities
                })
                
            return jsonify({
                'success': True,
                'devices': devices
            })
        else:
            return jsonify({
                'success': False,
                'devices': [],
                'message': 'Smart home not configured'
            })
            
    except Exception as e:
        logger.error(f"Smart home error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/smart-home/control', methods=['POST'])
@require_auth
def control_smart_device():
    """Control a smart home device"""
    try:
        data = request.json
        device_id = data.get('device_id')
        action = data.get('action')
        params = data.get('params', {})
        
        if not device_id or not action:
            return jsonify({'error': 'Missing device_id or action'}), 400
            
        if jarvis_instance and jarvis_instance.smart_home:
            # Execute control
            result = asyncio.run(
                jarvis_instance.smart_home.control_device(device_id, action, **params)
            )
            
            # Log action
            logger.info(f"Mobile control: {device_id} - {action} from {g.device_id}")
            
            return jsonify({
                'success': True,
                'device_id': device_id,
                'action': action,
                'result': result
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Smart home not available'
            }), 503
            
    except Exception as e:
        logger.error(f"Control error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/calendar/events', methods=['GET'])
@require_auth
def get_calendar_events():
    """Get calendar events"""
    try:
        days = int(request.args.get('days', 7))
        
        if jarvis_instance and jarvis_instance.calendar_ai:
            events = asyncio.run(
                jarvis_instance.calendar_ai.get_upcoming_events(days=days)
            )
            
            return jsonify({
                'success': True,
                'events': [event.to_dict() for event in events]
            })
        else:
            return jsonify({
                'success': False,
                'events': [],
                'message': 'Calendar not configured'
            })
            
    except Exception as e:
        logger.error(f"Calendar error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/shortcuts', methods=['GET'])
@require_auth
def get_shortcuts():
    """Get user's command shortcuts"""
    shortcuts = [
        {
            'id': 'morning',
            'name': 'Good Morning',
            'icon': 'sun',
            'command': 'Good morning JARVIS',
            'description': 'Start your morning routine'
        },
        {
            'id': 'lights_on',
            'name': 'Lights On',
            'icon': 'lightbulb',
            'command': 'Turn on all lights',
            'description': 'Turn on all lights in the house'
        },
        {
            'id': 'lights_off',
            'name': 'Lights Off',
            'icon': 'lightbulb-slash',
            'command': 'Turn off all lights',
            'description': 'Turn off all lights in the house'
        },
        {
            'id': 'check_calendar',
            'name': 'My Schedule',
            'icon': 'calendar',
            'command': 'What\'s on my calendar today?',
            'description': 'Check today\'s schedule'
        },
        {
            'id': 'check_email',
            'name': 'Check Email',
            'icon': 'envelope',
            'command': 'Check my emails',
            'description': 'Get email summary'
        },
        {
            'id': 'goodnight',
            'name': 'Good Night',
            'icon': 'moon',
            'command': 'Good night JARVIS',
            'description': 'Start bedtime routine'
        }
    ]
    
    # Add user's frequent commands
    user_shortcuts = get_user_shortcuts(g.device_id)
    shortcuts.extend(user_shortcuts)
    
    return jsonify({
        'success': True,
        'shortcuts': shortcuts
    })


# WebSocket events

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    logger.info(f"Mobile client connected: {request.sid}")
    emit('connected', {'status': 'Connected to JARVIS Mobile API'})


@socketio.on('authenticate')
def handle_authenticate(data):
    """Authenticate WebSocket connection"""
    try:
        token = data.get('token')
        
        # Verify token
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        device_id = payload['device_id']
        
        # Join device room
        join_room(device_id)
        
        emit('authenticated', {'success': True, 'device_id': device_id})
        
    except Exception as e:
        emit('authenticated', {'success': False, 'error': str(e)})


@socketio.on('voice_stream')
def handle_voice_stream(data):
    """Handle streaming voice data"""
    # This would process streaming audio for real-time voice commands
    # For now, just acknowledge
    emit('voice_ack', {'received': True})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info(f"Mobile client disconnected: {request.sid}")


# Helper functions

def get_available_features() -> Dict[str, bool]:
    """Get available JARVIS features"""
    features = {
        'voice_commands': True,
        'smart_home': False,
        'calendar': False,
        'email': False,
        'shortcuts': True,
        'voice_streaming': True,
        'push_notifications': False
    }
    
    if jarvis_instance:
        features['smart_home'] = jarvis_instance.smart_home is not None
        features['calendar'] = jarvis_instance.calendar_ai is not None
        features['email'] = jarvis_instance.email_ai is not None
        
    return features


def track_command(session: MobileSession, command: str, result: Dict[str, Any]):
    """Track command usage for analytics"""
    key = f"commands:{session.device_id}"
    
    command_data = {
        'command': command,
        'response': result.get('response'),
        'timestamp': datetime.now().isoformat(),
        'success': result.get('success', True)
    }
    
    if redis_client:
        redis_client.lpush(key, json.dumps(command_data))
        redis_client.ltrim(key, 0, 99)  # Keep last 100 commands
    else:
        if key not in sessions:
            sessions[key] = []
        sessions[key].insert(0, command_data)
        sessions[key] = sessions[key][:100]


def get_last_command(device_id: str) -> Optional[Dict[str, Any]]:
    """Get last command from device"""
    key = f"commands:{device_id}"
    
    if redis_client:
        data = redis_client.lindex(key, 0)
        return json.loads(data) if data else None
    else:
        commands = sessions.get(key, [])
        return commands[0] if commands else None


def get_suggestions(command: str, result: Dict[str, Any]) -> List[str]:
    """Get command suggestions based on context"""
    suggestions = []
    
    # Context-based suggestions
    if 'light' in command.lower():
        suggestions.extend([
            "Dim the lights to 50%",
            "Turn on movie mode",
            "Set lights to warm white"
        ])
    elif 'calendar' in command.lower():
        suggestions.extend([
            "Schedule a meeting",
            "What's free tomorrow?",
            "Remind me about my next meeting"
        ])
    elif 'email' in command.lower():
        suggestions.extend([
            "Any urgent emails?",
            "Summarize emails from boss",
            "Draft a reply"
        ])
        
    return suggestions[:3]  # Return top 3


def get_user_shortcuts(device_id: str) -> List[Dict[str, Any]]:
    """Get user's frequent commands as shortcuts"""
    shortcuts = []
    
    # Get command history
    key = f"commands:{device_id}"
    commands = []
    
    if redis_client:
        raw_commands = redis_client.lrange(key, 0, 99)
        commands = [json.loads(cmd) for cmd in raw_commands]
    else:
        commands = sessions.get(key, [])
        
    # Count frequency
    command_count = {}
    for cmd_data in commands:
        cmd = cmd_data['command']
        command_count[cmd] = command_count.get(cmd, 0) + 1
        
    # Get top commands
    top_commands = sorted(command_count.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Create shortcuts
    for i, (cmd, count) in enumerate(top_commands):
        if count >= 3:  # Used at least 3 times
            shortcuts.append({
                'id': f'frequent_{i}',
                'name': f'"{cmd[:20]}..."' if len(cmd) > 20 else f'"{cmd}"',
                'icon': 'star',
                'command': cmd,
                'description': f'Used {count} times'
            })
            
    return shortcuts


def set_jarvis_instance(instance):
    """Set the JARVIS instance for the API to use"""
    global jarvis_instance
    jarvis_instance = instance
    logger.info("JARVIS instance connected to Mobile API")


def broadcast_update(event_type: str, data: Dict[str, Any]):
    """Broadcast update to all connected mobile clients"""
    socketio.emit(event_type, data, namespace='/')


# API server runner
def run_mobile_api(host='0.0.0.0', port=5001, debug=False):
    """Run the mobile API server"""
    logger.info(f"Starting JARVIS Mobile API on {host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run server
    run_mobile_api(debug=True)