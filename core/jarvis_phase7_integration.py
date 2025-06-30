"""
JARVIS Phase 7: Complete Visual Integration
==========================================
Integrates visual feedback with all JARVIS systems
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import websockets
from pathlib import Path

# Import previous phases
from .jarvis_phase6_integration import JARVISPhase6Core
from .unified_input_pipeline import InputType
from .fluid_state_management import SystemState

# Import Phase 7 components
from .ui_components import JARVISUIComponents, UITheme, StatusIndicator
from .visual_feedback_system import VisualFeedbackSystem, UIBridge, InterventionType

class JARVISPhase7Core:
    """JARVIS with complete visual feedback integration"""
    
    def __init__(self, existing_jarvis=None):
        # Use Phase 6 as base
        self.core = existing_jarvis or JARVISPhase6Core()
        
        # Phase 7 components
        self.visual_feedback = VisualFeedbackSystem(UITheme.DARK)
        self.ui_bridge = UIBridge(self.visual_feedback)
        
        # WebSocket server for real-time UI updates
        self.websocket_clients = set()
        self.websocket_server = None
        
        # Visual preferences
        self.visual_preferences = {
            "show_sensor_status": True,
            "preview_interventions": True,
            "mode_indicators": True,
            "activity_timeline": True,
            "notification_duration": 5000,
            "theme": UITheme.DARK
        }
        
        # Performance metrics
        self.ui_metrics = {
            "updates_sent": 0,
            "interventions_shown": 0,
            "notifications_displayed": 0,
            "user_cancellations": 0
        }
        
        self._setup_event_handlers()
        
    def _setup_event_handlers(self):
        """Setup event handlers for visual updates"""
        # Register visual feedback callbacks
        self.visual_feedback.register_callback(
            "intervention_cancelled",
            self._handle_intervention_cancelled
        )
        
        # Add update listener for WebSocket broadcast
        self.visual_feedback.add_update_listener(self._broadcast_ui_update)
        
    async def initialize(self):
        """Initialize all components including visual system"""
        await self.core.initialize()
        
        # Start WebSocket server for UI
        if not self.websocket_server:
            self.websocket_server = asyncio.create_task(self._start_websocket_server())
            
        print("✅ JARVIS Phase 7 initialized with Visual Feedback System")
        
    async def _start_websocket_server(self, host="localhost", port=8765):
        """Start WebSocket server for real-time UI updates"""
        async def handle_client(websocket, path):
            self.websocket_clients.add(websocket)
            try:
                # Send initial UI state
                initial_state = {
                    "type": "initial_state",
                    "data": self.visual_feedback.get_current_ui_state(),
                    "html": self.visual_feedback.generate_full_ui()
                }
                await websocket.send(json.dumps(initial_state))
                
                # Handle client messages
                async for message in websocket:
                    await self._handle_client_message(websocket, json.loads(message))
                    
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.websocket_clients.remove(websocket)
                
        try:
            await websockets.serve(handle_client, host, port)
            print(f"🌐 WebSocket server started on ws://{host}:{port}")
        except Exception as e:
            print(f"Failed to start WebSocket server: {e}")
            
    async def _handle_client_message(self, websocket, message: Dict):
        """Handle messages from UI clients"""
        msg_type = message.get("type")
        
        if msg_type == "cancel_intervention":
            await self.visual_feedback.cancel_intervention(message["id"])
            self.ui_metrics["user_cancellations"] += 1
            
        elif msg_type == "update_preference":
            self.visual_preferences[message["preference"]] = message["value"]
            
        elif msg_type == "request_update":
            await self._send_ui_update(websocket)
            
    async def _broadcast_ui_update(self, update_type: str, data: Dict):
        """Broadcast UI updates to all connected clients"""
        if not self.websocket_clients:
            return
            
        message = json.dumps({
            "type": update_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
        
        # Send to all connected clients
        disconnected = set()
        for client in self.websocket_clients:
            try:
                await client.send(message)
                self.ui_metrics["updates_sent"] += 1
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
                
        # Remove disconnected clients
        self.websocket_clients -= disconnected
        
    async def _send_ui_update(self, websocket):
        """Send UI update to specific client"""
        update = {
            "type": "state_update",
            "data": self.visual_feedback.get_current_ui_state(),
            "html": self.visual_feedback.generate_full_ui()
        }
        await websocket.send(json.dumps(update))
        
    async def process_input(self, input_data: Dict, source: str = "unknown") -> Dict:
        """Process input with visual feedback"""
        # Update sensor status
        if "voice" in input_data:
            await self.visual_feedback.update_sensor_status(
                "voice", "processing", {"active": True}
            )
            
        if "biometric" in input_data:
            await self.visual_feedback.update_sensor_status(
                "biometric", "active", input_data["biometric"]
            )
            
        # Process through core
        result = await self.core.process_input(input_data, source)
        
        # Update visual feedback based on results
        await self._update_visual_feedback(result)
        
        # Check for needed interventions
        await self._check_interventions(result)
        
        # Update sensor status to idle
        if "voice" in input_data:
            await self.visual_feedback.update_sensor_status(
                "voice", "idle", {"active": False}
            )
            
        return result
        
    async def _update_visual_feedback(self, result: Dict):
        """Update visual feedback based on processing results"""
        # Update mode if changed
        new_mode = result.get("mode")
        if new_mode and new_mode != self.visual_feedback.current_mode:
            state_info = {
                "stress_level": result.get("emotional_state", {}).get("arousal", 0),
                "focus_level": 0.8 if new_mode == "flow" else 0.5,
                "emotional_state": result.get("emotional_state", {})
            }
            
            reason = self._get_mode_change_reason(new_mode, result)
            await self.visual_feedback.update_mode(new_mode, state_info, reason)
            
        # Update emotional sensor
        emotional_state = result.get("emotional_state", {})
        if emotional_state:
            await self.visual_feedback.update_sensor_status(
                "emotional",
                "active",
                {"state": emotional_state.get("quadrant", "neutral")}
            )
            
    async def _check_interventions(self, result: Dict):
        """Check if interventions are needed based on results"""
        if not self.visual_preferences["preview_interventions"]:
            return
            
        actions = result.get("actions", [])
        
        for action in actions:
            if action["type"] == "crisis_intervention":
                await self._show_crisis_intervention(action)
                
            elif action["type"] == "protect_flow":
                await self._show_flow_protection(action)
                
            elif action["type"] == "emotional_support":
                await self._show_emotional_support(action)
                
    async def _show_crisis_intervention(self, action: Dict):
        """Show crisis intervention preview"""
        interventions = action.get("actions", [])
        
        if "pause_notifications" in interventions:
            await self.visual_feedback.preview_intervention(
                InterventionType.BLOCK_NOTIFICATIONS,
                "Pausing all notifications to help you focus on what matters",
                countdown=2,
                can_cancel=False,
                callback=self._pause_notifications
            )
            
        if "offer_breathing_exercise" in interventions:
            await self.visual_feedback.preview_intervention(
                InterventionType.BREATHING_EXERCISE,
                "Let's take a moment to breathe together",
                countdown=5,
                can_cancel=True,
                callback=self._start_breathing_exercise
            )
            
        self.ui_metrics["interventions_shown"] += 1
        
    async def _show_flow_protection(self, action: Dict):
        """Show flow state protection"""
        await self.visual_feedback.preview_intervention(
            InterventionType.FOCUS_MODE,
            "Protecting your flow state - minimizing all distractions",
            countdown=3,
            can_cancel=True,
            callback=self._enable_focus_mode
        )
        
        self.ui_metrics["interventions_shown"] += 1
        
    async def _show_emotional_support(self, action: Dict):
        """Show emotional support interventions"""
        support_actions = action.get("actions", [])
        
        for support in support_actions:
            if support == "suggest_breathing_exercise":
                await self.visual_feedback.preview_intervention(
                    InterventionType.BREATHING_EXERCISE,
                    "A breathing exercise might help you feel better",
                    countdown=5,
                    can_cancel=True
                )
            elif support == "increase_check_ins":
                await self.visual_feedback.show_notification(
                    "I'll check in with you more frequently",
                    "info"
                )
                
    def _get_mode_change_reason(self, new_mode: str, result: Dict) -> str:
        """Generate reason for mode change"""
        emotional_state = result.get("emotional_state", {})
        
        reasons = {
            "flow": "Deep focus detected - optimizing for productivity",
            "crisis": "I'm here to support you through this",
            "rest": "Time to recharge - encouraging relaxation",
            "normal": "Returning to balanced operation"
        }
        
        return reasons.get(new_mode, f"Switching to {new_mode} mode")
        
    async def _handle_intervention_cancelled(self, intervention_id: str):
        """Handle when user cancels an intervention"""
        await self.visual_feedback.show_notification(
            "Intervention cancelled - let me know if you change your mind",
            "info"
        )
        
    # Intervention callbacks
    async def _pause_notifications(self):
        """Pause all notifications"""
        # Implementation would pause actual notifications
        await self.visual_feedback.show_notification(
            "Notifications paused",
            "success"
        )
        
    async def _start_breathing_exercise(self):
        """Start breathing exercise"""
        # Implementation would start actual breathing guide
        await self.visual_feedback.show_notification(
            "Starting 4-7-8 breathing exercise",
            "info"
        )
        
    async def _enable_focus_mode(self):
        """Enable focus mode"""
        # Implementation would enable actual focus mode
        await self.visual_feedback.show_notification(
            "Focus mode enabled - distractions minimized",
            "success"
        )
        
    def generate_dashboard_html(self) -> str:
        """Generate complete dashboard HTML"""
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>JARVIS Visual Dashboard</title>
            {self.visual_feedback.ui_components.generate_style_sheet()}
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    overflow-x: hidden;
                }}
                
                .dashboard-container {{
                    min-height: 100vh;
                    padding: 20px;
                }}
                
                .ui-header {{
                    margin-bottom: 30px;
                }}
                
                .ui-main {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                
                .ui-column {{
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }}
                
                @media (max-width: 1024px) {{
                    .ui-main {{
                        grid-template-columns: 1fr;
                    }}
                }}
                
                .connection-status {{
                    position: fixed;
                    top: 20px;
                    left: 20px;
                    padding: 8px 16px;
                    background: rgba(0, 0, 0, 0.8);
                    border-radius: 20px;
                    font-size: 0.875rem;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }}
                
                .status-dot {{
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: #4ade80;
                }}
                
                .status-dot.disconnected {{
                    background: #f87171;
                }}
                
                #notifications {{
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    width: 320px;
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                    z-index: 1000;
                }}
            </style>
        </head>
        <body>
            <div class="dashboard-container">
                <div class="connection-status">
                    <div class="status-dot" id="connectionDot"></div>
                    <span id="connectionText">Connecting...</span>
                </div>
                
                <div id="notifications"></div>
                <div id="dashboard">
                    {self.visual_feedback.generate_full_ui()}
                </div>
            </div>
            
            <script>
                // WebSocket connection
                let ws = null;
                let reconnectTimeout = null;
                
                function connect() {{
                    ws = new WebSocket('ws://localhost:8765');
                    
                    ws.onopen = function() {{
                        console.log('Connected to JARVIS');
                        document.getElementById('connectionDot').classList.remove('disconnected');
                        document.getElementById('connectionText').textContent = 'Connected';
                    }};
                    
                    ws.onmessage = function(event) {{
                        const message = JSON.parse(event.data);
                        handleMessage(message);
                    }};
                    
                    ws.onclose = function() {{
                        console.log('Disconnected from JARVIS');
                        document.getElementById('connectionDot').classList.add('disconnected');
                        document.getElementById('connectionText').textContent = 'Disconnected';
                        
                        // Reconnect after 3 seconds
                        clearTimeout(reconnectTimeout);
                        reconnectTimeout = setTimeout(connect, 3000);
                    }};
                    
                    ws.onerror = function(error) {{
                        console.error('WebSocket error:', error);
                    }};
                }}
                
                function handleMessage(message) {{
                    switch(message.type) {{
                        case 'initial_state':
                        case 'state_update':
                            updateDashboard(message.data, message.html);
                            break;
                            
                        case 'notification':
                            showNotification(message.data);
                            break;
                            
                        case 'intervention':
                            handleIntervention(message.data);
                            break;
                            
                        case 'status_update':
                        case 'mode_change':
                            updateSection(message.data);
                            break;
                    }}
                }}
                
                function updateDashboard(data, html) {{
                    if (html) {{
                        document.getElementById('dashboard').innerHTML = html;
                    }}
                }}
                
                function showNotification(data) {{
                    const container = document.getElementById('notifications');
                    const notif = document.createElement('div');
                    notif.innerHTML = data.html;
                    container.appendChild(notif.firstElementChild);
                    
                    // Auto remove after duration
                    setTimeout(() => {{
                        notif.remove();
                    }}, data.duration || 5000);
                }}
                
                function handleIntervention(data) {{
                    if (data.action === 'show') {{
                        showNotification(data);
                        
                        // Update countdown
                        if (data.countdown) {{
                            startCountdown(data.id, data.countdown);
                        }}
                    }} else if (data.action === 'update_countdown') {{
                        updateCountdown(data.id, data.remaining);
                    }} else if (data.action === 'complete' || data.action === 'cancel') {{
                        removeIntervention(data.id);
                    }}
                }}
                
                function startCountdown(id, seconds) {{
                    // Countdown logic
                }}
                
                function updateCountdown(id, remaining) {{
                    const el = document.querySelector(`[data-action="${{id}}"] .countdown-value`);
                    if (el) el.textContent = remaining;
                }}
                
                function removeIntervention(id) {{
                    const el = document.querySelector(`[data-action="${{id}}"]`);
                    if (el) el.remove();
                }}
                
                function cancelIntervention(id) {{
                    if (ws && ws.readyState === WebSocket.OPEN) {{
                        ws.send(JSON.stringify({{
                            type: 'cancel_intervention',
                            id: id
                        }}));
                    }}
                }}
                
                function dismissToast(button) {{
                    button.closest('.notification-toast').remove();
                }}
                
                // Connect on load
                connect();
                
                // Request updates every 30 seconds
                setInterval(() => {{
                    if (ws && ws.readyState === WebSocket.OPEN) {{
                        ws.send(JSON.stringify({{ type: 'request_update' }}));
                    }}
                }}, 30000);
            </script>
        </body>
        </html>
        """
        
    async def save_dashboard(self, filepath: str = "jarvis_dashboard.html"):
        """Save dashboard HTML to file"""
        html = self.generate_dashboard_html()
        
        path = Path(filepath)
        path.write_text(html)
        
        print(f"✅ Dashboard saved to {path.absolute()}")
        return str(path.absolute())
        
    async def demonstrate_phase7(self):
        """Demonstrate Phase 7 visual capabilities"""
        print("\n🎨 JARVIS Phase 7: Visual Feedback Demo")
        print("="*70)
        
        # Save dashboard
        dashboard_path = await self.save_dashboard()
        print(f"\n📊 Open {dashboard_path} in your browser to see the dashboard")
        
        # Wait for connection
        print("\nWaiting for dashboard connection...")
        await asyncio.sleep(3)
        
        # Run visual feedback demo
        await self.visual_feedback.demonstrate_visual_feedback()
        
        # Simulate various inputs
        test_scenarios = [
            {
                "name": "Normal Operation",
                "input": {
                    "voice": {"text": "Check my schedule for today"},
                    "biometric": {"heart_rate": 72, "stress_level": 0.3}
                }
            },
            {
                "name": "Flow State Detection",
                "input": {
                    "voice": {"text": "I'm really focused on this code"},
                    "biometric": {"heart_rate": 68, "stress_level": 0.2}
                }
            },
            {
                "name": "Stress Detection",
                "input": {
                    "voice": {"text": "This deadline is killing me!"},
                    "biometric": {"heart_rate": 95, "stress_level": 0.8}
                }
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n📍 Scenario: {scenario['name']}")
            result = await self.process_input(scenario["input"])
            await asyncio.sleep(3)
            
        # Show metrics
        print("\n📊 Visual System Metrics:")
        print(f"  Updates sent: {self.ui_metrics['updates_sent']}")
        print(f"  Interventions shown: {self.ui_metrics['interventions_shown']}")
        print(f"  Notifications displayed: {self.ui_metrics['notifications_displayed']}")
        print(f"  User cancellations: {self.ui_metrics['user_cancellations']}")


# Convenience functions
async def upgrade_to_phase7(existing_jarvis=None):
    """Upgrade existing JARVIS to Phase 7"""
    phase7 = JARVISPhase7Core(existing_jarvis)
    await phase7.initialize()
    return phase7


def create_phase7_jarvis():
    """Create new JARVIS with Phase 7 capabilities"""
    return JARVISPhase7Core()
