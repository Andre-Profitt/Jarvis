#!/usr/bin/env python3
"""
JARVIS Phase 1 Monitoring Server
WebSocket server for real-time dashboard updates
"""

import asyncio
import websockets
import json
import random
from datetime import datetime
from typing import Set, Dict, Any

# Import Phase 1 components
from core.jarvis_enhanced_core import JARVISEnhancedCore
from core.fluid_state_management import StateType, ResponseMode

class JARVISMonitoringServer:
    """WebSocket server for monitoring dashboard"""
    
    def __init__(self, jarvis_core: JARVISEnhancedCore = None):
        self.jarvis_core = jarvis_core
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.update_interval = 1.0  # seconds
        self.server = None
        
    async def register(self, websocket):
        """Register a new client"""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send initial state
        await self.send_current_state(websocket)
        
    async def unregister(self, websocket):
        """Unregister a client"""
        self.clients.remove(websocket)
        print(f"Client disconnected. Total clients: {len(self.clients)}")
        
    async def send_current_state(self, websocket):
        """Send current state to a specific client"""
        if self.jarvis_core and self.jarvis_core.current_state:
            data = {
                'states': {
                    state_type.name.lower(): value
                    for state_type, value in self.jarvis_core.current_state.values.items()
                },
                'mode': self.jarvis_core.response_mode.name,
                'modeDescription': self._get_mode_description(self.jarvis_core.response_mode),
                'metrics': {
                    'latency': self.jarvis_core.input_pipeline.metrics.get('avg_latency', 0),
                    'processed': self.jarvis_core.input_pipeline.metrics.get('total_processed', 0),
                    'queueSize': self.jarvis_core.input_pipeline.metrics.get('queue_size', 0),
                    'modeChanges': self.jarvis_core.integration_metrics.get('mode_changes', 0)
                }
            }
        else:
            # Send simulated data if no JARVIS core
            data = self._generate_simulated_data()
            
        await websocket.send(json.dumps(data))
        
    async def broadcast_update(self, data: Dict[str, Any]):
        """Broadcast update to all connected clients"""
        if self.clients:
            message = json.dumps(data)
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )
            
    async def handle_client(self, websocket, path):
        """Handle a client connection"""
        await self.register(websocket)
        try:
            # Keep connection alive
            async for message in websocket:
                # Handle any client messages if needed
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)
            
    async def update_loop(self):
        """Continuously send updates to all clients"""
        while True:
            await asyncio.sleep(self.update_interval)
            
            # Get current data
            if self.jarvis_core and self.jarvis_core.current_state:
                data = {
                    'states': {
                        state_type.name.lower(): value
                        for state_type, value in self.jarvis_core.current_state.values.items()
                    },
                    'mode': self.jarvis_core.response_mode.name,
                    'modeDescription': self._get_mode_description(self.jarvis_core.response_mode),
                    'metrics': {
                        'latency': self.jarvis_core.input_pipeline.metrics.get('avg_latency', 0),
                        'processed': self.jarvis_core.input_pipeline.metrics.get('total_processed', 0),
                        'queueSize': self.jarvis_core.input_pipeline.metrics.get('queue_size', 0),
                        'modeChanges': self.jarvis_core.integration_metrics.get('mode_changes', 0)
                    }
                }
            else:
                # Use simulated data
                data = self._generate_simulated_data()
                
            # Broadcast to all clients
            await self.broadcast_update(data)
            
    def _get_mode_description(self, mode: ResponseMode) -> str:
        """Get description for response mode"""
        descriptions = {
            ResponseMode.EMERGENCY: "Critical intervention required",
            ResponseMode.PROACTIVE: "Helpful suggestions active",
            ResponseMode.BACKGROUND: "Flow state - minimal intervention",
            ResponseMode.COLLABORATIVE: "Normal interaction mode",
            ResponseMode.SUPPORTIVE: "Emotional support mode",
            ResponseMode.PROTECTIVE: "Health protection active"
        }
        return descriptions.get(mode, "Unknown mode")
        
    def _generate_simulated_data(self) -> Dict[str, Any]:
        """Generate simulated data for demo purposes"""
        # Simulate states with some continuity
        if not hasattr(self, '_sim_states'):
            self._sim_states = {
                'stress': 0.3,
                'focus': 0.7,
                'energy': 0.6,
                'mood': 0.7,
                'creativity': 0.5,
                'productivity': 0.6,
                'social': 0.5,
                'health': 0.7
            }
            
        # Update states with small changes
        for state in self._sim_states:
            change = (random.random() - 0.5) * 0.05
            self._sim_states[state] = max(0, min(1, self._sim_states[state] + change))
            
        # Determine mode based on states
        if self._sim_states['stress'] > 0.8:
            mode = 'EMERGENCY'
        elif self._sim_states['focus'] > 0.85 and self._sim_states['stress'] < 0.3:
            mode = 'BACKGROUND'
        elif self._sim_states['energy'] < 0.3:
            mode = 'SUPPORTIVE'
        else:
            mode = 'COLLABORATIVE'
            
        return {
            'states': self._sim_states.copy(),
            'mode': mode,
            'modeDescription': self._get_mode_description(ResponseMode[mode]),
            'metrics': {
                'latency': random.uniform(0.01, 0.05),
                'processed': random.randint(100, 1000),
                'queueSize': random.randint(0, 10),
                'modeChanges': random.randint(0, 20)
            }
        }
        
    async def send_input_event(self, input_type: str, priority: str):
        """Send input pipeline event"""
        data = {
            'input': {
                'type': input_type,
                'priority': priority,
                'timestamp': datetime.now().isoformat()
            }
        }
        await self.broadcast_update(data)
        
    async def start(self, host='localhost', port=8765):
        """Start the WebSocket server"""
        print(f"Starting monitoring server on ws://{host}:{port}")
        
        # Start update loop
        asyncio.create_task(self.update_loop())
        
        # Start WebSocket server
        self.server = await websockets.serve(
            self.handle_client,
            host,
            port
        )
        
        print(f"✅ Monitoring server ready at ws://{host}:{port}")
        print(f"   Open jarvis-phase1-monitor.html to view the dashboard")
        
    async def stop(self):
        """Stop the server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()

# ============================================
# STANDALONE SERVER
# ============================================

async def run_standalone_server():
    """Run monitoring server in standalone mode"""
    print("🖥️  JARVIS Phase 1 Monitoring Server")
    print("="*40)
    
    # Create server without JARVIS core (will use simulated data)
    server = JARVISMonitoringServer()
    
    try:
        await server.start()
        
        # Keep server running
        await asyncio.Future()  # Run forever
        
    except KeyboardInterrupt:
        print("\n\nShutting down monitoring server...")
        await server.stop()

if __name__ == "__main__":
    asyncio.run(run_standalone_server())