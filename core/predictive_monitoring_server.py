"""
JARVIS Phase 4: Predictive Intelligence Monitoring Server
========================================================
WebSocket server for real-time dashboard updates.
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime
from typing import Set, Dict, Any
from pathlib import Path

# Import Phase 4 components
from .predictive_jarvis_integration import PredictiveJARVIS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictiveMonitoringServer:
    """WebSocket server for predictive intelligence monitoring"""
    
    def __init__(self, jarvis: PredictiveJARVIS, port: int = 8766):
        self.jarvis = jarvis
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.update_interval = 2  # seconds
        self.running = False
    
    async def register(self, websocket):
        """Register new client"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send initial data
        await self.send_update(websocket)
    
    async def unregister(self, websocket):
        """Unregister client"""
        self.clients.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def send_update(self, websocket):
        """Send current state to a client"""
        try:
            data = await self.get_current_data()
            await websocket.send(json.dumps(data))
        except Exception as e:
            logger.error(f"Error sending update: {e}")
    
    async def broadcast_update(self):
        """Broadcast update to all clients"""
        if self.clients:
            data = await self.get_current_data()
            message = json.dumps(data)
            
            # Send to all connected clients
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )
    
    async def get_current_data(self) -> Dict[str, Any]:
        """Get current system data"""
        # Get current context
        context = None
        if self.jarvis.core.initialized:
            current_state = self.jarvis.core.state_manager.get_current_state_name()
            context = {
                'user_state': current_state,
                'active_task': self.jarvis.core.current_task,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add recent action if available
            if self.jarvis.action_buffer:
                context['recent_action'] = self.jarvis.action_buffer[-1]
        
        # Get prediction insights
        insights = self.jarvis.get_prediction_insights()
        
        # Get active predictions (simplified for demo)
        predictions = []
        
        # Format preloader stats
        stats = insights.get('preloader_stats', {})
        stats.update({
            'total_predictions': insights.get('total_contexts', 0) * 3,  # Estimate
            'patterns_learned': insights.get('unique_patterns', 0),
            'contexts_analyzed': insights.get('total_contexts', 0)
        })
        
        # Format patterns for visualization
        patterns = []
        transitions = insights.get('top_transitions', {})
        for transition, count in list(transitions.items())[:5]:
            if '→' in transition:
                from_state, to_state = transition.split('→')
                patterns.append({
                    'from': from_state,
                    'to': to_state,
                    'count': count
                })
        
        # Format state transitions
        transition_list = []
        for trans, count in transitions.items():
            if '→' in trans:
                from_state, to_state = trans.split('→')
                transition_list.append({
                    'from': from_state,
                    'to': to_state,
                    'count': count
                })
        
        return {
            'context': context,
            'predictions': predictions,
            'stats': stats,
            'patterns': patterns,
            'transitions': transition_list,
            'timestamp': datetime.now().isoformat()
        }
    
    async def handle_client(self, websocket, path):
        """Handle client connection"""
        await self.register(websocket)
        try:
            async for message in websocket:
                # Handle client messages if needed
                data = json.loads(message)
                logger.info(f"Received from client: {data}")
        finally:
            await self.unregister(websocket)
    
    async def update_loop(self):
        """Background loop for broadcasting updates"""
        while self.running:
            try:
                await self.broadcast_update()
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
            
            await asyncio.sleep(self.update_interval)
    
    async def start(self):
        """Start the monitoring server"""
        self.running = True
        
        # Start update loop
        asyncio.create_task(self.update_loop())
        
        # Start WebSocket server
        async with websockets.serve(self.handle_client, "localhost", self.port):
            logger.info(f"Predictive monitoring server started on port {self.port}")
            await asyncio.Future()  # Run forever
    
    def stop(self):
        """Stop the monitoring server"""
        self.running = False


async def run_phase4_with_monitoring():
    """Run JARVIS Phase 4 with monitoring dashboard"""
    
    print("🚀 Starting JARVIS Phase 4 with Predictive Intelligence")
    print("=" * 60)
    
    # Initialize JARVIS with predictive intelligence
    jarvis = PredictiveJARVIS()
    await jarvis.initialize()
    
    # Create monitoring server
    monitor = PredictiveMonitoringServer(jarvis)
    
    # Start monitoring server in background
    monitor_task = asyncio.create_task(monitor.start())
    
    print("\n✅ Systems online!")
    print(f"📊 Dashboard: http://localhost:{monitor.port}")
    print("🔮 Predictive Intelligence: Active")
    print("\n💡 Open jarvis-phase4-predictive-dashboard.html in your browser")
    
    # Simulate some activity
    test_scenarios = [
        {
            'scenario': 'Morning Routine',
            'inputs': [
                {'voice': "Good morning JARVIS", 'biometric': {'heart_rate': 65}},
                {'voice': "Check my emails", 'biometric': {'heart_rate': 68}},
                {'voice': "What's on my calendar today?", 'biometric': {'heart_rate': 70}}
            ]
        },
        {
            'scenario': 'Work Session',
            'inputs': [
                {'voice': "Open the quarterly report", 'biometric': {'heart_rate': 72}},
                {'voice': "I need to focus on this analysis", 'biometric': {'heart_rate': 75}},
                {'voice': "Block notifications for the next hour", 'biometric': {'heart_rate': 70}}
            ]
        },
        {
            'scenario': 'Stress Detection',
            'inputs': [
                {'voice': "This deadline is tight", 'biometric': {'heart_rate': 85, 'stress_level': 0.8}},
                {'voice': "I need help organizing this", 'biometric': {'heart_rate': 88, 'stress_level': 0.85}},
                {'voice': "Can you prioritize my tasks?", 'biometric': {'heart_rate': 82, 'stress_level': 0.75}}
            ]
        }
    ]
    
    print("\n📝 Running test scenarios...")
    
    for scenario_data in test_scenarios:
        print(f"\n🎬 Scenario: {scenario_data['scenario']}")
        
        for input_data in scenario_data['inputs']:
            # Process input
            result = await jarvis.process_input(input_data, source='demo')
            
            # Show what happened
            voice_input = input_data.get('voice', '')
            print(f"   You: {voice_input}")
            print(f"   JARVIS: State={result.get('current_state')}, "
                  f"Mode={result.get('response_mode')}")
            
            # Small delay between inputs
            await asyncio.sleep(3)
    
    print("\n📊 Final Insights:")
    insights = jarvis.get_prediction_insights()
    print(f"   Patterns learned: {insights['unique_patterns']}")
    print(f"   Contexts analyzed: {insights['total_contexts']}")
    print(f"   Preload hit rate: {insights['preloader_stats']['hit_rate']:.1%}")
    
    print("\n⏳ System running... Press Ctrl+C to stop")
    
    try:
        # Keep running
        await asyncio.Future()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        monitor.stop()
        await jarvis.shutdown()
        print("✅ Shutdown complete!")


if __name__ == "__main__":
    asyncio.run(run_phase4_with_monitoring())
