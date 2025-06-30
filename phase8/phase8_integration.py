"""
JARVIS Phase 8: UX Enhancement Integration
=========================================
Integrates all Phase 8 components into JARVIS
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import websockets
import logging

# Import Phase 8 components
from .visual_state_indicator import VisualStateIndicatorSystem
from .intervention_preview_system import InterventionPreviewSystem, InterventionType
from .cognitive_load_reducer import CognitiveLoadReducer, CognitiveLoadLevel

# Import existing JARVIS components
import sys
sys.path.append('..')
from core.jarvis_enhanced_core import JARVISEnhancedCore
from core.fluid_state_management import FluidStateManager

logger = logging.getLogger(__name__)

class JARVISPhase8UXEnhancement:
    """Main integration class for Phase 8 UX enhancements"""
    
    def __init__(self, jarvis_core: JARVISEnhancedCore):
        self.jarvis_core = jarvis_core
        
        # Initialize Phase 8 components
        self.visual_indicators = VisualStateIndicatorSystem()
        self.intervention_preview = InterventionPreviewSystem()
        self.cognitive_load_reducer = CognitiveLoadReducer()
        
        # WebSocket clients for real-time updates
        self.websocket_clients = set()
        
        # Current system state
        self.current_ux_state = {
            'visual_state': 'flow',
            'mode': 'adaptive',
            'cognitive_load': CognitiveLoadLevel.LOW,
            'active_interventions': [],
            'monitoring_status': {}
        }
        
    async def initialize(self):
        """Initialize UX enhancement system"""
        logger.info("Initializing Phase 8 UX Enhancements...")
        
        # Hook into JARVIS state changes
        self.jarvis_core.state_manager.on_state_change = self._on_state_change
        
        # Hook into intervention system
        self._wrap_intervention_system()
        
        # Start WebSocket server for dashboard
        asyncio.create_task(self._start_websocket_server())
        
        # Start monitoring
        asyncio.create_task(self._monitor_system())
        
        logger.info("Phase 8 UX Enhancements initialized successfully!")
        
    async def _on_state_change(self, old_state: str, new_state: str, metadata: Dict[str, Any]):
        """Handle JARVIS state changes"""
        # Update visual indicator
        intensity = metadata.get('confidence', 1.0)
        await self.visual_indicators.update_state_indicator(new_state, intensity)
        
        # Determine appropriate mode
        mode = self._determine_mode(new_state, metadata)
        await self.visual_indicators.show_mode_indicator(mode)
        
        # Assess cognitive load
        user_state = metadata.get('user_state', {})
        cognitive_load = await self.cognitive_load_reducer.assess_cognitive_load(user_state)
        
        # Adapt interface
        adaptations = await self.cognitive_load_reducer.adapt_interface(cognitive_load)
        
        # Update current state
        self.current_ux_state.update({
            'visual_state': new_state,
            'mode': mode,
            'cognitive_load': cognitive_load,
            'timestamp': datetime.now().isoformat()
        })
        
        # Broadcast updates
        await self._broadcast_state_update()
        
    def _determine_mode(self, state: str, metadata: Dict[str, Any]) -> str:
        """Determine appropriate mode based on state"""
        mode_mapping = {
            'flow': 'focused',
            'crisis': 'emergency',
            'creative': 'creative',
            'rest': 'minimal',
            'social': 'supportive'
        }
        
        # Check for emergency conditions
        if metadata.get('stress_level', 0) > 0.8:
            return 'emergency'
            
        return mode_mapping.get(state, 'adaptive')
        
    def _wrap_intervention_system(self):
        """Wrap JARVIS intervention system with preview capability"""
        original_execute = self.jarvis_core.execute_intervention
        
        async def wrapped_execute(intervention: Dict[str, Any]):
            """Execute intervention with preview"""
            # Map to intervention type
            intervention_type = self._map_intervention_type(intervention)
            
            # Create preview
            preview_data = {
                'type': intervention_type.value,
                'title': intervention.get('action', 'JARVIS Action'),
                'description': intervention.get('description', ''),
                'urgency': intervention.get('priority', 0.5) / 5.0,
                'context': intervention.get('context', {}),
                'can_cancel': intervention.get('allow_cancel', True),
                'execute_function': lambda ctx: original_execute(intervention)
            }
            
            # Show preview
            preview = await self.intervention_preview.preview_intervention(preview_data)
            
            # Broadcast preview
            await self._broadcast_intervention_preview(preview)
            
            # Preview handles execution after countdown
            
        self.jarvis_core.execute_intervention = wrapped_execute
        
    def _map_intervention_type(self, intervention: Dict[str, Any]) -> InterventionType:
        """Map JARVIS intervention to preview type"""
        action_type = intervention.get('action_type', '')
        
        mapping = {
            'block_notifications': InterventionType.NOTIFICATION_BLOCK,
            'suggest_break': InterventionType.BREAK_REMINDER,
            'emergency': InterventionType.EMERGENCY_ACTION,
            'health': InterventionType.HEALTH_INTERVENTION,
            'optimize': InterventionType.PRODUCTIVITY_BOOST,
            'focus_mode': InterventionType.FOCUS_MODE
        }
        
        for key, value in mapping.items():
            if key in action_type.lower():
                return value
                
        return InterventionType.SUGGESTION
        
    async def _monitor_system(self):
        """Monitor system and update indicators"""
        while True:
            try:
                # Update monitoring status
                active_inputs = self.jarvis_core.pipeline.get_active_inputs()
                for modality in ['voice', 'biometric', 'vision', 'environment', 'temporal']:
                    is_active = modality in active_inputs
                    await self.visual_indicators.update_monitoring_status(modality, is_active)
                    
                # Update cognitive load based on pipeline activity
                pipeline_load = len(self.jarvis_core.pipeline.processing_queue) / 10.0
                current_load = self.current_ux_state['cognitive_load']
                
                # Adjust load based on pipeline
                if pipeline_load > 0.7 and current_load != CognitiveLoadLevel.HIGH:
                    await self._update_cognitive_load(CognitiveLoadLevel.HIGH)
                    
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(5)
                
    async def _update_cognitive_load(self, load_level: CognitiveLoadLevel):
        """Update cognitive load and adapt interface"""
        adaptations = await self.cognitive_load_reducer.adapt_interface(load_level)
        self.current_ux_state['cognitive_load'] = load_level
        
        # Broadcast cognitive load update
        await self._broadcast_update('cognitive_load', {
            'level': load_level.value,
            'adaptations': adaptations,
            'mode': self.cognitive_load_reducer._get_interface_mode()
        })
        
    async def _start_websocket_server(self):
        """Start WebSocket server for dashboard"""
        async def handle_client(websocket, path):
            """Handle WebSocket client connection"""
            self.websocket_clients.add(websocket)
            try:
                # Send current state
                await websocket.send(json.dumps({
                    'type': 'initial_state',
                    'data': self.current_ux_state
                }))
                
                # Keep connection alive
                await websocket.wait_closed()
            finally:
                self.websocket_clients.remove(websocket)
                
        try:
            await websockets.serve(handle_client, 'localhost', 8890)
            logger.info("WebSocket server started on ws://localhost:8890")
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            
    async def _broadcast_update(self, update_type: str, data: Any):
        """Broadcast update to all WebSocket clients"""
        if not self.websocket_clients:
            return
            
        message = json.dumps({
            'type': update_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
        
        # Send to all connected clients
        disconnected = set()
        for client in self.websocket_clients:
            try:
                await client.send(message)
            except:
                disconnected.add(client)
                
        # Remove disconnected clients
        self.websocket_clients -= disconnected
        
    async def _broadcast_state_update(self):
        """Broadcast current state update"""
        await self._broadcast_update('state', {
            'state': self.current_ux_state['visual_state'],
            'intensity': 1.0
        })
        
        await self._broadcast_update('mode', {
            'name': self.current_ux_state['mode']
        })
        
        await self._broadcast_update('monitoring', 
            self.visual_indicators.status_indicators['monitoring']
        )
        
    async def _broadcast_intervention_preview(self, preview):
        """Broadcast intervention preview"""
        await self._broadcast_update('intervention_preview', {
            'description': preview.title,
            'reason': preview.reason,
            'countdown': preview.countdown_seconds,
            'visual_hint': preview.visual_preview,
            'can_cancel': preview.can_cancel,
            'can_delay': preview.can_delay
        })
        
    def get_ux_metrics(self) -> Dict[str, Any]:
        """Get UX enhancement metrics"""
        return {
            'current_state': self.current_ux_state,
            'intervention_history': len(self.intervention_preview.intervention_history),
            'cognitive_adaptations': len(self.cognitive_load_reducer.adaptation_history),
            'connected_clients': len(self.websocket_clients),
            'visual_indicators': self.visual_indicators.get_current_display_state()
        }


async def integrate_phase8(jarvis_core: JARVISEnhancedCore):
    """Main integration function for Phase 8"""
    logger.info("Integrating Phase 8 UX Enhancements...")
    
    # Create enhancement system
    ux_enhancement = JARVISPhase8UXEnhancement(jarvis_core)
    
    # Initialize
    await ux_enhancement.initialize()
    
    # Attach to JARVIS core
    jarvis_core.ux_enhancement = ux_enhancement
    
    logger.info("Phase 8 integration complete!")
    
    return ux_enhancement


if __name__ == "__main__":
    # Demo/test mode
    async def demo():
        print("🎨 JARVIS Phase 8 UX Enhancement Demo")
        print("=" * 50)
        
        # Create mock JARVIS core
        class MockJARVIS:
            def __init__(self):
                self.state_manager = MockStateManager()
                self.pipeline = MockPipeline()
                
            async def execute_intervention(self, intervention):
                print(f"Executing: {intervention}")
                
        class MockStateManager:
            def __init__(self):
                self.on_state_change = None
                
        class MockPipeline:
            def get_active_inputs(self):
                return ['voice', 'biometric']
                
            @property
            def processing_queue(self):
                return [1, 2, 3]  # Mock queue
                
        # Create and integrate
        mock_jarvis = MockJARVIS()
        ux_enhancement = JARVISPhase8UXEnhancement(mock_jarvis)
        await ux_enhancement.initialize()
        
        print("\n✅ Phase 8 UX Enhancements Active!")
        print("\n📊 Open jarvis-ux-dashboard.html in your browser")
        print("🔗 WebSocket server running on ws://localhost:8890")
        
        # Simulate state changes
        await asyncio.sleep(2)
        
        print("\n🔄 Simulating state changes...")
        states = ['flow', 'focus', 'creative', 'rest', 'crisis']
        
        for state in states:
            print(f"\n➡️  Transitioning to: {state}")
            if ux_enhancement.jarvis_core.state_manager.on_state_change:
                await ux_enhancement.jarvis_core.state_manager.on_state_change(
                    'previous', state, {
                        'confidence': 0.9,
                        'user_state': {
                            'stress_level': 0.8 if state == 'crisis' else 0.3,
                            'focus_level': 0.9 if state == 'flow' else 0.5
                        }
                    }
                )
            await asyncio.sleep(3)
            
        print("\n\n✨ Demo complete! Check the dashboard for visual updates.")
        
    asyncio.run(demo())
