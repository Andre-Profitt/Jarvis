# JARVIS Enhanced Core - Phase 1 Integration
# Connects unified pipeline and fluid states to existing JARVIS

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

# Import the new Phase 1 components
from .unified_input_pipeline import (
    UnifiedInputPipeline, 
    InputType, 
    Priority,
    UnifiedInput
)
from .fluid_state_management import (
    FluidStateManager,
    StateVector,
    ResponseMode,
    StateType
)

# Import existing JARVIS components if available
try:
    from .minimal_jarvis import MinimalJARVIS
    from .consciousness_jarvis import ConsciousnessCore
    from .jarvis_memory import JARVISMemory
    HAS_EXISTING_JARVIS = True
except ImportError:
    HAS_EXISTING_JARVIS = False

# ============================================
# INTEGRATION LAYER
# ============================================

class JARVISEnhancedCore:
    """
    Enhanced JARVIS core that integrates Phase 1 improvements
    with existing functionality
    """
    
    def __init__(self, existing_jarvis=None):
        # Phase 1 Components
        self.input_pipeline = UnifiedInputPipeline()
        self.state_manager = FluidStateManager()
        
        # Reference to existing JARVIS (if available)
        self.legacy_jarvis = existing_jarvis
        
        # Integration state
        self.current_state = None
        self.response_mode = ResponseMode.COLLABORATIVE
        self.active_processors = {}
        
        # Metrics
        self.integration_metrics = {
            'inputs_processed': 0,
            'state_updates': 0,
            'mode_changes': 0,
            'legacy_fallbacks': 0
        }
        
        # Response handlers for each mode
        self.response_handlers = {
            ResponseMode.EMERGENCY: self._handle_emergency,
            ResponseMode.PROACTIVE: self._handle_proactive,
            ResponseMode.BACKGROUND: self._handle_background,
            ResponseMode.COLLABORATIVE: self._handle_collaborative,
            ResponseMode.SUPPORTIVE: self._handle_supportive,
            ResponseMode.PROTECTIVE: self._handle_protective
        }
        
    async def initialize(self):
        """Initialize the enhanced JARVIS system"""
        print("🚀 Initializing Enhanced JARVIS...")
        
        # Start the unified pipeline
        await self.input_pipeline.start()
        
        # Initialize existing components if available
        if self.legacy_jarvis:
            await self._initialize_legacy_components()
        elif HAS_EXISTING_JARVIS:
            # Try to initialize from available modules
            await self._auto_initialize_components()
            
        print("✅ Enhanced JARVIS Ready!")
        
    async def _initialize_legacy_components(self):
        """Initialize and connect legacy components"""
        # Connect existing processors to new pipeline
        if hasattr(self.legacy_jarvis, 'fusion_engine'):
            # Map fusion engine to new processors
            self.input_pipeline.processors[InputType.VOICE] = \
                self._wrap_legacy_processor(self.legacy_jarvis.fusion_engine.voice_processor)
            self.input_pipeline.processors[InputType.VISION] = \
                self._wrap_legacy_processor(self.legacy_jarvis.fusion_engine.vision_processor)
                
    async def _auto_initialize_components(self):
        """Auto-initialize from available JARVIS modules"""
        try:
            # Try to create minimal JARVIS instance
            self.legacy_jarvis = MinimalJARVIS()
            await self._initialize_legacy_components()
        except:
            print("  ⚠️  Could not auto-initialize legacy components")
    
    def _wrap_legacy_processor(self, legacy_processor):
        """Wrap legacy processor to work with new pipeline"""
        class LegacyWrapper:
            def __init__(self, processor):
                self.processor = processor
                
            async def process(self, data, context):
                # Convert new format to legacy format if needed
                legacy_data = self._convert_to_legacy(data)
                result = await self.processor.process(legacy_data)
                return self._convert_from_legacy(result)
                
            def _convert_to_legacy(self, data):
                # Implement conversion logic
                return data
                
            def _convert_from_legacy(self, result):
                # Implement conversion logic
                return result
                
        return LegacyWrapper(legacy_processor)
    
    # ============================================
    # MAIN PROCESSING FLOW
    # ============================================
    
    async def process_input(self, raw_input: Any, source: str = "unknown") -> Dict[str, Any]:
        """
        Main entry point for all inputs - replaces direct calls to JARVIS
        
        Example:
            # Old way:
            jarvis.process_voice(audio_data)
            
            # New way:
            enhanced_jarvis.process_input(
                {"waveform": audio_data, "sample_rate": 16000},
                source="microphone"
            )
        """
        # Track metrics
        self.integration_metrics['inputs_processed'] += 1
        
        # Add metadata
        input_with_metadata = {
            **raw_input,
            'source': source,
            'timestamp': datetime.now()
        }
        
        # Process through unified pipeline
        pipeline_result = await self.input_pipeline.process_input(
            input_with_metadata,
            {'source': source}
        )
        
        # Update state based on input
        await self._update_state_from_input(raw_input)
        
        # Determine response based on current state
        response = await self._generate_response(pipeline_result)
        
        return response
    
    async def _update_state_from_input(self, raw_input: Any):
        """Update fluid state based on input"""
        # Convert input to state manager format
        state_input = self._extract_state_features(raw_input)
        
        # Update state
        self.current_state = await self.state_manager.update_state(state_input)
        self.integration_metrics['state_updates'] += 1
        
        # Check for response mode change
        new_mode = self.state_manager.get_response_mode(self.current_state)
        if new_mode != self.response_mode:
            await self._handle_mode_change(new_mode)
            
    def _extract_state_features(self, raw_input: Any) -> Dict[str, Any]:
        """Extract features for state calculation from raw input"""
        features = {}
        
        # Extract biometric features
        if 'heart_rate' in raw_input or 'biometric' in raw_input:
            features['biometric'] = raw_input.get('biometric', raw_input)
            
        # Extract voice features
        if 'waveform' in raw_input or 'voice' in raw_input:
            features['voice'] = raw_input.get('voice', {
                'features': raw_input.get('features', {})
            })
            
        # Extract activity features
        if 'activity' in raw_input:
            features['activity'] = raw_input['activity']
            
        # Add temporal context
        features['temporal'] = {
            'hour': datetime.now().hour,
            'day_of_week': datetime.now().weekday()
        }
        
        return features
    
    async def _generate_response(self, pipeline_result: Any) -> Dict[str, Any]:
        """Generate response based on state and pipeline result"""
        response = {
            'result': pipeline_result,
            'state': self._get_state_summary(),
            'mode': self.response_mode.name,
            'recommendations': []
        }
        
        # Get mode-specific handler
        handler = self.response_handlers.get(
            self.response_mode, 
            self._handle_collaborative
        )
        
        # Apply mode-specific enhancements
        mode_response = await handler(pipeline_result)
        response.update(mode_response)
        
        return response
    
    # ============================================
    # RESPONSE MODE HANDLERS
    # ============================================
    
    async def _handle_emergency(self, pipeline_result: Any) -> Dict[str, Any]:
        """Handle emergency mode responses"""
        return {
            'priority': 'critical',
            'actions': await self._get_emergency_actions(),
            'notifications': ['emergency_contact', 'health_monitor'],
            'override_all': True
        }
    
    async def _handle_proactive(self, pipeline_result: Any) -> Dict[str, Any]:
        """Handle proactive mode responses"""
        return {
            'suggestions': await self._get_proactive_suggestions(),
            'prepare_resources': True,
            'anticipate_needs': True
        }
    
    async def _handle_background(self, pipeline_result: Any) -> Dict[str, Any]:
        """Handle background mode (flow state)"""
        return {
            'visibility': 'minimal',
            'defer_non_critical': True,
            'protect_flow': True,
            'silent_optimizations': True
        }
    
    async def _handle_collaborative(self, pipeline_result: Any) -> Dict[str, Any]:
        """Handle collaborative mode (normal)"""
        return {
            'interactive': True,
            'suggestions_available': True,
            'normal_notifications': True
        }
    
    async def _handle_supportive(self, pipeline_result: Any) -> Dict[str, Any]:
        """Handle supportive mode"""
        return {
            'emotional_support': True,
            'gentle_guidance': True,
            'stress_reduction': await self._get_stress_reduction_actions()
        }
    
    async def _handle_protective(self, pipeline_result: Any) -> Dict[str, Any]:
        """Handle protective mode"""
        return {
            'protective_measures': True,
            'limit_stressors': True,
            'health_monitoring': 'active',
            'recovery_suggestions': await self._get_recovery_suggestions()
        }
    
    # ============================================
    # MODE CHANGE HANDLING
    # ============================================
    
    async def _handle_mode_change(self, new_mode: ResponseMode):
        """Handle response mode changes"""
        old_mode = self.response_mode
        self.response_mode = new_mode
        self.integration_metrics['mode_changes'] += 1
        
        print(f"\n🔄 Response Mode Change: {old_mode.name} → {new_mode.name}")
        
        # Mode-specific actions
        if new_mode == ResponseMode.BACKGROUND:
            print("  → Entering minimal intervention mode (Flow State Protected)")
            await self._enable_do_not_disturb()
            
        elif new_mode == ResponseMode.EMERGENCY:
            print("  → ⚠️  Emergency mode activated!")
            await self._trigger_emergency_protocol()
            
        elif old_mode == ResponseMode.BACKGROUND and new_mode != ResponseMode.BACKGROUND:
            print("  → Exiting flow state")
            await self._disable_do_not_disturb()
    
    # ============================================
    # HELPER METHODS
    # ============================================
    
    def _get_state_summary(self) -> Dict[str, float]:
        """Get current state summary"""
        if not self.current_state:
            return {}
            
        return {
            state_type.name.lower(): value 
            for state_type, value in self.current_state.values.items()
        }
    
    async def _get_emergency_actions(self) -> list:
        """Get emergency actions based on state"""
        actions = []
        
        if self.current_state.values[StateType.STRESS] > 0.9:
            actions.extend([
                "pause_all_notifications",
                "activate_calming_protocol",
                "alert_emergency_contact",
                "initiate_breathing_guide"
            ])
            
        if self.current_state.values[StateType.HEALTH] < 0.2:
            actions.extend([
                "check_vital_signs",
                "suggest_medical_attention",
                "log_health_event"
            ])
            
        return actions
    
    async def _get_proactive_suggestions(self) -> list:
        """Get proactive suggestions based on state"""
        suggestions = []
        
        if self.current_state.values[StateType.ENERGY] < 0.3:
            suggestions.append("Take a 15-minute break - your energy is low")
            
        if self.current_state.values[StateType.FOCUS] < 0.4:
            suggestions.append("Try the Pomodoro technique to regain focus")
            
        if self.current_state.values[StateType.MOOD] < 0.4:
            suggestions.append("Play some uplifting music or take a walk")
            
        if self.current_state.values[StateType.PRODUCTIVITY] < 0.3:
            suggestions.append("Consider reorganizing your task list")
            
        return suggestions
    
    async def _get_stress_reduction_actions(self) -> list:
        """Get stress reduction recommendations"""
        return [
            "guided_breathing_exercise",
            "calming_music_playlist",
            "notification_reduction",
            "break_reminder"
        ]
    
    async def _get_recovery_suggestions(self) -> list:
        """Get recovery suggestions for protective mode"""
        return [
            "hydration_reminder",
            "posture_check",
            "eye_rest_timer",
            "micro_break_schedule"
        ]
    
    async def _enable_do_not_disturb(self):
        """Enable do not disturb mode"""
        print("    ✓ Do Not Disturb enabled")
        # Implement DND logic
        pass
    
    async def _disable_do_not_disturb(self):
        """Disable do not disturb mode"""
        print("    ✓ Do Not Disturb disabled")
        # Implement logic
        pass
    
    async def _trigger_emergency_protocol(self):
        """Trigger emergency protocol"""
        print("    ✓ Emergency protocol initiated")
        # Implement emergency logic
        pass
    
    # ============================================
    # CONVENIENCE METHODS
    # ============================================
    
    async def get_current_status(self) -> Dict[str, Any]:
        """Get comprehensive current status"""
        return {
            'state': self._get_state_summary() if self.current_state else {},
            'mode': self.response_mode.name,
            'trends': self.state_manager.get_state_trends() if self.current_state else {},
            'pipeline_metrics': self.input_pipeline.get_metrics(),
            'integration_metrics': self.integration_metrics
        }
    
    async def predict_future_state(self, minutes: int = 30) -> Dict[str, Any]:
        """Predict future state"""
        prediction = self.state_manager.get_state_prediction(minutes)
        if prediction:
            return {
                'predicted_state': {
                    state_type.name.lower(): value 
                    for state_type, value in prediction.values.items()
                },
                'confidence': prediction.confidence,
                'timestamp': prediction.timestamp.isoformat()
            }
        return {}

# ============================================
# MIGRATION HELPERS
# ============================================

class JARVISMigration:
    """Helper class for migrating existing JARVIS code"""
    
    @staticmethod
    def create_migration_wrapper(enhanced_jarvis):
        """Create a wrapper that maintains backward compatibility"""
        
        class MigrationWrapper:
            def __init__(self, enhanced):
                self.enhanced = enhanced
                
            # Wrap old method signatures
            async def process_voice(self, audio_data, sample_rate=16000):
                """Legacy voice processing method"""
                return await self.enhanced.process_input({
                    'waveform': audio_data,
                    'sample_rate': sample_rate
                }, source='voice_legacy')
                
            async def process_biometric(self, heart_rate, **kwargs):
                """Legacy biometric processing"""
                biometric_data = {'heart_rate': heart_rate, **kwargs}
                return await self.enhanced.process_input({
                    'biometric': biometric_data
                }, source='biometric_legacy')
                
            async def analyze_fusion(self, multi_modal_input):
                """Legacy fusion analysis"""
                return await self.enhanced.process_input(
                    multi_modal_input,
                    source='fusion_legacy'
                )
                
            # Add more legacy method wrappers as needed
            
        return MigrationWrapper(enhanced_jarvis)

# Export main components
__all__ = [
    'JARVISEnhancedCore',
    'JARVISMigration'
]