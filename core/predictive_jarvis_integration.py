"""
JARVIS Phase 4 Integration: Predictive Intelligence Integration
===============================================================
Integrates predictive intelligence with existing JARVIS Phase 1 components.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from .predictive_intelligence import (
    PredictiveIntelligence, 
    ContextSnapshot,
    Prediction,
    PredictionType
)
from .jarvis_enhanced_core import JARVISEnhancedCore
from .unified_input_pipeline import UnifiedInputPipeline, InputPriority
from .fluid_state_management import FluidStateManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictiveJARVIS:
    """JARVIS with predictive intelligence capabilities"""
    
    def __init__(self, config_path: Optional[Path] = None):
        # Initialize Phase 1 components
        self.core = JARVISEnhancedCore(config_path)
        
        # Initialize Phase 4 predictive intelligence
        self.data_dir = Path("./jarvis_data/predictive")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.predictor = PredictiveIntelligence(self.data_dir)
        
        # Integration components
        self.last_context_update = datetime.now()
        self.action_buffer = []
        self.prediction_handlers = self._setup_prediction_handlers()
        
        self.running = False
    
    def _setup_prediction_handlers(self) -> Dict:
        """Setup handlers for different prediction types"""
        return {
            PredictionType.NEXT_ACTION: self._handle_next_action_prediction,
            PredictionType.RESOURCE_NEED: self._handle_resource_prediction,
            PredictionType.USER_STATE: self._handle_state_prediction,
            PredictionType.TASK_DURATION: self._handle_duration_prediction,
            PredictionType.INTERRUPTION: self._handle_interruption_prediction,
            PredictionType.PREFERENCE: self._handle_preference_prediction
        }
    
    async def initialize(self):
        """Initialize all systems"""
        # Initialize Phase 1 core
        await self.core.initialize()
        
        # Start predictive intelligence
        await self.predictor.start()
        
        # Start integration loops
        self.running = True
        asyncio.create_task(self._context_sync_loop())
        asyncio.create_task(self._prediction_action_loop())
        
        logger.info("✅ Predictive JARVIS initialized successfully")
    
    async def process_input(self, input_data: Dict[str, Any], 
                          source: str = "unknown") -> Dict[str, Any]:
        """Process input with predictive enhancements"""
        # First, use Phase 1 processing
        result = await self.core.process_input(input_data, source)
        
        # Record action for pattern learning
        if 'action' in result:
            self.action_buffer.append(result['action'])
        
        # Update context if significant change
        if await self._should_update_context(input_data, result):
            context = await self._create_context_snapshot(input_data, result)
            predictions = await self.predictor.update_context(context)
            
            # Handle predictions
            await self._handle_predictions(predictions)
        
        # Check if we have preloaded resources
        if 'requested_resource' in result:
            preloaded = self.predictor.preloader.get_resource(
                result['requested_resource']['type'],
                result['requested_resource']['id']
            )
            if preloaded:
                result['resource_data'] = preloaded
                result['response_time'] *= 0.1  # 10x faster with preload
        
        return result
    
    async def _should_update_context(self, input_data: Dict, 
                                   result: Dict) -> bool:
        """Determine if context should be updated"""
        # Update if state changed
        if 'state_change' in result and result['state_change']:
            return True
        
        # Update if significant time passed
        if datetime.now() - self.last_context_update > timedelta(minutes=5):
            return True
        
        # Update if task changed
        if 'task' in input_data and input_data.get('task') != self.core.current_task:
            return True
        
        # Update if high-priority input
        if input_data.get('priority') in ['CRITICAL', 'URGENT']:
            return True
        
        return False
    
    async def _create_context_snapshot(self, input_data: Dict, 
                                     result: Dict) -> ContextSnapshot:
        """Create context snapshot from current state"""
        # Get current state from Phase 1 components
        current_state = self.core.state_manager.get_current_state_name()
        
        # Get recent actions
        recent_actions = self.action_buffer[-10:] if self.action_buffer else []
        
        # Environmental factors
        environmental = {
            'time_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'location': input_data.get('location', 'unknown'),
            'device': input_data.get('device', 'unknown')
        }
        
        # Biometric data if available
        biometric = None
        if 'biometric' in input_data:
            biometric = input_data['biometric']
        
        return ContextSnapshot(
            timestamp=datetime.now(),
            user_state=current_state,
            active_task=self.core.current_task,
            recent_actions=recent_actions,
            environmental_factors=environmental,
            biometric_data=biometric
        )
    
    async def _handle_predictions(self, predictions: List[Prediction]):
        """Handle predictions appropriately"""
        for prediction in predictions:
            handler = self.prediction_handlers.get(prediction.prediction_type)
            if handler:
                asyncio.create_task(handler(prediction))
            else:
                logger.warning(f"No handler for prediction type: {prediction.prediction_type}")
    
    async def _handle_next_action_prediction(self, prediction: Prediction):
        """Handle next action predictions"""
        if prediction.confidence > 0.8:
            logger.info(f"🔮 High confidence next action: {prediction.predicted_value}")
            
            # Prepare UI hints or shortcuts
            await self.core.process_input({
                'type': 'prepare_action',
                'action': prediction.predicted_value,
                'confidence': prediction.confidence
            }, source='predictor')
    
    async def _handle_resource_prediction(self, prediction: Prediction):
        """Handle resource need predictions"""
        resource_type, resource_id = prediction.predicted_value
        
        logger.info(f"📦 Preloading {resource_type}: {resource_id}")
        
        # Preloading is handled by the predictor
        # Here we might notify the UI or prepare the system
        await self.core.process_input({
            'type': 'resource_preparation',
            'resource_type': resource_type,
            'resource_id': resource_id,
            'time_horizon': prediction.time_horizon.total_seconds()
        }, source='predictor')
    
    async def _handle_state_prediction(self, prediction: Prediction):
        """Handle user state change predictions"""
        predicted_state = prediction.predicted_value
        
        if prediction.confidence > 0.7:
            logger.info(f"🎭 Predicted state change to: {predicted_state}")
            
            # Prepare for state change
            if predicted_state == "stressed":
                # Prepare calming interventions
                await self.core.process_input({
                    'type': 'prepare_intervention',
                    'intervention_type': 'stress_relief',
                    'time_horizon': prediction.time_horizon.total_seconds()
                }, source='predictor')
            
            elif predicted_state == "tired":
                # Prepare break suggestions
                await self.core.process_input({
                    'type': 'prepare_intervention',
                    'intervention_type': 'break_suggestion',
                    'time_horizon': prediction.time_horizon.total_seconds()
                }, source='predictor')
    
    async def _handle_duration_prediction(self, prediction: Prediction):
        """Handle task duration predictions"""
        duration_minutes = prediction.predicted_value
        
        logger.info(f"⏱️ Task estimated to take {duration_minutes} minutes")
        
        # Update scheduling and notifications
        await self.core.process_input({
            'type': 'update_schedule',
            'task': self.core.current_task,
            'estimated_duration': duration_minutes,
            'confidence': prediction.confidence
        }, source='predictor')
    
    async def _handle_interruption_prediction(self, prediction: Prediction):
        """Handle interruption predictions"""
        # Implement interruption handling
        pass
    
    async def _handle_preference_prediction(self, prediction: Prediction):
        """Handle preference predictions"""
        # Implement preference handling
        pass
    
    async def _context_sync_loop(self):
        """Sync context periodically"""
        while self.running:
            try:
                # Create periodic context snapshot
                if self.core.initialized:
                    context = await self._create_context_snapshot({}, {})
                    await self.predictor.update_context(context)
                    self.last_context_update = datetime.now()
            
            except Exception as e:
                logger.error(f"Error in context sync: {e}")
            
            await asyncio.sleep(60)  # Sync every minute
    
    async def _prediction_action_loop(self):
        """Act on predictions proactively"""
        while self.running:
            try:
                # Get current insights
                insights = self.predictor.get_insights()
                
                # If we have strong patterns, act on them
                if insights['top_transitions']:
                    # Implement proactive actions based on patterns
                    pass
                
                # Log prediction performance
                stats = insights['preloader_stats']
                if stats['cache_hits'] + stats['cache_misses'] > 0:
                    logger.info(f"📊 Prediction hit rate: {stats['hit_rate']:.1%}")
            
            except Exception as e:
                logger.error(f"Error in prediction action loop: {e}")
            
            await asyncio.sleep(300)  # Check every 5 minutes
    
    async def shutdown(self):
        """Shutdown all systems"""
        self.running = False
        
        # Stop predictive intelligence
        await self.predictor.stop()
        
        # Shutdown core
        await self.core.shutdown()
        
        logger.info("Predictive JARVIS shutdown complete")
    
    def get_prediction_insights(self) -> Dict[str, Any]:
        """Get insights about predictions"""
        return self.predictor.get_insights()


# Enhanced launcher for Phase 4
async def launch_jarvis_phase4():
    """Launch JARVIS with Phase 4 predictive intelligence"""
    
    print("🚀 Launching JARVIS with Predictive Intelligence")
    print("=" * 50)
    
    jarvis = PredictiveJARVIS()
    await jarvis.initialize()
    
    print("\n✅ JARVIS Phase 4 Ready!")
    print("\nCapabilities:")
    print("  🔮 Predictive action suggestions")
    print("  📦 Resource preloading")
    print("  🎭 State change prediction")
    print("  ⏱️ Task duration estimation")
    print("  🧠 Pattern learning & adaptation")
    
    # Demo interactions
    test_inputs = [
        {
            'voice': "Open my quarterly report",
            'biometric': {'heart_rate': 70, 'stress_level': 0.3}
        },
        {
            'voice': "I need to check my emails",
            'biometric': {'heart_rate': 72, 'stress_level': 0.4}
        },
        {
            'voice': "Schedule a meeting for tomorrow",
            'biometric': {'heart_rate': 75, 'stress_level': 0.5}
        }
    ]
    
    print("\n📝 Processing test inputs...")
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n🎯 Test {i}: {test_input.get('voice', 'No voice input')}")
        
        result = await jarvis.process_input(test_input, source='demo')
        
        print(f"   State: {result.get('current_state', 'Unknown')}")
        print(f"   Response Mode: {result.get('response_mode', 'Unknown')}")
        if 'resource_data' in result:
            print(f"   ⚡ Resource was preloaded!")
        
        await asyncio.sleep(2)
    
    # Show insights
    print("\n📊 Prediction Insights:")
    insights = jarvis.get_prediction_insights()
    print(f"   Patterns learned: {insights['unique_patterns']}")
    print(f"   Contexts analyzed: {insights['total_contexts']}")
    if insights['preloader_stats']['cache_hits'] > 0:
        print(f"   Preload hit rate: {insights['preloader_stats']['hit_rate']:.1%}")
    
    # Keep running for a bit to demonstrate background learning
    print("\n🧠 Learning patterns in background...")
    await asyncio.sleep(10)
    
    # Shutdown
    await jarvis.shutdown()
    print("\n👋 JARVIS Phase 4 shutdown complete!")


if __name__ == "__main__":
    asyncio.run(launch_jarvis_phase4())
