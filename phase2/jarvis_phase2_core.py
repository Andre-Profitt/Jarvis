#!/usr/bin/env python3
"""
JARVIS Phase 2: Enhanced Core Integration
Integrates Context Persistence, Predictive Pre-loading, Temporal Processing, and Vision
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import json
from pathlib import Path

# Import Phase 1 components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.unified_input_pipeline import UnifiedInputPipeline, InputPacket
from core.fluid_state_management import FluidStateManager
from core.jarvis_enhanced_core import JARVISEnhancedCore

# Import Phase 2 components
from context_persistence import get_context_persistence, ContextPersistenceSystem
from predictive_preloading import get_predictive_system, PredictivePreloadingSystem
from temporal_processing import get_temporal_system, TemporalProcessingSystem
from vision_processing import get_vision_system, VisionProcessingSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Phase2Config:
    """Configuration for Phase 2 features"""
    enable_context_persistence: bool = True
    enable_predictive_preload: bool = True
    enable_temporal_processing: bool = True
    enable_vision_processing: bool = True
    
    # Context settings
    context_memory_size: int = 1000
    context_relevance_threshold: float = 0.6
    
    # Prediction settings
    prediction_confidence_threshold: float = 0.5
    max_preload_items: int = 5
    
    # Temporal settings
    temporal_pattern_min_confidence: float = 0.7
    temporal_anomaly_threshold: float = 3.0
    
    # Vision settings
    vision_capture_interval: float = 2.0  # seconds
    vision_roi_enabled: bool = True

class JARVISPhase2Core:
    """Enhanced JARVIS core with Phase 2 intelligent processing"""
    
    def __init__(self, config: Optional[Phase2Config] = None):
        self.config = config or Phase2Config()
        
        # Phase 1 components (already initialized)
        self.enhanced_core = None
        
        # Phase 2 components
        self.context_system = None
        self.predictive_system = None
        self.temporal_system = None
        self.vision_system = None
        
        # Integration state
        self.active_context = {}
        self.predicted_actions = []
        self.temporal_patterns = []
        self.visual_context = None
        
        # Background tasks
        self.background_tasks = []
        
    async def initialize(self):
        """Initialize all Phase 2 components"""
        logger.info("Initializing JARVIS Phase 2...")
        
        # Initialize Phase 1 core
        self.enhanced_core = JARVISEnhancedCore()
        await self.enhanced_core.initialize()
        
        # Initialize Phase 2 components
        if self.config.enable_context_persistence:
            self.context_system = await get_context_persistence()
            logger.info("✓ Context Persistence initialized")
        
        if self.config.enable_predictive_preload:
            self.predictive_system = await get_predictive_system()
            logger.info("✓ Predictive Pre-loading initialized")
        
        if self.config.enable_temporal_processing:
            self.temporal_system = await get_temporal_system()
            logger.info("✓ Temporal Processing initialized")
        
        if self.config.enable_vision_processing:
            self.vision_system = await get_vision_system()
            logger.info("✓ Vision Processing initialized")
        
        # Start background processes
        await self._start_background_processes()
        
        logger.info("JARVIS Phase 2 initialization complete!")
    
    async def process_with_intelligence(self, input_data: Dict[str, Any], 
                                      source: str = "user") -> Dict[str, Any]:
        """Process input with full Phase 2 intelligence"""
        start_time = datetime.now()
        
        # Record input for context
        if self.context_system:
            await self.context_system.add_context(
                context_type='input',
                content=input_data,
                source=source
            )
        
        # Get relevant context
        context = await self._gather_intelligent_context(input_data)
        
        # Predict likely actions
        predictions = await self._predict_next_actions(input_data, context)
        
        # Pre-load predicted resources
        if predictions and self.predictive_system:
            preloaded = await self.predictive_system.pre_load_resources(predictions[:3])
            context['preloaded_resources'] = preloaded
        
        # Process with Phase 1 pipeline (enhanced with context)
        enhanced_input = {
            **input_data,
            'context': context,
            'predictions': [p.__dict__ for p in predictions[:3]]
        }
        
        result = await self.enhanced_core.process_input(enhanced_input, source)
        
        # Record result for learning
        if self.context_system:
            await self.context_system.add_context(
                context_type='response',
                content=result,
                source='jarvis',
                metadata={'processing_time': (datetime.now() - start_time).total_seconds()}
            )
        
        # Update temporal patterns
        if self.temporal_system:
            await self.temporal_system.add_temporal_event(
                event_type='interaction',
                value={
                    'input': input_data.get('type', 'unknown'),
                    'response': result.get('type', 'unknown')
                },
                duration=datetime.now() - start_time
            )
        
        # Track action for prediction learning
        if self.predictive_system and 'action' in result:
            await self.predictive_system.record_action(
                action_type=result['action'].get('type', 'unknown'),
                target=result['action'].get('target', ''),
                context=context
            )
        
        return {
            'result': result,
            'intelligence': {
                'context_used': len(context),
                'predictions_made': len(predictions),
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'confidence': self._calculate_confidence(result, context, predictions)
            }
        }
    
    async def _gather_intelligent_context(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Gather intelligent context from all systems"""
        context = {}
        
        # Get persistent context
        if self.context_system:
            query = str(input_data.get('query', '')) or str(input_data)
            relevant_contexts = await self.context_system.get_relevant_context(
                query=query,
                time_window=timedelta(hours=24)
            )
            
            context['history'] = [
                {
                    'type': ctx.type,
                    'content': ctx.content,
                    'timestamp': ctx.timestamp.isoformat(),
                    'relevance': ctx.confidence
                }
                for ctx in relevant_contexts[:5]
            ]
            
            # Get user preferences
            context['preferences'] = await self.context_system.get_user_preferences()
        
        # Get temporal context
        if self.temporal_system:
            temporal_context = await self.temporal_system.get_temporal_context()
            context['temporal'] = temporal_context
            
            # Check for routines
            routines = await self.temporal_system.detect_routines(lookback_days=7)
            if routines:
                context['routines'] = routines[:3]
        
        # Get visual context
        if self.vision_system and self.visual_context:
            context['visual'] = {
                'active_window': self.visual_context.active_window,
                'activity': self.visual_context.user_activity,
                'focus_area': self.visual_context.focus_area,
                'element_count': len(self.visual_context.visible_elements)
            }
            
            # Check for visual patterns
            ui_patterns = await self.vision_system.detect_ui_patterns()
            if ui_patterns:
                context['ui_patterns'] = [p.pattern_id for p in ui_patterns[:3]]
        
        # Get state context from Phase 1
        state_context = await self.enhanced_core.state_manager.get_state_context()
        context['state'] = state_context
        
        return context
    
    async def _predict_next_actions(self, input_data: Dict[str, Any], 
                                  context: Dict[str, Any]) -> List:
        """Predict likely next actions"""
        if not self.predictive_system:
            return []
        
        # Build prediction context
        pred_context = {
            'current_input': input_data,
            'state': context.get('state', {}),
            'visual': context.get('visual', {}),
            'temporal': context.get('temporal', {})
        }
        
        # Get predictions
        predictions = await self.predictive_system.predict_next_actions(
            current_context=pred_context,
            top_k=self.config.max_preload_items
        )
        
        # Filter by confidence
        confident_predictions = [
            p for p in predictions 
            if p.probability >= self.config.prediction_confidence_threshold
        ]
        
        return confident_predictions
    
    async def _start_background_processes(self):
        """Start background intelligence processes"""
        # Context relevance updater
        if self.context_system:
            task = asyncio.create_task(self._update_context_relevance())
            self.background_tasks.append(task)
        
        # Visual context monitor
        if self.vision_system:
            task = asyncio.create_task(self._monitor_visual_context())
            self.background_tasks.append(task)
        
        # Temporal pattern detector
        if self.temporal_system:
            task = asyncio.create_task(self._detect_temporal_patterns())
            self.background_tasks.append(task)
        
        # Predictive model trainer
        if self.predictive_system:
            task = asyncio.create_task(self._train_predictive_models())
            self.background_tasks.append(task)
    
    async def _update_context_relevance(self):
        """Periodically update context relevance"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self.context_system.update_context_relevance()
                logger.debug("Updated context relevance")
            except Exception as e:
                logger.error(f"Context relevance update failed: {e}")
    
    async def _monitor_visual_context(self):
        """Monitor visual context continuously"""
        while True:
            try:
                await asyncio.sleep(self.config.vision_capture_interval)
                
                # Capture screen context
                self.visual_context = await self.vision_system.capture_screen_context()
                
                # Track visual workflow
                workflow_id = await self.vision_system.track_visual_workflow(
                    self.visual_context
                )
                
                if workflow_id:
                    # Record workflow in context
                    await self.context_system.add_context(
                        context_type='workflow',
                        content=workflow_id,
                        source='vision',
                        metadata={'window': self.visual_context.active_window}
                    )
                
            except Exception as e:
                logger.error(f"Visual monitoring failed: {e}")
    
    async def _detect_temporal_patterns(self):
        """Detect temporal patterns periodically"""
        while True:
            try:
                await asyncio.sleep(600)  # Every 10 minutes
                
                # Detect routines
                routines = await self.temporal_system.detect_routines(lookback_days=7)
                
                if routines:
                    # Store significant routines
                    for routine in routines[:5]:
                        if routine['confidence'] > self.config.temporal_pattern_min_confidence:
                            await self.context_system.add_context(
                                context_type='pattern',
                                content=routine,
                                source='temporal',
                                confidence=routine['confidence']
                            )
                
                logger.debug(f"Detected {len(routines)} temporal patterns")
                
            except Exception as e:
                logger.error(f"Temporal pattern detection failed: {e}")
    
    async def _train_predictive_models(self):
        """Train predictive models periodically"""
        while True:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes
                
                # Let predictive system train itself
                await self.predictive_system._train_prediction_model()
                
                logger.debug("Predictive models updated")
                
            except Exception as e:
                logger.error(f"Predictive model training failed: {e}")
    
    def _calculate_confidence(self, result: Dict[str, Any], 
                            context: Dict[str, Any], 
                            predictions: List) -> float:
        """Calculate overall confidence in response"""
        confidence_factors = []
        
        # Base confidence from result
        if 'confidence' in result:
            confidence_factors.append(result['confidence'])
        
        # Context relevance
        if context.get('history'):
            avg_relevance = sum(h['relevance'] for h in context['history']) / len(context['history'])
            confidence_factors.append(avg_relevance)
        
        # Prediction accuracy
        if predictions:
            avg_prediction_conf = sum(p.probability for p in predictions[:3]) / min(3, len(predictions))
            confidence_factors.append(avg_prediction_conf)
        
        # State stability
        if 'state' in context and 'confidence' in context['state']:
            confidence_factors.append(context['state']['confidence'])
        
        # Calculate weighted average
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        
        return 0.5  # Default confidence
    
    async def get_intelligence_summary(self) -> Dict[str, Any]:
        """Get summary of current intelligence state"""
        summary = {
            'phase': 2,
            'components': {
                'context_persistence': self.context_system is not None,
                'predictive_preload': self.predictive_system is not None,
                'temporal_processing': self.temporal_system is not None,
                'vision_processing': self.vision_system is not None
            },
            'statistics': {}
        }
        
        # Context statistics
        if self.context_system:
            summary['statistics']['context'] = {
                'working_memory_size': len(self.context_system.memory.working),
                'short_term_size': len(self.context_system.memory.short_term),
                'long_term_size': len(self.context_system.memory.long_term),
                'patterns_detected': len(self.context_system.context_patterns)
            }
        
        # Prediction statistics
        if self.predictive_system:
            summary['statistics']['predictions'] = {
                'actions_recorded': len(self.predictive_system.action_history),
                'patterns_learned': len(self.predictive_system.user_patterns),
                'model_trained': self.predictive_system.model_trained,
                'cache_size_mb': self.predictive_system.current_cache_size_mb
            }
        
        # Temporal statistics
        if self.temporal_system:
            event_count = sum(len(events) for events in self.temporal_system.events.values())
            summary['statistics']['temporal'] = {
                'total_events': event_count,
                'event_types': len(self.temporal_system.events),
                'patterns_detected': len(self.temporal_system.detected_patterns),
                'routines_found': len(self.temporal_system.routine_patterns)
            }
        
        # Vision statistics
        if self.vision_system:
            summary['statistics']['vision'] = {
                'screen_history_size': len(self.vision_system.screen_history),
                'ui_patterns': sum(len(patterns) for patterns in self.vision_system.pattern_library.values()),
                'current_activity': self.visual_context.user_activity if self.visual_context else None
            }
        
        return summary
    
    async def shutdown(self):
        """Gracefully shutdown Phase 2 components"""
        logger.info("Shutting down JARVIS Phase 2...")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Save states
        if self.context_system:
            # Context is already persisted to DB
            pass
        
        if self.predictive_system:
            # Models are already saved periodically
            pass
        
        logger.info("JARVIS Phase 2 shutdown complete")

# Factory function
async def create_jarvis_phase2(config: Optional[Phase2Config] = None) -> JARVISPhase2Core:
    """Create and initialize JARVIS Phase 2"""
    jarvis = JARVISPhase2Core(config)
    await jarvis.initialize()
    return jarvis
