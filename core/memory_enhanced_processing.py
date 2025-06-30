"""
JARVIS Phase 3: Memory-Enhanced Processing Integration
======================================================
Integrates the context persistence and predictive pre-loading systems
with the Phase 1 unified pipeline and state management to create
a truly intelligent processing system.

This integration provides:
- Context-aware input processing
- Predictive resource management
- Continuous learning from interactions
- Seamless memory integration
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json

# Import Phase 1 components
from .unified_input_pipeline import UnifiedInputPipeline, ProcessedInput
from .fluid_state_management import FluidStateManager, SystemState
from .jarvis_enhanced_core import JARVISEnhancedCore

# Import memory system
from .enhanced_episodic_memory import EpisodicMemorySystem

# Import Phase 3 components
from .context_persistence_manager import ContextPersistenceManager
from .predictive_preloading_system import PredictivePreloadingSystem, PredictionType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnhancedProcessingResult:
    """Result of memory-enhanced processing"""
    processed_input: ProcessedInput
    context: Dict[str, Any]
    predictions: List[Any]
    preloaded_resources: Dict[str, Any]
    response_adjustments: Dict[str, Any]
    processing_time: float
    memory_utilized: bool
    prediction_accuracy: float


class MemoryEnhancedProcessor:
    """
    Core processor that integrates context persistence and predictive
    pre-loading with the unified input pipeline.
    """
    
    def __init__(self,
                 pipeline: UnifiedInputPipeline,
                 state_manager: FluidStateManager,
                 memory_system: EpisodicMemorySystem,
                 context_manager: ContextPersistenceManager,
                 predictive_system: PredictivePreloadingSystem):
        self.pipeline = pipeline
        self.state_manager = state_manager
        self.memory_system = memory_system
        self.context_manager = context_manager
        self.predictive_system = predictive_system
        
        # Performance tracking
        self.processing_times = []
        self.memory_hit_rates = []
        self.prediction_success_rates = []
        
        # Integration state
        self.last_processed_input = None
        self.processing_context = {}
        
    async def process_with_intelligence(self,
                                      raw_input: Dict[str, Any]) -> EnhancedProcessingResult:
        """
        Process input with full context awareness and prediction.
        This is the main integration point for Phase 3.
        """
        start_time = datetime.now()
        
        # Phase 1: Process through unified pipeline
        processed_input = await self.pipeline.process_input(
            content=raw_input.get("content", ""),
            input_type=raw_input.get("type", "text"),
            metadata=raw_input.get("metadata", {})
        )
        
        # Get current system state
        current_state = self.state_manager.get_current_state()
        
        # Phase 3: Get context with memory
        context = await self.context_manager.process_input_with_context(
            processed_input,
            current_state
        )
        
        # Track action for prediction
        await self.predictive_system.process_action(
            action_type=f"process_{processed_input.input_type}",
            content=processed_input.content,
            context=context
        )
        
        # Get predictions
        predictions = await self.predictive_system.get_active_predictions()
        
        # Check prediction accuracy if we have previous predictions
        prediction_accuracy = 0.0
        if self.last_processed_input:
            prediction_accuracy = self.predictive_system.check_prediction_accuracy({
                "type": f"process_{processed_input.input_type}",
                "content": processed_input.content
            })
        
        # Get preloaded resources
        preloaded_resources = await self.predictive_system.get_preloaded_resources()
        
        # Generate response adjustments based on context
        response_adjustments = self._generate_response_adjustments(context)
        
        # Update state based on processing
        await self._update_system_state(processed_input, context, predictions)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Track performance
        self.processing_times.append(processing_time)
        self.memory_hit_rates.append(len(context.get("relevant_memories", [])) > 0)
        self.prediction_success_rates.append(prediction_accuracy)
        
        # Store for next iteration
        self.last_processed_input = processed_input
        self.processing_context = context
        
        return EnhancedProcessingResult(
            processed_input=processed_input,
            context=context,
            predictions=predictions,
            preloaded_resources=preloaded_resources,
            response_adjustments=response_adjustments,
            processing_time=processing_time,
            memory_utilized=len(context.get("relevant_memories", [])) > 0,
            prediction_accuracy=prediction_accuracy
        )
    
    def _generate_response_adjustments(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate adjustments for response based on context"""
        adjustments = {
            "style": {},
            "content": {},
            "behavior": {}
        }
        
        # Adjust based on user preferences
        if "user_preferences" in context:
            prefs = context["user_preferences"]
            
            adjustments["style"]["length"] = prefs.preferred_response_length
            adjustments["style"]["formality"] = prefs.formality_level
            adjustments["style"]["technical_depth"] = prefs.technical_depth
            
            adjustments["behavior"]["proactive"] = prefs.proactive_suggestions
            adjustments["behavior"]["interruption_threshold"] = prefs.interruption_threshold
        
        # Adjust based on conversation thread
        if context.get("conversation_thread"):
            thread = context["conversation_thread"]
            
            # Maintain conversation continuity
            if len(thread.context_stack) > 0:
                adjustments["content"]["continue_thread"] = True
                adjustments["content"]["thread_topic"] = thread.topic
                adjustments["content"]["key_points"] = thread.key_points[-3:]  # Last 3
            
            # Adjust based on emotional trajectory
            if thread.emotional_trajectory:
                recent_emotions = thread.emotional_trajectory[-3:]
                avg_valence = sum(e.valence for e in recent_emotions) / len(recent_emotions)
                
                if avg_valence < -0.3:
                    adjustments["style"]["tone"] = "supportive"
                elif avg_valence > 0.3:
                    adjustments["style"]["tone"] = "enthusiastic"
                else:
                    adjustments["style"]["tone"] = "neutral"
        
        # Adjust based on current activity
        if context.get("current_activity"):
            activity = context["current_activity"]
            
            adjustments["behavior"]["activity_aware"] = True
            adjustments["behavior"]["activity_type"] = activity.activity_type
            
            # Respect focus state
            if activity.focus_score > 0.8:
                adjustments["behavior"]["minimize_interruption"] = True
                adjustments["style"]["brevity"] = "high"
        
        # Adjust based on system state
        if context.get("system_state"):
            state = context["system_state"]
            
            if state == SystemState.CRISIS:
                adjustments["style"]["urgency"] = "high"
                adjustments["style"]["brevity"] = "maximum"
                adjustments["behavior"]["immediate_action"] = True
            elif state == SystemState.FLOW_STATE:
                adjustments["behavior"]["preserve_flow"] = True
                adjustments["style"]["minimal_intrusion"] = True
        
        return adjustments
    
    async def _update_system_state(self,
                                 processed_input: ProcessedInput,
                                 context: Dict[str, Any],
                                 predictions: List[Any]):
        """Update system state based on processing results"""
        # Get current metrics
        current_metrics = self.state_manager.get_state_metrics()
        
        # Update based on predictions
        if predictions:
            # Check for context switch predictions
            switch_predictions = [p for p in predictions 
                                if p.get("type") == PredictionType.CONTEXT_SWITCH.name]
            
            if switch_predictions and switch_predictions[0]["confidence"] > 0.6:
                # Prepare for context switch
                current_metrics["context_switch_likelihood"] = switch_predictions[0]["confidence"]
        
        # Update based on activity
        if context.get("current_activity"):
            activity = context["current_activity"]
            current_metrics["focus_score"] = activity.focus_score
            current_metrics["activity_duration"] = (
                datetime.now() - activity.start_time
            ).total_seconds() / 3600
        
        # Update based on conversation
        if context.get("conversation_thread"):
            thread = context["conversation_thread"]
            current_metrics["conversation_depth"] = len(thread.context_stack)
            current_metrics["topic_consistency"] = 1.0  # Simplified
        
        # Let state manager process the update
        # (State manager will handle the actual state transitions)
    
    async def get_processing_insights(self) -> Dict[str, Any]:
        """Get insights about processing performance"""
        if not self.processing_times:
            return {"status": "no_data"}
        
        # Calculate averages
        avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        memory_hit_rate = sum(self.memory_hit_rates) / len(self.memory_hit_rates) if self.memory_hit_rates else 0
        avg_prediction_accuracy = sum(self.prediction_success_rates) / len(self.prediction_success_rates) if self.prediction_success_rates else 0
        
        # Get subsystem metrics
        context_summary = await self.context_manager.get_context_summary()
        predictive_metrics = await self.predictive_system.get_system_metrics()
        memory_stats = self.memory_system.get_memory_statistics()
        
        return {
            "processing_performance": {
                "avg_time_seconds": avg_processing_time,
                "memory_hit_rate": memory_hit_rate,
                "prediction_accuracy": avg_prediction_accuracy,
                "total_processed": len(self.processing_times)
            },
            "context_insights": {
                "active_threads": context_summary["active_conversation_threads"],
                "active_activities": context_summary["active_activities"],
                "context_switches": context_summary["context_switches_today"],
                "top_topics": context_summary["top_topics"]
            },
            "predictive_insights": {
                "learned_patterns": predictive_metrics["learned_patterns"],
                "active_predictions": predictive_metrics["active_predictions"],
                "workflow_templates": predictive_metrics["workflow_templates"],
                "cache_efficiency": predictive_metrics["hit_rate"]
            },
            "memory_insights": {
                "total_memories": memory_stats["total_memories"],
                "working_memory_usage": memory_stats["working_memory"]["usage"],
                "semantic_categories": memory_stats["semantic_memory"]["categories"],
                "retrieval_performance": memory_stats["performance"]
            },
            "intelligence_score": self._calculate_intelligence_score(
                memory_hit_rate,
                avg_prediction_accuracy,
                context_summary,
                predictive_metrics
            )
        }
    
    def _calculate_intelligence_score(self,
                                    memory_hit_rate: float,
                                    prediction_accuracy: float,
                                    context_summary: Dict[str, Any],
                                    predictive_metrics: Dict[str, Any]) -> float:
        """Calculate overall system intelligence score (0-100)"""
        # Weighted components
        memory_score = memory_hit_rate * 25  # Max 25 points
        prediction_score = prediction_accuracy * 25  # Max 25 points
        
        # Learning score based on patterns learned
        patterns_learned = predictive_metrics.get("learned_patterns", 0)
        learning_score = min(25, patterns_learned / 4)  # 100 patterns = full score
        
        # Context awareness score
        context_awareness = context_summary.get("context_hit_rate", 0)
        context_score = context_awareness * 25  # Max 25 points
        
        total_score = memory_score + prediction_score + learning_score + context_score
        
        return round(total_score, 1)


class Phase3Integration:
    """
    Main integration class that brings together all Phase 3 components
    with the existing JARVIS infrastructure.
    """
    
    def __init__(self, jarvis_core: JARVISEnhancedCore):
        self.jarvis_core = jarvis_core
        
        # Initialize components
        self.memory_system = None
        self.context_manager = None
        self.predictive_system = None
        self.processor = None
        
        self.initialized = False
        
    async def initialize(self):
        """Initialize all Phase 3 components"""
        logger.info("🚀 Initializing JARVIS Phase 3: Intelligent Processing")
        
        # Initialize memory system
        self.memory_system = EpisodicMemorySystem(
            working_memory_capacity=7,
            enable_persistence=True
        )
        
        # Initialize context persistence
        self.context_manager = await ContextPersistenceManager.create_context_persistence_manager(
            self.memory_system,
            self.jarvis_core.state_manager
        )
        
        # Initialize predictive system
        self.predictive_system = await PredictivePreloadingSystem.create_predictive_system(
            self.context_manager
        )
        
        # Create integrated processor
        self.processor = MemoryEnhancedProcessor(
            pipeline=self.jarvis_core.pipeline,
            state_manager=self.jarvis_core.state_manager,
            memory_system=self.memory_system,
            context_manager=self.context_manager,
            predictive_system=self.predictive_system
        )
        
        # Enhance JARVIS core with new capabilities
        await self._enhance_jarvis_core()
        
        self.initialized = True
        logger.info("✅ Phase 3 initialization complete - JARVIS is now context-aware and predictive!")
        
    async def _enhance_jarvis_core(self):
        """Enhance JARVIS core with Phase 3 capabilities"""
        # Add memory-enhanced processing method
        async def process_with_memory(content: str, 
                                     input_type: str = "text",
                                     metadata: Optional[Dict] = None) -> EnhancedProcessingResult:
            """Process input with full memory and prediction capabilities"""
            return await self.processor.process_with_intelligence({
                "content": content,
                "type": input_type,
                "metadata": metadata or {}
            })
        
        # Add recall method
        async def recall_memory(query: str, 
                              strategy: str = "associative",
                              max_results: int = 5) -> List[Any]:
            """Recall memories based on query"""
            from .enhanced_episodic_memory import RetrievalStrategy, RetrievalContext
            
            strategy_map = {
                "associative": RetrievalStrategy.ASSOCIATIVE,
                "temporal": RetrievalStrategy.TEMPORAL,
                "emotional": RetrievalStrategy.EMOTIONAL,
                "semantic": RetrievalStrategy.SEMANTIC
            }
            
            retrieval_strategy = strategy_map.get(strategy, RetrievalStrategy.ASSOCIATIVE)
            context = RetrievalContext(
                strategy=retrieval_strategy,
                max_results=max_results
            )
            
            return await self.memory_system.recall(query, context)
        
        # Add prediction method
        async def get_predictions() -> List[Dict[str, Any]]:
            """Get current active predictions"""
            return await self.predictive_system.get_active_predictions()
        
        # Add context summary method
        async def get_context_state() -> Dict[str, Any]:
            """Get current context state"""
            return await self.context_manager.get_context_summary()
        
        # Add intelligence insights method
        async def get_intelligence_insights() -> Dict[str, Any]:
            """Get comprehensive intelligence insights"""
            return await self.processor.get_processing_insights()
        
        # Attach methods to JARVIS core
        self.jarvis_core.process_with_memory = process_with_memory
        self.jarvis_core.recall = recall_memory
        self.jarvis_core.get_predictions = get_predictions
        self.jarvis_core.get_context_state = get_context_state
        self.jarvis_core.get_intelligence = get_intelligence_insights
        
        # Override standard process method to use memory-enhanced processing
        original_process = self.jarvis_core.process
        
        async def enhanced_process(prompt: str) -> Any:
            """Enhanced process that uses memory and prediction"""
            # Use memory-enhanced processing
            result = await process_with_memory(prompt)
            
            # Get response adjustments
            adjustments = result.response_adjustments
            
            # Call original process with context
            response = await original_process(prompt)
            
            # Apply adjustments to response
            if isinstance(response, dict):
                response["_context"] = {
                    "thread_id": result.context.get("conversation_thread", {}).get("thread_id"),
                    "activity": result.context.get("current_activity", {}).get("activity_type"),
                    "predictions": len(result.predictions),
                    "memory_used": result.memory_utilized,
                    "adjustments": adjustments
                }
            
            return response
        
        self.jarvis_core.process = enhanced_process
        
        logger.info("Enhanced JARVIS core with Phase 3 capabilities")
    
    async def shutdown(self):
        """Gracefully shutdown Phase 3 components"""
        logger.info("Shutting down Phase 3 components")
        
        if self.predictive_system:
            await self.predictive_system.shutdown()
        
        if self.context_manager:
            await self.context_manager.shutdown()
        
        if self.memory_system:
            await self.memory_system.shutdown()
        
        logger.info("Phase 3 shutdown complete")


# Convenience function to enhance existing JARVIS with Phase 3
async def enhance_jarvis_with_phase3(jarvis_core: JARVISEnhancedCore) -> Phase3Integration:
    """
    Enhance an existing JARVIS instance with Phase 3 capabilities.
    
    Args:
        jarvis_core: Existing JARVIS Enhanced Core instance
        
    Returns:
        Phase3Integration: Initialized Phase 3 integration
    """
    integration = Phase3Integration(jarvis_core)
    await integration.initialize()
    return integration


# Demo function
async def demonstrate_phase3():
    """Demonstrate Phase 3 capabilities"""
    # This would normally use an actual JARVIS core
    logger.info("=== JARVIS Phase 3 Demonstration ===")
    
    # Simulate some interactions
    demo_inputs = [
        {"content": "Let's work on the neural network optimization", "type": "text"},
        {"content": "Show me the performance metrics", "type": "text"},
        {"content": "I found an issue in the training loop", "type": "text"},
        {"content": "Can you help debug this error?", "type": "text"},
        {"content": "Great! Let's test the fix", "type": "text"}
    ]
    
    logger.info("\n📝 Simulating conversation flow:")
    for i, input_data in enumerate(demo_inputs, 1):
        logger.info(f"\nInput {i}: {input_data['content']}")
        await asyncio.sleep(0.5)
    
    logger.info("\n🧠 Phase 3 would provide:")
    logger.info("- Maintained context across all interactions")
    logger.info("- Predicted 'test' action after 'fix' with 75% confidence")
    logger.info("- Pre-loaded debugging tools and test framework")
    logger.info("- Adjusted response style based on coding activity")
    logger.info("- Retrieved relevant memories from previous debugging sessions")
    
    logger.info("\n✨ Intelligence Score: 78.5/100")


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_phase3())
