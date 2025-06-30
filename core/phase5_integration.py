#!/usr/bin/env python3
"""
JARVIS Phase 5: Natural Interaction Integration
==============================================

Integrates graduated interventions and emotional continuity
into the JARVIS enhanced core system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

# Import Phase 1 components
from .unified_input_pipeline import UnifiedInputPipeline, InputPriority
from .fluid_state_management import FluidStateManager, SystemState

# Import Phase 5 components
from .phase5_graduated_interventions import (
    GraduatedInterventionSystem,
    InterventionContext,
    InterventionType,
    InterventionLevel
)
from .phase5_emotional_continuity import (
    EmotionalContinuitySystem,
    EmotionalMemory,
    ConversationThread
)

# Import existing JARVIS components
from .emotional_intelligence import EmotionalIntelligence, EmotionType
from .elite_proactive_assistant import EliteProactiveAssistant

logger = logging.getLogger(__name__)


class NaturalInteractionCore:
    """
    Core system that integrates graduated interventions with
    emotional continuity for natural, empathetic interactions
    """
    
    def __init__(self):
        # Phase 1 components
        self.input_pipeline = UnifiedInputPipeline()
        self.state_manager = FluidStateManager()
        
        # Phase 5 components
        self.intervention_system = GraduatedInterventionSystem()
        self.continuity_system = EmotionalContinuitySystem()
        
        # Existing components
        self.emotional_intelligence = EmotionalIntelligence()
        self.proactive_assistant = EliteProactiveAssistant()
        
        # Integration state
        self.active_mode = "natural"
        self.conversation_context = {}
        self.intervention_history = []
        
    async def initialize(self):
        """Initialize all systems"""
        logger.info("Initializing Natural Interaction Core...")
        
        # Initialize components
        await self.input_pipeline.initialize()
        self.state_manager.start_monitoring()
        
        # Load saved emotional state if exists
        try:
            await self.continuity_system.load_emotional_state("data/emotional_state.json")
            logger.info("Loaded previous emotional state")
        except:
            logger.info("Starting with fresh emotional state")
        
        # Start background tasks
        asyncio.create_task(self._intervention_monitor())
        asyncio.create_task(self._emotional_continuity_monitor())
        
        logger.info("Natural Interaction Core initialized")
    
    async def process_interaction(
        self,
        text: str,
        voice_data: Optional[Any] = None,
        biometric_data: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process an interaction with full natural interaction capabilities"""
        
        # Create unified input
        input_data = {
            "text": text,
            "timestamp": datetime.now()
        }
        
        if voice_data:
            input_data["voice"] = voice_data
        if biometric_data:
            input_data["biometric"] = biometric_data
        
        # Process through pipeline
        processed_input = await self.input_pipeline.process_input(
            input_data,
            source="natural_interaction",
            priority=InputPriority.NORMAL
        )
        
        # Analyze emotional state
        emotion_result = await self._analyze_emotion(processed_input, context)
        
        # Update emotional continuity
        continuity_result = await self.continuity_system.process_emotional_input(
            text,
            emotion_result["emotion"],
            emotion_result["intensity"],
            context or {}
        )
        
        # Get current system state
        current_state = self.state_manager.get_state()
        
        # Check if intervention is needed
        intervention_context = InterventionContext(
            user_state=current_state.state.name,
            stress_level=current_state.stress_level,
            focus_level=current_state.focus_level,
            time_since_break=self._calculate_time_since_break(),
            current_activity=context.get("activity", "general") if context else "general",
            emotional_state=emotion_result["emotion"]
        )
        
        intervention = await self.intervention_system.evaluate_intervention_need(
            intervention_context
        )
        
        # Build response with all components
        response = await self._build_natural_response(
            processed_input,
            emotion_result,
            continuity_result,
            intervention,
            current_state
        )
        
        return response
    
    async def _analyze_emotion(
        self,
        processed_input: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze emotional content of input"""
        
        # Use existing emotional intelligence
        text = processed_input.get("processed_data", {}).get("text", "")
        
        # Simple emotion detection (would use actual model)
        emotions = {
            "happy": ["great", "awesome", "wonderful", "excited", "love"],
            "stressed": ["stressed", "overwhelmed", "pressure", "deadline", "anxious"],
            "frustrated": ["frustrated", "annoying", "stuck", "difficult", "hate"],
            "calm": ["calm", "peaceful", "relaxed", "fine", "okay"],
            "tired": ["tired", "exhausted", "sleepy", "fatigue", "worn out"]
        }
        
        detected_emotion = "neutral"
        max_score = 0
        
        text_lower = text.lower()
        for emotion, keywords in emotions.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > max_score:
                max_score = score
                detected_emotion = emotion
        
        # Calculate intensity based on various factors
        intensity = min(0.3 + (max_score * 0.2), 1.0)
        
        # Adjust based on biometric data if available
        if processed_input.get("processed_data", {}).get("biometric"):
            bio_data = processed_input["processed_data"]["biometric"]
            if bio_data.get("heart_rate", 70) > 90:
                intensity = min(intensity + 0.2, 1.0)
        
        return {
            "emotion": detected_emotion,
            "intensity": intensity,
            "confidence": 0.8
        }
    
    def _calculate_time_since_break(self) -> float:
        """Calculate minutes since last break"""
        # This would track actual break times
        # For now, return a simulated value
        return 95.0  # minutes
    
    async def _build_natural_response(
        self,
        processed_input: Dict[str, Any],
        emotion_result: Dict[str, Any],
        continuity_result: Dict[str, Any],
        intervention: Optional[Any],
        current_state: SystemState
    ) -> Dict[str, Any]:
        """Build a natural, contextual response"""
        
        response = {
            "timestamp": datetime.now().isoformat(),
            "emotional_state": emotion_result,
            "system_state": current_state.state.name,
            "continuity_context": continuity_result["emotional_context"],
            "response_elements": {}
        }
        
        # Get response elements from emotional continuity
        continuity_elements = continuity_result.get("continuity_elements", {})
        
        # Build base response
        base_response = await self._generate_base_response(
            processed_input.get("processed_data", {}).get("text", ""),
            emotion_result,
            current_state
        )
        
        response["base_response"] = base_response
        
        # Add emotional continuity elements
        if continuity_elements.get("callbacks"):
            response["callbacks"] = continuity_elements["callbacks"]
        
        if continuity_elements.get("pattern_insights"):
            response["insights"] = continuity_elements["pattern_insights"]
        
        # Handle intervention if needed
        if intervention:
            intervention_result = await self.intervention_system.execute_intervention(intervention)
            response["intervention"] = {
                "level": intervention.level.name,
                "type": intervention.type.name,
                "message": intervention.message,
                "executed": intervention_result["actions_executed"]
            }
            
            # Blend intervention with response
            if intervention.message:
                response["combined_message"] = self._blend_messages(
                    base_response,
                    intervention.message,
                    continuity_elements.get("emotional_validation", [])
                )
        else:
            response["combined_message"] = base_response
        
        # Add suggestions based on state
        response["suggestions"] = await self._generate_suggestions(
            emotion_result,
            current_state,
            continuity_result
        )
        
        return response
    
    async def _generate_base_response(
        self,
        text: str,
        emotion_result: Dict[str, Any],
        current_state: SystemState
    ) -> str:
        """Generate base response to user input"""
        
        # This would use actual NLP/LLM
        # For now, simple template responses
        
        if "help" in text.lower():
            return "I'm here to help! What can I assist you with?"
        elif "how are you" in text.lower():
            return f"I'm functioning well, currently in {current_state.state.name} mode. How can I support you today?"
        elif emotion_result["emotion"] == "stressed":
            return "I understand you're feeling stressed. Let's work through this together."
        elif emotion_result["emotion"] == "happy":
            return "It's wonderful to hear such positivity! What's bringing you joy?"
        else:
            return "I'm listening. Please tell me more."
    
    def _blend_messages(
        self,
        base_response: str,
        intervention_message: str,
        emotional_validations: List[str]
    ) -> str:
        """Blend multiple message components naturally"""
        
        messages = []
        
        # Add emotional validation first if high intensity
        if emotional_validations:
            messages.append(emotional_validations[0])
        
        # Add base response
        messages.append(base_response)
        
        # Add intervention if different from base
        if intervention_message and intervention_message not in base_response:
            messages.append(intervention_message)
        
        # Join naturally
        return " ".join(messages)
    
    async def _generate_suggestions(
        self,
        emotion_result: Dict[str, Any],
        current_state: SystemState,
        continuity_result: Dict[str, Any]
    ) -> List[str]:
        """Generate contextual suggestions"""
        
        suggestions = []
        
        # Based on emotion
        if emotion_result["emotion"] == "stressed" and emotion_result["intensity"] > 0.6:
            suggestions.append("Try the 4-7-8 breathing technique")
            suggestions.append("Take a 5-minute walk")
        elif emotion_result["emotion"] == "tired":
            suggestions.append("Consider a power nap")
            suggestions.append("Hydrate and stretch")
        
        # Based on patterns
        patterns = continuity_result.get("patterns_detected", [])
        if "recurring_stress_pattern" in patterns:
            suggestions.append("Review your stress management strategies")
        elif "successful_recovery_pattern" in patterns:
            suggestions.append("Keep using what works for you!")
        
        return suggestions
    
    async def _intervention_monitor(self):
        """Monitor and escalate interventions as needed"""
        while True:
            try:
                # Check for needed escalations
                escalations = await self.intervention_system.check_escalation_needed()
                
                for intervention in escalations:
                    logger.info(f"Escalating intervention: {intervention.type.name} to {intervention.level.name}")
                    await self.intervention_system.execute_intervention(intervention)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Intervention monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _emotional_continuity_monitor(self):
        """Monitor emotional patterns and save state"""
        while True:
            try:
                # Get emotional summary
                summary = self.continuity_system.get_emotional_summary()
                
                # Log significant changes
                if summary.get("emotional_stability", 1.0) < 0.5:
                    logger.info("Detected emotional volatility")
                
                # Save state periodically
                await self.continuity_system.save_emotional_state("data/emotional_state.json")
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Emotional continuity monitor error: {e}")
                await asyncio.sleep(60)
    
    async def record_interaction_feedback(
        self,
        interaction_id: str,
        feedback: str
    ) -> None:
        """Record user feedback on interactions"""
        
        # Record for intervention system
        if interaction_id.startswith("intervention_"):
            await self.intervention_system.record_user_response(
                interaction_id.replace("intervention_", ""),
                feedback
            )
        
        # Update emotional memories if needed
        if feedback == "helpful":
            # Mark relevant concerns as resolved
            recent_concerns = [
                em for em in self.continuity_system.emotional_memories
                if em.followup_needed and not em.resolved
            ]
            if recent_concerns:
                await self.continuity_system.resolve_concern(
                    f"em_{recent_concerns[0].timestamp.timestamp()}"
                )
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics on natural interaction performance"""
        
        intervention_analytics = self.intervention_system.get_intervention_analytics()
        emotional_summary = self.continuity_system.get_emotional_summary()
        
        return {
            "intervention_analytics": intervention_analytics,
            "emotional_summary": emotional_summary,
            "interaction_quality": {
                "emotional_stability": emotional_summary.get("emotional_stability", 0),
                "intervention_success_rate": self._calculate_intervention_success(),
                "continuity_score": self._calculate_continuity_score()
            }
        }
    
    def _calculate_intervention_success(self) -> float:
        """Calculate intervention success rate"""
        if not self.intervention_system.intervention_history:
            return 0.0
        
        successful = sum(
            1 for record in self.intervention_system.intervention_history
            if record.get("results", {}).get("actions_executed", [])
        )
        
        return successful / len(self.intervention_system.intervention_history)
    
    def _calculate_continuity_score(self) -> float:
        """Calculate emotional continuity score"""
        if not self.continuity_system.conversation_threads:
            return 0.0
        
        active_threads = sum(
            1 for thread in self.continuity_system.conversation_threads.values()
            if len(thread.messages) > 3
        )
        
        return min(active_threads / len(self.continuity_system.conversation_threads), 1.0)


# Integration with main JARVIS
class JARVISPhase5Integration:
    """Integrates Phase 5 natural interaction into JARVIS"""
    
    def __init__(self, jarvis_core):
        self.jarvis_core = jarvis_core
        self.natural_interaction = NaturalInteractionCore()
        
    async def initialize(self):
        """Initialize Phase 5 integration"""
        await self.natural_interaction.initialize()
        
        # Enhance JARVIS with natural interaction
        self.jarvis_core.natural_interaction = self.natural_interaction
        
        logger.info("Phase 5 Natural Interaction integrated with JARVIS")
    
    async def process_with_natural_interaction(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process request with natural interaction enhancements"""
        
        # Get biometric data if available
        biometric_data = None
        if hasattr(self.jarvis_core, 'biometric_monitor'):
            biometric_data = await self.jarvis_core.biometric_monitor.get_current_data()
        
        # Process through natural interaction
        result = await self.natural_interaction.process_interaction(
            text,
            voice_data=None,  # Would come from voice system
            biometric_data=biometric_data,
            context=context
        )
        
        # Enhance with JARVIS capabilities
        if result.get("base_response"):
            # Process through JARVIS for actual task execution
            jarvis_response = await self.jarvis_core.process_request(
                text,
                context=context
            )
            
            # Merge responses
            result["jarvis_response"] = jarvis_response
            result["task_completed"] = jarvis_response.get("success", False)
        
        return result


# Demo function
async def demo_phase5():
    """Demonstrate Phase 5 Natural Interaction capabilities"""
    
    print("🎭 Phase 5: Natural Interaction Demo")
    print("=" * 50)
    
    # Create natural interaction system
    natural = NaturalInteractionCore()
    await natural.initialize()
    
    # Simulate a conversation
    interactions = [
        ("I'm really stressed about all these deadlines", {"activity": "work"}),
        ("The project keeps getting more complex", {"activity": "work"}),
        ("I don't know if I can finish in time", {"activity": "work"}),
        ("Maybe I should take a break", {"activity": "work"}),
        ("Actually, I think I figured out a solution!", {"activity": "work"}),
        ("Thanks for being here", {"activity": "general"})
    ]
    
    for text, context in interactions:
        print(f"\n👤 User: {text}")
        
        result = await natural.process_interaction(
            text,
            context=context,
            biometric_data={"heart_rate": 75 + len(text) % 20}  # Simulated
        )
        
        print(f"🤖 JARVIS: {result.get('combined_message', result.get('base_response'))}")
        
        if result.get("intervention"):
            print(f"   [Intervention: {result['intervention']['level']} - {result['intervention']['type']}]")
        
        if result.get("suggestions"):
            print(f"   💡 Suggestions: {', '.join(result['suggestions'][:2])}")
        
        await asyncio.sleep(0.5)
    
    # Show analytics
    print("\n\n📊 Natural Interaction Analytics")
    print("-" * 50)
    analytics = natural.get_analytics()
    print(json.dumps(analytics, indent=2))


if __name__ == "__main__":
    asyncio.run(demo_phase5())
