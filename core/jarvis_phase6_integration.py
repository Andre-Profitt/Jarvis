"""
JARVIS Phase 6: Integration Layer
================================
Integrates Natural Language Flow and Emotional Intelligence with existing JARVIS
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

# Import Phase 1 components
from .unified_input_pipeline import UnifiedInputPipeline
from .fluid_state_management import FluidStateManager
from .jarvis_enhanced_core import JARVISEnhancedCore

# Import Phase 6 components
from .natural_language_flow import NaturalLanguageFlow, EmotionalTone, ConversationTopic
from .emotional_intelligence import EmotionalIntelligence, EmotionalState

class JARVISPhase6Core:
    """Enhanced JARVIS with Natural Language Flow and Emotional Intelligence"""
    
    def __init__(self, existing_jarvis=None):
        # Phase 1 components
        self.pipeline = UnifiedInputPipeline()
        self.state_manager = FluidStateManager()
        
        # Phase 6 components
        self.nlp_flow = NaturalLanguageFlow()
        self.emotional_engine = EmotionalIntelligence()
        
        # Existing JARVIS connection
        self.legacy_jarvis = existing_jarvis or JARVISEnhancedCore()
        
        # Integration state
        self.conversation_mode = "adaptive"  # adaptive, focused, casual, crisis
        self.emotional_awareness_enabled = True
        self.context_persistence_enabled = True
        
        # Performance tracking
        self.metrics = {
            "conversations": 0,
            "emotional_interventions": 0,
            "context_switches": 0,
            "user_satisfaction": []
        }
        
    async def initialize(self):
        """Initialize all components"""
        await self.legacy_jarvis.initialize()
        print("✅ JARVIS Phase 6 initialized with Natural Language Flow and Emotional Intelligence")
        
    async def process_input(self, input_data: Dict, source: str = "unknown") -> Dict:
        """Process input with full emotional and conversational awareness"""
        start_time = datetime.now()
        
        # Extract components from input
        text = input_data.get("voice", {}).get("text", "")
        voice_features = input_data.get("voice", {}).get("features", {})
        biometrics = input_data.get("biometric", {})
        
        # 1. Emotional Analysis
        emotional_analysis = await self.emotional_engine.analyze_emotional_content(
            text=text,
            voice_features=voice_features,
            biometrics=biometrics,
            context={"source": source, "timestamp": start_time}
        )
        
        # 2. Update Fluid State based on emotions
        emotional_state = emotional_analysis["current_state"]
        if emotional_state["quadrant"] == "excited_unhappy":
            await self.state_manager.update_state({
                "heart_rate": 100,  # Simulate elevated state
                "stress_level": 0.8
            })
        elif emotional_state["quadrant"] == "calm_unhappy":
            await self.state_manager.update_state({
                "heart_rate": 65,
                "stress_level": 0.6
            })
            
        # 3. Process through Natural Language Flow
        nlp_result = await self.nlp_flow.process_input(
            text=text,
            metadata={
                "emotional_state": emotional_state,
                "biometrics": biometrics,
                "jarvis_state": self.state_manager.get_state()
            }
        )
        
        # 4. Generate base response using legacy JARVIS
        legacy_response = await self.legacy_jarvis.process_input(input_data, source)
        
        # 5. Enhance response with emotional awareness
        if self.emotional_awareness_enabled:
            enhanced_response = self.emotional_engine.generate_emotionally_aware_response(
                base_response=legacy_response.get("response", ""),
                emotional_analysis=emotional_analysis
            )
        else:
            enhanced_response = legacy_response.get("response", "")
            
        # 6. Apply conversational flow
        if self.context_persistence_enabled:
            final_response = self._apply_conversational_context(
                enhanced_response,
                nlp_result
            )
        else:
            final_response = enhanced_response
            
        # 7. Determine actions based on emotional state
        actions = self._determine_emotional_actions(emotional_analysis, nlp_result)
        
        # 8. Update metrics
        self._update_metrics(emotional_analysis, nlp_result)
        
        # Build comprehensive response
        response = {
            "response": final_response,
            "emotional_state": emotional_analysis["current_state"],
            "conversation_context": nlp_result["context"],
            "actions": actions,
            "mode": self._determine_conversation_mode(emotional_analysis, nlp_result),
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "continuity_score": nlp_result["continuity_score"],
            "emotional_trajectory": emotional_analysis["trajectory"]
        }
        
        return response
        
    def _apply_conversational_context(self, response: str, nlp_result: Dict) -> str:
        """Apply conversational context to response"""
        # Add context bridges if topic recently changed
        if nlp_result["context"].get("interrupt_stack"):
            # Acknowledge pending topics
            pending = len(nlp_result["context"]["interrupt_stack"])
            if pending > 0:
                response += f" (I'm keeping track of {pending} other things we can circle back to)"
                
        # Reference entities if relevant
        entities = nlp_result["context"].get("entities", {})
        if entities and not any(entity in response for entity in entities):
            # Response doesn't mention key entities, add context
            people = [e for e, t in entities.items() if t == "person"]
            if people:
                response = f"Regarding {people[0]}, " + response
                
        return response
        
    def _determine_emotional_actions(self, 
                                   emotional_analysis: Dict,
                                   nlp_result: Dict) -> List[Dict]:
        """Determine actions based on emotional and conversational analysis"""
        actions = []
        
        # Crisis detection
        if emotional_analysis["current_state"]["quadrant"] == "excited_unhappy":
            if emotional_analysis["trajectory"]["velocity"] > 0.5:
                actions.append({
                    "type": "crisis_intervention",
                    "priority": "immediate",
                    "actions": ["pause_notifications", "offer_breathing_exercise", "alert_support"]
                })
                
        # Flow state protection
        current_state = self.state_manager.get_state()
        if current_state == "flow":
            actions.append({
                "type": "protect_flow",
                "actions": ["minimize_interruptions", "queue_non_urgent"]
            })
            
        # Emotional support actions
        recommendations = emotional_analysis["recommended_response"]
        if "crisis_support" in recommendations.get("approach", ""):
            actions.append({
                "type": "emotional_support",
                "actions": recommendations["actions"]
            })
            
        # Context management actions
        if nlp_result["context"].get("interrupt_stack"):
            actions.append({
                "type": "context_management",
                "action": "track_interrupted_threads",
                "threads": len(nlp_result["context"]["interrupt_stack"])
            })
            
        return actions
        
    def _determine_conversation_mode(self, 
                                   emotional_analysis: Dict,
                                   nlp_result: Dict) -> str:
        """Determine appropriate conversation mode"""
        emotional_state = emotional_analysis["current_state"]
        trajectory = emotional_analysis["trajectory"]
        
        # Crisis mode
        if trajectory["prediction"] == "needs_intervention":
            return "crisis"
            
        # Focused mode for work
        if nlp_result["topic"] == ConversationTopic.WORK.value:
            return "focused"
            
        # Casual mode for positive states
        if emotional_state["valence"] > 0.3 and abs(emotional_state["arousal"]) < 0.5:
            return "casual"
            
        # Adaptive mode for everything else
        return "adaptive"
        
    def _update_metrics(self, emotional_analysis: Dict, nlp_result: Dict):
        """Update performance metrics"""
        self.metrics["conversations"] += 1
        
        if emotional_analysis["trajectory"]["prediction"] == "needs_intervention":
            self.metrics["emotional_interventions"] += 1
            
        if nlp_result["context"].get("interrupt_stack"):
            self.metrics["context_switches"] += 1
            
    async def handle_feedback(self, feedback: Dict):
        """Handle user feedback to improve responses"""
        # Update emotional intelligence
        if "effectiveness" in feedback:
            # Update response effectiveness in emotional memory
            if self.emotional_engine.emotional_history:
                self.emotional_engine.emotional_history[-1].response_effectiveness = feedback["effectiveness"]
                
        # Update conversation satisfaction
        if "satisfaction" in feedback:
            self.metrics["user_satisfaction"].append({
                "score": feedback["satisfaction"],
                "timestamp": datetime.now(),
                "context": self.nlp_flow.context.topic.value
            })
            
    async def get_conversation_summary(self) -> Dict:
        """Get comprehensive conversation summary"""
        return {
            "conversation_flow": self.nlp_flow.get_conversation_summary(),
            "emotional_journey": self.emotional_engine.get_emotional_summary(),
            "state_progression": await self.state_manager.get_state_history(),
            "metrics": self.metrics,
            "active_mode": self.conversation_mode
        }
        
    async def resume_conversation(self, context_id: str) -> str:
        """Resume a previous conversation thread"""
        return await self.nlp_flow.resume_context(context_id)
        
    def set_emotional_awareness(self, enabled: bool):
        """Toggle emotional awareness"""
        self.emotional_awareness_enabled = enabled
        
    def set_context_persistence(self, enabled: bool):
        """Toggle context persistence"""
        self.context_persistence_enabled = enabled
        
    async def demonstrate_phase6(self):
        """Demonstrate Phase 6 capabilities"""
        print("\n🎭 JARVIS Phase 6: Natural Language Flow & Emotional Intelligence Demo")
        print("="*70)
        
        # Demo scenarios
        scenarios = [
            {
                "name": "Emotional Support During Stress",
                "input": {
                    "voice": {
                        "text": "I'm completely overwhelmed with this project deadline",
                        "features": {"pitch_ratio": 1.3, "rate_ratio": 1.4}
                    },
                    "biometric": {"heart_rate": 95, "breathing_rate": 22}
                }
            },
            {
                "name": "Topic Interruption Handling",
                "sequence": [
                    {
                        "voice": {"text": "Can you help me plan tomorrow's presentation?"}
                    },
                    {
                        "voice": {"text": "Actually wait, first I need to find the quarterly report"}
                    },
                    {
                        "voice": {"text": "Never mind, back to the presentation planning"}
                    }
                ]
            },
            {
                "name": "Emotional Continuity",
                "sequence": [
                    {
                        "voice": {"text": "I had a really tough day at work"},
                        "biometric": {"heart_rate": 85}
                    },
                    {
                        "voice": {"text": "The meeting didn't go well at all"}
                    },
                    {
                        "voice": {"text": "I don't know what to do next"}
                    }
                ]
            }
        ]
        
        for scenario in scenarios:
            print(f"\n📍 Scenario: {scenario['name']}")
            print("-" * 50)
            
            if "sequence" in scenario:
                # Multi-turn conversation
                for i, turn in enumerate(scenario["sequence"]):
                    print(f"\n👤 Turn {i+1}: {turn['voice']['text']}")
                    result = await self.process_input(turn)
                    print(f"🤖 JARVIS: {result['response']}")
                    print(f"   Emotion: {result['emotional_state']['primary_emotion']}")
                    print(f"   Context: {result['conversation_context']['topic']}")
                    await asyncio.sleep(1)
            else:
                # Single turn
                print(f"👤 User: {scenario['input']['voice']['text']}")
                result = await self.process_input(scenario["input"])
                print(f"🤖 JARVIS: {result['response']}")
                print(f"   Emotional State: {result['emotional_state']}")
                print(f"   Mode: {result['mode']}")
                print(f"   Actions: {[a['type'] for a in result['actions']]}")
                
        # Show conversation summary
        print("\n📊 Conversation Summary:")
        summary = await self.get_conversation_summary()
        print(json.dumps(summary, indent=2))


# Convenience functions for integration
async def upgrade_to_phase6(existing_jarvis=None):
    """Upgrade existing JARVIS to Phase 6"""
    phase6 = JARVISPhase6Core(existing_jarvis)
    await phase6.initialize()
    return phase6


def create_phase6_jarvis():
    """Create new JARVIS with Phase 6 capabilities"""
    return JARVISPhase6Core()
