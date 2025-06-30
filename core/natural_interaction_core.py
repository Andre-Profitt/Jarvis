"""
JARVIS Phase 5: Natural Interaction Core
Integrates conversational memory, emotional continuity, and natural language flow
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

# Import Phase 1 components
from .unified_input_pipeline import UnifiedInputPipeline
from .fluid_state_management import FluidStateManager

# Import Phase 5 components
from .conversational_memory import ConversationalMemory, MemoryType
from .emotional_continuity import EmotionalContinuity, EmotionType
from .natural_language_flow import NaturalLanguageFlow, ResponseStyle

class InteractionMode(Enum):
    CONVERSATION = "conversation"
    TASK = "task"
    LEARNING = "learning"
    CREATIVE = "creative"
    CRISIS = "crisis"
    FLOW = "flow"

class NaturalInteractionCore:
    """Core system for Phase 5 - Natural Interaction Flow"""
    
    def __init__(self):
        # Phase 1 components
        self.input_pipeline = UnifiedInputPipeline()
        self.state_manager = FluidStateManager()
        
        # Phase 5 components
        self.memory = ConversationalMemory(memory_window_minutes=30)
        self.emotional_continuity = EmotionalContinuity()
        self.language_flow = NaturalLanguageFlow()
        
        # Interaction state
        self.current_mode = InteractionMode.CONVERSATION
        self.interaction_context = {
            "start_time": datetime.now(),
            "turns": 0,
            "mode_history": [],
            "satisfaction_score": 0.7
        }
        
        # Natural interaction settings
        self.settings = {
            "memory_depth": 5,  # How many exchanges to consider
            "emotional_weight": 0.3,  # How much emotion affects responses
            "personality_traits": {
                "enthusiasm": 0.6,
                "humor": 0.4,
                "empathy": 0.8,
                "formality": 0.3
            }
        }
        
        print("✨ Natural Interaction Core initialized with:")
        print("  - Conversational Memory System")
        print("  - Emotional Continuity Tracking")
        print("  - Natural Language Flow Engine")
    
    async def process_interaction(self, 
                                user_input: str,
                                multimodal_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process user interaction with full natural flow"""
        
        start_time = asyncio.get_event_loop().time()
        
        # Process through Phase 1 pipeline
        pipeline_result = await self.input_pipeline.process(
            multimodal_inputs or {"text": user_input}
        )
        
        # Update emotional state
        emotional_state = await self.emotional_continuity.update_emotional_state(
            pipeline_result["processed_inputs"],
            self.interaction_context
        )
        
        # Add to conversational memory
        memory_segment = await self.memory.add_memory(
            content=user_input,
            context={
                "pipeline_result": pipeline_result,
                "emotional_state": emotional_state,
                "mode": self.current_mode
            },
            importance=pipeline_result.get("priority_score", 0.5)
        )
        
        # Get conversation context
        conversation_context = await self.memory.get_conversation_context(
            depth=self.settings["memory_depth"]
        )
        
        # Detect interaction mode
        self.current_mode = await self._detect_interaction_mode(
            user_input,
            conversation_context,
            emotional_state
        )
        
        # Generate natural response
        response = await self._generate_natural_response(
            user_input,
            pipeline_result,
            conversation_context,
            emotional_state
        )
        
        # Update interaction tracking
        self.interaction_context["turns"] += 1
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return {
            "response": response,
            "mode": self.current_mode.value,
            "emotional_state": {
                "current": emotional_state.primary_emotion.value,
                "intensity": emotional_state.intensity,
                "trajectory": await self._get_emotional_trajectory()
            },
            "context": {
                "active_topics": conversation_context["active_topics"],
                "conversation_duration": conversation_context["conversation_duration"],
                "related_memories": await self._get_related_memories(memory_segment)
            },
            "performance": {
                "processing_time": processing_time,
                "memory_utilization": len(self.memory.working_memory),
                "emotional_stability": await self._calculate_emotional_stability()
            }
        }
    
    async def handle_conversation_flow(self,
                                     user_input: str,
                                     interrupt: bool = False) -> str:
        """Handle natural conversation flow including interruptions"""
        
        if interrupt and self.interaction_context.get("last_response"):
            # Handle interruption naturally
            return await self.language_flow.handle_interruption(
                self.interaction_context["last_response"],
                user_input
            )
        
        # Normal flow
        result = await self.process_interaction(user_input)
        self.interaction_context["last_response"] = result["response"]
        
        return result["response"]
    
    async def get_interaction_insights(self) -> Dict[str, Any]:
        """Get insights about current interaction"""
        
        # Detect patterns
        emotional_patterns = await self.emotional_continuity.detect_emotional_patterns()
        memory_patterns = await self.memory.consolidate_memories()
        
        # Analyze satisfaction
        satisfaction = await self._analyze_interaction_satisfaction()
        
        return {
            "emotional_patterns": emotional_patterns,
            "conversation_topics": list(self.memory.active_topics),
            "interaction_quality": {
                "satisfaction_score": satisfaction,
                "engagement_level": self._calculate_engagement(),
                "rapport_score": await self._calculate_rapport()
            },
            "recommendations": await self._generate_interaction_recommendations()
        }
    
    async def adapt_personality(self, feedback: Dict[str, Any]):
        """Adapt personality based on user feedback"""
        
        # Update personality traits
        if "preferred_style" in feedback:
            style_map = {
                "casual": {"formality": 0.2, "humor": 0.5},
                "professional": {"formality": 0.8, "humor": 0.2},
                "friendly": {"enthusiasm": 0.7, "empathy": 0.8}
            }
            
            if feedback["preferred_style"] in style_map:
                self.settings["personality_traits"].update(
                    style_map[feedback["preferred_style"]]
                )
        
        # Update response preferences
        if "verbosity" in feedback:
            self.language_flow.verbosity_preference = feedback["verbosity"]
        
        # Update emotional baseline
        if "emotional_preference" in feedback:
            self.emotional_continuity.user_baseline.update(feedback["emotional_preference"])
    
    async def _detect_interaction_mode(self,
                                     user_input: str,
                                     context: Dict[str, Any],
                                     emotional_state: Any) -> InteractionMode:
        """Detect current interaction mode"""
        
        # Crisis detection
        if emotional_state.intensity > 0.8 and emotional_state.valence < -0.5:
            return InteractionMode.CRISIS
        
        # Flow state detection
        if context.get("conversation_duration", 0) > 20 and emotional_state.arousal < 0.4:
            return InteractionMode.FLOW
        
        # Task detection
        task_keywords = ["do", "create", "make", "build", "fix", "solve"]
        if any(keyword in user_input.lower() for keyword in task_keywords):
            return InteractionMode.TASK
        
        # Learning detection
        learning_keywords = ["how", "why", "explain", "understand", "learn"]
        if any(keyword in user_input.lower() for keyword in learning_keywords):
            return InteractionMode.LEARNING
        
        # Creative detection
        creative_keywords = ["imagine", "what if", "idea", "design", "create"]
        if any(keyword in user_input.lower() for keyword in creative_keywords):
            return InteractionMode.CREATIVE
        
        return InteractionMode.CONVERSATION
    
    async def _generate_natural_response(self,
                                       user_input: str,
                                       pipeline_result: Dict[str, Any],
                                       conversation_context: Dict[str, Any],
                                       emotional_state: Any) -> str:
        """Generate natural response considering all factors"""
        
        # Get empathetic response strategy
        empathy_response = await self.emotional_continuity.get_empathetic_response(
            emotional_state,
            conversation_context
        )
        
        # Determine intent (simplified)
        intent = {
            "type": self.current_mode.value,
            "confidence": 0.8,
            "entities": conversation_context.get("active_topics", [])
        }
        
        # Generate response with natural flow
        response = await self.language_flow.generate_response(
            user_input,
            intent,
            conversation_context,
            {
                "valence": emotional_state.valence,
                "arousal": emotional_state.arousal,
                "intensity": emotional_state.intensity
            }
        )
        
        # Apply empathetic adjustments
        if empathy_response["intervention_level"] in ["high", "moderate"]:
            # Prepend empathetic acknowledgment
            empathy_phrase = empathy_response["suggested_phrases"][0] if empathy_response["suggested_phrases"] else ""
            if empathy_phrase:
                response = f"{empathy_phrase} {response}"
        
        # Apply personality
        response = await self.language_flow.inject_personality(
            response,
            self.settings["personality_traits"]
        )
        
        return response
    
    async def _get_emotional_trajectory(self) -> List[str]:
        """Get predicted emotional trajectory"""
        
        predictions = await self.emotional_continuity.predict_emotional_trajectory(
            self.emotional_continuity.current_state,
            time_horizon=5
        )
        
        return [
            f"{p.primary_emotion.value} ({p.confidence:.1f})"
            for p in predictions[:3]
        ]
    
    async def _get_related_memories(self, 
                                  current_memory: Any,
                                  limit: int = 3) -> List[Dict[str, Any]]:
        """Get memories related to current interaction"""
        
        related = await self.memory.find_related_memories(current_memory, limit)
        
        return [
            {
                "content": m.content,
                "timestamp": m.timestamp.isoformat(),
                "relevance": m.importance
            }
            for m in related
        ]
    
    async def _calculate_emotional_stability(self) -> float:
        """Calculate emotional stability score"""
        
        if len(self.emotional_continuity.emotional_history) < 3:
            return 1.0
        
        # Check variance in recent emotions
        recent = self.emotional_continuity.emotional_history[-5:]
        intensities = [s.intensity for s in recent]
        
        # Lower variance = higher stability
        variance = np.std(intensities) if intensities else 0
        stability = 1.0 - min(variance, 1.0)
        
        return stability
    
    async def _analyze_interaction_satisfaction(self) -> float:
        """Analyze user satisfaction with interaction"""
        
        # Factors indicating satisfaction
        factors = {
            "positive_emotions": 0.0,
            "engagement_duration": 0.0,
            "topic_coherence": 0.0,
            "response_acceptance": 0.0
        }
        
        # Check emotional valence
        if self.emotional_continuity.current_state.valence > 0:
            factors["positive_emotions"] = 0.3
        
        # Check engagement duration
        duration = self.memory._get_conversation_duration()
        if duration > 5:
            factors["engagement_duration"] = min(0.3, duration / 20)
        
        # Check topic coherence
        transitions = self.memory._get_topic_transitions()
        if len(transitions) < 3:
            factors["topic_coherence"] = 0.2
        
        # Assume response acceptance (would track in production)
        factors["response_acceptance"] = 0.2
        
        return sum(factors.values())
    
    def _calculate_engagement(self) -> float:
        """Calculate user engagement level"""
        
        if not self.memory.working_memory:
            return 0.0
        
        # Factors for engagement
        recent_memories = list(self.memory.working_memory)[-5:]
        
        # Question frequency
        questions = sum(1 for m in recent_memories if "?" in m.content)
        question_rate = questions / len(recent_memories) if recent_memories else 0
        
        # Response length (longer = more engaged)
        avg_length = sum(len(m.content.split()) for m in recent_memories) / len(recent_memories)
        length_score = min(1.0, avg_length / 20)
        
        # Topic diversity
        topics = set()
        for m in recent_memories:
            topics.update(m.topics)
        diversity_score = min(1.0, len(topics) / 5)
        
        return (question_rate + length_score + diversity_score) / 3
    
    async def _calculate_rapport(self) -> float:
        """Calculate rapport score"""
        
        # Simplified rapport calculation
        factors = {
            "emotional_sync": 0.0,
            "conversation_flow": 0.0,
            "mutual_understanding": 0.0
        }
        
        # Emotional synchrony
        if self.emotional_continuity.current_state.valence > -0.3:
            factors["emotional_sync"] = 0.4
        
        # Smooth conversation flow
        if self.interaction_context["turns"] > 5:
            factors["conversation_flow"] = 0.3
        
        # Assume understanding (would use NLU in production)
        factors["mutual_understanding"] = 0.3
        
        return sum(factors.values())
    
    async def _generate_interaction_recommendations(self) -> List[str]:
        """Generate recommendations for improving interaction"""
        
        recommendations = []
        
        # Check emotional state
        if self.emotional_continuity.current_state.intensity > 0.7:
            recommendations.append("Consider calming techniques or breaks")
        
        # Check engagement
        if self._calculate_engagement() < 0.3:
            recommendations.append("Try asking more engaging questions")
        
        # Check mode distribution
        mode_counts = {}
        for m in self.memory.working_memory:
            mode = m.context.get("mode", InteractionMode.CONVERSATION)
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        if len(mode_counts) == 1:
            recommendations.append("Vary interaction types for richer experience")
        
        return recommendations


# Example usage demonstration
async def demonstrate_natural_interaction():
    """Demonstrate Phase 5 Natural Interaction capabilities"""
    
    print("\n🌟 JARVIS Phase 5: Natural Interaction Flow Demo")
    print("=" * 60)
    
    # Initialize system
    interaction_core = NaturalInteractionCore()
    
    # Simulate natural conversation
    conversations = [
        # Opening
        {
            "user": "Hey JARVIS, I'm feeling a bit overwhelmed with this project",
            "multimodal": {
                "voice": {"features": {"pitch_variance": 0.6, "speaking_rate": 0.8}},
                "biometric": {"heart_rate": 85, "skin_conductance": 0.6}
            }
        },
        # Follow-up
        {
            "user": "There's just so much to do and I don't know where to start",
            "multimodal": {
                "voice": {"features": {"pitch_variance": 0.7, "volume": 0.4}},
                "biometric": {"heart_rate": 90, "skin_conductance": 0.7}
            }
        },
        # Interruption
        {
            "user": "Actually wait... can you just help me organize my tasks first?",
            "multimodal": {
                "voice": {"features": {"speaking_rate": 1.2}},
                "biometric": {"heart_rate": 88}
            }
        },
        # Continuation
        {
            "user": "I have the presentation, the code review, and three meetings",
            "multimodal": {
                "voice": {"features": {"pitch_variance": 0.5, "speaking_rate": 1.0}},
                "biometric": {"heart_rate": 82}
            }
        },
        # Emotional shift
        {
            "user": "You know what, breaking it down like this actually helps. Thanks!",
            "multimodal": {
                "voice": {"features": {"pitch_variance": 0.4, "volume": 0.6}},
                "biometric": {"heart_rate": 75, "skin_conductance": 0.4}
            }
        }
    ]
    
    # Process conversations
    for i, conv in enumerate(conversations):
        print(f"\n👤 User: {conv['user']}")
        
        # Check if this is an interruption
        is_interruption = "wait" in conv["user"].lower() or "actually" in conv["user"].lower()
        
        # Process interaction
        result = await interaction_core.process_interaction(
            conv["user"],
            conv.get("multimodal", {})
        )
        
        print(f"🤖 JARVIS: {result['response']}")
        print(f"   [Mode: {result['mode']}, Emotion: {result['emotional_state']['current']}]")
        
        await asyncio.sleep(1)
    
    # Show interaction insights
    print("\n\n📊 Interaction Analysis:")
    print("=" * 60)
    
    insights = await interaction_core.get_interaction_insights()
    
    print(f"\n🎭 Emotional Journey:")
    for pattern in insights["emotional_patterns"][:3]:
        print(f"  - {pattern}")
    
    print(f"\n💬 Conversation Topics:")
    for topic in insights["conversation_topics"]:
        print(f"  - {topic}")
    
    print(f"\n📈 Interaction Quality:")
    quality = insights["interaction_quality"]
    print(f"  - Satisfaction: {quality['satisfaction_score']:.2f}")
    print(f"  - Engagement: {quality['engagement_level']:.2f}")
    print(f"  - Rapport: {quality['rapport_score']:.2f}")
    
    print(f"\n💡 Recommendations:")
    for rec in insights["recommendations"]:
        print(f"  - {rec}")
    
    # Demonstrate memory recall
    print("\n\n🧠 Conversational Memory Demo:")
    print("=" * 60)
    
    # Recall related memories
    memories = await interaction_core.memory.recall(
        "project organization",
        context={"activity": "planning"}
    )
    
    print(f"\nRelated memories found: {len(memories)}")
    for mem in memories[:3]:
        print(f"  - {mem.timestamp.strftime('%H:%M')}: {mem.content[:50]}...")
    
    # Show emotional continuity
    print("\n\n❤️ Emotional Continuity:")
    print("=" * 60)
    
    trajectory = await interaction_core.emotional_continuity.predict_emotional_trajectory(
        interaction_core.emotional_continuity.current_state,
        time_horizon=5
    )
    
    print("Predicted emotional trajectory (next 5 minutes):")
    for i, state in enumerate(trajectory):
        print(f"  +{i+1} min: {state.primary_emotion.value} "
              f"(intensity: {state.intensity:.2f})")
    
    print("\n✨ Natural Interaction Flow Successfully Demonstrated!")
    print("The system maintains context, tracks emotions, and responds naturally.")

if __name__ == "__main__":
    # Run demonstration
    import numpy as np  # For stability calculation
    asyncio.run(demonstrate_natural_interaction())
