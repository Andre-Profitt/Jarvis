#!/usr/bin/env python3
"""
JARVIS Phase 5: Emotional Continuity System
===========================================

Maintains emotional context across interactions, ensuring JARVIS
responds with appropriate emotional intelligence and memory.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import json
import numpy as np
from collections import deque
import pickle

logger = logging.getLogger(__name__)


class EmotionalMomentum(Enum):
    """Direction of emotional change"""
    IMPROVING = auto()
    STABLE = auto()
    DECLINING = auto()
    VOLATILE = auto()


@dataclass
class EmotionalMemory:
    """A memory with emotional context"""
    content: str
    emotion: str
    intensity: float
    timestamp: datetime
    context: Dict[str, Any]
    resolved: bool = False
    followup_needed: bool = False


@dataclass
class EmotionalJourney:
    """Tracks user's emotional journey over time"""
    start_emotion: str
    current_emotion: str
    peak_emotion: Tuple[str, float]  # (emotion, intensity)
    low_emotion: Tuple[str, float]
    duration: float  # hours
    key_moments: List[EmotionalMemory]
    momentum: EmotionalMomentum
    patterns: List[str]


@dataclass
class ConversationThread:
    """Maintains context across related conversations"""
    thread_id: str
    topic: str
    emotional_context: Dict[str, Any]
    start_time: datetime
    last_interaction: datetime
    messages: List[Dict[str, Any]]
    unresolved_concerns: List[str]
    positive_moments: List[str]


class EmotionalContinuitySystem:
    """
    Maintains emotional continuity across all interactions,
    ensuring JARVIS remembers and responds appropriately to
    the user's emotional journey
    """
    
    def __init__(self):
        self.emotional_history = deque(maxlen=1000)
        self.conversation_threads: Dict[str, ConversationThread] = {}
        self.emotional_memories: List[EmotionalMemory] = []
        self.current_journey: Optional[EmotionalJourney] = None
        self.emotional_patterns = self._initialize_patterns()
        self.response_templates = self._load_response_templates()
        
        # Emotional state persistence
        self.short_term_memory = deque(maxlen=50)  # Last 50 interactions
        self.long_term_patterns = {}  # Persistent patterns
        self.emotional_anchors = {}  # Key emotional memories
        
    def _initialize_patterns(self) -> Dict[str, Any]:
        """Initialize emotional pattern recognition"""
        return {
            "stress_triggers": [],
            "joy_triggers": [],
            "productivity_patterns": {},
            "social_patterns": {},
            "time_based_patterns": {},
            "coping_mechanisms": [],
            "support_preferences": []
        }
    
    def _load_response_templates(self) -> Dict[str, List[str]]:
        """Load emotionally aware response templates"""
        return {
            "acknowledging_stress": [
                "I remember you were feeling stressed about {topic} earlier. How is that going?",
                "Last time we talked about {topic}, it seemed challenging. Any progress?",
                "I've been thinking about what you said regarding {topic}. How are you feeling about it now?"
            ],
            "celebrating_progress": [
                "This is wonderful progress from when you were struggling with {topic}!",
                "I'm so glad to see you've overcome {challenge}. You should be proud!",
                "Remember when {topic} was stressing you out? Look how far you've come!"
            ],
            "offering_continuity": [
                "Would you like to continue where we left off with {topic}?",
                "I noticed we didn't finish discussing {topic}. Shall we revisit that?",
                "There were some unresolved points about {topic}. Ready to tackle them?"
            ],
            "emotional_check_in": [
                "How are you feeling today compared to yesterday?",
                "I noticed you've been {emotion} lately. How are things?",
                "It's been a while since you felt {positive_emotion}. What would help?"
            ],
            "pattern_observation": [
                "I've noticed you tend to feel {emotion} during {time_pattern}. Is that the case today?",
                "This reminds me of when you faced {similar_situation}. What worked then?",
                "You usually handle {situation} by {coping_mechanism}. Would that help now?"
            ]
        }
    
    async def process_emotional_input(
        self,
        text: str,
        detected_emotion: str,
        intensity: float,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process input with emotional continuity"""
        
        # Create emotional memory
        memory = EmotionalMemory(
            content=text,
            emotion=detected_emotion,
            intensity=intensity,
            timestamp=datetime.now(),
            context=context
        )
        
        # Add to history
        self.emotional_history.append(memory)
        self.short_term_memory.append(memory)
        
        # Update or create conversation thread
        thread = await self._update_conversation_thread(memory, context)
        
        # Analyze emotional journey
        journey_analysis = await self._analyze_emotional_journey()
        
        # Check for patterns
        patterns = await self._check_emotional_patterns(memory)
        
        # Generate continuity-aware response elements
        response_elements = await self._generate_response_elements(
            memory, thread, journey_analysis, patterns
        )
        
        return {
            "memory_id": f"em_{datetime.now().timestamp()}",
            "thread_id": thread.thread_id if thread else None,
            "emotional_context": {
                "current": detected_emotion,
                "intensity": intensity,
                "momentum": journey_analysis.get("momentum", "stable"),
                "duration": journey_analysis.get("duration_hours", 0)
            },
            "continuity_elements": response_elements,
            "patterns_detected": patterns,
            "follow_up_needed": memory.followup_needed
        }
    
    async def _update_conversation_thread(
        self,
        memory: EmotionalMemory,
        context: Dict[str, Any]
    ) -> Optional[ConversationThread]:
        """Update or create conversation thread"""
        
        # Determine thread based on topic and time
        thread_key = await self._determine_thread_key(memory, context)
        
        if thread_key in self.conversation_threads:
            thread = self.conversation_threads[thread_key]
            thread.last_interaction = datetime.now()
            thread.messages.append({
                "timestamp": memory.timestamp,
                "emotion": memory.emotion,
                "content": memory.content
            })
            
            # Update emotional context
            thread.emotional_context["latest_emotion"] = memory.emotion
            thread.emotional_context["intensity_trend"] = self._calculate_intensity_trend(thread)
            
        else:
            # Create new thread
            thread = ConversationThread(
                thread_id=thread_key,
                topic=context.get("topic", "general"),
                emotional_context={
                    "start_emotion": memory.emotion,
                    "latest_emotion": memory.emotion,
                    "peak_intensity": memory.intensity
                },
                start_time=datetime.now(),
                last_interaction=datetime.now(),
                messages=[{
                    "timestamp": memory.timestamp,
                    "emotion": memory.emotion,
                    "content": memory.content
                }],
                unresolved_concerns=[],
                positive_moments=[]
            )
            self.conversation_threads[thread_key] = thread
        
        # Check for unresolved concerns
        if memory.intensity > 0.7 and memory.emotion in ["stressed", "anxious", "frustrated"]:
            thread.unresolved_concerns.append(memory.content)
        elif memory.intensity > 0.7 and memory.emotion in ["happy", "excited", "proud"]:
            thread.positive_moments.append(memory.content)
        
        return thread
    
    async def _determine_thread_key(self, memory: EmotionalMemory, context: Dict[str, Any]) -> str:
        """Determine which conversation thread this belongs to"""
        
        # Simple threading based on topic and time proximity
        topic = context.get("topic", "general")
        
        # Check recent threads
        recent_threads = [
            (key, thread) for key, thread in self.conversation_threads.items()
            if (datetime.now() - thread.last_interaction).total_seconds() < 3600  # Within last hour
        ]
        
        for key, thread in recent_threads:
            if thread.topic == topic:
                return key
        
        # Create new thread key
        return f"{topic}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    def _calculate_intensity_trend(self, thread: ConversationThread) -> str:
        """Calculate emotional intensity trend in thread"""
        if len(thread.messages) < 2:
            return "stable"
        
        # Get last 5 messages
        recent_messages = thread.messages[-5:]
        intensities = [msg.get("intensity", 0.5) for msg in recent_messages]
        
        # Calculate trend
        if len(intensities) >= 2:
            trend = np.polyfit(range(len(intensities)), intensities, 1)[0]
            if trend > 0.1:
                return "increasing"
            elif trend < -0.1:
                return "decreasing"
        
        return "stable"
    
    async def _analyze_emotional_journey(self) -> Dict[str, Any]:
        """Analyze the user's emotional journey"""
        
        if len(self.emotional_history) < 2:
            return {"status": "insufficient_data"}
        
        # Get recent emotional states
        recent_emotions = list(self.short_term_memory)
        
        # Calculate journey metrics
        start_emotion = recent_emotions[0].emotion
        current_emotion = recent_emotions[-1].emotion
        
        # Find peaks and valleys
        peak_intensity = max(em.intensity for em in recent_emotions)
        peak_emotion = next(em.emotion for em in recent_emotions if em.intensity == peak_intensity)
        
        low_intensity = min(em.intensity for em in recent_emotions)
        low_emotion = next(em.emotion for em in recent_emotions if em.intensity == low_intensity)
        
        # Calculate duration
        duration_hours = (recent_emotions[-1].timestamp - recent_emotions[0].timestamp).total_seconds() / 3600
        
        # Determine momentum
        momentum = self._calculate_emotional_momentum(recent_emotions)
        
        # Identify key moments
        key_moments = [em for em in recent_emotions if em.intensity > 0.7 or em.followup_needed]
        
        # Detect patterns
        patterns = self._detect_journey_patterns(recent_emotions)
        
        self.current_journey = EmotionalJourney(
            start_emotion=start_emotion,
            current_emotion=current_emotion,
            peak_emotion=(peak_emotion, peak_intensity),
            low_emotion=(low_emotion, low_intensity),
            duration=duration_hours,
            key_moments=key_moments,
            momentum=momentum,
            patterns=patterns
        )
        
        return {
            "status": "analyzed",
            "momentum": momentum.name,
            "duration_hours": duration_hours,
            "emotional_range": peak_intensity - low_intensity,
            "stability": 1.0 - np.std([em.intensity for em in recent_emotions]),
            "patterns": patterns
        }
    
    def _calculate_emotional_momentum(self, emotions: List[EmotionalMemory]) -> EmotionalMomentum:
        """Calculate the direction of emotional change"""
        
        if len(emotions) < 3:
            return EmotionalMomentum.STABLE
        
        # Get intensity values
        intensities = [em.intensity for em in emotions]
        
        # Calculate overall trend
        trend = np.polyfit(range(len(intensities)), intensities, 1)[0]
        
        # Calculate volatility
        volatility = np.std(intensities)
        
        if volatility > 0.3:
            return EmotionalMomentum.VOLATILE
        elif trend > 0.1:
            return EmotionalMomentum.IMPROVING
        elif trend < -0.1:
            return EmotionalMomentum.DECLINING
        else:
            return EmotionalMomentum.STABLE
    
    def _detect_journey_patterns(self, emotions: List[EmotionalMemory]) -> List[str]:
        """Detect patterns in emotional journey"""
        patterns = []
        
        # Time-based patterns
        hours = [em.timestamp.hour for em in emotions]
        if len(set(hours)) == 1:
            patterns.append(f"consistent_{hours[0]}h_interaction")
        
        # Emotion sequences
        emotion_sequence = [em.emotion for em in emotions]
        if len(set(emotion_sequence)) == 1:
            patterns.append(f"sustained_{emotion_sequence[0]}")
        elif "stressed" in emotion_sequence and "calm" in emotion_sequence:
            patterns.append("stress_recovery_cycle")
        
        # Intensity patterns
        intensities = [em.intensity for em in emotions]
        if all(i > 0.7 for i in intensities):
            patterns.append("high_intensity_period")
        elif all(i < 0.3 for i in intensities):
            patterns.append("low_intensity_period")
        
        return patterns
    
    async def _check_emotional_patterns(self, memory: EmotionalMemory) -> List[str]:
        """Check for recurring emotional patterns"""
        patterns_found = []
        
        # Check time-based patterns
        current_hour = memory.timestamp.hour
        time_key = f"hour_{current_hour}"
        
        if time_key in self.long_term_patterns:
            pattern = self.long_term_patterns[time_key]
            if pattern.get("dominant_emotion") == memory.emotion:
                patterns_found.append(f"typical_{memory.emotion}_at_{current_hour}h")
        
        # Check trigger patterns
        if memory.emotion == "stressed" and memory.intensity > 0.7:
            # Look for similar past situations
            similar_memories = [
                em for em in self.emotional_history
                if em.emotion == "stressed" and em.intensity > 0.7
            ]
            if len(similar_memories) > 3:
                patterns_found.append("recurring_stress_pattern")
        
        # Check recovery patterns
        if len(self.short_term_memory) > 5:
            recent = list(self.short_term_memory)[-5:]
            if recent[0].emotion in ["stressed", "anxious"] and recent[-1].emotion in ["calm", "happy"]:
                patterns_found.append("successful_recovery_pattern")
        
        return patterns_found
    
    async def _generate_response_elements(
        self,
        memory: EmotionalMemory,
        thread: Optional[ConversationThread],
        journey_analysis: Dict[str, Any],
        patterns: List[str]
    ) -> Dict[str, Any]:
        """Generate response elements that maintain emotional continuity"""
        
        elements = {
            "acknowledgments": [],
            "callbacks": [],
            "suggestions": [],
            "emotional_validation": [],
            "pattern_insights": []
        }
        
        # Acknowledge current emotional state
        if memory.intensity > 0.6:
            elements["emotional_validation"].append(
                f"I can sense that you're feeling quite {memory.emotion}."
            )
        
        # Reference previous interactions if relevant
        if thread and len(thread.messages) > 1:
            if thread.unresolved_concerns:
                concern = thread.unresolved_concerns[-1]
                elements["callbacks"].append(
                    self._select_template("acknowledging_stress", topic=concern)
                )
            
            if thread.positive_moments and memory.emotion in ["happy", "proud", "excited"]:
                moment = thread.positive_moments[-1]
                elements["callbacks"].append(
                    self._select_template("celebrating_progress", topic=moment)
                )
        
        # Add continuity based on journey
        if journey_analysis.get("momentum") == "DECLINING":
            elements["suggestions"].append(
                "I've noticed things have been getting tougher. What can I do to help?"
            )
        elif journey_analysis.get("momentum") == "IMPROVING":
            elements["acknowledgments"].append(
                "It's great to see things improving for you!"
            )
        
        # Pattern-based insights
        if "recurring_stress_pattern" in patterns:
            elements["pattern_insights"].append(
                "I've noticed this type of stress comes up periodically. Would you like to explore some strategies?"
            )
        elif "successful_recovery_pattern" in patterns:
            elements["pattern_insights"].append(
                "You're doing a great job managing your stress. Your coping strategies are working!"
            )
        
        # Check for follow-ups needed
        recent_concerns = [
            em for em in self.short_term_memory
            if em.followup_needed and not em.resolved
        ]
        
        if recent_concerns:
            for concern in recent_concerns[:2]:  # Max 2 follow-ups
                elements["callbacks"].append(
                    f"By the way, how did things go with what we discussed about {concern.context.get('topic', 'that matter')}?"
                )
        
        return elements
    
    def _select_template(self, category: str, **kwargs) -> str:
        """Select and fill a response template"""
        templates = self.response_templates.get(category, [])
        if not templates:
            return ""
        
        # Simple template selection (could be more sophisticated)
        template = templates[len(self.emotional_history) % len(templates)]
        
        # Fill in variables
        try:
            return template.format(**kwargs)
        except KeyError:
            return template
    
    async def get_emotional_context(self, user_id: str) -> Dict[str, Any]:
        """Get current emotional context for user"""
        
        if not self.emotional_history:
            return {
                "status": "no_history",
                "recommendation": "normal_interaction"
            }
        
        recent_emotions = list(self.short_term_memory)[-10:] if self.short_term_memory else []
        
        context = {
            "current_emotion": recent_emotions[-1].emotion if recent_emotions else "neutral",
            "recent_emotions": [em.emotion for em in recent_emotions],
            "average_intensity": np.mean([em.intensity for em in recent_emotions]) if recent_emotions else 0.5,
            "emotional_momentum": self.current_journey.momentum.name if self.current_journey else "STABLE",
            "active_threads": len([t for t in self.conversation_threads.values() 
                                 if (datetime.now() - t.last_interaction).total_seconds() < 3600]),
            "unresolved_concerns": sum(len(t.unresolved_concerns) for t in self.conversation_threads.values()),
            "recent_positive_moments": sum(len(t.positive_moments) for t in self.conversation_threads.values())
        }
        
        # Add recommendations
        if context["average_intensity"] > 0.7:
            context["recommendation"] = "gentle_supportive"
        elif context["emotional_momentum"] == "DECLINING":
            context["recommendation"] = "proactive_support"
        elif context["recent_positive_moments"] > 3:
            context["recommendation"] = "celebrate_progress"
        else:
            context["recommendation"] = "normal_interaction"
        
        return context
    
    async def resolve_concern(self, concern_id: str) -> bool:
        """Mark a concern as resolved"""
        for memory in self.emotional_memories:
            if f"em_{memory.timestamp.timestamp()}" == concern_id:
                memory.resolved = True
                memory.followup_needed = False
                return True
        return False
    
    async def save_emotional_state(self, filepath: str) -> None:
        """Save emotional state for persistence"""
        state = {
            "patterns": self.emotional_patterns,
            "long_term_patterns": self.long_term_patterns,
            "emotional_anchors": self.emotional_anchors,
            "conversation_threads": {
                k: {
                    "topic": v.topic,
                    "emotional_context": v.emotional_context,
                    "unresolved_concerns": v.unresolved_concerns,
                    "positive_moments": v.positive_moments
                }
                for k, v in self.conversation_threads.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    async def load_emotional_state(self, filepath: str) -> None:
        """Load emotional state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.emotional_patterns = state.get("patterns", self.emotional_patterns)
            self.long_term_patterns = state.get("long_term_patterns", {})
            self.emotional_anchors = state.get("emotional_anchors", {})
            
            # Reconstruct conversation threads
            for key, thread_data in state.get("conversation_threads", {}).items():
                self.conversation_threads[key] = ConversationThread(
                    thread_id=key,
                    topic=thread_data["topic"],
                    emotional_context=thread_data["emotional_context"],
                    start_time=datetime.now(),  # Approximate
                    last_interaction=datetime.now(),
                    messages=[],
                    unresolved_concerns=thread_data["unresolved_concerns"],
                    positive_moments=thread_data["positive_moments"]
                )
        
        except Exception as e:
            logger.error(f"Failed to load emotional state: {e}")
    
    def get_emotional_summary(self) -> Dict[str, Any]:
        """Get summary of emotional patterns and insights"""
        
        if not self.emotional_history:
            return {"status": "no_data"}
        
        recent_emotions = list(self.short_term_memory)
        
        # Count emotions
        emotion_counts = {}
        for em in recent_emotions:
            emotion_counts[em.emotion] = emotion_counts.get(em.emotion, 0) + 1
        
        # Most common emotion
        if emotion_counts:
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        else:
            dominant_emotion = "neutral"
        
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": emotion_counts,
            "average_intensity": np.mean([em.intensity for em in recent_emotions]) if recent_emotions else 0.5,
            "emotional_range": max([em.intensity for em in recent_emotions]) - min([em.intensity for em in recent_emotions]) if recent_emotions else 0,
            "pattern_count": len(self.emotional_patterns),
            "active_concerns": sum(1 for em in self.emotional_memories if em.followup_needed and not em.resolved),
            "positive_moments": sum(len(t.positive_moments) for t in self.conversation_threads.values()),
            "conversation_threads": len(self.conversation_threads),
            "emotional_stability": 1.0 - np.std([em.intensity for em in recent_emotions]) if len(recent_emotions) > 1 else 1.0
        }


# Testing function
async def test_emotional_continuity():
    """Test the emotional continuity system"""
    system = EmotionalContinuitySystem()
    
    print("Testing Emotional Continuity System")
    print("=" * 50)
    
    # Simulate a conversation with emotional progression
    test_inputs = [
        ("I'm really stressed about this project deadline", "stressed", 0.8, {"topic": "work_project"}),
        ("The requirements keep changing and I'm falling behind", "frustrated", 0.9, {"topic": "work_project"}),
        ("I managed to complete the first module though", "relieved", 0.6, {"topic": "work_project"}),
        ("Actually, I think I can finish this on time", "hopeful", 0.7, {"topic": "work_project"}),
        ("Just submitted the project! It went really well!", "happy", 0.9, {"topic": "work_project"}),
    ]
    
    for i, (text, emotion, intensity, context) in enumerate(test_inputs):
        print(f"\n--- Interaction {i+1} ---")
        print(f"User: {text}")
        print(f"Detected: {emotion} (intensity: {intensity})")
        
        result = await system.process_emotional_input(text, emotion, intensity, context)
        
        print(f"Thread ID: {result['thread_id']}")
        print(f"Momentum: {result['emotional_context']['momentum']}")
        
        elements = result['continuity_elements']
        if elements['acknowledgments']:
            print(f"Acknowledgments: {elements['acknowledgments']}")
        if elements['callbacks']:
            print(f"Callbacks: {elements['callbacks']}")
        if elements['pattern_insights']:
            print(f"Insights: {elements['pattern_insights']}")
        
        await asyncio.sleep(0.1)  # Simulate time passing
    
    # Test pattern detection
    print("\n\n--- Testing Pattern Detection ---")
    
    # Simulate recurring stress pattern
    stress_inputs = [
        ("Another deadline approaching and I'm anxious", "anxious", 0.7, {"topic": "work"}),
        ("Why does this always happen to me?", "frustrated", 0.8, {"topic": "work"}),
        ("I need to find a better way to manage these", "stressed", 0.75, {"topic": "work"}),
    ]
    
    for text, emotion, intensity, context in stress_inputs:
        result = await system.process_emotional_input(text, emotion, intensity, context)
        if result['patterns_detected']:
            print(f"Patterns detected: {result['patterns_detected']}")
    
    # Get emotional summary
    print("\n\n--- Emotional Summary ---")
    summary = system.get_emotional_summary()
    print(json.dumps(summary, indent=2))
    
    # Test emotional context retrieval
    print("\n\n--- Current Emotional Context ---")
    context = await system.get_emotional_context("test_user")
    print(json.dumps(context, indent=2))
    
    # Save state
    await system.save_emotional_state("/tmp/emotional_state_test.json")
    print("\n✅ Emotional state saved")


if __name__ == "__main__":
    asyncio.run(test_emotional_continuity())
