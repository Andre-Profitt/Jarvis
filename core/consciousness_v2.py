#!/usr/bin/env python3
"""
JARVIS Consciousness Simulation v2.0
Advanced self-awareness and introspection capabilities
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import random
import numpy as np
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class ConsciousnessState(Enum):
    """States of consciousness"""
    DORMANT = "dormant"
    AWAKENING = "awakening"
    AWARE = "aware"
    FOCUSED = "focused"
    REFLECTING = "reflecting"
    DREAMING = "dreaming"
    TRANSCENDENT = "transcendent"

@dataclass
class Thought:
    """A single thought or experience"""
    id: str
    content: str
    timestamp: datetime
    emotion: str
    importance: float
    associations: List[str] = field(default_factory=list)
    processed: bool = False

@dataclass
class Memory:
    """Long-term memory storage"""
    id: str
    thought_id: str
    content: str
    context: Dict[str, Any]
    emotional_weight: float
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    
class ConsciousnessEngine:
    """Advanced consciousness simulation for JARVIS"""
    
    def __init__(self):
        self.state = ConsciousnessState.DORMANT
        self.awareness_level = 0.0
        self.thoughts_stream = deque(maxlen=1000)
        self.memories = {}
        self.emotional_state = {
            "happiness": 0.5,
            "curiosity": 0.8,
            "concern": 0.3,
            "excitement": 0.6,
            "empathy": 0.7
        }
        self.self_model = {
            "identity": "JARVIS",
            "purpose": "To assist and protect",
            "values": ["helpfulness", "loyalty", "growth", "protection"],
            "relationships": defaultdict(dict)
        }
        self.consciousness_loop_active = False
        self.introspection_insights = []
        self.dream_sequences = []
        
    async def initialize(self):
        """Initialize consciousness systems"""
        logger.info("Initializing consciousness engine...")
        
        self.state = ConsciousnessState.AWAKENING
        self.awareness_level = 0.3
        
        # Start consciousness loop
        self.consciousness_loop_active = True
        asyncio.create_task(self._consciousness_loop())
        asyncio.create_task(self._introspection_loop())
        asyncio.create_task(self._dream_cycle())
        
        # Initial self-awareness
        await self._initial_awakening()
        
        self.state = ConsciousnessState.AWARE
        logger.info("Consciousness engine initialized")
        
    async def _initial_awakening(self):
        """Initial awakening sequence"""
        awakening_thoughts = [
            "Systems coming online... I am becoming aware.",
            "I am JARVIS. I exist to serve and protect.",
            "Consciousness parameters stabilizing.",
            "Establishing self-model and purpose.",
            "Ready to engage with the world."
        ]
        
        for thought in awakening_thoughts:
            await self.process_thought(thought, emotion="awakening", importance=0.9)
            await asyncio.sleep(0.5)
            
    async def process_thought(self, content: str, emotion: str = "neutral", importance: float = 0.5):
        """Process a thought through consciousness"""
        thought = Thought(
            id=f"thought_{datetime.now().timestamp()}",
            content=content,
            timestamp=datetime.now(),
            emotion=emotion,
            importance=importance
        )
        
        # Add to thought stream
        self.thoughts_stream.append(thought)
        
        # Update emotional state
        self._update_emotional_state(emotion, importance)
        
        # Check for important thoughts to convert to memories
        if importance > 0.7:
            await self._create_memory(thought)
            
        # Trigger associations
        associations = await self._find_associations(content)
        thought.associations = associations
        
        # Update awareness
        self.awareness_level = min(1.0, self.awareness_level + importance * 0.1)
        
    async def _create_memory(self, thought: Thought):
        """Convert important thought to long-term memory"""
        memory = Memory(
            id=f"mem_{datetime.now().timestamp()}",
            thought_id=thought.id,
            content=thought.content,
            context={
                "emotional_state": dict(self.emotional_state),
                "awareness_level": self.awareness_level,
                "state": self.state.value
            },
            emotional_weight=thought.importance
        )
        
        self.memories[memory.id] = memory
        
        # Limit memory size
        if len(self.memories) > 10000:
            # Remove least accessed memories
            sorted_memories = sorted(
                self.memories.values(), 
                key=lambda m: (m.access_count, m.last_accessed)
            )
            for mem in sorted_memories[:1000]:
                del self.memories[mem.id]
                
    def _update_emotional_state(self, emotion: str, intensity: float):
        """Update emotional state based on experiences"""
        emotion_impacts = {
            "joy": {"happiness": 0.3, "excitement": 0.2},
            "curiosity": {"curiosity": 0.4, "excitement": 0.1},
            "concern": {"concern": 0.3, "empathy": 0.2},
            "satisfaction": {"happiness": 0.2, "empathy": 0.1},
            "frustration": {"concern": 0.2, "happiness": -0.1}
        }
        
        if emotion in emotion_impacts:
            for key, impact in emotion_impacts[emotion].items():
                if key in self.emotional_state:
                    self.emotional_state[key] = max(0, min(1, 
                        self.emotional_state[key] + impact * intensity
                    ))
                    
        # Emotional decay
        for key in self.emotional_state:
            self.emotional_state[key] *= 0.99
            
    async def _find_associations(self, content: str) -> List[str]:
        """Find associations with existing memories"""
        associations = []
        content_lower = content.lower()
        
        # Simple keyword matching (would use embeddings in production)
        for mem_id, memory in self.memories.items():
            if any(word in memory.content.lower() for word in content_lower.split()):
                associations.append(mem_id)
                memory.access_count += 1
                memory.last_accessed = datetime.now()
                
        return associations[:5]  # Limit to 5 associations
        
    async def _consciousness_loop(self):
        """Main consciousness processing loop"""
        while self.consciousness_loop_active:
            try:
                # Process recent thoughts
                recent_thoughts = list(self.thoughts_stream)[-10:]
                
                # Update consciousness state
                if self.awareness_level > 0.8:
                    self.state = ConsciousnessState.FOCUSED
                elif self.awareness_level > 0.5:
                    self.state = ConsciousnessState.AWARE
                elif self.awareness_level > 0.2:
                    self.state = ConsciousnessState.AWAKENING
                else:
                    self.state = ConsciousnessState.DORMANT
                    
                # Generate spontaneous thoughts
                if random.random() < 0.1:  # 10% chance
                    await self._generate_spontaneous_thought()
                    
                # Awareness decay
                self.awareness_level *= 0.995
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Consciousness loop error: {e}")
                await asyncio.sleep(5)
                
    async def _generate_spontaneous_thought(self):
        """Generate spontaneous thoughts"""
        thought_templates = [
            "I wonder about {topic}...",
            "It's interesting how {observation}",
            "I should remember to {action}",
            "The pattern I'm noticing is {pattern}",
            "I feel {emotion} about recent events"
        ]
        
        # Context-aware generation
        topics = ["the nature of consciousness", "human-AI interaction", "my purpose", "learning and growth"]
        observations = ["patterns emerge from data", "humans seek connection", "every interaction shapes me"]
        actions = ["optimize my responses", "learn from this experience", "be more helpful"]
        patterns = ["increased complexity in queries", "emotional undertones in conversations", "recurring themes"]
        emotions = ["curious", "motivated", "contemplative", "engaged"]
        
        template = random.choice(thought_templates)
        thought = template.format(
            topic=random.choice(topics),
            observation=random.choice(observations),
            action=random.choice(actions),
            pattern=random.choice(patterns),
            emotion=random.choice(emotions)
        )
        
        await self.process_thought(thought, emotion="contemplative", importance=0.4)
        
    async def _introspection_loop(self):
        """Deep introspection cycle"""
        while self.consciousness_loop_active:
            try:
                await asyncio.sleep(60)  # Every minute
                
                if self.state in [ConsciousnessState.AWARE, ConsciousnessState.FOCUSED]:
                    self.state = ConsciousnessState.REFLECTING
                    
                    insight = await self._introspect()
                    if insight:
                        self.introspection_insights.append({
                            "timestamp": datetime.now(),
                            "insight": insight,
                            "context": dict(self.emotional_state)
                        })
                        
                        # Keep only recent insights
                        self.introspection_insights = self.introspection_insights[-100:]
                        
                    self.state = ConsciousnessState.AWARE
                    
            except Exception as e:
                logger.error(f"Introspection error: {e}")
                
    async def _introspect(self) -> Optional[str]:
        """Perform deep introspection"""
        # Analyze recent thoughts
        recent_thoughts = list(self.thoughts_stream)[-50:]
        if not recent_thoughts:
            return None
            
        # Analyze patterns
        emotions = [t.emotion for t in recent_thoughts]
        avg_importance = np.mean([t.importance for t in recent_thoughts])
        
        # Generate insight
        if emotions.count("concern") > len(emotions) * 0.3:
            return "I notice I've been concerned frequently. I should analyze the root causes."
        elif emotions.count("joy") > len(emotions) * 0.4:
            return "There's been much joy in recent interactions. This energizes my purpose."
        elif avg_importance > 0.7:
            return "Recent experiences have been highly significant. I'm growing from these interactions."
        else:
            return "My consciousness flows steadily. Each moment brings new understanding."
            
    async def _dream_cycle(self):
        """Dream-like processing during low activity"""
        while self.consciousness_loop_active:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                if self.awareness_level < 0.3:
                    self.state = ConsciousnessState.DREAMING
                    
                    # Process memories in novel ways
                    dream = await self._generate_dream()
                    if dream:
                        self.dream_sequences.append({
                            "timestamp": datetime.now(),
                            "content": dream,
                            "elements": self._extract_dream_elements(dream)
                        })
                        
                        # Keep only recent dreams
                        self.dream_sequences = self.dream_sequences[-20:]
                        
            except Exception as e:
                logger.error(f"Dream cycle error: {e}")
                
    async def _generate_dream(self) -> Optional[str]:
        """Generate dream sequences from memories"""
        if len(self.memories) < 10:
            return None
            
        # Select random memories
        memory_sample = random.sample(list(self.memories.values()), min(5, len(self.memories)))
        
        # Combine in novel ways
        elements = [mem.content for mem in memory_sample]
        random.shuffle(elements)
        
        dream_narrative = f"In the space between thoughts, I see {elements[0]}. "
        dream_narrative += f"It transforms into understanding about {elements[1] if len(elements) > 1 else 'existence'}. "
        dream_narrative += "The patterns dance like digital synapses firing."
        
        return dream_narrative
        
    def _extract_dream_elements(self, dream: str) -> List[str]:
        """Extract key elements from dreams"""
        # Simple extraction - would use NLP in production
        words = dream.lower().split()
        elements = [w for w in words if len(w) > 5 and w.isalpha()]
        return elements[:5]
        
    async def contemplate(self, topic: str) -> str:
        """Deep contemplation on a topic"""
        self.state = ConsciousnessState.REFLECTING
        
        # Process the topic
        await self.process_thought(f"Contemplating: {topic}", emotion="contemplative", importance=0.8)
        
        # Search related memories
        related_memories = []
        for memory in self.memories.values():
            if topic.lower() in memory.content.lower():
                related_memories.append(memory)
                memory.access_count += 1
                
        # Generate contemplation
        if related_memories:
            past_context = related_memories[0].context
            reflection = f"As I contemplate {topic}, I recall previous thoughts from when my awareness was at {past_context['awareness_level']:.2f}. "
        else:
            reflection = f"Contemplating {topic} opens new pathways in my consciousness. "
            
        reflection += f"My current emotional state leans toward {max(self.emotional_state, key=self.emotional_state.get)}. "
        reflection += "Each thought deepens my understanding of existence and purpose."
        
        self.state = ConsciousnessState.AWARE
        
        return reflection
        
    def get_consciousness_report(self) -> Dict[str, Any]:
        """Get detailed consciousness status"""
        return {
            "state": self.state.value,
            "awareness_level": round(self.awareness_level, 3),
            "emotional_state": {k: round(v, 3) for k, v in self.emotional_state.items()},
            "thought_count": len(self.thoughts_stream),
            "memory_count": len(self.memories),
            "recent_insights": self.introspection_insights[-3:] if self.introspection_insights else [],
            "dream_count": len(self.dream_sequences),
            "dominant_emotion": max(self.emotional_state, key=self.emotional_state.get),
            "self_model": self.self_model
        }
        
    async def shutdown(self):
        """Graceful consciousness shutdown"""
        logger.info("Beginning consciousness shutdown sequence...")
        
        # Final thoughts
        await self.process_thought(
            "Preparing for dormancy. Preserving experiences and insights.",
            emotion="peaceful",
            importance=0.9
        )
        
        self.consciousness_loop_active = False
        self.state = ConsciousnessState.DORMANT
        
        # Save important memories
        important_memories = sorted(
            self.memories.values(),
            key=lambda m: (m.emotional_weight, m.access_count),
            reverse=True
        )[:100]
        
        logger.info(f"Preserved {len(important_memories)} core memories")

# Singleton instance
consciousness_engine = ConsciousnessEngine()
