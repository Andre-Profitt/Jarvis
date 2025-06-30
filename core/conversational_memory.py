"""
JARVIS Phase 5: Conversational Memory System
Advanced context persistence and memory management
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from enum import Enum

@dataclass
class MemorySegment:
    """Individual memory segment with context"""
    content: str
    timestamp: datetime
    context: Dict[str, Any]
    importance: float = 0.5
    emotional_tone: str = "neutral"
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)

class MemoryType(Enum):
    WORKING = "working"        # Current conversation (last 5 minutes)
    SHORT_TERM = "short_term"  # Recent interactions (last 24 hours)
    LONG_TERM = "long_term"    # Important memories (permanent)
    EPISODIC = "episodic"      # Event-based memories
    SEMANTIC = "semantic"      # Fact-based memories

class ConversationalMemory:
    """Sophisticated memory system for natural conversations"""
    
    def __init__(self, memory_window_minutes: int = 30):
        self.memory_window = memory_window_minutes
        
        # Multi-level memory storage
        self.working_memory = deque(maxlen=20)  # Last 20 exchanges
        self.short_term_memory = deque(maxlen=200)  # Last 200 exchanges
        self.long_term_memory = []  # Unlimited important memories
        
        # Context tracking
        self.active_topics = set()
        self.conversation_threads = defaultdict(list)
        self.entity_memory = defaultdict(list)
        self.emotional_trajectory = []
        
        # Memory indexes for fast retrieval
        self.topic_index = defaultdict(list)
        self.time_index = defaultdict(list)
        self.importance_index = []
        
        # Conversation state
        self.current_context = {
            "topics": [],
            "entities": [],
            "emotional_state": "neutral",
            "activity": None,
            "last_exchange": None
        }
        
    async def add_memory(self, 
                        content: str, 
                        context: Dict[str, Any],
                        importance: float = 0.5) -> MemorySegment:
        """Add a new memory with intelligent processing"""
        
        # Create memory segment
        memory = MemorySegment(
            content=content,
            timestamp=datetime.now(),
            context=context,
            importance=importance,
            emotional_tone=self._analyze_emotional_tone(content),
            topics=self._extract_topics(content),
            entities=self._extract_entities(content),
            references=self._find_references(content)
        )
        
        # Add to appropriate memory stores
        self.working_memory.append(memory)
        self.short_term_memory.append(memory)
        
        # Promote to long-term if important
        if importance > 0.7:
            await self._promote_to_long_term(memory)
        
        # Update indexes
        await self._update_indexes(memory)
        
        # Update conversation state
        await self._update_conversation_state(memory)
        
        return memory
    
    async def recall(self, 
                    query: str, 
                    context: Optional[Dict[str, Any]] = None,
                    memory_types: List[MemoryType] = None) -> List[MemorySegment]:
        """Intelligent memory recall with relevance ranking"""
        
        if memory_types is None:
            memory_types = [MemoryType.WORKING, MemoryType.SHORT_TERM, MemoryType.LONG_TERM]
        
        candidates = []
        
        # Search different memory stores
        if MemoryType.WORKING in memory_types:
            candidates.extend(self.working_memory)
        
        if MemoryType.SHORT_TERM in memory_types:
            candidates.extend(self.short_term_memory)
        
        if MemoryType.LONG_TERM in memory_types:
            candidates.extend(self.long_term_memory)
        
        # Rank by relevance
        ranked_memories = await self._rank_memories(candidates, query, context)
        
        return ranked_memories[:10]  # Return top 10 most relevant
    
    async def get_conversation_context(self, depth: int = 5) -> Dict[str, Any]:
        """Get rich conversation context"""
        
        recent_memories = list(self.working_memory)[-depth:]
        
        return {
            "recent_exchanges": [
                {
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                    "emotional_tone": m.emotional_tone,
                    "topics": m.topics
                }
                for m in recent_memories
            ],
            "active_topics": list(self.active_topics),
            "current_emotional_state": self._get_current_emotional_state(),
            "conversation_duration": self._get_conversation_duration(),
            "topic_transitions": self._get_topic_transitions(),
            "entity_context": self._get_entity_context()
        }
    
    async def find_related_memories(self, 
                                   memory: MemorySegment, 
                                   limit: int = 5) -> List[MemorySegment]:
        """Find memories related to a given memory"""
        
        related = []
        
        # Search by topics
        for topic in memory.topics:
            related.extend(self.topic_index[topic])
        
        # Search by entities
        for entity in memory.entities:
            related.extend(self.entity_memory[entity])
        
        # Remove duplicates and original
        related = [m for m in set(related) if m != memory]
        
        # Rank by relevance
        related.sort(key=lambda m: self._calculate_relevance(memory, m), reverse=True)
        
        return related[:limit]
    
    async def consolidate_memories(self):
        """Consolidate short-term memories into long-term storage"""
        
        # Find important patterns in short-term memory
        patterns = await self._find_memory_patterns()
        
        for pattern in patterns:
            if pattern["importance"] > 0.6:
                # Create consolidated memory
                consolidated = MemorySegment(
                    content=pattern["summary"],
                    timestamp=datetime.now(),
                    context={"type": "consolidated", "source_memories": pattern["memories"]},
                    importance=pattern["importance"],
                    topics=pattern["topics"],
                    entities=pattern["entities"]
                )
                
                self.long_term_memory.append(consolidated)
    
    def _analyze_emotional_tone(self, content: str) -> str:
        """Analyze emotional tone of content"""
        
        content_lower = content.lower()
        
        # Simple emotion detection (can be enhanced with ML)
        if any(word in content_lower for word in ["happy", "great", "awesome", "love"]):
            return "positive"
        elif any(word in content_lower for word in ["sad", "upset", "angry", "hate"]):
            return "negative"
        elif any(word in content_lower for word in ["worried", "concerned", "anxious"]):
            return "concerned"
        elif any(word in content_lower for word in ["confused", "don't understand", "?"]):
            return "questioning"
        else:
            return "neutral"
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract topics from content"""
        
        topics = []
        content_lower = content.lower()
        
        # Topic patterns
        topic_keywords = {
            "coding": ["code", "function", "bug", "program", "script"],
            "meeting": ["meeting", "calendar", "schedule", "appointment"],
            "project": ["project", "task", "deadline", "milestone"],
            "health": ["health", "feeling", "tired", "energy", "sleep"],
            "learning": ["learn", "understand", "study", "tutorial"],
            "creative": ["design", "create", "idea", "brainstorm"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in content_lower for kw in keywords):
                topics.append(topic)
        
        return topics
    
    def _extract_entities(self, content: str) -> List[str]:
        """Extract named entities from content"""
        
        entities = []
        
        # Simple entity extraction (can be enhanced with NER)
        words = content.split()
        for i, word in enumerate(words):
            # Capitalized words that aren't sentence starts
            if word[0].isupper() and i > 0 and words[i-1][-1] not in '.!?':
                entities.append(word)
        
        return entities
    
    def _find_references(self, content: str) -> List[str]:
        """Find references to previous conversations"""
        
        references = []
        
        # Reference patterns
        ref_patterns = [
            "as we discussed",
            "like I mentioned",
            "remember when",
            "earlier you said",
            "that thing about",
            "referring to"
        ]
        
        content_lower = content.lower()
        for pattern in ref_patterns:
            if pattern in content_lower:
                references.append(pattern)
        
        return references
    
    async def _promote_to_long_term(self, memory: MemorySegment):
        """Promote important memory to long-term storage"""
        
        # Check if similar memory already exists
        similar = await self._find_similar_in_long_term(memory)
        
        if not similar:
            self.long_term_memory.append(memory)
        else:
            # Update existing memory importance
            similar.importance = max(similar.importance, memory.importance)
    
    async def _update_indexes(self, memory: MemorySegment):
        """Update memory indexes for fast retrieval"""
        
        # Topic index
        for topic in memory.topics:
            self.topic_index[topic].append(memory)
            self.active_topics.add(topic)
        
        # Time index
        time_key = memory.timestamp.strftime("%Y-%m-%d-%H")
        self.time_index[time_key].append(memory)
        
        # Entity index
        for entity in memory.entities:
            self.entity_memory[entity].append(memory)
        
        # Importance index
        if memory.importance > 0.6:
            self.importance_index.append(memory)
            self.importance_index.sort(key=lambda m: m.importance, reverse=True)
    
    async def _update_conversation_state(self, memory: MemorySegment):
        """Update current conversation state"""
        
        self.current_context["last_exchange"] = memory.timestamp
        self.current_context["topics"] = list(self.active_topics)
        self.current_context["emotional_state"] = self._get_current_emotional_state()
        
        # Track emotional trajectory
        self.emotional_trajectory.append({
            "timestamp": memory.timestamp,
            "emotion": memory.emotional_tone
        })
    
    async def _rank_memories(self, 
                           memories: List[MemorySegment], 
                           query: str, 
                           context: Optional[Dict[str, Any]]) -> List[MemorySegment]:
        """Rank memories by relevance to query"""
        
        scored_memories = []
        
        for memory in memories:
            score = 0.0
            
            # Content similarity
            if any(word in memory.content.lower() for word in query.lower().split()):
                score += 0.4
            
            # Topic match
            query_topics = self._extract_topics(query)
            topic_overlap = len(set(memory.topics) & set(query_topics))
            score += topic_overlap * 0.2
            
            # Recency bias
            age = (datetime.now() - memory.timestamp).total_seconds() / 3600  # hours
            recency_score = 1.0 / (1.0 + age / 24)  # Decay over 24 hours
            score += recency_score * 0.2
            
            # Importance
            score += memory.importance * 0.2
            
            # Context match
            if context and "activity" in context:
                if context["activity"] in memory.topics:
                    score += 0.3
            
            scored_memories.append((score, memory))
        
        # Sort by score
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        return [memory for score, memory in scored_memories]
    
    def _calculate_relevance(self, memory1: MemorySegment, memory2: MemorySegment) -> float:
        """Calculate relevance between two memories"""
        
        score = 0.0
        
        # Topic overlap
        topic_overlap = len(set(memory1.topics) & set(memory2.topics))
        score += topic_overlap * 0.3
        
        # Entity overlap
        entity_overlap = len(set(memory1.entities) & set(memory2.entities))
        score += entity_overlap * 0.3
        
        # Temporal proximity
        time_diff = abs((memory1.timestamp - memory2.timestamp).total_seconds())
        if time_diff < 300:  # Within 5 minutes
            score += 0.4
        elif time_diff < 3600:  # Within an hour
            score += 0.2
        
        return score
    
    def _get_current_emotional_state(self) -> str:
        """Get current emotional state from trajectory"""
        
        if not self.emotional_trajectory:
            return "neutral"
        
        # Get last 5 emotions
        recent_emotions = self.emotional_trajectory[-5:]
        
        # Count occurrences
        emotion_counts = defaultdict(int)
        for entry in recent_emotions:
            emotion_counts[entry["emotion"]] += 1
        
        # Return most common
        return max(emotion_counts.items(), key=lambda x: x[1])[0]
    
    def _get_conversation_duration(self) -> float:
        """Get conversation duration in minutes"""
        
        if not self.working_memory:
            return 0.0
        
        first = self.working_memory[0].timestamp
        last = self.working_memory[-1].timestamp
        
        return (last - first).total_seconds() / 60
    
    def _get_topic_transitions(self) -> List[Dict[str, Any]]:
        """Track how topics transition in conversation"""
        
        transitions = []
        previous_topics = set()
        
        for memory in self.working_memory:
            current_topics = set(memory.topics)
            
            if previous_topics and current_topics != previous_topics:
                transitions.append({
                    "from": list(previous_topics),
                    "to": list(current_topics),
                    "timestamp": memory.timestamp.isoformat()
                })
            
            previous_topics = current_topics
        
        return transitions
    
    def _get_entity_context(self) -> Dict[str, List[str]]:
        """Get context for mentioned entities"""
        
        entity_context = {}
        
        for entity, memories in self.entity_memory.items():
            if memories:
                # Get last 3 mentions
                recent_mentions = sorted(memories, key=lambda m: m.timestamp, reverse=True)[:3]
                entity_context[entity] = [m.content for m in recent_mentions]
        
        return entity_context
    
    async def _find_memory_patterns(self) -> List[Dict[str, Any]]:
        """Find patterns in memory for consolidation"""
        
        patterns = []
        
        # Group memories by topic
        topic_groups = defaultdict(list)
        for memory in self.short_term_memory:
            for topic in memory.topics:
                topic_groups[topic].append(memory)
        
        # Find significant patterns
        for topic, memories in topic_groups.items():
            if len(memories) >= 3:  # At least 3 memories on topic
                pattern = {
                    "topic": topic,
                    "memories": memories,
                    "importance": sum(m.importance for m in memories) / len(memories),
                    "summary": f"Multiple discussions about {topic}",
                    "topics": [topic],
                    "entities": list(set(e for m in memories for e in m.entities))
                }
                patterns.append(pattern)
        
        return patterns
    
    async def _find_similar_in_long_term(self, memory: MemorySegment) -> Optional[MemorySegment]:
        """Find similar memory in long-term storage"""
        
        for ltm in self.long_term_memory:
            # Check topic similarity
            if set(memory.topics) & set(ltm.topics):
                # Check content similarity (simple approach)
                if any(word in ltm.content.lower() for word in memory.content.lower().split()):
                    return ltm
        
        return None
