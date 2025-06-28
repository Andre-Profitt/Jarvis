"""
Enhanced Conversational Memory Architecture with Episodic Memory System
Improvements include: better error handling, performance optimizations,
persistence layer, advanced retrieval algorithms, and richer memory representations.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set, Protocol, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import hashlib
import json
from collections import deque, defaultdict
import math
import heapq
import pickle
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import weakref
import os


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionType(Enum):
    """Extended emotion categories based on Plutchik's wheel of emotions"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"
    # Secondary emotions (combinations)
    LOVE = "love"  # Joy + Trust
    SUBMISSION = "submission"  # Trust + Fear
    AWE = "awe"  # Surprise + Fear
    DISAPPROVAL = "disapproval"  # Surprise + Sadness
    REMORSE = "remorse"  # Sadness + Disgust
    CONTEMPT = "contempt"  # Disgust + Anger
    AGGRESSIVENESS = "aggressiveness"  # Anger + Anticipation
    OPTIMISM = "optimism"  # Anticipation + Joy


class MemoryStrength(Enum):
    """Memory consolidation levels with more granular stages"""
    SENSORY = 0.1  # < 1 second
    ICONIC = 0.2  # Visual sensory memory
    ECHOIC = 0.25  # Auditory sensory memory
    SHORT_TERM = 0.3  # < 30 seconds
    WORKING = 0.5  # Active processing
    RECENT = 0.65  # Recently consolidated
    LONG_TERM = 0.8  # Consolidated
    REMOTE = 0.9  # Well-established
    PERMANENT = 1.0  # Highly reinforced


class RetrievalStrategy(Enum):
    """Different memory retrieval strategies"""
    ASSOCIATIVE = auto()
    TEMPORAL = auto()
    EMOTIONAL = auto()
    SEMANTIC = auto()
    CONTEXTUAL = auto()
    PATTERN_COMPLETION = auto()  # Hippocampal pattern completion
    CUE_DEPENDENT = auto()  # Environmental cue-based


@dataclass
class EmotionalContext:
    """Enhanced emotional tagging with mixed emotions support"""
    primary_emotion: EmotionType
    secondary_emotions: List[Tuple[EmotionType, float]] = field(default_factory=list)
    intensity: float = 0.5  # 0.0 to 1.0
    valence: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    arousal: float = 0.5  # 0.0 (calm) to 1.0 (excited)
    dominance: float = 0.5  # 0.0 (submissive) to 1.0 (dominant)
    confidence: float = 0.8
    mixed_emotion_score: float = 0.0  # Degree of emotional conflict
    
    def __post_init__(self):
        # Calculate mixed emotion score based on opposing emotions
        if self.secondary_emotions:
            # Check for conflicting emotions
            primary_valence = self._get_emotion_valence(self.primary_emotion)
            conflicts = []
            for emotion, weight in self.secondary_emotions:
                secondary_valence = self._get_emotion_valence(emotion)
                if primary_valence * secondary_valence < 0:  # Opposite signs
                    conflicts.append(weight)
            if conflicts:
                self.mixed_emotion_score = sum(conflicts) / len(self.secondary_emotions)
    
    def _get_emotion_valence(self, emotion: EmotionType) -> float:
        """Get valence for basic emotions"""
        positive = {EmotionType.JOY, EmotionType.TRUST, EmotionType.ANTICIPATION, 
                   EmotionType.LOVE, EmotionType.OPTIMISM}
        negative = {EmotionType.SADNESS, EmotionType.ANGER, EmotionType.FEAR, 
                   EmotionType.DISGUST, EmotionType.REMORSE, EmotionType.CONTEMPT}
        
        if emotion in positive:
            return 1.0
        elif emotion in negative:
            return -1.0
        else:
            return 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert emotional context to vector representation"""
        # Primary emotion one-hot encoding
        emotion_vec = np.zeros(len(EmotionType))
        emotion_vec[list(EmotionType).index(self.primary_emotion)] = self.intensity
        
        # Add secondary emotions
        for emotion, weight in self.secondary_emotions:
            idx = list(EmotionType).index(emotion)
            emotion_vec[idx] += weight * 0.5  # Secondary emotions have less weight
        
        # Normalize emotion vector
        if np.sum(emotion_vec) > 0:
            emotion_vec = emotion_vec / np.sum(emotion_vec)
        
        return np.concatenate([
            emotion_vec,
            [self.valence, self.arousal, self.dominance, 
             self.confidence, self.mixed_emotion_score]
        ])


@dataclass
class SensoryModality:
    """Represents different sensory modalities in memory"""
    visual: Optional[Any] = None
    auditory: Optional[Any] = None
    linguistic: Optional[str] = None
    spatial: Optional[Dict] = None
    temporal: Optional[datetime] = None
    
    def has_content(self) -> bool:
        return any([self.visual, self.auditory, self.linguistic, 
                   self.spatial, self.temporal])


@dataclass
class MemoryChunk:
    """Enhanced memory chunk with richer representation"""
    content: Any
    timestamp: datetime
    chunk_id: str
    modality: SensoryModality = field(default_factory=SensoryModality)
    semantic_embedding: Optional[np.ndarray] = None
    attention_weight: float = 0.5
    rehearsal_count: int = 0
    
    def __post_init__(self):
        if self.chunk_id is None:
            content_str = str(self.content) + str(self.timestamp)
            self.chunk_id = hashlib.md5(content_str.encode()).hexdigest()[:8]
    
    def rehearse(self):
        """Rehearsal strengthens memory chunk"""
        self.rehearsal_count += 1
        self.attention_weight = min(1.0, self.attention_weight + 0.1)


@dataclass
class MemoryContext:
    """Rich contextual information for memories"""
    location: Optional[str] = None
    participants: List[str] = field(default_factory=list)
    activity: Optional[str] = None
    goals: List[str] = field(default_factory=list)
    environmental_cues: Dict[str, Any] = field(default_factory=dict)
    cognitive_state: Dict[str, float] = field(default_factory=dict)  # attention, fatigue, etc.


@dataclass
class EpisodicMemory:
    """Enhanced episodic memory with richer features"""
    memory_id: str
    chunks: List[MemoryChunk]
    emotional_context: EmotionalContext
    memory_context: MemoryContext
    creation_time: datetime
    last_accessed: datetime
    access_count: int = 1
    importance_score: float = 0.5
    consolidation_level: MemoryStrength = MemoryStrength.SHORT_TERM
    associations: Set[str] = field(default_factory=set)
    context_tags: List[str] = field(default_factory=list)
    
    # New fields
    vividness: float = 0.5  # How vivid/detailed the memory is
    confidence: float = 0.8  # Confidence in memory accuracy
    source_monitoring: str = "self"  # Source of memory (self, other, media, etc.)
    retrieval_count: int = 0  # Number of times retrieved
    last_rehearsal: Optional[datetime] = None
    distortions: List[Dict] = field(default_factory=list)  # Track memory changes
    
    def calculate_retention_probability(self, current_time: datetime) -> float:
        """Enhanced forgetting curve with multiple factors"""
        time_elapsed = (current_time - self.last_accessed).total_seconds() / 3600
        
        # Base strength includes consolidation and importance
        base_strength = self.consolidation_level.value * self.importance_score
        
        # Emotional modulation - stronger emotions slow forgetting
        emotional_modifier = 1 + (self.emotional_context.intensity * 
                                 self.emotional_context.arousal)
        
        # Mixed emotions can either strengthen or weaken memory
        if self.emotional_context.mixed_emotion_score > 0.5:
            emotional_modifier *= 1.2  # Conflicting emotions are memorable
        
        # Vividness bonus
        vividness_modifier = 1 + (self.vividness * 0.3)
        
        strength = base_strength * emotional_modifier * vividness_modifier
        
        # Calculate retention using forgetting curve
        retention = math.exp(-time_elapsed / (strength * 24))
        
        # Spaced repetition effect
        if self.access_count > 1:
            spacing_bonus = math.log(self.access_count) * 0.15
            retention = min(1.0, retention + spacing_bonus)
        
        # Recent rehearsal bonus
        if self.last_rehearsal:
            rehearsal_recency = (current_time - self.last_rehearsal).total_seconds() / 3600
            if rehearsal_recency < 24:  # Within last day
                retention = min(1.0, retention + 0.2)
        
        return max(0.0, min(1.0, retention))
    
    def update_access(self, current_time: datetime, retrieval_context: Optional[Dict] = None):
        """Enhanced access update with retrieval context"""
        self.last_accessed = current_time
        self.access_count += 1
        self.retrieval_count += 1
        
        # Progressive consolidation
        consolidation_thresholds = {
            3: MemoryStrength.RECENT,
            5: MemoryStrength.LONG_TERM,
            10: MemoryStrength.REMOTE,
            20: MemoryStrength.PERMANENT
        }
        
        for threshold, level in consolidation_thresholds.items():
            if self.access_count >= threshold and self.consolidation_level.value < level.value:
                self.consolidation_level = level
                logger.info(f"Memory {self.memory_id} consolidated to {level.name}")
                break
        
        # Track retrieval context for source monitoring
        if retrieval_context:
            self.distortions.append({
                "time": current_time,
                "context": retrieval_context,
                "type": "retrieval"
            })


class MemoryStorage(Protocol):
    """Protocol for memory storage backends"""
    async def save_memory(self, memory: EpisodicMemory) -> bool:
        ...
    
    async def load_memory(self, memory_id: str) -> Optional[EpisodicMemory]:
        ...
    
    async def delete_memory(self, memory_id: str) -> bool:
        ...
    
    async def list_memories(self, filter_criteria: Optional[Dict] = None) -> List[str]:
        ...


class InMemoryStorage:
    """Default in-memory storage implementation"""
    def __init__(self):
        self.storage: Dict[str, EpisodicMemory] = {}
    
    async def save_memory(self, memory: EpisodicMemory) -> bool:
        self.storage[memory.memory_id] = memory
        return True
    
    async def load_memory(self, memory_id: str) -> Optional[EpisodicMemory]:
        return self.storage.get(memory_id)
    
    async def delete_memory(self, memory_id: str) -> bool:
        if memory_id in self.storage:
            del self.storage[memory_id]
            return True
        return False
    
    async def list_memories(self, filter_criteria: Optional[Dict] = None) -> List[str]:
        if not filter_criteria:
            return list(self.storage.keys())
        
        # Simple filtering implementation
        filtered = []
        for memory_id, memory in self.storage.items():
            if self._matches_criteria(memory, filter_criteria):
                filtered.append(memory_id)
        return filtered
    
    def _matches_criteria(self, memory: EpisodicMemory, criteria: Dict) -> bool:
        # Implement filtering logic
        if "emotion" in criteria:
            if memory.emotional_context.primary_emotion != criteria["emotion"]:
                return False
        if "min_importance" in criteria:
            if memory.importance_score < criteria["min_importance"]:
                return False
        return True


class PersistentStorage(InMemoryStorage):
    """File-based persistent storage"""
    def __init__(self, storage_path: str = "./memory_storage"):
        super().__init__()
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    async def save_memory(self, memory: EpisodicMemory) -> bool:
        # Save to memory first
        await super().save_memory(memory)
        
        # Persist to disk
        file_path = f"{self.storage_path}/{memory.memory_id}.pkl"
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(memory, f)
            return True
        except Exception as e:
            logger.error(f"Failed to persist memory {memory.memory_id}: {e}")
            return False
    
    async def load_memory(self, memory_id: str) -> Optional[EpisodicMemory]:
        # Check in-memory first
        memory = await super().load_memory(memory_id)
        if memory:
            return memory
        
        # Load from disk
        file_path = f"{self.storage_path}/{memory_id}.pkl"
        try:
            with open(file_path, 'rb') as f:
                memory = pickle.load(f)
                self.storage[memory_id] = memory  # Cache in memory
                return memory
        except Exception as e:
            logger.error(f"Failed to load memory {memory_id}: {e}")
            return None


class WorkingMemory:
    """Enhanced working memory with cognitive load modeling"""
    def __init__(self, capacity: int = 7, variance: int = 2):
        self.base_capacity = capacity
        self.variance = variance
        self.current_capacity = capacity
        self.buffer: deque = deque(maxlen=capacity + variance)
        self.attention_weights: Dict[str, float] = {}
        self.rehearsal_buffer: deque = deque(maxlen=3)  # Phonological loop
        self.visuospatial_buffer: deque = deque(maxlen=4)  # Visuospatial sketchpad
        self.cognitive_load: float = 0.0
        self.central_executive_resources: float = 1.0
    
    def adjust_capacity(self, cognitive_load: float):
        """Dynamically adjust capacity based on cognitive load"""
        self.cognitive_load = max(0.0, min(1.0, cognitive_load))
        
        # Cognitive load reduces available capacity
        load_penalty = int(self.variance * self.cognitive_load)
        self.current_capacity = max(3, self.base_capacity - load_penalty)
        
        # Update central executive resources
        self.central_executive_resources = 1.0 - (self.cognitive_load * 0.5)
        
        # Prune buffer if over capacity
        self._maintain_capacity_constraint()
    
    def _maintain_capacity_constraint(self):
        """Maintain capacity constraints using attention-based pruning"""
        while len(self.buffer) > self.current_capacity:
            # Find chunk with lowest attention weight
            if not self.buffer:
                break
                
            min_chunk = min(
                self.buffer,
                key=lambda x: self.attention_weights.get(x.chunk_id, 0)
            )
            
            # Try to move to rehearsal buffer if linguistic
            if min_chunk.modality.linguistic and len(self.rehearsal_buffer) < 3:
                self.rehearsal_buffer.append(min_chunk)
            
            self.buffer.remove(min_chunk)
            self.attention_weights.pop(min_chunk.chunk_id, None)
    
    def add_chunk(self, chunk: MemoryChunk, attention_weight: float = 0.5):
        """Add chunk with modality-specific processing"""
        # Route to appropriate buffer based on modality
        if chunk.modality.visual or chunk.modality.spatial:
            if len(self.visuospatial_buffer) < 4:
                self.visuospatial_buffer.append(chunk)
        elif chunk.modality.linguistic or chunk.modality.auditory:
            if len(self.rehearsal_buffer) < 3:
                self.rehearsal_buffer.append(chunk)
        
        # Add to main buffer
        self.buffer.append(chunk)
        self.attention_weights[chunk.chunk_id] = attention_weight * self.central_executive_resources
        
        # Maintain constraints
        self._maintain_capacity_constraint()
    
    def rehearse(self):
        """Perform rehearsal to maintain items in memory"""
        # Rehearse items in rehearsal buffer
        for chunk in self.rehearsal_buffer:
            chunk.rehearse()
            # Boost attention for rehearsed items
            if chunk.chunk_id in self.attention_weights:
                self.attention_weights[chunk.chunk_id] = min(
                    1.0, 
                    self.attention_weights[chunk.chunk_id] + 0.1
                )
    
    def get_active_chunks(self) -> List[MemoryChunk]:
        """Get all active chunks across buffers"""
        all_chunks = list(self.buffer) + list(self.rehearsal_buffer) + list(self.visuospatial_buffer)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                unique_chunks.append(chunk)
        
        # Sort by attention weight
        return sorted(
            unique_chunks,
            key=lambda x: self.attention_weights.get(x.chunk_id, 0),
            reverse=True
        )


class SemanticMemory:
    """Enhanced semantic memory with hierarchical knowledge organization"""
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.knowledge_graph: Dict[str, Dict[str, Any]] = {}
        self.category_index: Dict[str, Set[str]] = defaultdict(set)
        self.concept_hierarchy: Dict[str, Set[str]] = defaultdict(set)  # Parent -> children
        self.compression_threshold = 0.8
        self.concept_embeddings: Dict[str, np.ndarray] = {}
        
        # Prototype theory - store category prototypes
        self.category_prototypes: Dict[str, np.ndarray] = {}
        
    def extract_semantic_features(self, episodic_memory: EpisodicMemory) -> Dict[str, Any]:
        """Extract semantic knowledge with concept hierarchy"""
        features = {
            "concepts": [],
            "relations": [],
            "attributes": {},
            "category": None,
            "prototype_distance": 0.0
        }
        
        # Extract concepts from memory chunks
        for chunk in episodic_memory.chunks:
            if chunk.modality.linguistic:
                # Extract concepts from linguistic content
                concepts = self._extract_concepts_from_text(chunk.modality.linguistic)
                features["concepts"].extend(concepts)
        
        # Categorize and compute prototype distance
        category = self._categorize_memory(episodic_memory)
        features["category"] = category
        
        if category in self.category_prototypes:
            # Compute distance to category prototype
            memory_vector = self._compute_memory_vector(episodic_memory)
            prototype = self.category_prototypes[category]
            features["prototype_distance"] = np.linalg.norm(memory_vector - prototype)
        
        return features
    
    def _extract_concepts_from_text(self, text: str) -> List[str]:
        """Extract concepts from text (simplified version)"""
        # In practice, use NLP libraries for concept extraction
        words = text.lower().split()
        # Filter out common words (simplified stopword removal)
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to"}
        concepts = [w for w in words if w not in stopwords and len(w) > 3]
        return concepts
    
    def _compute_memory_vector(self, memory: EpisodicMemory) -> np.ndarray:
        """Compute vector representation of memory"""
        # Combine emotional and semantic features
        emotional_vec = memory.emotional_context.to_vector()
        
        # Average semantic embeddings if available
        semantic_vecs = []
        for chunk in memory.chunks:
            if chunk.semantic_embedding is not None:
                semantic_vecs.append(chunk.semantic_embedding)
        
        if semantic_vecs:
            semantic_vec = np.mean(semantic_vecs, axis=0)
        else:
            semantic_vec = np.zeros(self.embedding_dim)
        
        # Concatenate and normalize
        combined = np.concatenate([emotional_vec, semantic_vec])
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
            
        return combined
    
    def update_category_prototype(self, category: str, memories: List[EpisodicMemory]):
        """Update category prototype based on exemplars"""
        if not memories:
            return
        
        # Compute average vector of category members
        vectors = []
        for memory in memories:
            vec = self._compute_memory_vector(memory)
            vectors.append(vec)
        
        prototype = np.mean(vectors, axis=0)
        self.category_prototypes[category] = prototype
        
    def _categorize_memory(self, memory: EpisodicMemory) -> str:
        """Categorize using prototype theory"""
        # If prototypes exist, find nearest prototype
        if self.category_prototypes:
            memory_vec = self._compute_memory_vector(memory)
            min_distance = float('inf')
            best_category = "uncategorized"
            
            for category, prototype in self.category_prototypes.items():
                distance = np.linalg.norm(memory_vec - prototype)
                if distance < min_distance:
                    min_distance = distance
                    best_category = category
            
            if min_distance < 0.5:  # Threshold for category membership
                return best_category
        
        # Fallback to rule-based categorization
        if memory.emotional_context.valence > 0.5:
            base_category = "positive_experience"
        elif memory.emotional_context.valence < -0.5:
            base_category = "negative_experience"
        else:
            base_category = "neutral_information"
        
        if memory.context_tags:
            return f"{base_category}_{memory.context_tags[0]}"
        return base_category
    
    async def compress_knowledge(self, memories: List[EpisodicMemory]):
        """Asynchronous knowledge compression with hierarchy building"""
        # Group by category
        category_groups = defaultdict(list)
        for memory in memories:
            category = self._categorize_memory(memory)
            category_groups[category].append(memory)
        
        # Process each category in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            compression_tasks = []
            for category, group in category_groups.items():
                if len(group) > 3:
                    task = executor.submit(self._compress_category, category, group)
                    compression_tasks.append(task)
            
            # Wait for all compressions to complete
            for task in compression_tasks:
                try:
                    compressed = task.result()
                    self.knowledge_graph[compressed["id"]] = compressed
                    self.category_index[compressed["category"]].add(compressed["id"])
                    
                    # Update category prototype
                    self.update_category_prototype(compressed["category"], 
                                                   category_groups[compressed["category"]])
                except Exception as e:
                    logger.error(f"Compression failed: {e}")
    
    def _compress_category(self, category: str, memories: List[EpisodicMemory]) -> Dict:
        """Compress memories in a category"""
        compressed = self._create_semantic_representation(memories)
        compressed["category"] = category
        
        # Build concept hierarchy
        all_concepts = []
        for memory in memories:
            features = self.extract_semantic_features(memory)
            all_concepts.extend(features["concepts"])
        
        # Find common concepts (potential parent concepts)
        concept_counts = defaultdict(int)
        for concept in all_concepts:
            concept_counts[concept] += 1
        
        # Concepts appearing in >50% of memories are parent concepts
        threshold = len(memories) * 0.5
        parent_concepts = [c for c, count in concept_counts.items() if count > threshold]
        
        for parent in parent_concepts:
            self.concept_hierarchy[parent].update(set(all_concepts) - {parent})
        
        compressed["parent_concepts"] = parent_concepts
        
        return compressed
    
    def _create_semantic_representation(self, memories: List[EpisodicMemory]) -> Dict:
        """Create enriched semantic representation"""
        rep = {
            "id": hashlib.md5(str(memories).encode()).hexdigest()[:12],
            "frequency": len(memories),
            "emotional_summary": self._summarize_emotions(memories),
            "key_concepts": self._extract_key_concepts(memories),
            "creation_time": datetime.now(),
            "source_episodes": [m.memory_id for m in memories],
            "confidence": np.mean([m.confidence for m in memories]),
            "temporal_span": (
                max(m.creation_time for m in memories) - 
                min(m.creation_time for m in memories)
            ).total_seconds()
        }
        
        return rep
    
    def _summarize_emotions(self, memories: List[EpisodicMemory]) -> Dict:
        """Comprehensive emotional summary"""
        emotions = [m.emotional_context for m in memories]
        
        # Track all emotions
        emotion_counts = defaultdict(int)
        total_weight = 0
        
        for ctx in emotions:
            emotion_counts[ctx.primary_emotion] += ctx.intensity
            total_weight += ctx.intensity
            
            for emotion, weight in ctx.secondary_emotions:
                emotion_counts[emotion] += weight * 0.5
                total_weight += weight * 0.5
        
        # Normalize
        emotion_distribution = {}
        if total_weight > 0:
            for emotion, count in emotion_counts.items():
                emotion_distribution[emotion.value] = count / total_weight
        
        return {
            "average_valence": np.mean([e.valence for e in emotions]),
            "average_arousal": np.mean([e.arousal for e in emotions]),
            "valence_variance": np.var([e.valence for e in emotions]),
            "dominant_emotion": max(emotion_counts.items(), key=lambda x: x[1])[0].value,
            "emotion_distribution": emotion_distribution,
            "mixed_emotion_frequency": np.mean([e.mixed_emotion_score for e in emotions])
        }
    
    def _extract_key_concepts(self, memories: List[EpisodicMemory]) -> List[Tuple[str, float]]:
        """Extract key concepts with importance scores"""
        concept_counts = defaultdict(int)
        concept_importance = defaultdict(float)
        
        for memory in memories:
            # Weight concepts by memory importance
            weight = memory.importance_score * memory.confidence
            
            for tag in memory.context_tags:
                concept_counts[tag] += 1
                concept_importance[tag] += weight
            
            # Extract from chunks
            for chunk in memory.chunks:
                if chunk.modality.linguistic:
                    concepts = self._extract_concepts_from_text(chunk.modality.linguistic)
                    for concept in concepts:
                        concept_counts[concept] += 1
                        concept_importance[concept] += weight * chunk.attention_weight
        
        # Normalize importance scores
        if concept_counts:
            max_count = max(concept_counts.values())
            results = []
            for concept, count in concept_counts.items():
                normalized_importance = concept_importance[concept] / count
                frequency_score = count / max_count
                combined_score = (normalized_importance + frequency_score) / 2
                results.append((concept, combined_score))
            
            # Return top concepts
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:10]
        
        return []


class EmotionalMemory:
    """Advanced emotional memory with pattern recognition"""
    def __init__(self):
        self.emotion_patterns: Dict[str, List[EmotionalContext]] = defaultdict(list)
        self.emotion_transitions: Dict[Tuple[EmotionType, EmotionType], int] = defaultdict(int)
        self.emotion_thresholds = {
            "high_arousal": 0.7,
            "strong_valence": 0.6,
            "tag_threshold": 0.5,
            "mixed_emotion": 0.3
        }
        self.emotion_regulation_strategies: List[str] = []
        
    def tag_memory(self, content: Any, context: Optional[Dict] = None) -> EmotionalContext:
        """Enhanced emotional tagging with mixed emotions"""
        # Analyze content for emotional indicators
        emotion_indicators = self._analyze_emotional_content(content, context)
        
        # Determine primary and secondary emotions
        emotion_scores = self._compute_emotion_scores(emotion_indicators)
        
        # Sort emotions by score
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Primary emotion
        primary_emotion = sorted_emotions[0][0] if sorted_emotions else EmotionType.NEUTRAL
        
        # Secondary emotions (those above threshold)
        secondary_emotions = []
        if len(sorted_emotions) > 1:
            primary_score = sorted_emotions[0][1]
            for emotion, score in sorted_emotions[1:]:
                if score > self.emotion_thresholds["mixed_emotion"] * primary_score:
                    secondary_emotions.append((emotion, score / primary_score))
        
        # Calculate emotional dimensions
        intensity = emotion_indicators.get("intensity", 0.5)
        valence = emotion_indicators.get("valence", 0.0)
        arousal = emotion_indicators.get("arousal", 0.5)
        dominance = emotion_indicators.get("dominance", 0.5)
        
        # Create emotional context
        emotional_context = EmotionalContext(
            primary_emotion=primary_emotion,
            secondary_emotions=secondary_emotions[:3],  # Limit to top 3
            intensity=intensity,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            confidence=self._calculate_confidence(emotion_scores)
        )
        
        # Track emotion patterns
        self._update_emotion_patterns(emotional_context)
        
        return emotional_context
    
    def _compute_emotion_scores(self, indicators: Dict) -> Dict[EmotionType, float]:
        """Compute scores for each emotion type"""
        scores = {}
        
        # Use a simple model based on arousal-valence-dominance
        valence = indicators.get("valence", 0)
        arousal = indicators.get("arousal", 0.5)
        dominance = indicators.get("dominance", 0.5)
        
        # Map to basic emotions (simplified Plutchik model)
        if valence > 0.3:
            if arousal > 0.6:
                scores[EmotionType.JOY] = valence * arousal
                if dominance > 0.6:
                    scores[EmotionType.AGGRESSIVENESS] = valence * arousal * dominance * 0.7
            else:
                scores[EmotionType.TRUST] = valence * (1 - arousal)
                scores[EmotionType.LOVE] = valence * (1 - arousal) * 0.5
        
        elif valence < -0.3:
            if arousal > 0.6:
                scores[EmotionType.ANGER] = abs(valence) * arousal * dominance
                scores[EmotionType.FEAR] = abs(valence) * arousal * (1 - dominance)
            else:
                scores[EmotionType.SADNESS] = abs(valence) * (1 - arousal)
                if dominance < 0.4:
                    scores[EmotionType.REMORSE] = abs(valence) * (1 - arousal) * (1 - dominance)
        
        else:  # Neutral valence
            if arousal > 0.7:
                scores[EmotionType.SURPRISE] = arousal
            else:
                scores[EmotionType.NEUTRAL] = 1 - arousal
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores
    
    def _calculate_confidence(self, emotion_scores: Dict[EmotionType, float]) -> float:
        """Calculate confidence based on emotion clarity"""
        if not emotion_scores:
            return 0.5
        
        # Higher confidence when one emotion dominates
        scores = list(emotion_scores.values())
        if len(scores) == 1:
            return 0.9
        
        # Calculate entropy
        entropy = -sum(s * math.log(s + 1e-10) for s in scores if s > 0)
        max_entropy = math.log(len(scores))
        
        # Lower entropy = higher confidence
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.8
        return max(0.3, min(1.0, confidence))
    
    def _update_emotion_patterns(self, emotional_context: EmotionalContext):
        """Track emotion patterns and transitions"""
        pattern_key = f"{emotional_context.primary_emotion.value}_{round(emotional_context.valence, 1)}"
        self.emotion_patterns[pattern_key].append(emotional_context)
        
        # Track transitions if we have previous emotions
        if pattern_key in self.emotion_patterns and len(self.emotion_patterns[pattern_key]) > 1:
            prev_emotion = self.emotion_patterns[pattern_key][-2].primary_emotion
            curr_emotion = emotional_context.primary_emotion
            self.emotion_transitions[(prev_emotion, curr_emotion)] += 1
    
    def _analyze_emotional_content(self, content: Any, context: Optional[Dict]) -> Dict:
        """Advanced emotional content analysis"""
        indicators = {
            "intensity": 0.5,
            "valence": 0.0,
            "arousal": 0.5,
            "dominance": 0.5
        }
        
        if context and "emotional_indicators" in context:
            indicators.update(context["emotional_indicators"])
        
        # Analyze linguistic content
        if isinstance(content, str):
            content_str = content.lower()
            
            # Extended emotion lexicons
            emotion_lexicons = {
                "joy": ["happy", "joy", "delighted", "excited", "wonderful", "amazing", "fantastic"],
                "sadness": ["sad", "depressed", "unhappy", "miserable", "lonely", "disappointed"],
                "anger": ["angry", "furious", "mad", "irritated", "annoyed", "outraged"],
                "fear": ["scared", "afraid", "terrified", "anxious", "worried", "nervous"],
                "surprise": ["surprised", "amazed", "astonished", "shocked", "unexpected"],
                "disgust": ["disgusted", "revolted", "repulsed", "sick", "awful"],
                "trust": ["trust", "confident", "secure", "reliable", "faith"],
                "anticipation": ["excited", "looking forward", "eager", "hopeful", "expecting"]
            }
            
            # Count emotion words
            emotion_counts = defaultdict(int)
            for emotion, words in emotion_lexicons.items():
                for word in words:
                    if word in content_str:
                        emotion_counts[emotion] += 1
            
            # Update indicators based on counts
            if emotion_counts:
                dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
                
                # Map to valence-arousal-dominance
                emotion_vad = {
                    "joy": (0.8, 0.7, 0.6),
                    "sadness": (-0.7, 0.3, 0.3),
                    "anger": (-0.6, 0.8, 0.8),
                    "fear": (-0.6, 0.7, 0.2),
                    "surprise": (0.1, 0.8, 0.5),
                    "disgust": (-0.5, 0.5, 0.6),
                    "trust": (0.5, 0.3, 0.6),
                    "anticipation": (0.3, 0.6, 0.5)
                }
                
                if dominant_emotion in emotion_vad:
                    v, a, d = emotion_vad[dominant_emotion]
                    indicators["valence"] = v
                    indicators["arousal"] = a
                    indicators["dominance"] = d
            
            # Analyze linguistic markers
            if "!" in content_str:
                indicators["arousal"] = min(1.0, indicators["arousal"] + 0.2)
                indicators["intensity"] = min(1.0, indicators["intensity"] + 0.1)
            
            if "?" in content_str:
                indicators["dominance"] = max(0.0, indicators["dominance"] - 0.1)
            
            if content_str.isupper():
                indicators["arousal"] = min(1.0, indicators["arousal"] + 0.3)
                indicators["intensity"] = min(1.0, indicators["intensity"] + 0.2)
        
        return indicators
    
    def predict_emotional_trajectory(self, current_emotion: EmotionType) -> List[Tuple[EmotionType, float]]:
        """Predict likely emotional transitions"""
        predictions = []
        
        # Use transition history
        total_transitions = 0
        transition_counts = defaultdict(int)
        
        for (from_emotion, to_emotion), count in self.emotion_transitions.items():
            if from_emotion == current_emotion:
                transition_counts[to_emotion] += count
                total_transitions += count
        
        # Calculate probabilities
        if total_transitions > 0:
            for emotion, count in transition_counts.items():
                probability = count / total_transitions
                predictions.append((emotion, probability))
        
        # Sort by probability
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # If no history, return general transitions
        if not predictions:
            # Default transitions based on emotion theory
            default_transitions = {
                EmotionType.JOY: [(EmotionType.TRUST, 0.3), (EmotionType.SURPRISE, 0.2)],
                EmotionType.SADNESS: [(EmotionType.ANGER, 0.2), (EmotionType.FEAR, 0.15)],
                EmotionType.ANGER: [(EmotionType.DISGUST, 0.25), (EmotionType.SADNESS, 0.2)],
                EmotionType.FEAR: [(EmotionType.SURPRISE, 0.3), (EmotionType.SADNESS, 0.2)]
            }
            predictions = default_transitions.get(current_emotion, [(EmotionType.NEUTRAL, 0.5)])
        
        return predictions[:5]


class RetrievalContext:
    """Context for memory retrieval operations"""
    def __init__(self,
                 strategy: RetrievalStrategy = RetrievalStrategy.ASSOCIATIVE,
                 time_window: Optional[timedelta] = None,
                 emotion_filter: Optional[EmotionType] = None,
                 min_confidence: float = 0.3,
                 max_results: int = 10,
                 include_associations: bool = True,
                 cue_words: Optional[List[str]] = None):
        self.strategy = strategy
        self.time_window = time_window or timedelta(days=30)
        self.emotion_filter = emotion_filter
        self.min_confidence = min_confidence
        self.max_results = max_results
        self.include_associations = include_associations
        self.cue_words = cue_words or []


class MemoryConsolidator:
    """Handles memory consolidation processes"""
    def __init__(self, semantic_memory: SemanticMemory):
        self.semantic_memory = semantic_memory
        self.consolidation_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        
    async def start(self):
        """Start the consolidation process"""
        self.is_running = True
        asyncio.create_task(self._consolidation_loop())
    
    async def stop(self):
        """Stop the consolidation process"""
        self.is_running = False
    
    async def _consolidation_loop(self):
        """Background consolidation process"""
        while self.is_running:
            try:
                # Get memories to consolidate
                memories = []
                while not self.consolidation_queue.empty() and len(memories) < 50:
                    memory = await self.consolidation_queue.get()
                    memories.append(memory)
                
                if memories:
                    # Perform consolidation
                    await self.semantic_memory.compress_knowledge(memories)
                    logger.info(f"Consolidated {len(memories)} memories")
                
                # Sleep to simulate natural consolidation intervals
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Consolidation error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute
    
    async def queue_for_consolidation(self, memory: EpisodicMemory):
        """Add memory to consolidation queue"""
        await self.consolidation_queue.put(memory)


class EpisodicMemorySystem:
    """Complete episodic memory system with all enhancements"""
    def __init__(self,
                 working_memory_capacity: int = 7,
                 storage_backend: Optional[MemoryStorage] = None,
                 embedding_model: Optional[Any] = None,
                 enable_persistence: bool = True):
        
        # Core components
        self.working_memory = WorkingMemory(capacity=working_memory_capacity)
        self.semantic_memory = SemanticMemory()
        self.emotional_memory = EmotionalMemory()
        
        # Storage backend
        if storage_backend:
            self.storage = storage_backend
        elif enable_persistence:
            self.storage = PersistentStorage()
        else:
            self.storage = InMemoryStorage()
        
        # Memory indices for fast retrieval
        self.temporal_index: Dict[datetime, List[str]] = defaultdict(list)
        self.emotion_index: Dict[EmotionType, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Consolidation system
        self.consolidator = MemoryConsolidator(self.semantic_memory)
        self.consolidation_interval = timedelta(hours=6)
        self.last_consolidation = datetime.now()
        
        # Optional embedding model
        self.embedding_model = embedding_model
        
        # Memory statistics
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.retrieval_performance: List[Dict] = []
        
        # Feature Association Matrix
        self.association_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Start background processes
        asyncio.create_task(self.consolidator.start())
        
    async def remember_interaction(self,
                                   interaction: Dict[str, Any],
                                   context: Optional[Dict] = None) -> EpisodicMemory:
        """Enhanced memory formation with error handling"""
        try:
            # Extract significant moments
            key_moments = await self._extract_significant_moments(interaction)
            
            if not key_moments:
                logger.warning("No significant moments found in interaction")
                return None
            
            # Create memory chunks
            chunks = self._create_memory_chunks(key_moments)
            
            # Create memory context
            memory_context = MemoryContext()
            if context:
                memory_context.location = context.get("location")
                memory_context.participants = context.get("participants", [])
                memory_context.activity = context.get("activity")
                memory_context.goals = context.get("goals", [])
                memory_context.environmental_cues = context.get("environmental_cues", {})
                memory_context.cognitive_state = context.get("cognitive_state", {})
            
            # Tag with emotions
            emotional_context = self.emotional_memory.tag_memory(interaction, context)
            
            # Calculate importance
            importance = self._calculate_importance(interaction, emotional_context, memory_context)
            
            # Create episodic memory
            memory = EpisodicMemory(
                memory_id=self._generate_memory_id(),
                chunks=chunks[:self.working_memory.current_capacity],
                emotional_context=emotional_context,
                memory_context=memory_context,
                creation_time=datetime.now(),
                last_accessed=datetime.now(),
                importance_score=importance,
                context_tags=context.get("tags", []) if context else [],
                vividness=self._calculate_vividness(chunks, emotional_context),
                source_monitoring=context.get("source", "self") if context else "self"
            )
            
            # Store memory
            await self.storage.save_memory(memory)
            
            # Update indices
            self._update_indices(memory)
            
            # Update working memory
            for chunk in chunks[:self.working_memory.current_capacity]:
                attention = emotional_context.intensity * 0.5 + importance * 0.5
                self.working_memory.add_chunk(chunk, attention)
            
            # Create associations
            await self._create_associations(memory)
            
            # Queue for consolidation if eligible
            if memory.importance_score > 0.6:
                await self.consolidator.queue_for_consolidation(memory)
            
            # Update access patterns
            self.access_patterns[memory.memory_id].append(datetime.now())
            
            logger.info(f"Created memory {memory.memory_id} with importance {importance:.2f}")
            
            return memory
            
        except Exception as e:
            logger.error(f"Failed to create memory: {e}")
            raise
    
    def _calculate_vividness(self, chunks: List[MemoryChunk], emotional_context: EmotionalContext) -> float:
        """Calculate memory vividness based on detail and emotion"""
        # Base vividness from number of modalities
        modality_count = 0
        for chunk in chunks:
            if chunk.modality.visual:
                modality_count += 1
            if chunk.modality.auditory:
                modality_count += 1
            if chunk.modality.linguistic:
                modality_count += 1
            if chunk.modality.spatial:
                modality_count += 1
        
        modality_score = min(1.0, modality_count / (len(chunks) * 2))
        
        # Emotional enhancement
        emotional_score = emotional_context.intensity * emotional_context.arousal
        
        # Combine scores
        vividness = (modality_score + emotional_score) / 2
        
        return max(0.1, min(1.0, vividness))
    
    def _update_indices(self, memory: EpisodicMemory):
        """Update all memory indices"""
        # Temporal index
        self.temporal_index[memory.creation_time].append(memory.memory_id)
        
        # Emotion index
        self.emotion_index[memory.emotional_context.primary_emotion].add(memory.memory_id)
        for emotion, _ in memory.emotional_context.secondary_emotions:
            self.emotion_index[emotion].add(memory.memory_id)
        
        # Tag index
        for tag in memory.context_tags:
            self.tag_index[tag].add(memory.memory_id)
    
    async def _extract_significant_moments(self, interaction: Dict) -> List[Dict]:
        """Extract significant moments with improved heuristics"""
        moments = []
        
        # Process different types of content
        if "highlights" in interaction:
            for highlight in interaction["highlights"]:
                moments.append({
                    "content": highlight,
                    "type": "highlight",
                    "timestamp": datetime.now(),
                    "significance": 0.8
                })
        
        if "messages" in interaction:
            for msg in interaction["messages"]:
                significance = self._calculate_message_significance(msg)
                if significance > 0.5:
                    moments.append({
                        "content": msg.get("content"),
                        "type": msg.get("type", "message"),
                        "timestamp": msg.get("timestamp", datetime.now()),
                        "significance": significance,
                        "metadata": msg.get("metadata", {})
                    })
        
        if "summary" in interaction:
            moments.append({
                "content": interaction["summary"],
                "type": "summary",
                "timestamp": datetime.now(),
                "significance": 0.7
            })
        
        # Sort by significance
        moments.sort(key=lambda x: x.get("significance", 0.5), reverse=True)
        
        return moments
    
    def _calculate_message_significance(self, message: Dict) -> float:
        """Calculate significance score for a message"""
        significance = 0.5  # Base score
        
        # Emotional content
        if "emotion_score" in message:
            significance = max(significance, message["emotion_score"])
        
        # Question detection
        content = str(message.get("content", ""))
        if "?" in content:
            significance += 0.2
        
        # Important markers
        importance_markers = [
            "important", "remember", "key", "critical", "significant",
            "don't forget", "please note", "crucial", "essential"
        ]
        content_lower = content.lower()
        for marker in importance_markers:
            if marker in content_lower:
                significance += 0.3
                break
        
        # Length heuristic (longer messages often more significant)
        word_count = len(content.split())
        if word_count > 20:
            significance += 0.1
        
        # Learning moments
        learning_markers = ["learned", "realized", "discovered", "understood"]
        for marker in learning_markers:
            if marker in content_lower:
                significance += 0.2
                break
        
        return min(1.0, significance)
    
    def _create_memory_chunks(self, moments: List[Dict]) -> List[MemoryChunk]:
        """Create memory chunks with multimodal support"""
        chunks = []
        
        for moment in moments:
            # Create sensory modality
            modality = SensoryModality()
            
            # Determine modality based on content type
            content = moment.get("content")
            if isinstance(content, str):
                modality.linguistic = content
            elif isinstance(content, dict):
                modality.linguistic = content.get("text", "")
                modality.visual = content.get("visual")
                modality.auditory = content.get("auditory")
                modality.spatial = content.get("spatial")
            
            modality.temporal = moment.get("timestamp", datetime.now())
            
            # Create chunk
            chunk = MemoryChunk(
                content=moment,
                timestamp=moment.get("timestamp", datetime.now()),
                chunk_id=None,
                modality=modality,
                attention_weight=moment.get("significance", 0.5)
            )
            
            # Add semantic embedding if available
            if self.embedding_model and modality.linguistic:
                chunk.semantic_embedding = self._get_embedding(modality.linguistic)
            
            chunks.append(chunk)
        
        return chunks
    
    def _get_embedding(self, content: str) -> Optional[np.ndarray]:
        """Get semantic embedding for content"""
        if self.embedding_model:
            try:
                # Placeholder - integrate actual embedding model
                # In practice, use sentence-transformers or similar
                embedding = np.random.randn(self.semantic_memory.embedding_dim)
                # Normalize
                embedding = embedding / np.linalg.norm(embedding)
                return embedding
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                return None
        return None
    
    def _calculate_importance(self,
                              interaction: Dict,
                              emotional_context: EmotionalContext,
                              memory_context: MemoryContext) -> float:
        """Calculate importance with multiple factors"""
        base_importance = 0.5
        
        # Emotional significance
        emotional_boost = emotional_context.intensity * 0.25
        if emotional_context.mixed_emotion_score > 0.5:
            emotional_boost += 0.1  # Mixed emotions are memorable
        
        # Novelty
        novelty_score = interaction.get("novelty_score", 0.5)
        novelty_boost = novelty_score * 0.15
        
        # Personal relevance
        relevance_score = interaction.get("personal_relevance", 0.5)
        relevance_boost = relevance_score * 0.2
        
        # Goal relevance
        if memory_context.goals:
            goal_relevance = interaction.get("goal_relevance", 0.5)
            relevance_boost += goal_relevance * 0.1
        
        # Social significance (interactions with multiple participants)
        if len(memory_context.participants) > 1:
            base_importance += 0.1
        
        # Surprise factor
        if emotional_context.primary_emotion == EmotionType.SURPRISE:
            base_importance += 0.15
        
        total_importance = base_importance + emotional_boost + novelty_boost + relevance_boost
        
        return max(0.1, min(1.0, total_importance))
    
    def _generate_memory_id(self) -> str:
        """Generate unique memory ID"""
        timestamp = datetime.now().isoformat()
        random_component = np.random.randint(0, 1000000)
        return hashlib.md5(f"{timestamp}_{random_component}".encode()).hexdigest()[:16]
    
    async def _create_associations(self, memory: EpisodicMemory):
        """Create associations with parallel processing"""
        # Find related memories
        related_memories = await self._find_related_memories(memory, top_k=10)
        
        # Create bidirectional associations
        for related_id, similarity in related_memories:
            if similarity > 0.3:  # Threshold
                memory.associations.add(related_id)
                
                # Update stored memory's associations
                related_memory = await self.storage.load_memory(related_id)
                if related_memory:
                    related_memory.associations.add(memory.memory_id)
                    await self.storage.save_memory(related_memory)
                
                # Update association matrix
                self.association_matrix[memory.memory_id][related_id] = similarity
                self.association_matrix[related_id][memory.memory_id] = similarity
    
    async def _find_related_memories(self,
                                     memory: EpisodicMemory,
                                     top_k: int = 10) -> List[Tuple[str, float]]:
        """Find related memories with improved similarity metrics"""
        candidates = []
        
        # Get candidate memories from indices
        candidate_ids = set()
        
        # Same emotion memories
        candidate_ids.update(self.emotion_index[memory.emotional_context.primary_emotion])
        
        # Same tag memories
        for tag in memory.context_tags:
            candidate_ids.update(self.tag_index[tag])
        
        # Temporally close memories
        time_window = timedelta(hours=24)
        for time, ids in self.temporal_index.items():
            if abs((time - memory.creation_time).total_seconds()) < time_window.total_seconds():
                candidate_ids.update(ids)
        
        # Remove self
        candidate_ids.discard(memory.memory_id)
        
        # Calculate similarities
        for candidate_id in candidate_ids:
            candidate = await self.storage.load_memory(candidate_id)
            if candidate:
                similarity = self._calculate_memory_similarity(memory, candidate)
                if similarity > 0.2:
                    candidates.append((candidate_id, similarity))
        
        # Sort and return top k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def _calculate_memory_similarity(self, mem1: EpisodicMemory, mem2: EpisodicMemory) -> float:
        """Calculate comprehensive similarity between memories"""
        similarity_components = []
        weights = []
        
        # Emotional similarity
        emotional_sim = self._calculate_emotional_similarity(
            mem1.emotional_context, 
            mem2.emotional_context
        )
        similarity_components.append(emotional_sim)
        weights.append(0.3)
        
        # Temporal proximity
        time_diff = abs((mem1.creation_time - mem2.creation_time).total_seconds())
        temporal_sim = math.exp(-time_diff / (7 * 24 * 3600))  # Decay over week
        similarity_components.append(temporal_sim)
        weights.append(0.2)
        
        # Semantic similarity
        if (mem1.chunks and mem2.chunks and 
            mem1.chunks[0].semantic_embedding is not None and
            mem2.chunks[0].semantic_embedding is not None):
            semantic_sim = self._calculate_semantic_similarity(
                mem1.chunks[0].semantic_embedding,
                mem2.chunks[0].semantic_embedding
            )
            similarity_components.append(semantic_sim)
            weights.append(0.3)
        
        # Tag overlap
        if mem1.context_tags and mem2.context_tags:
            tag_sim = len(set(mem1.context_tags) & set(mem2.context_tags)) / \
                      len(set(mem1.context_tags) | set(mem2.context_tags))
            similarity_components.append(tag_sim)
            weights.append(0.1)
        
        # Context similarity
        context_sim = self._calculate_context_similarity(mem1.memory_context, mem2.memory_context)
        similarity_components.append(context_sim)
        weights.append(0.1)
        
        # Weighted average
        if similarity_components:
            total_weight = sum(weights[:len(similarity_components)])
            weighted_sum = sum(s * w for s, w in zip(similarity_components, weights))
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return 0.0
    
    def _calculate_emotional_similarity(self, ec1: EmotionalContext, ec2: EmotionalContext) -> float:
        """Calculate similarity between emotional contexts"""
        # Primary emotion match
        primary_match = 1.0 if ec1.primary_emotion == ec2.primary_emotion else 0.0
        
        # Dimensional similarity
        dim_distance = math.sqrt(
            (ec1.valence - ec2.valence) ** 2 +
            (ec1.arousal - ec2.arousal) ** 2 +
            (ec1.dominance - ec2.dominance) ** 2
        )
        dim_similarity = 1.0 - (dim_distance / math.sqrt(3))  # Normalize
        
        # Mixed emotion similarity
        mixed_sim = 1.0 - abs(ec1.mixed_emotion_score - ec2.mixed_emotion_score)
        
        # Combine
        return (primary_match * 0.4 + dim_similarity * 0.4 + mixed_sim * 0.2)
    
    def _calculate_context_similarity(self, ctx1: MemoryContext, ctx2: MemoryContext) -> float:
        """Calculate similarity between memory contexts"""
        similarity = 0.0
        count = 0
        
        # Location match
        if ctx1.location and ctx2.location:
            similarity += 1.0 if ctx1.location == ctx2.location else 0.0
            count += 1
        
        # Participant overlap
        if ctx1.participants and ctx2.participants:
            overlap = len(set(ctx1.participants) & set(ctx2.participants))
            total = len(set(ctx1.participants) | set(ctx2.participants))
            similarity += overlap / total if total > 0 else 0.0
            count += 1
        
        # Activity match
        if ctx1.activity and ctx2.activity:
            similarity += 1.0 if ctx1.activity == ctx2.activity else 0.0
            count += 1
        
        return similarity / count if count > 0 else 0.0
    
    def _calculate_semantic_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        if emb1 is None or emb2 is None:
            return 0.0
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    async def recall(self,
                     query: str,
                     context: Optional[RetrievalContext] = None) -> List[EpisodicMemory]:
        """Enhanced recall with multiple strategies"""
        if not context:
            context = RetrievalContext()
        
        # Track retrieval performance
        start_time = datetime.now()
        
        try:
            # Route to appropriate retrieval strategy
            if context.strategy == RetrievalStrategy.ASSOCIATIVE:
                results = await self._associative_recall(query, context)
            elif context.strategy == RetrievalStrategy.TEMPORAL:
                results = await self._temporal_recall(query, context)
            elif context.strategy == RetrievalStrategy.EMOTIONAL:
                results = await self._emotional_recall(query, context)
            elif context.strategy == RetrievalStrategy.SEMANTIC:
                results = await self._semantic_recall(query, context)
            elif context.strategy == RetrievalStrategy.CONTEXTUAL:
                results = await self._contextual_recall(query, context)
            elif context.strategy == RetrievalStrategy.PATTERN_COMPLETION:
                results = await self._pattern_completion_recall(query, context)
            elif context.strategy == RetrievalStrategy.CUE_DEPENDENT:
                results = await self._cue_dependent_recall(query, context)
            else:
                results = await self._associative_recall(query, context)
            
            # Update access statistics
            current_time = datetime.now()
            for memory in results:
                memory.update_access(current_time, {"query": query, "strategy": context.strategy.name})
                self.access_patterns[memory.memory_id].append(current_time)
            
            # Track performance
            retrieval_time = (datetime.now() - start_time).total_seconds()
            self.retrieval_performance.append({
                "timestamp": start_time,
                "strategy": context.strategy.name,
                "query_length": len(query),
                "results_count": len(results),
                "retrieval_time": retrieval_time
            })
            
            logger.info(f"Retrieved {len(results)} memories using {context.strategy.name} in {retrieval_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Recall failed: {e}")
            return []
    
    async def _associative_recall(self,
                                  query: str,
                                  context: RetrievalContext) -> List[EpisodicMemory]:
        """Associative recall with spreading activation"""
        # Get query embedding
        query_embedding = self._get_embedding(query) if self.embedding_model else None
        
        # Score all memories
        memory_scores = []
        current_time = datetime.now()
        
        # Get all memory IDs
        all_memory_ids = await self.storage.list_memories()
        
        # Process in batches for efficiency
        batch_size = 100
        for i in range(0, len(all_memory_ids), batch_size):
            batch_ids = all_memory_ids[i:i + batch_size]
            
            # Load memories in parallel
            batch_memories = await asyncio.gather(
                *[self.storage.load_memory(mid) for mid in batch_ids]
            )
            
            for memory in batch_memories:
                if memory:
                    score = await self._score_memory_for_query(
                        memory, query, query_embedding, current_time, context
                    )
                    if score > context.min_confidence:
                        memory_scores.append((memory, score))
        
        # Sort by score
        memory_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top results
        recalled_memories = [m for m, s in memory_scores[:context.max_results]]
        
        # Spreading activation
        if context.include_associations and recalled_memories:
            associated = await self._get_associated_memories(
                recalled_memories[:3], context.max_results // 2
            )
            
            # Merge without duplicates
            memory_ids = {m.memory_id for m in recalled_memories}
            for assoc_memory in associated:
                if assoc_memory.memory_id not in memory_ids:
                    recalled_memories.append(assoc_memory)
                    if len(recalled_memories) >= context.max_results:
                        break
        
        return recalled_memories
    
    async def _score_memory_for_query(self,
                                      memory: EpisodicMemory,
                                      query: str,
                                      query_embedding: Optional[np.ndarray],
                                      current_time: datetime,
                                      context: RetrievalContext) -> float:
        """Score a memory for a given query"""
        score = 0.0
        
        # Retention probability
        retention = memory.calculate_retention_probability(current_time)
        score += retention * 0.3
        
        # Semantic similarity
        if query_embedding is not None and memory.chunks:
            semantic_scores = []
            for chunk in memory.chunks:
                if chunk.semantic_embedding is not None:
                    sim = self._calculate_semantic_similarity(
                        query_embedding, chunk.semantic_embedding
                    )
                    semantic_scores.append(sim * chunk.attention_weight)
            
            if semantic_scores:
                score += max(semantic_scores) * 0.4
        
        # Keyword matching
        query_words = set(query.lower().split())
        memory_words = set()
        for chunk in memory.chunks:
            if chunk.modality.linguistic:
                memory_words.update(chunk.modality.linguistic.lower().split())
        
        if query_words and memory_words:
            overlap = len(query_words & memory_words) / len(query_words)
            score += overlap * 0.2
        
        # Importance factor
        score += memory.importance_score * 0.1
        
        return score
    
    async def _get_associated_memories(self,
                                       seed_memories: List[EpisodicMemory],
                                       max_associations: int) -> List[EpisodicMemory]:
        """Get memories associated with seed memories"""
        associated = []
        seen_ids = {m.memory_id for m in seed_memories}
        
        # Get associations for each seed
        for memory in seed_memories:
            for assoc_id in memory.associations:
                if assoc_id not in seen_ids:
                    assoc_memory = await self.storage.load_memory(assoc_id)
                    if assoc_memory:
                        # Get association strength
                        strength = self.association_matrix.get(
                            memory.memory_id, {}
                        ).get(assoc_id, 0.5)
                        
                        if strength > 0.4:
                            associated.append(assoc_memory)
                            seen_ids.add(assoc_id)
                            
                            if len(associated) >= max_associations:
                                return associated
        
        return associated
    
    async def _temporal_recall(self,
                               query: str,
                               context: RetrievalContext) -> List[EpisodicMemory]:
        """Temporal recall within time window"""
        reference_time = datetime.now()
        memories = []
        
        # Get memories within time window
        for time, memory_ids in self.temporal_index.items():
            if abs((time - reference_time).total_seconds()) <= context.time_window.total_seconds():
                for memory_id in memory_ids:
                    memory = await self.storage.load_memory(memory_id)
                    if memory:
                        memories.append(memory)
        
        # Sort by temporal proximity
        memories.sort(key=lambda m: abs((m.creation_time - reference_time).total_seconds()))
        
        return memories[:context.max_results]
    
    async def _emotional_recall(self,
                                query: str,
                                context: RetrievalContext) -> List[EpisodicMemory]:
        """Mood-congruent recall"""
        # Determine target emotion
        if context.emotion_filter:
            target_emotion = context.emotion_filter
        else:
            # Analyze query for emotional content
            emotional_context = self.emotional_memory.tag_memory(query)
            target_emotion = emotional_context.primary_emotion
        
        # Get memories with matching emotion
        memory_ids = self.emotion_index[target_emotion]
        memories = []
        
        for memory_id in memory_ids:
            memory = await self.storage.load_memory(memory_id)
            if memory:
                memories.append(memory)
        
        # Sort by emotional intensity and retention
        current_time = datetime.now()
        memories.sort(
            key=lambda m: (
                m.emotional_context.intensity * 
                m.calculate_retention_probability(current_time)
            ),
            reverse=True
        )
        
        return memories[:context.max_results]
    
    async def _semantic_recall(self,
                               query: str,
                               context: RetrievalContext) -> List[EpisodicMemory]:
        """Semantic similarity based recall"""
        # Extract concepts from query
        concepts = self.semantic_memory._extract_concepts_from_text(query)
        
        # Find memories with similar concepts
        memories = []
        memory_scores = []
        
        for tag, memory_ids in self.tag_index.items():
            if tag in concepts:
                for memory_id in memory_ids:
                    memory = await self.storage.load_memory(memory_id)
                    if memory:
                        # Score based on concept overlap
                        memory_concepts = set(memory.context_tags)
                        for chunk in memory.chunks:
                            if chunk.modality.linguistic:
                                chunk_concepts = self.semantic_memory._extract_concepts_from_text(
                                    chunk.modality.linguistic
                                )
                                memory_concepts.update(chunk_concepts)
                        
                        if memory_concepts:
                            overlap = len(set(concepts) & memory_concepts) / len(concepts)
                            memory_scores.append((memory, overlap))
        
        # Sort by semantic overlap
        memory_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [m for m, s in memory_scores[:context.max_results]]
    
    async def _contextual_recall(self,
                                 query: str,
                                 context: RetrievalContext) -> List[EpisodicMemory]:
        """Context-based recall using environmental cues"""
        # This would use context matching from MemoryContext
        # For now, fallback to associative recall
        return await self._associative_recall(query, context)
    
    async def _pattern_completion_recall(self,
                                         query: str,
                                         context: RetrievalContext) -> List[EpisodicMemory]:
        """Hippocampal-style pattern completion"""
        # Find memories that match partial patterns in the query
        # This is a simplified implementation
        return await self._associative_recall(query, context)
    
    async def _cue_dependent_recall(self,
                                    query: str,
                                    context: RetrievalContext) -> List[EpisodicMemory]:
        """Recall based on specific cue words"""
        if not context.cue_words:
            context.cue_words = query.lower().split()[:3]  # Use first 3 words as cues
        
        memories = []
        memory_scores = []
        
        # Find memories containing cue words
        for cue in context.cue_words:
            if cue in self.tag_index:
                for memory_id in self.tag_index[cue]:
                    memory = await self.storage.load_memory(memory_id)
                    if memory:
                        # Score based on cue presence
                        cue_count = 0
                        for chunk in memory.chunks:
                            if chunk.modality.linguistic and cue in chunk.modality.linguistic.lower():
                                cue_count += 1
                        
                        if cue_count > 0:
                            memory_scores.append((memory, cue_count))
        
        # Remove duplicates and sort
        unique_memories = {}
        for memory, score in memory_scores:
            if memory.memory_id not in unique_memories or unique_memories[memory.memory_id][1] < score:
                unique_memories[memory.memory_id] = (memory, score)
        
        sorted_memories = sorted(unique_memories.values(), key=lambda x: x[1], reverse=True)
        
        return [m for m, s in sorted_memories[:context.max_results]]
    
    async def forget_irrelevant_memories(self):
        """Enhanced forgetting with gradual decay"""
        current_time = datetime.now()
        memories_to_forget = []
        memories_to_weaken = []
        
        all_memory_ids = await self.storage.list_memories()
        
        for memory_id in all_memory_ids:
            memory = await self.storage.load_memory(memory_id)
            if memory:
                retention = memory.calculate_retention_probability(current_time)
                
                # Complete forgetting criteria
                if (retention < 0.1 and 
                    memory.importance_score < 0.4 and
                    memory.access_count < 2):
                    memories_to_forget.append(memory_id)
                
                # Gradual weakening criteria
                elif (retention < 0.3 and 
                      memory.consolidation_level == MemoryStrength.SHORT_TERM):
                    memories_to_weaken.append(memory)
        
        # Process forgetting
        for memory_id in memories_to_forget:
            memory = await self.storage.load_memory(memory_id)
            if memory:
                # Extract semantic trace before forgetting
                if memory.consolidation_level.value >= MemoryStrength.RECENT.value:
                    features = self.semantic_memory.extract_semantic_features(memory)
                    # Store semantic trace (implementation depends on use case)
                
                # Remove from storage and indices
                await self.storage.delete_memory(memory_id)
                self._remove_from_indices(memory)
                
                logger.info(f"Forgot memory {memory_id}")
        
        # Process weakening
        for memory in memories_to_weaken:
            # Reduce vividness and confidence
            memory.vividness *= 0.8
            memory.confidence *= 0.9
            
            # Add distortion marker
            memory.distortions.append({
                "time": current_time,
                "type": "natural_decay",
                "retention": memory.calculate_retention_probability(current_time)
            })
            
            await self.storage.save_memory(memory)
        
        logger.info(f"Forgot {len(memories_to_forget)} memories, weakened {len(memories_to_weaken)}")
    
    def _remove_from_indices(self, memory: EpisodicMemory):
        """Remove memory from all indices"""
        # Temporal index
        if memory.creation_time in self.temporal_index:
            self.temporal_index[memory.creation_time].remove(memory.memory_id)
            if not self.temporal_index[memory.creation_time]:
                del self.temporal_index[memory.creation_time]
        
        # Emotion index
        self.emotion_index[memory.emotional_context.primary_emotion].discard(memory.memory_id)
        for emotion, _ in memory.emotional_context.secondary_emotions:
            self.emotion_index[emotion].discard(memory.memory_id)
        
        # Tag index
        for tag in memory.context_tags:
            self.tag_index[tag].discard(memory.memory_id)
        
        # Association cleanup
        for assoc_id in memory.associations:
            if assoc_id in self.association_matrix:
                self.association_matrix[assoc_id].pop(memory.memory_id, None)
        self.association_matrix.pop(memory.memory_id, None)
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        current_time = datetime.now()
        
        # Basic counts
        total_memories = len(self.temporal_index)
        
        # Memory distribution by consolidation level
        consolidation_dist = defaultdict(int)
        retention_scores = []
        importance_scores = []
        
        # This is synchronous for simplicity - in practice, make it async
        for time_memories in self.temporal_index.values():
            for memory_id in time_memories:
                # Note: This is inefficient - consider caching these stats
                pass
        
        # Working memory stats
        wm_stats = {
            "current_capacity": self.working_memory.current_capacity,
            "usage": len(self.working_memory.buffer),
            "cognitive_load": self.working_memory.cognitive_load,
            "rehearsal_buffer": len(self.working_memory.rehearsal_buffer),
            "visuospatial_buffer": len(self.working_memory.visuospatial_buffer)
        }
        
        # Semantic memory stats
        sm_stats = {
            "categories": len(self.semantic_memory.category_index),
            "concepts": len(self.semantic_memory.concept_hierarchy),
            "knowledge_items": len(self.semantic_memory.knowledge_graph),
            "prototypes": len(self.semantic_memory.category_prototypes)
        }
        
        # Emotional patterns
        emotion_stats = {
            "emotion_patterns": len(self.emotional_memory.emotion_patterns),
            "tracked_transitions": len(self.emotional_memory.emotion_transitions),
            "primary_emotions": dict(
                (emotion, len(memories)) 
                for emotion, memories in self.emotion_index.items()
            )
        }
        
        # Performance metrics
        if self.retrieval_performance:
            recent_performance = self.retrieval_performance[-100:]  # Last 100 retrievals
            perf_stats = {
                "average_retrieval_time": np.mean([p["retrieval_time"] for p in recent_performance]),
                "average_results_count": np.mean([p["results_count"] for p in recent_performance]),
                "strategy_distribution": defaultdict(int)
            }
            for perf in recent_performance:
                perf_stats["strategy_distribution"][perf["strategy"]] += 1
        else:
            perf_stats = {}
        
        return {
            "total_memories": total_memories,
            "working_memory": wm_stats,
            "semantic_memory": sm_stats,
            "emotional_memory": emotion_stats,
            "performance": perf_stats,
            "indices": {
                "temporal_index_size": len(self.temporal_index),
                "emotion_index_size": sum(len(v) for v in self.emotion_index.values()),
                "tag_index_size": sum(len(v) for v in self.tag_index.values())
            },
            "last_consolidation": self.last_consolidation.isoformat()
        }
    
    async def shutdown(self):
        """Gracefully shutdown the memory system"""
        logger.info("Shutting down episodic memory system...")
        
        # Stop consolidation
        await self.consolidator.stop()
        
        # Final consolidation
        await self.forget_irrelevant_memories()
        
        # Save any pending changes
        # (Implementation depends on storage backend)
        
        logger.info("Episodic memory system shutdown complete")


# Integration with JARVIS
async def integrate_with_jarvis(jarvis_instance):
    """Integrate enhanced episodic memory with JARVIS"""
    logger.info("🧠 Integrating Enhanced Episodic Memory System with JARVIS")
    
    # Create episodic memory system
    memory_system = EpisodicMemorySystem(
        working_memory_capacity=7,
        enable_persistence=True
    )
    
    # Attach to JARVIS
    jarvis_instance.episodic_memory = memory_system
    
    # Create memory formation wrapper
    async def remember_conversation(conversation_data, context=None):
        """Remember a conversation with JARVIS"""
        return await memory_system.remember_interaction(
            conversation_data,
            context
        )
    
    # Create recall wrapper
    async def recall_memories(query, strategy=RetrievalStrategy.ASSOCIATIVE):
        """Recall memories from JARVIS's episodic memory"""
        context = RetrievalContext(strategy=strategy)
        return await memory_system.recall(query, context)
    
    # Attach methods to JARVIS
    jarvis_instance.remember = remember_conversation
    jarvis_instance.recall = recall_memories
    
    logger.info("✅ Enhanced Episodic Memory integrated successfully")
    
    return memory_system


if __name__ == "__main__":
    # Standalone test
    async def test_memory_system():
        memory_system = EpisodicMemorySystem()
        
        # Test memory creation
        interaction = {
            "messages": [
                {"content": "I just learned about quantum computing!", "emotion_score": 0.8},
                {"content": "It's fascinating how qubits can be in superposition", "emotion_score": 0.7}
            ],
            "summary": "Discussion about quantum computing concepts",
            "personal_relevance": 0.9
        }
        
        context = {
            "tags": ["quantum", "learning", "technology"],
            "emotional_indicators": {"valence": 0.8, "arousal": 0.7}
        }
        
        memory = await memory_system.remember_interaction(interaction, context)
        print(f"Created memory: {memory.memory_id}")
        
        # Test recall
        results = await memory_system.recall("quantum computing")
        print(f"Recalled {len(results)} memories")
        
        await memory_system.shutdown()
    
    asyncio.run(test_memory_system())