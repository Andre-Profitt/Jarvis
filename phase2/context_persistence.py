#!/usr/bin/env python3
"""
JARVIS Phase 2: Context Persistence System
Maintains conversation and activity context between interactions
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import json
import pickle
import os
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContextItem:
    """Single context item with metadata"""
    id: str
    type: str  # conversation, activity, preference, pattern
    content: Any
    timestamp: datetime
    confidence: float = 1.0
    source: str = "unknown"
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    decay_rate: float = 0.95  # How fast this context becomes less relevant

@dataclass
class ContextMemory:
    """Hierarchical context memory structure"""
    working: deque  # Last 5-10 items (immediate context)
    short_term: deque  # Last hour of context
    long_term: List[ContextItem]  # Persistent patterns and preferences
    
    def __init__(self):
        self.working = deque(maxlen=10)
        self.short_term = deque(maxlen=100)
        self.long_term = []

class ContextPersistenceSystem:
    """Advanced context persistence with hierarchical memory"""
    
    def __init__(self, db_path: str = "jarvis_context.db"):
        self.db_path = db_path
        self.memory = ContextMemory()
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.context_embeddings = {}
        self.active_contexts = {}
        self.context_patterns = {}
        
        # Initialize database
        self._init_database()
        
        # Load existing context
        self._load_context_from_db()
        
        # Context relevance thresholds
        self.relevance_thresholds = {
            'working': 0.8,
            'short_term': 0.6,
            'long_term': 0.4
        }
        
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS context_items (
                id TEXT PRIMARY KEY,
                type TEXT,
                content TEXT,
                timestamp REAL,
                confidence REAL,
                source TEXT,
                embedding BLOB,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS context_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT,
                frequency INTEGER,
                last_seen REAL,
                pattern_data TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def add_context(self, context_type: str, content: Any, 
                         source: str = "user", confidence: float = 1.0,
                         metadata: Optional[Dict] = None) -> str:
        """Add new context item to memory hierarchy"""
        # Create context item
        context_id = f"{context_type}_{datetime.now().timestamp()}"
        context_item = ContextItem(
            id=context_id,
            type=context_type,
            content=content,
            timestamp=datetime.now(),
            confidence=confidence,
            source=source,
            metadata=metadata or {}
        )
        
        # Generate embedding for text content
        if isinstance(content, str) and len(content) > 10:
            context_item.embedding = await self._generate_embedding(content)
        
        # Add to memory hierarchy
        self.memory.working.append(context_item)
        self.memory.short_term.append(context_item)
        
        # Determine if should go to long-term memory
        if await self._should_persist_long_term(context_item):
            self.memory.long_term.append(context_item)
            await self._save_to_database(context_item)
        
        # Update active contexts
        self.active_contexts[context_type] = context_item
        
        # Detect patterns
        await self._detect_patterns(context_item)
        
        logger.info(f"Added context: {context_type} from {source}")
        return context_id
    
    async def get_relevant_context(self, query: str, 
                                  context_types: Optional[List[str]] = None,
                                  time_window: Optional[timedelta] = None) -> List[ContextItem]:
        """Retrieve relevant context based on query"""
        relevant_contexts = []
        query_embedding = await self._generate_embedding(query)
        
        # Search in memory hierarchy
        for memory_level, contexts in [
            ('working', list(self.memory.working)),
            ('short_term', list(self.memory.short_term)),
            ('long_term', self.memory.long_term)
        ]:
            threshold = self.relevance_thresholds[memory_level]
            
            for context in contexts:
                # Filter by type if specified
                if context_types and context.type not in context_types:
                    continue
                
                # Filter by time window
                if time_window and (datetime.now() - context.timestamp) > time_window:
                    continue
                
                # Calculate relevance
                relevance = await self._calculate_relevance(
                    query_embedding, context, memory_level
                )
                
                if relevance > threshold:
                    relevant_contexts.append((context, relevance))
        
        # Sort by relevance and return
        relevant_contexts.sort(key=lambda x: x[1], reverse=True)
        return [ctx for ctx, _ in relevant_contexts[:10]]
    
    async def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        history = []
        
        for context in reversed(list(self.memory.short_term)):
            if context.type == 'conversation':
                history.append({
                    'timestamp': context.timestamp.isoformat(),
                    'content': context.content,
                    'source': context.source,
                    'metadata': context.metadata
                })
                
                if len(history) >= limit:
                    break
        
        return history
    
    async def get_user_preferences(self) -> Dict[str, Any]:
        """Extract learned user preferences from long-term memory"""
        preferences = {
            'communication_style': {},
            'activity_patterns': {},
            'response_preferences': {},
            'topic_interests': {}
        }
        
        # Analyze long-term memory for patterns
        for context in self.memory.long_term:
            if context.type == 'preference':
                pref_type = context.metadata.get('preference_type', 'general')
                if pref_type not in preferences:
                    preferences[pref_type] = {}
                
                preferences[pref_type][context.content] = {
                    'confidence': context.confidence,
                    'last_updated': context.timestamp
                }
        
        # Add detected patterns
        for pattern_id, pattern_data in self.context_patterns.items():
            if pattern_data['type'] in preferences:
                preferences[pattern_data['type']][pattern_id] = {
                    'frequency': pattern_data['frequency'],
                    'confidence': pattern_data['confidence']
                }
        
        return preferences
    
    async def update_context_relevance(self):
        """Apply time decay to context relevance"""
        current_time = datetime.now()
        
        # Update short-term memory
        updated_short_term = deque(maxlen=100)
        for context in self.memory.short_term:
            time_diff = (current_time - context.timestamp).total_seconds() / 3600  # hours
            decay_factor = context.decay_rate ** time_diff
            context.confidence *= decay_factor
            
            if context.confidence > 0.1:  # Keep if still relevant
                updated_short_term.append(context)
        
        self.memory.short_term = updated_short_term
        
        # Update long-term memory (slower decay)
        for context in self.memory.long_term:
            time_diff = (current_time - context.timestamp).total_seconds() / 86400  # days
            decay_factor = 0.99 ** time_diff  # Slower decay for long-term
            context.confidence *= decay_factor
    
    async def merge_contexts(self, contexts: List[ContextItem]) -> ContextItem:
        """Merge multiple related contexts into a unified context"""
        if not contexts:
            return None
        
        # Find most recent as base
        base_context = max(contexts, key=lambda c: c.timestamp)
        
        # Merge content
        merged_content = {
            'base': base_context.content,
            'related': [c.content for c in contexts if c != base_context]
        }
        
        # Average confidence
        avg_confidence = np.mean([c.confidence for c in contexts])
        
        # Merge metadata
        merged_metadata = {}
        for context in contexts:
            merged_metadata.update(context.metadata)
        
        # Create merged context
        merged = ContextItem(
            id=f"merged_{datetime.now().timestamp()}",
            type=base_context.type,
            content=merged_content,
            timestamp=datetime.now(),
            confidence=avg_confidence,
            source="context_merger",
            metadata=merged_metadata
        )
        
        return merged
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate text embedding for similarity matching"""
        try:
            # Simple TF-IDF embedding (in production, use better embeddings)
            if not hasattr(self, '_fitted_vectorizer'):
                # Fit on some initial data
                sample_texts = [text] + [c.content for c in self.memory.long_term 
                               if isinstance(c.content, str)][:100]
                if len(sample_texts) > 1:
                    self.vectorizer.fit(sample_texts)
                    self._fitted_vectorizer = True
                else:
                    return np.zeros(100)  # Default embedding
            
            embedding = self.vectorizer.transform([text]).toarray()[0]
            return embedding
        except:
            # Fallback to simple character-based embedding
            return np.array([ord(c) for c in text[:100]])
    
    async def _calculate_relevance(self, query_embedding: np.ndarray, 
                                  context: ContextItem, 
                                  memory_level: str) -> float:
        """Calculate relevance score between query and context"""
        relevance = 0.0
        
        # Text similarity if embeddings available
        if context.embedding is not None and query_embedding is not None:
            try:
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    context.embedding.reshape(1, -1)
                )[0][0]
                relevance += similarity * 0.6
            except:
                pass
        
        # Time relevance
        time_diff = (datetime.now() - context.timestamp).total_seconds() / 3600
        time_relevance = np.exp(-time_diff / 24)  # Exponential decay over 24 hours
        relevance += time_relevance * 0.2
        
        # Confidence factor
        relevance += context.confidence * 0.2
        
        # Memory level bonus
        level_bonuses = {'working': 0.3, 'short_term': 0.1, 'long_term': 0.0}
        relevance += level_bonuses.get(memory_level, 0)
        
        return min(relevance, 1.0)
    
    async def _should_persist_long_term(self, context: ContextItem) -> bool:
        """Determine if context should be saved to long-term memory"""
        # Always save preferences and patterns
        if context.type in ['preference', 'pattern']:
            return True
        
        # Save high-confidence items
        if context.confidence > 0.8:
            return True
        
        # Save if similar to existing patterns
        for pattern in self.context_patterns.values():
            if pattern['type'] == context.type and pattern['confidence'] > 0.7:
                return True
        
        return False
    
    async def _detect_patterns(self, context: ContextItem):
        """Detect patterns in context over time"""
        # Group contexts by type
        similar_contexts = [c for c in self.memory.short_term 
                           if c.type == context.type]
        
        if len(similar_contexts) >= 3:
            # Look for repeated patterns
            pattern_id = f"pattern_{context.type}_{len(self.context_patterns)}"
            
            self.context_patterns[pattern_id] = {
                'type': context.type,
                'frequency': len(similar_contexts),
                'confidence': 0.7,
                'last_seen': datetime.now(),
                'examples': [c.content for c in similar_contexts[-3:]]
            }
    
    async def _save_to_database(self, context: ContextItem):
        """Save context item to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embedding_blob = pickle.dumps(context.embedding) if context.embedding is not None else None
        
        cursor.execute('''
            INSERT OR REPLACE INTO context_items 
            (id, type, content, timestamp, confidence, source, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            context.id,
            context.type,
            json.dumps(context.content) if not isinstance(context.content, str) else context.content,
            context.timestamp.timestamp(),
            context.confidence,
            context.source,
            embedding_blob,
            json.dumps(context.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def _load_context_from_db(self):
        """Load existing context from database"""
        if not os.path.exists(self.db_path):
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load recent context items
        week_ago = (datetime.now() - timedelta(days=7)).timestamp()
        cursor.execute('''
            SELECT * FROM context_items 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC 
            LIMIT 1000
        ''', (week_ago,))
        
        for row in cursor.fetchall():
            context = ContextItem(
                id=row[0],
                type=row[1],
                content=json.loads(row[2]) if row[2].startswith('{') else row[2],
                timestamp=datetime.fromtimestamp(row[3]),
                confidence=row[4],
                source=row[5],
                embedding=pickle.loads(row[6]) if row[6] else None,
                metadata=json.loads(row[7]) if row[7] else {}
            )
            
            # Add to appropriate memory level
            time_diff = datetime.now() - context.timestamp
            if time_diff < timedelta(hours=1):
                self.memory.short_term.append(context)
            
            self.memory.long_term.append(context)
        
        conn.close()
        logger.info(f"Loaded {len(self.memory.long_term)} contexts from database")

# Create singleton instance
_context_persistence = None

async def get_context_persistence() -> ContextPersistenceSystem:
    """Get or create context persistence system"""
    global _context_persistence
    if _context_persistence is None:
        _context_persistence = ContextPersistenceSystem()
    return _context_persistence
