"""
JARVIS Long-Term Memory System
Persistent memory with semantic understanding and retrieval
"""

import os
import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import hashlib
from collections import defaultdict
import threading
import pickle

class MemoryVector:
    """Vector representation of memories for semantic search"""
    
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.word_vectors = {}
        self._initialize_basic_vectors()
        
    def _initialize_basic_vectors(self):
        """Initialize basic word vectors"""
        # Simple word embeddings (in production, use Word2Vec or BERT)
        base_concepts = {
            'user': np.random.randn(self.dimension),
            'system': np.random.randn(self.dimension),
            'error': np.random.randn(self.dimension),
            'success': np.random.randn(self.dimension),
            'task': np.random.randn(self.dimension),
            'help': np.random.randn(self.dimension),
            'data': np.random.randn(self.dimension),
            'process': np.random.randn(self.dimension)
        }
        
        # Normalize vectors
        for word, vector in base_concepts.items():
            self.word_vectors[word] = vector / np.linalg.norm(vector)
            
    def text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to vector representation"""
        words = text.lower().split()
        vectors = []
        
        for word in words:
            if word in self.word_vectors:
                vectors.append(self.word_vectors[word])
            else:
                # Create new vector for unknown word
                new_vector = np.random.randn(self.dimension)
                new_vector = new_vector / np.linalg.norm(new_vector)
                self.word_vectors[word] = new_vector
                vectors.append(new_vector)
                
        if not vectors:
            return np.zeros(self.dimension)
            
        # Average vectors
        return np.mean(vectors, axis=0)
        
    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        if np.any(vec1) and np.any(vec2):
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return 0.0


class LongTermMemory:
    """Long-term persistent memory storage and retrieval"""
    
    def __init__(self, db_path: str = "jarvis_memory.db"):
        self.db_path = db_path
        self.vector_engine = MemoryVector()
        self.memory_index = {}
        self.connection = None
        self._lock = threading.Lock()
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize SQLite database for memory storage"""
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.connection.cursor()
        
        # Create memories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                context TEXT,
                category TEXT,
                importance REAL DEFAULT 0.5,
                emotional_weight REAL DEFAULT 0.0,
                access_count INTEGER DEFAULT 0,
                created_at TIMESTAMP,
                last_accessed TIMESTAMP,
                vector BLOB,
                metadata TEXT
            )
        ''')
        
        # Create associations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS associations (
                memory1_id TEXT,
                memory2_id TEXT,
                strength REAL DEFAULT 0.5,
                association_type TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (memory1_id) REFERENCES memories(id),
                FOREIGN KEY (memory2_id) REFERENCES memories(id)
            )
        ''')
        
        # Create indices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON memories(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at)')
        
        self.connection.commit()
        
    def store_memory(self, content: str, context: str = "", category: str = "general",
                    importance: float = 0.5, emotional_weight: float = 0.0,
                    metadata: Dict[str, Any] = None) -> str:
        """Store a new memory"""
        with self._lock:
            # Generate unique ID
            memory_id = hashlib.sha256(
                f"{content}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]
            
            # Create vector representation
            vector = self.vector_engine.text_to_vector(content)
            vector_blob = pickle.dumps(vector)
            
            # Store in database
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO memories 
                (id, content, context, category, importance, emotional_weight,
                 created_at, last_accessed, vector, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                memory_id, content, context, category, importance, emotional_weight,
                datetime.now(), datetime.now(), vector_blob,
                json.dumps(metadata or {})
            ))
            
            self.connection.commit()
            
            # Update memory index
            self.memory_index[memory_id] = vector
            
            # Find and create associations
            self._create_associations(memory_id, content, vector)
            
            return memory_id
            
    def _create_associations(self, memory_id: str, content: str, vector: np.ndarray):
        """Create associations with related memories"""
        cursor = self.connection.cursor()
        
        # Find similar memories
        similar_memories = self.search_memories(content, limit=5, exclude_id=memory_id)
        
        for similar in similar_memories:
            if similar['similarity'] > 0.6:  # Threshold for association
                cursor.execute('''
                    INSERT INTO associations (memory1_id, memory2_id, strength, 
                                            association_type, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    memory_id, similar['id'], similar['similarity'],
                    'semantic', datetime.now()
                ))
                
        self.connection.commit()
        
    def retrieve_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory"""
        with self._lock:
            cursor = self.connection.cursor()
            
            # Update access count and last accessed
            cursor.execute('''
                UPDATE memories 
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
            ''', (datetime.now(), memory_id))
            
            # Retrieve memory
            cursor.execute('SELECT * FROM memories WHERE id = ?', (memory_id,))
            row = cursor.fetchone()
            
            if row:
                columns = [desc[0] for desc in cursor.description]
                memory = dict(zip(columns, row))
                
                # Deserialize vector
                memory['vector'] = pickle.loads(memory['vector'])
                memory['metadata'] = json.loads(memory['metadata'])
                
                return memory
                
            return None
            
    def search_memories(self, query: str, limit: int = 10, 
                       category: Optional[str] = None,
                       time_range: Optional[Tuple[datetime, datetime]] = None,
                       exclude_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search memories using semantic similarity"""
        query_vector = self.vector_engine.text_to_vector(query)
        results = []
        
        with self._lock:
            cursor = self.connection.cursor()
            
            # Build query
            sql = 'SELECT * FROM memories WHERE 1=1'
            params = []
            
            if category:
                sql += ' AND category = ?'
                params.append(category)
                
            if time_range:
                sql += ' AND created_at BETWEEN ? AND ?'
                params.extend(time_range)
                
            if exclude_id:
                sql += ' AND id != ?'
                params.append(exclude_id)
                
            cursor.execute(sql, params)
            
            # Calculate similarities
            for row in cursor.fetchall():
                columns = [desc[0] for desc in cursor.description]
                memory = dict(zip(columns, row))
                
                # Deserialize vector
                memory_vector = pickle.loads(memory['vector'])
                
                # Calculate similarity
                similarity = self.vector_engine.similarity(query_vector, memory_vector)
                
                # Weight by importance and recency
                recency_weight = self._calculate_recency_weight(memory['last_accessed'])
                final_score = (similarity * 0.7 + 
                             memory['importance'] * 0.2 + 
                             recency_weight * 0.1)
                
                memory['similarity'] = similarity
                memory['score'] = final_score
                memory['metadata'] = json.loads(memory['metadata'])
                
                results.append(memory)
                
        # Sort by score and limit
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
        
    def _calculate_recency_weight(self, last_accessed: str) -> float:
        """Calculate weight based on recency"""
        if isinstance(last_accessed, str):
            last_accessed = datetime.fromisoformat(last_accessed)
            
        age = datetime.now() - last_accessed
        
        # Decay function
        if age < timedelta(hours=1):
            return 1.0
        elif age < timedelta(days=1):
            return 0.8
        elif age < timedelta(weeks=1):
            return 0.6
        elif age < timedelta(days=30):
            return 0.4
        else:
            return 0.2
            
    def consolidate_memories(self, threshold: float = 0.8):
        """Consolidate similar memories to prevent redundancy"""
        with self._lock:
            cursor = self.connection.cursor()
            cursor.execute('SELECT id, content, vector FROM memories')
            
            memories = []
            for row in cursor.fetchall():
                memories.append({
                    'id': row[0],
                    'content': row[1],
                    'vector': pickle.loads(row[2])
                })
                
            # Find highly similar memories
            consolidations = []
            for i, mem1 in enumerate(memories):
                for j, mem2 in enumerate(memories[i+1:], i+1):
                    similarity = self.vector_engine.similarity(
                        mem1['vector'], mem2['vector']
                    )
                    
                    if similarity > threshold:
                        consolidations.append((mem1, mem2, similarity))
                        
            # Merge similar memories
            for mem1, mem2, similarity in consolidations:
                # Keep the more important/accessed memory
                cursor.execute('''
                    SELECT importance, access_count FROM memories WHERE id IN (?, ?)
                ''', (mem1['id'], mem2['id']))
                
                stats = cursor.fetchall()
                if stats[0][0] + stats[0][1] > stats[1][0] + stats[1][1]:
                    keep_id, remove_id = mem1['id'], mem2['id']
                else:
                    keep_id, remove_id = mem2['id'], mem1['id']
                    
                # Update associations
                cursor.execute('''
                    UPDATE associations SET memory1_id = ? WHERE memory1_id = ?
                ''', (keep_id, remove_id))
                
                cursor.execute('''
                    UPDATE associations SET memory2_id = ? WHERE memory2_id = ?
                ''', (keep_id, remove_id))
                
                # Remove duplicate
                cursor.execute('DELETE FROM memories WHERE id = ?', (remove_id,))
                
            self.connection.commit()
            
    def get_memory_graph(self, start_id: str, depth: int = 2) -> Dict[str, Any]:
        """Get memory association graph"""
        with self._lock:
            visited = set()
            graph = {'nodes': [], 'edges': []}
            
            def explore_node(memory_id: str, current_depth: int):
                if current_depth > depth or memory_id in visited:
                    return
                    
                visited.add(memory_id)
                
                # Get memory
                memory = self.retrieve_memory(memory_id)
                if memory:
                    graph['nodes'].append({
                        'id': memory_id,
                        'content': memory['content'][:50] + '...',
                        'category': memory['category'],
                        'importance': memory['importance']
                    })
                    
                    # Get associations
                    cursor = self.connection.cursor()
                    cursor.execute('''
                        SELECT memory2_id, strength, association_type 
                        FROM associations 
                        WHERE memory1_id = ?
                    ''', (memory_id,))
                    
                    for assoc in cursor.fetchall():
                        graph['edges'].append({
                            'source': memory_id,
                            'target': assoc[0],
                            'strength': assoc[1],
                            'type': assoc[2]
                        })
                        
                        explore_node(assoc[0], current_depth + 1)
                        
            explore_node(start_id, 0)
            return graph
            
    def forget_old_memories(self, days: int = 365, importance_threshold: float = 0.3):
        """Forget old, unimportant memories"""
        with self._lock:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            cursor = self.connection.cursor()
            cursor.execute('''
                DELETE FROM memories 
                WHERE last_accessed < ? 
                AND importance < ? 
                AND access_count < 5
            ''', (cutoff_date, importance_threshold))
            
            deleted = cursor.rowcount
            self.connection.commit()
            
            return deleted
            
    def export_memories(self, filepath: str):
        """Export memories to file"""
        with self._lock:
            cursor = self.connection.cursor()
            cursor.execute('SELECT * FROM memories')
            
            memories = []
            for row in cursor.fetchall():
                columns = [desc[0] for desc in cursor.description]
                memory = dict(zip(columns, row))
                
                # Don't export vector blob
                memory.pop('vector', None)
                memories.append(memory)
                
            with open(filepath, 'w') as f:
                json.dump(memories, f, indent=2, default=str)
                
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        with self._lock:
            cursor = self.connection.cursor()
            
            stats = {}
            
            # Total memories
            cursor.execute('SELECT COUNT(*) FROM memories')
            stats['total_memories'] = cursor.fetchone()[0]
            
            # Categories
            cursor.execute('SELECT category, COUNT(*) FROM memories GROUP BY category')
            stats['categories'] = dict(cursor.fetchall())
            
            # Average importance
            cursor.execute('SELECT AVG(importance) FROM memories')
            stats['avg_importance'] = cursor.fetchone()[0] or 0
            
            # Most accessed
            cursor.execute('''
                SELECT content, access_count 
                FROM memories 
                ORDER BY access_count DESC 
                LIMIT 5
            ''')
            stats['most_accessed'] = [
                {'content': row[0][:50] + '...', 'count': row[1]}
                for row in cursor.fetchall()
            ]
            
            # Memory age
            cursor.execute('''
                SELECT MIN(created_at), MAX(created_at) FROM memories
            ''')
            oldest, newest = cursor.fetchone()
            stats['oldest_memory'] = oldest
            stats['newest_memory'] = newest
            
            return stats


# Integration with JARVIS
def integrate_long_term_memory(jarvis_core):
    """Integrate long-term memory into JARVIS"""
    memory = LongTermMemory()
    
    # Hook into consciousness for automatic memory storage
    consciousness = jarvis_core.component_manager.get_component('consciousness')
    if consciousness:
        original_think = consciousness.think_about
        
        def think_and_remember(topic: str, category: str = 'general'):
            # Original thinking
            original_think(topic, category)
            
            # Store important thoughts
            if any(keyword in topic.lower() for keyword in ['error', 'success', 'learned', 'user']):
                importance = 0.7 if 'error' in topic.lower() else 0.6
                memory.store_memory(
                    content=topic,
                    category=category,
                    importance=importance,
                    emotional_weight=consciousness.stream.emotions.get('concern', 0)
                )
                
        consciousness.think_about = think_and_remember
        
    jarvis_core.component_manager.register_component('long_term_memory', memory)
    
    print("✅ Long-term memory integrated!")
    
    return jarvis_core


if __name__ == "__main__":
    # Test long-term memory
    print("🧠 Testing Long-Term Memory...")
    
    memory = LongTermMemory("test_memory.db")
    
    # Store some memories
    id1 = memory.store_memory(
        "User asked about quantum computing",
        context="Technical discussion",
        category="user_interaction",
        importance=0.8
    )
    
    id2 = memory.store_memory(
        "Successfully explained quantum superposition",
        context="Educational response",
        category="success",
        importance=0.7
    )
    
    id3 = memory.store_memory(
        "Error in processing natural language query",
        context="System error",
        category="error",
        importance=0.9,
        emotional_weight=0.6
    )
    
    # Search memories
    results = memory.search_memories("quantum", limit=5)
    print(f"\nSearch results for 'quantum': {len(results)} found")
    for result in results:
        print(f"- {result['content'][:50]}... (score: {result['score']:.2f})")
        
    # Get statistics
    stats = memory.get_statistics()
    print(f"\nMemory Statistics: {json.dumps(stats, indent=2, default=str)}")
    
    print("\n✅ Long-term memory test complete!")
