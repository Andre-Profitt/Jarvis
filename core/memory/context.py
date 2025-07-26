"""
Context Manager - Handles conversation memory and learning
Short-term, long-term, and episodic memory with vector search
"""
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
import numpy as np

import redis
from sentence_transformers import SentenceTransformer
import faiss

from ..logger import setup_logger

logger = setup_logger(__name__)


class ContextManager:
    """Manages conversation context and memory"""
    
    def __init__(self, config):
        self.config = config
        self.db_path = Path(config.get("memory.db_path", "jarvis_memory.db"))
        
        # Memory components
        self.redis_client = None  # Short-term memory
        self.db_conn = None      # Long-term memory
        self.vector_index = None # Semantic search
        
        # Models
        self.embedding_model = None
        
        # Current session
        self.conversation_id = None
        self.last_response = ""
        
    async def initialize(self):
        """Initialize memory systems"""
        logger.info("Initializing context manager...")
        
        # Initialize components
        await asyncio.gather(
            self._init_redis(),
            self._init_sqlite(),
            self._init_vector_search()
        )
        
        logger.info("Context manager ready")
        
    async def _init_redis(self):
        """Initialize Redis for short-term memory"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.get("memory.redis_host", "localhost"),
                port=self.config.get("memory.redis_port", 6379),
                decode_responses=True
            )
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.ping
            )
            logger.info("✓ Redis connected")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_client = None
            
    async def _init_sqlite(self):
        """Initialize SQLite for long-term memory"""
        self.db_path.parent.mkdir(exist_ok=True)
        self.db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Create tables
        self.db_conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                started_at TIMESTAMP,
                ended_at TIMESTAMP,
                summary TEXT,
                metadata TEXT
            );
            
            CREATE TABLE IF NOT EXISTS exchanges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                timestamp TIMESTAMP,
                user_input TEXT,
                assistant_response TEXT,
                intent TEXT,
                entities TEXT,
                embeddings BLOB,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );
            
            CREATE TABLE IF NOT EXISTS user_preferences (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS learned_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT,
                response_template TEXT,
                confidence REAL,
                usage_count INTEGER DEFAULT 0,
                created_at TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_exchanges_conversation 
            ON exchanges(conversation_id);
            
            CREATE INDEX IF NOT EXISTS idx_exchanges_timestamp 
            ON exchanges(timestamp);
        """)
        self.db_conn.commit()
        logger.info("✓ SQLite initialized")
        
    async def _init_vector_search(self):
        """Initialize vector search for semantic memory"""
        # Load embedding model
        model_name = self.config.get("memory.embedding_model", "all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize FAISS index
        dimension = 384  # all-MiniLM-L6-v2 dimension
        self.vector_index = faiss.IndexFlatL2(dimension)
        
        # Load existing embeddings
        await self._load_embeddings()
        
        logger.info("✓ Vector search initialized")
        
    async def _load_embeddings(self):
        """Load existing embeddings into vector index"""
        cursor = self.db_conn.execute(
            "SELECT id, embeddings FROM exchanges WHERE embeddings IS NOT NULL"
        )
        
        for row in cursor:
            if row[1]:
                embedding = np.frombuffer(row[1], dtype=np.float32)
                self.vector_index.add(embedding.reshape(1, -1))
                
    def create_conversation(self) -> str:
        """Create new conversation session"""
        import uuid
        self.conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        
        self.db_conn.execute(
            "INSERT INTO conversations (id, started_at) VALUES (?, ?)",
            (self.conversation_id, datetime.now())
        )
        self.db_conn.commit()
        
        # Initialize Redis session
        if self.redis_client:
            self.redis_client.hset(
                f"conversation:{self.conversation_id}",
                mapping={
                    "started_at": datetime.now().isoformat(),
                    "exchange_count": 0
                }
            )
            self.redis_client.expire(f"conversation:{self.conversation_id}", 3600)
            
        return self.conversation_id
        
    async def update_context(
        self,
        text: str,
        intent: str,
        entities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update context with new input"""
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(text)
        
        # Search similar past exchanges
        similar_exchanges = await self._search_similar_exchanges(embeddings)
        
        # Get recent history
        recent_history = await self._get_recent_history()
        
        # Build context
        context = {
            "conversation_id": self.conversation_id,
            "current_intent": intent,
            "entities": entities,
            "similar_exchanges": similar_exchanges,
            "history": recent_history,
            "user_preferences": await self._get_user_preferences(),
            "current_context": await self._build_current_context(intent, entities)
        }
        
        # Update short-term memory
        if self.redis_client:
            self.redis_client.hset(
                f"context:{self.conversation_id}",
                mapping={
                    "last_intent": intent,
                    "last_entities": json.dumps(entities),
                    "last_input": text,
                    "timestamp": datetime.now().isoformat()
                }
            )
            self.redis_client.expire(f"context:{self.conversation_id}", 3600)
            
        return context
        
    async def add_exchange(self, user_input: str, assistant_response: str):
        """Store conversation exchange"""
        self.last_response = assistant_response
        
        # Generate embeddings for the exchange
        combined_text = f"{user_input} {assistant_response}"
        embeddings = self.embedding_model.encode(combined_text)
        
        # Store in database
        self.db_conn.execute("""
            INSERT INTO exchanges 
            (conversation_id, timestamp, user_input, assistant_response, embeddings)
            VALUES (?, ?, ?, ?, ?)
        """, (
            self.conversation_id,
            datetime.now(),
            user_input,
            assistant_response,
            embeddings.tobytes()
        ))
        self.db_conn.commit()
        
        # Add to vector index
        self.vector_index.add(embeddings.reshape(1, -1))
        
        # Update short-term memory
        if self.redis_client:
            exchange_count = self.redis_client.hincrby(
                f"conversation:{self.conversation_id}",
                "exchange_count",
                1
            )
            
            # Store recent exchange
            self.redis_client.lpush(
                f"recent:{self.conversation_id}",
                json.dumps({
                    "user": user_input,
                    "assistant": assistant_response,
                    "timestamp": datetime.now().isoformat()
                })
            )
            self.redis_client.ltrim(f"recent:{self.conversation_id}", 0, 9)
            
        # Learn from patterns
        await self._learn_from_exchange(user_input, assistant_response)
        
    async def _search_similar_exchanges(
        self, 
        embeddings: np.ndarray, 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar past exchanges"""
        if self.vector_index.ntotal == 0:
            return []
            
        # Search in vector index
        distances, indices = self.vector_index.search(embeddings.reshape(1, -1), k)
        
        # Retrieve exchanges
        similar = []
        cursor = self.db_conn.execute(
            "SELECT user_input, assistant_response, intent FROM exchanges LIMIT ?",
            (k,)
        )
        
        for i, row in enumerate(cursor):
            if i in indices[0]:
                similar.append({
                    "user_input": row[0],
                    "assistant_response": row[1],
                    "intent": row[2],
                    "similarity": float(1 / (1 + distances[0][i]))
                })
                
        return similar
        
    async def _get_recent_history(self, limit: int = 5) -> List[Dict[str, str]]:
        """Get recent conversation history"""
        if self.redis_client:
            # Try Redis first
            recent = self.redis_client.lrange(
                f"recent:{self.conversation_id}",
                0,
                limit - 1
            )
            if recent:
                return [json.loads(r) for r in recent]
                
        # Fallback to database
        cursor = self.db_conn.execute("""
            SELECT user_input, assistant_response 
            FROM exchanges 
            WHERE conversation_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (self.conversation_id, limit))
        
        return [
            {"user": row[0], "assistant": row[1]}
            for row in reversed(cursor.fetchall())
        ]
        
    async def _get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences"""
        cursor = self.db_conn.execute("SELECT key, value FROM user_preferences")
        return {row[0]: json.loads(row[1]) for row in cursor}
        
    async def _build_current_context(
        self, 
        intent: str, 
        entities: Dict[str, Any]
    ) -> str:
        """Build current context summary"""
        context_parts = []
        
        # Add time context
        now = datetime.now()
        context_parts.append(f"Current time: {now.strftime('%I:%M %p')}")
        context_parts.append(f"Date: {now.strftime('%A, %B %d, %Y')}")
        
        # Add location if available
        if "location" in entities:
            context_parts.append(f"Location: {entities['location']}")
            
        # Add any active reminders or tasks
        # (would check calendar/reminders here)
        
        return ". ".join(context_parts)
        
    async def _learn_from_exchange(
        self, 
        user_input: str, 
        assistant_response: str
    ):
        """Learn patterns from successful exchanges"""
        # Simple pattern learning - in production, use more sophisticated ML
        
        # Check if this exchange was successful (no follow-up correction)
        # For now, assume all exchanges are successful
        
        # Extract potential patterns
        words = user_input.lower().split()
        if len(words) >= 3:
            # Store trigrams as patterns
            for i in range(len(words) - 2):
                pattern = " ".join(words[i:i+3])
                
                # Check if pattern exists
                cursor = self.db_conn.execute(
                    "SELECT id, usage_count FROM learned_patterns WHERE pattern = ?",
                    (pattern,)
                )
                row = cursor.fetchone()
                
                if row:
                    # Update usage count
                    self.db_conn.execute(
                        "UPDATE learned_patterns SET usage_count = ? WHERE id = ?",
                        (row[1] + 1, row[0])
                    )
                else:
                    # Insert new pattern
                    self.db_conn.execute("""
                        INSERT INTO learned_patterns 
                        (pattern, response_template, confidence, created_at)
                        VALUES (?, ?, ?, ?)
                    """, (pattern, assistant_response, 0.5, datetime.now()))
                    
            self.db_conn.commit()
            
    async def save_conversation(self, conversation_id: str):
        """Save conversation summary"""
        # Generate summary (in production, use AI to summarize)
        cursor = self.db_conn.execute(
            "SELECT COUNT(*) FROM exchanges WHERE conversation_id = ?",
            (conversation_id,)
        )
        exchange_count = cursor.fetchone()[0]
        
        summary = f"Conversation with {exchange_count} exchanges"
        
        # Update conversation record
        self.db_conn.execute("""
            UPDATE conversations 
            SET ended_at = ?, summary = ?
            WHERE id = ?
        """, (datetime.now(), summary, conversation_id))
        self.db_conn.commit()
        
        # Clear Redis data
        if self.redis_client:
            self.redis_client.delete(
                f"conversation:{conversation_id}",
                f"context:{conversation_id}",
                f"recent:{conversation_id}"
            )
            
    async def shutdown(self):
        """Cleanup memory systems"""
        logger.info("Shutting down context manager...")
        
        if self.db_conn:
            self.db_conn.close()
            
        if self.redis_client:
            self.redis_client.close()