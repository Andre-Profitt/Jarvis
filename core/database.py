#!/usr/bin/env python3
"""
Database Layer for JARVIS
Persistent storage for memories, learning, and state
"""

import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, Column, String, DateTime, Float, JSON, Text, Integer, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import QueuePool
import json
import logging
from contextlib import contextmanager
import pickle
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

# Database Models

class Conversation(Base):
    """Store conversation history"""
    __tablename__ = 'conversations'
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    context = Column(JSON, default=dict)
    summary = Column(Text, nullable=True)
    sentiment_score = Column(Float, default=0.0)
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    learnings = relationship("Learning", back_populates="conversation")


class Message(Base):
    """Individual messages in conversations"""
    __tablename__ = 'messages'
    
    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey('conversations.id'), nullable=False)
    role = Column(String, nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_used = Column(String, nullable=True)
    confidence_score = Column(Float, nullable=True)
    feedback_score = Column(Float, nullable=True)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")


class Learning(Base):
    """Things JARVIS has learned"""
    __tablename__ = 'learnings'
    
    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey('conversations.id'), nullable=True)
    learned_at = Column(DateTime, default=datetime.utcnow)
    learning_type = Column(String, nullable=False)  # 'fact', 'preference', 'correction', 'pattern'
    content = Column(JSON, nullable=False)
    confidence = Column(Float, default=0.5)
    reinforcement_count = Column(Integer, default=0)
    last_used = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="learnings")


class Memory(Base):
    """Long-term memory storage"""
    __tablename__ = 'memories'
    
    id = Column(String, primary_key=True)
    memory_type = Column(String, nullable=False)  # 'episodic', 'semantic', 'procedural'
    content = Column(JSON, nullable=False)
    importance_score = Column(Float, default=0.5)
    access_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    embedding = Column(Text, nullable=True)  # Serialized vector
    tags = Column(JSON, default=list)
    
    # Methods
    def get_embedding(self) -> Optional[np.ndarray]:
        """Deserialize embedding vector"""
        if self.embedding:
            return pickle.loads(self.embedding.encode('latin1'))
        return None
    
    def set_embedding(self, vector: np.ndarray):
        """Serialize embedding vector"""
        self.embedding = pickle.dumps(vector).decode('latin1')


class AgentState(Base):
    """Persistent state for swarm agents"""
    __tablename__ = 'agent_states'
    
    agent_id = Column(String, primary_key=True)
    agent_type = Column(String, nullable=False)
    capabilities = Column(JSON, default=list)
    position = Column(JSON, nullable=True)  # For swarm behaviors
    reputation = Column(Float, default=1.0)
    task_history = Column(JSON, default=list)
    knowledge_base = Column(JSON, default=dict)
    last_active = Column(DateTime, default=datetime.utcnow)
    total_tasks_completed = Column(Integer, default=0)
    average_performance = Column(Float, default=0.0)


class Task(Base):
    """Task tracking and history"""
    __tablename__ = 'tasks'
    
    id = Column(String, primary_key=True)
    task_type = Column(String, nullable=False)
    status = Column(String, default='pending')
    priority = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    assigned_agents = Column(JSON, default=list)
    payload = Column(JSON, default=dict)
    result = Column(JSON, nullable=True)
    performance_metrics = Column(JSON, default=dict)


class UserPreference(Base):
    """User preferences and settings"""
    __tablename__ = 'user_preferences'
    
    user_id = Column(String, primary_key=True)
    preferences = Column(JSON, default=dict)
    communication_style = Column(String, default='friendly')
    timezone = Column(String, default='UTC')
    language = Column(String, default='en')
    voice_settings = Column(JSON, default=dict)
    privacy_settings = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ModelCheckpoint(Base):
    """Model versioning and checkpoints"""
    __tablename__ = 'model_checkpoints'
    
    id = Column(String, primary_key=True)
    model_name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    metrics = Column(JSON, default=dict)
    parameters = Column(JSON, default=dict)
    file_path = Column(String, nullable=False)
    is_active = Column(Boolean, default=False)
    rollback_count = Column(Integer, default=0)


# Database Manager

class DatabaseManager:
    """Manage database connections and operations"""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv('DATABASE_URL', 'sqlite:///jarvis.db')
        
        # Create engine with connection pooling
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=40,
            pool_pre_ping=True,  # Verify connections before using
            echo=False
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False
        )
        
        # Create tables
        self.create_tables()
        
        logger.info(f"Database initialized: {self.database_url}")
    
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created/verified")
    
    @contextmanager
    def get_session(self) -> Session:
        """Context manager for database sessions"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    # Conversation Management
    
    def create_conversation(self, user_id: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Create new conversation"""
        with self.get_session() as session:
            conversation = Conversation(
                id=f"conv_{datetime.utcnow().timestamp()}",
                user_id=user_id,
                context=context or {}
            )
            session.add(conversation)
            return conversation.id
    
    def add_message(self, 
                   conversation_id: str,
                   role: str,
                   content: str,
                   model_used: Optional[str] = None,
                   confidence: Optional[float] = None) -> str:
        """Add message to conversation"""
        with self.get_session() as session:
            message = Message(
                id=f"msg_{datetime.utcnow().timestamp()}",
                conversation_id=conversation_id,
                role=role,
                content=content,
                model_used=model_used,
                confidence_score=confidence
            )
            session.add(message)
            return message.id
    
    def get_conversation_history(self, 
                               conversation_id: str,
                               limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation messages"""
        with self.get_session() as session:
            query = session.query(Message).filter_by(
                conversation_id=conversation_id
            ).order_by(Message.timestamp)
            
            if limit:
                query = query.limit(limit)
            
            messages = query.all()
            
            return [{
                'id': msg.id,
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat(),
                'model_used': msg.model_used,
                'confidence': msg.confidence_score
            } for msg in messages]
    
    # Learning Management
    
    def record_learning(self,
                       learning_type: str,
                       content: Dict[str, Any],
                       conversation_id: Optional[str] = None,
                       confidence: float = 0.5) -> str:
        """Record something JARVIS learned"""
        with self.get_session() as session:
            learning = Learning(
                id=f"learn_{datetime.utcnow().timestamp()}",
                conversation_id=conversation_id,
                learning_type=learning_type,
                content=content,
                confidence=confidence
            )
            session.add(learning)
            return learning.id
    
    def get_learnings(self,
                     learning_type: Optional[str] = None,
                     min_confidence: float = 0.0,
                     limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve learnings"""
        with self.get_session() as session:
            query = session.query(Learning).filter(
                Learning.confidence >= min_confidence
            )
            
            if learning_type:
                query = query.filter_by(learning_type=learning_type)
            
            learnings = query.order_by(
                Learning.confidence.desc()
            ).limit(limit).all()
            
            return [{
                'id': l.id,
                'type': l.learning_type,
                'content': l.content,
                'confidence': l.confidence,
                'learned_at': l.learned_at.isoformat()
            } for l in learnings]
    
    def reinforce_learning(self, learning_id: str, positive: bool = True):
        """Reinforce or weaken a learning"""
        with self.get_session() as session:
            learning = session.query(Learning).filter_by(id=learning_id).first()
            if learning:
                if positive:
                    learning.confidence = min(1.0, learning.confidence + 0.1)
                    learning.reinforcement_count += 1
                else:
                    learning.confidence = max(0.0, learning.confidence - 0.1)
                learning.last_used = datetime.utcnow()
    
    # Memory Management
    
    def store_memory(self,
                    memory_type: str,
                    content: Dict[str, Any],
                    importance: float = 0.5,
                    tags: Optional[List[str]] = None,
                    embedding: Optional[np.ndarray] = None) -> str:
        """Store long-term memory"""
        with self.get_session() as session:
            memory = Memory(
                id=f"mem_{datetime.utcnow().timestamp()}",
                memory_type=memory_type,
                content=content,
                importance_score=importance,
                tags=tags or []
            )
            
            if embedding is not None:
                memory.set_embedding(embedding)
            
            session.add(memory)
            return memory.id
    
    def search_memories(self,
                       query_embedding: Optional[np.ndarray] = None,
                       memory_type: Optional[str] = None,
                       tags: Optional[List[str]] = None,
                       min_importance: float = 0.0,
                       limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories by various criteria"""
        with self.get_session() as session:
            query = session.query(Memory).filter(
                Memory.importance_score >= min_importance
            )
            
            if memory_type:
                query = query.filter_by(memory_type=memory_type)
            
            memories = query.order_by(
                Memory.importance_score.desc(),
                Memory.last_accessed.desc()
            ).limit(limit * 2).all()  # Get more for embedding filtering
            
            results = []
            for memory in memories:
                # Update access count
                memory.access_count += 1
                memory.last_accessed = datetime.utcnow()
                
                result = {
                    'id': memory.id,
                    'type': memory.memory_type,
                    'content': memory.content,
                    'importance': memory.importance_score,
                    'tags': memory.tags,
                    'created_at': memory.created_at.isoformat()
                }
                
                # Calculate similarity if embedding provided
                if query_embedding is not None and memory.embedding:
                    mem_embedding = memory.get_embedding()
                    similarity = np.dot(query_embedding, mem_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(mem_embedding)
                    )
                    result['similarity'] = float(similarity)
                
                results.append(result)
            
            # Sort by similarity if embedding search
            if query_embedding is not None:
                results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            return results[:limit]
    
    # Agent State Management
    
    def save_agent_state(self, agent_id: str, state_data: Dict[str, Any]):
        """Save agent state"""
        with self.get_session() as session:
            agent_state = session.query(AgentState).filter_by(
                agent_id=agent_id
            ).first()
            
            if not agent_state:
                agent_state = AgentState(agent_id=agent_id)
                session.add(agent_state)
            
            # Update fields
            for key, value in state_data.items():
                if hasattr(agent_state, key):
                    setattr(agent_state, key, value)
            
            agent_state.last_active = datetime.utcnow()
    
    def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent state"""
        with self.get_session() as session:
            agent_state = session.query(AgentState).filter_by(
                agent_id=agent_id
            ).first()
            
            if agent_state:
                return {
                    'agent_id': agent_state.agent_id,
                    'agent_type': agent_state.agent_type,
                    'capabilities': agent_state.capabilities,
                    'reputation': agent_state.reputation,
                    'task_history': agent_state.task_history,
                    'knowledge_base': agent_state.knowledge_base,
                    'performance': agent_state.average_performance
                }
            
            return None
    
    # Task Management
    
    def create_task(self, task_data: Dict[str, Any]) -> str:
        """Create new task"""
        with self.get_session() as session:
            task = Task(
                id=f"task_{datetime.utcnow().timestamp()}",
                **task_data
            )
            session.add(task)
            return task.id
    
    def update_task_status(self, 
                          task_id: str,
                          status: str,
                          result: Optional[Dict[str, Any]] = None):
        """Update task status"""
        with self.get_session() as session:
            task = session.query(Task).filter_by(id=task_id).first()
            if task:
                task.status = status
                if status == 'completed':
                    task.completed_at = datetime.utcnow()
                    if result:
                        task.result = result
                elif status == 'in_progress':
                    task.started_at = datetime.utcnow()
    
    # Analytics
    
    def get_conversation_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get conversation analytics for user"""
        with self.get_session() as session:
            conversations = session.query(Conversation).filter_by(
                user_id=user_id
            ).all()
            
            total_messages = session.query(Message).join(
                Conversation
            ).filter(Conversation.user_id == user_id).count()
            
            avg_sentiment = session.query(
                func.avg(Conversation.sentiment_score)
            ).filter_by(user_id=user_id).scalar() or 0.0
            
            return {
                'total_conversations': len(conversations),
                'total_messages': total_messages,
                'average_sentiment': float(avg_sentiment),
                'active_conversations': sum(1 for c in conversations if c.ended_at is None)
            }
    
    def cleanup_old_data(self, days: int = 90):
        """Clean up old data"""
        with self.get_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Delete old completed tasks
            session.query(Task).filter(
                Task.completed_at < cutoff_date,
                Task.status == 'completed'
            ).delete()
            
            # Reduce importance of old memories
            old_memories = session.query(Memory).filter(
                Memory.last_accessed < cutoff_date
            ).all()
            
            for memory in old_memories:
                memory.importance_score *= 0.9


# Singleton instance
db_manager = DatabaseManager()


# Helper functions
def init_database():
    """Initialize database with tables"""
    db_manager.create_tables()
    logger.info("Database initialized successfully")


if __name__ == "__main__":
    init_database()