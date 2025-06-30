#!/usr/bin/env python3
"""
Fix database module to match test expectations exactly
"""

import shutil
from pathlib import Path

def backup_and_fix_database():
    """Backup current database.py and replace with test-compatible version"""
    
    db_file = Path("core/database.py")
    
    # Backup existing file
    if db_file.exists():
        backup_file = db_file.with_suffix('.py.backup')
        shutil.copy(db_file, backup_file)
        print(f"✅ Backed up existing database.py to {backup_file}")
    
    # Write test-compatible database implementation
    test_compatible_db = '''#!/usr/bin/env python3
"""
Database Layer for JARVIS - Test Compatible Version
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid
import json
import logging
from contextlib import contextmanager
import numpy as np

from sqlalchemy import (
    create_engine, Column, String, DateTime, Float, JSON, Integer, 
    LargeBinary, Text, Boolean, ForeignKey, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)
Base = declarative_base()


class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    context = Column(JSON, default=dict)


class Message(Base):
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True)
    conversation_id = Column(String, nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, default=dict)


class Learning(Base):
    __tablename__ = "learnings"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    learning_type = Column(String, nullable=False)
    content = Column(JSON, nullable=False)
    context = Column(JSON, default=dict)
    confidence_score = Column(Float, default=0.5)
    reinforcement_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Memory(Base):
    __tablename__ = "memories"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    memory_type = Column(String, nullable=False)
    content = Column(JSON, nullable=False)
    importance_score = Column(Float, default=0.5)
    access_count = Column(Integer, default=0)
    tags = Column(JSON, default=list)
    embedding = Column(LargeBinary, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    
    def set_embedding(self, embedding: np.ndarray):
        """Store numpy array as binary"""
        self.embedding = embedding.tobytes()
    
    def get_embedding(self) -> Optional[np.ndarray]:
        """Retrieve binary as numpy array"""
        if self.embedding:
            return np.frombuffer(self.embedding, dtype=np.float32)
        return None


class AgentState(Base):
    __tablename__ = "agent_states"
    
    id = Column(String, primary_key=True)
    agent_id = Column(String, nullable=False, unique=True)
    state_data = Column(JSON, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Task(Base):
    __tablename__ = "tasks"
    
    id = Column(String, primary_key=True)
    data = Column(JSON, nullable=False)
    status = Column(String, default='pending')
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class DatabaseManager:
    """Manages database operations with test-compatible interface"""
    
    def __init__(self, connection_string: str = None):
        if connection_string is None:
            connection_string = os.environ.get(
                "DATABASE_URL", "sqlite:///jarvis.db"
            )
        
        self.engine = create_engine(
            connection_string,
            poolclass=QueuePool if not connection_string.startswith("sqlite") else None,
            pool_size=10 if not connection_string.startswith("sqlite") else None,
            echo=False
        )
        
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.logger = logger
    
    @contextmanager
    def get_session(self) -> Session:
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    def create_conversation(self, user_id: str, context: Optional[Dict] = None) -> str:
        """Create new conversation"""
        conv_id = str(uuid.uuid4())
        
        with self.get_session() as session:
            conversation = Conversation(
                id=conv_id,
                user_id=user_id,
                context=context or {}
            )
            session.add(conversation)
        
        return conv_id
    
    def add_message(self, conversation_id: str, role: str, content: str,
                   metadata: Optional[Dict] = None) -> str:
        """Add message to conversation"""
        msg_id = str(uuid.uuid4())
        
        with self.get_session() as session:
            message = Message(
                id=msg_id,
                conversation_id=conversation_id,
                role=role,
                content=content,
                metadata=metadata or {}
            )
            session.add(message)
        
        return msg_id
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Get all messages in conversation"""
        with self.get_session() as session:
            messages = session.query(Message).filter_by(
                conversation_id=conversation_id
            ).order_by(Message.timestamp).all()
            
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                }
                for msg in messages
            ]
    
    def record_learning(self, user_id: str, learning_type: str, 
                       content: Dict[str, Any], context: Optional[Dict] = None,
                       confidence: float = 0.5) -> str:
        """Record new learning"""
        learning_id = str(uuid.uuid4())
        
        with self.get_session() as session:
            learning = Learning(
                id=learning_id,
                user_id=user_id,
                learning_type=learning_type,
                content=content,
                context=context or {},
                confidence_score=confidence
            )
            session.add(learning)
        
        return learning_id
    
    def get_learnings(self, user_id: str, learning_type: Optional[str] = None) -> List[Dict]:
        """Get user learnings"""
        with self.get_session() as session:
            query = session.query(Learning).filter_by(user_id=user_id)
            
            if learning_type:
                query = query.filter_by(learning_type=learning_type)
            
            learnings = query.all()
            
            return [
                {
                    "id": l.id,
                    "learning_type": l.learning_type,
                    "content": l.content,
                    "context": l.context,
                    "confidence_score": l.confidence_score,
                    "reinforcement_count": l.reinforcement_count
                }
                for l in learnings
            ]
    
    def reinforce_learning(self, learning_id: str, positive: bool = True):
        """Reinforce or weaken a learning"""
        with self.get_session() as session:
            learning = session.query(Learning).filter_by(id=learning_id).first()
            
            if learning:
                if positive:
                    learning.reinforcement_count += 1
                    learning.confidence_score = min(1.0, learning.confidence_score + 0.1)
                else:
                    learning.reinforcement_count = max(0, learning.reinforcement_count - 1)
                    learning.confidence_score = max(0.0, learning.confidence_score - 0.1)
                
                learning.updated_at = datetime.utcnow()
    
    def store_memory(self, user_id: str, memory_type: str, content: Dict[str, Any],
                    importance: float = 0.5, tags: Optional[List[str]] = None) -> str:
        """Store new memory"""
        memory_id = str(uuid.uuid4())
        
        with self.get_session() as session:
            memory = Memory(
                id=memory_id,
                user_id=user_id,
                memory_type=memory_type,
                content=content,
                importance_score=importance,
                tags=tags or []
            )
            session.add(memory)
        
        return memory_id
    
    def search_memories(self, user_id: str, memory_type: Optional[str] = None,
                       tag: Optional[str] = None) -> List[Dict]:
        """Search user memories"""
        with self.get_session() as session:
            query = session.query(Memory).filter_by(user_id=user_id)
            
            if memory_type:
                query = query.filter_by(memory_type=memory_type)
            
            memories = query.all()
            
            # Filter by tag if specified
            if tag:
                memories = [m for m in memories if tag in (m.tags or [])]
            
            return [
                {
                    "id": m.id,
                    "memory_type": m.memory_type,
                    "content": m.content,
                    "importance_score": m.importance_score,
                    "tags": m.tags,
                    "access_count": m.access_count
                }
                for m in memories
            ]
    
    def save_agent_state(self, agent_id: str, state_data: Dict[str, Any]):
        """Save or update agent state"""
        with self.get_session() as session:
            # Check if state exists
            state = session.query(AgentState).filter_by(agent_id=agent_id).first()
            
            if state:
                state.state_data = state_data
                state.updated_at = datetime.utcnow()
            else:
                state = AgentState(
                    id=str(uuid.uuid4()),
                    agent_id=agent_id,
                    state_data=state_data
                )
                session.add(state)
    
    def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent state"""
        with self.get_session() as session:
            state = session.query(AgentState).filter_by(agent_id=agent_id).first()
            return state.state_data if state else None
    
    def create_task(self, task_data: Dict[str, Any]) -> str:
        """Create new task"""
        task_id = str(uuid.uuid4())
        
        with self.get_session() as session:
            task = Task(
                id=task_id,
                data=task_data,
                status='pending'
            )
            session.add(task)
        
        return task_id
    
    def update_task_status(self, task_id: str, status: str, updates: Optional[Dict] = None):
        """Update task status"""
        with self.get_session() as session:
            task = session.query(Task).filter_by(id=task_id).first()
            
            if task:
                task.status = status
                if updates:
                    task.data = {**task.data, **updates}
                if status == 'completed':
                    task.completed_at = datetime.utcnow()
    
    def get_conversation_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get conversation analytics for user"""
        with self.get_session() as session:
            # Total conversations
            total_convs = session.query(func.count(Conversation.id)).filter_by(
                user_id=user_id
            ).scalar() or 0
            
            # Active conversations
            active_convs = session.query(func.count(Conversation.id)).filter_by(
                user_id=user_id, ended_at=None
            ).scalar() or 0
            
            # Get conversation IDs
            conv_ids = session.query(Conversation.id).filter_by(user_id=user_id).all()
            conv_ids = [c[0] for c in conv_ids]
            
            # Total messages
            total_messages = 0
            if conv_ids:
                total_messages = session.query(func.count(Message.id)).filter(
                    Message.conversation_id.in_(conv_ids)
                ).scalar() or 0
            
            avg_messages = total_messages / total_convs if total_convs > 0 else 0
            
            return {
                "total_conversations": total_convs,
                "active_conversations": active_convs,
                "total_messages": total_messages,
                "avg_messages_per_conversation": avg_messages
            }
    
    def cleanup_old_data(self, days: int = 90) -> Dict[str, int]:
        """Clean up old data"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        with self.get_session() as session:
            # Find old conversations
            old_convs = session.query(Conversation).filter(
                Conversation.ended_at < cutoff_date
            ).all()
            
            deleted_convs = len(old_convs)
            
            # Delete associated messages and conversations
            for conv in old_convs:
                session.query(Message).filter_by(conversation_id=conv.id).delete()
                session.delete(conv)
            
            return {"conversations": deleted_convs}


# Global instance
db_manager = None


def init_database(connection_string: str = None):
    """Initialize global database manager"""
    global db_manager
    import os
    
    if connection_string is None:
        connection_string = os.environ.get("DATABASE_URL", "sqlite:///jarvis.db")
    
    db_manager = DatabaseManager(connection_string)
    return db_manager


# Import os at module level
import os
'''
    
    db_file.write_text(test_compatible_db)
    print("✅ Replaced database.py with test-compatible version")

if __name__ == "__main__":
    backup_and_fix_database()