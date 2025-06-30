"""
JARVIS Phase 3: Context Persistence Manager
===========================================
Advanced context management that maintains conversation and activity context
across interactions, building on the enhanced episodic memory system.

This manager provides:
- Cross-session context preservation
- Activity pattern recognition
- Context-aware response adaptation
- Seamless conversation continuity
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import numpy as np
from pathlib import Path

# Import Phase 1 components
from .unified_input_pipeline import UnifiedInputPipeline, ProcessedInput
from .fluid_state_management import FluidStateManager, SystemState

# Import existing memory system
from .enhanced_episodic_memory import (
    EpisodicMemorySystem,
    MemoryContext,
    EmotionalContext,
    RetrievalStrategy,
    RetrievalContext,
    EmotionType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConversationThread:
    """Represents a continuous conversation thread"""
    thread_id: str
    start_time: datetime
    last_update: datetime
    topic: str
    participants: List[str]
    context_stack: deque = field(default_factory=lambda: deque(maxlen=10))
    key_points: List[Dict[str, Any]] = field(default_factory=list)
    emotional_trajectory: List[EmotionalContext] = field(default_factory=list)
    is_active: bool = True
    importance_score: float = 0.5


@dataclass
class ActivityContext:
    """Represents an ongoing activity or task"""
    activity_id: str
    activity_type: str  # coding, research, communication, planning, etc.
    start_time: datetime
    last_action: datetime
    related_files: List[str] = field(default_factory=list)
    related_entities: List[str] = field(default_factory=list)
    progress_markers: List[Dict[str, Any]] = field(default_factory=list)
    interruption_count: int = 0
    focus_score: float = 1.0  # Decreases with interruptions


@dataclass
class UserPreferences:
    """Learned user preferences"""
    preferred_response_length: str = "medium"  # short, medium, long
    formality_level: float = 0.5  # 0 = casual, 1 = formal
    technical_depth: float = 0.7  # 0 = simple, 1 = expert
    proactive_suggestions: bool = True
    interruption_threshold: float = 0.7  # How important something must be to interrupt
    working_hours: List[Tuple[int, int]] = field(default_factory=lambda: [(9, 17)])
    communication_style: Dict[str, float] = field(default_factory=dict)


class ContextPersistenceManager:
    """
    Advanced context persistence manager for JARVIS Phase 3.
    Maintains conversation threads, activity contexts, and user preferences
    across sessions using the enhanced episodic memory system.
    """
    
    def __init__(self, 
                 memory_system: EpisodicMemorySystem,
                 state_manager: FluidStateManager,
                 persistence_path: str = "./context_persistence"):
        self.memory_system = memory_system
        self.state_manager = state_manager
        self.persistence_path = Path(persistence_path)
        self.persistence_path.mkdir(exist_ok=True)
        
        # Active contexts
        self.conversation_threads: Dict[str, ConversationThread] = {}
        self.activity_contexts: Dict[str, ActivityContext] = {}
        self.user_preferences = UserPreferences()
        
        # Context indices
        self.topic_index: Dict[str, Set[str]] = defaultdict(set)  # topic -> thread_ids
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)  # entity -> activity_ids
        self.temporal_index: Dict[datetime, List[str]] = defaultdict(list)
        
        # Pattern tracking
        self.interaction_patterns: deque = deque(maxlen=1000)
        self.context_switches: List[Dict[str, Any]] = []
        self.focus_periods: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.context_hit_rate = 0.0
        self.prediction_accuracy = 0.0
        self.continuity_score = 0.0
        
        # Background tasks
        self.running = False
        
    async def initialize(self):
        """Initialize context persistence manager"""
        logger.info("🎯 Initializing Context Persistence Manager")
        
        # Load persisted contexts
        await self._load_persisted_contexts()
        
        # Load user preferences
        await self._load_user_preferences()
        
        # Start background tasks
        self.running = True
        asyncio.create_task(self._context_maintenance_loop())
        asyncio.create_task(self._pattern_analysis_loop())
        
        logger.info("✅ Context Persistence Manager initialized")
    
    async def process_input_with_context(self, 
                                        processed_input: ProcessedInput,
                                        current_state: SystemState) -> Dict[str, Any]:
        """
        Process input with full context awareness.
        Returns enriched context for the current interaction.
        """
        # Identify relevant conversation thread
        thread = await self._identify_conversation_thread(processed_input)
        
        # Identify current activity
        activity = await self._identify_current_activity(processed_input)
        
        # Retrieve relevant memories
        memories = await self._retrieve_relevant_memories(processed_input, thread, activity)
        
        # Build comprehensive context
        context = {
            "conversation_thread": thread,
            "current_activity": activity,
            "relevant_memories": memories,
            "user_preferences": self.user_preferences,
            "system_state": current_state,
            "timestamp": datetime.now()
        }
        
        # Update thread and activity
        if thread:
            await self._update_conversation_thread(thread, processed_input, context)
        if activity:
            await self._update_activity_context(activity, processed_input)
        
        # Track interaction pattern
        self._track_interaction_pattern(processed_input, context)
        
        # Store in episodic memory
        await self._store_interaction_memory(processed_input, context)
        
        return context
    
    async def _identify_conversation_thread(self, 
                                          processed_input: ProcessedInput) -> Optional[ConversationThread]:
        """Identify which conversation thread this input belongs to"""
        # Check for explicit thread continuation
        if processed_input.metadata.get("thread_id"):
            thread_id = processed_input.metadata["thread_id"]
            if thread_id in self.conversation_threads:
                return self.conversation_threads[thread_id]
        
        # Find thread by topic similarity
        input_topics = await self._extract_topics(processed_input)
        best_thread = None
        best_score = 0.0
        
        for thread in self.conversation_threads.values():
            if not thread.is_active:
                continue
                
            # Check temporal proximity
            time_diff = (datetime.now() - thread.last_update).total_seconds()
            if time_diff > 3600:  # 1 hour gap = likely new thread
                continue
            
            # Check topic overlap
            thread_topics = {thread.topic} | set(kp.get("topic", "") for kp in thread.key_points)
            overlap = len(input_topics & thread_topics) / max(len(input_topics), 1)
            
            # Calculate similarity score
            temporal_score = 1.0 - (time_diff / 3600)  # Decay over 1 hour
            topic_score = overlap
            score = temporal_score * 0.3 + topic_score * 0.7
            
            if score > best_score and score > 0.5:
                best_score = score
                best_thread = thread
        
        # Create new thread if no match
        if not best_thread:
            best_thread = await self._create_conversation_thread(processed_input, input_topics)
        
        return best_thread
    
    async def _extract_topics(self, processed_input: ProcessedInput) -> Set[str]:
        """Extract topics from input"""
        topics = set()
        
        # Extract from different input types
        if processed_input.input_type == "voice":
            # Would use NLP here
            text = processed_input.processed_data.get("text", "")
            topics.update(self._simple_topic_extraction(text))
        elif processed_input.input_type == "text":
            topics.update(self._simple_topic_extraction(processed_input.content))
        
        return topics
    
    def _simple_topic_extraction(self, text: str) -> Set[str]:
        """Simple topic extraction (would use NLP in production)"""
        # Keywords that often indicate topics
        keywords = []
        words = text.lower().split()
        
        # Extract capitalized words (likely proper nouns)
        for word in text.split():
            if word and word[0].isupper() and len(word) > 3:
                keywords.append(word.lower())
        
        # Extract long words (likely important)
        keywords.extend([w for w in words if len(w) > 7])
        
        return set(keywords[:5])  # Limit to top 5
    
    async def _create_conversation_thread(self, 
                                        processed_input: ProcessedInput,
                                        topics: Set[str]) -> ConversationThread:
        """Create a new conversation thread"""
        thread_id = f"thread_{datetime.now().timestamp()}"
        main_topic = list(topics)[0] if topics else "general"
        
        thread = ConversationThread(
            thread_id=thread_id,
            start_time=datetime.now(),
            last_update=datetime.now(),
            topic=main_topic,
            participants=["user", "jarvis"],
            importance_score=processed_input.priority.value / 5.0
        )
        
        # Add initial context
        thread.context_stack.append({
            "input": processed_input.content,
            "timestamp": datetime.now(),
            "topics": list(topics)
        })
        
        self.conversation_threads[thread_id] = thread
        
        # Update indices
        for topic in topics:
            self.topic_index[topic].add(thread_id)
        
        logger.info(f"Created new conversation thread: {thread_id} about {main_topic}")
        
        return thread
    
    async def _identify_current_activity(self, 
                                       processed_input: ProcessedInput) -> Optional[ActivityContext]:
        """Identify current user activity"""
        # Check for explicit activity markers
        if processed_input.metadata.get("activity_type"):
            activity_type = processed_input.metadata["activity_type"]
            
            # Find matching activity
            for activity in self.activity_contexts.values():
                if (activity.activity_type == activity_type and 
                    (datetime.now() - activity.last_action).total_seconds() < 1800):  # 30 min
                    return activity
        
        # Infer activity from input
        activity_type = self._infer_activity_type(processed_input)
        
        if activity_type:
            # Find or create activity
            for activity in self.activity_contexts.values():
                if (activity.activity_type == activity_type and 
                    (datetime.now() - activity.last_action).total_seconds() < 1800):
                    return activity
            
            # Create new activity
            return await self._create_activity_context(activity_type, processed_input)
        
        return None
    
    def _infer_activity_type(self, processed_input: ProcessedInput) -> Optional[str]:
        """Infer activity type from input"""
        content = processed_input.content.lower()
        
        # Simple heuristics (would use ML in production)
        if any(word in content for word in ["code", "function", "debug", "implement"]):
            return "coding"
        elif any(word in content for word in ["research", "paper", "study", "analyze"]):
            return "research"
        elif any(word in content for word in ["email", "message", "reply", "send"]):
            return "communication"
        elif any(word in content for word in ["plan", "schedule", "meeting", "task"]):
            return "planning"
        elif any(word in content for word in ["learn", "tutorial", "course", "understand"]):
            return "learning"
        
        return None
    
    async def _create_activity_context(self, 
                                     activity_type: str,
                                     processed_input: ProcessedInput) -> ActivityContext:
        """Create new activity context"""
        activity_id = f"activity_{activity_type}_{datetime.now().timestamp()}"
        
        activity = ActivityContext(
            activity_id=activity_id,
            activity_type=activity_type,
            start_time=datetime.now(),
            last_action=datetime.now()
        )
        
        self.activity_contexts[activity_id] = activity
        
        logger.info(f"Created new activity context: {activity_type}")
        
        return activity
    
    async def _retrieve_relevant_memories(self,
                                        processed_input: ProcessedInput,
                                        thread: Optional[ConversationThread],
                                        activity: Optional[ActivityContext]) -> List[Any]:
        """Retrieve relevant memories for current context"""
        memories = []
        
        # Build query from context
        query_parts = [processed_input.content]
        
        if thread:
            query_parts.append(thread.topic)
            # Add recent context
            for ctx in list(thread.context_stack)[-3:]:
                query_parts.append(ctx.get("input", ""))
        
        if activity:
            query_parts.append(activity.activity_type)
        
        query = " ".join(query_parts)
        
        # Retrieve using different strategies
        retrieval_context = RetrievalContext(
            strategy=RetrievalStrategy.ASSOCIATIVE,
            max_results=5,
            time_window=timedelta(days=7)
        )
        
        # Get recent relevant memories
        recent_memories = await self.memory_system.recall(query, retrieval_context)
        memories.extend(recent_memories)
        
        # Get emotionally similar memories if high emotion
        if processed_input.urgency > 0.7:
            emotion_context = RetrievalContext(
                strategy=RetrievalStrategy.EMOTIONAL,
                max_results=3
            )
            emotional_memories = await self.memory_system.recall(query, emotion_context)
            memories.extend(emotional_memories)
        
        return memories[:8]  # Limit total
    
    async def _update_conversation_thread(self,
                                        thread: ConversationThread,
                                        processed_input: ProcessedInput,
                                        context: Dict[str, Any]):
        """Update conversation thread with new input"""
        thread.last_update = datetime.now()
        
        # Add to context stack
        thread.context_stack.append({
            "input": processed_input.content,
            "timestamp": datetime.now(),
            "state": context["system_state"].name,
            "response": None  # Will be filled by response
        })
        
        # Extract key points
        if processed_input.priority.value >= 3:  # HIGH or above
            thread.key_points.append({
                "content": processed_input.content,
                "timestamp": datetime.now(),
                "importance": processed_input.priority.value / 5.0
            })
        
        # Track emotional trajectory
        if "emotional_context" in context:
            thread.emotional_trajectory.append(context["emotional_context"])
        
        # Update importance
        thread.importance_score = max(
            thread.importance_score,
            processed_input.priority.value / 5.0
        )
    
    async def _update_activity_context(self,
                                     activity: ActivityContext,
                                     processed_input: ProcessedInput):
        """Update activity context"""
        activity.last_action = datetime.now()
        
        # Check for interruption
        time_since_last = (datetime.now() - activity.last_action).total_seconds()
        if time_since_last > 300:  # 5 minute gap
            activity.interruption_count += 1
            activity.focus_score *= 0.9  # Reduce focus score
        
        # Extract related entities
        entities = self._extract_entities(processed_input)
        activity.related_entities.extend(entities)
        
        # Track progress markers
        if any(word in processed_input.content.lower() 
               for word in ["done", "complete", "finished", "solved"]):
            activity.progress_markers.append({
                "type": "completion",
                "content": processed_input.content,
                "timestamp": datetime.now()
            })
    
    def _extract_entities(self, processed_input: ProcessedInput) -> List[str]:
        """Extract entities from input (simplified)"""
        entities = []
        
        # Extract file paths
        import re
        file_pattern = r'[\w/\\]+\.\w+'
        files = re.findall(file_pattern, processed_input.content)
        entities.extend(files)
        
        # Extract URLs
        url_pattern = r'https?://\S+'
        urls = re.findall(url_pattern, processed_input.content)
        entities.extend(urls)
        
        return entities
    
    def _track_interaction_pattern(self,
                                 processed_input: ProcessedInput,
                                 context: Dict[str, Any]):
        """Track interaction patterns for analysis"""
        pattern = {
            "timestamp": datetime.now(),
            "input_type": processed_input.input_type,
            "priority": processed_input.priority.value,
            "state": context["system_state"].name,
            "has_thread": context["conversation_thread"] is not None,
            "has_activity": context["current_activity"] is not None,
            "memory_count": len(context["relevant_memories"])
        }
        
        self.interaction_patterns.append(pattern)
        
        # Track context switches
        if len(self.interaction_patterns) > 1:
            prev = self.interaction_patterns[-2]
            if prev["state"] != pattern["state"]:
                self.context_switches.append({
                    "from_state": prev["state"],
                    "to_state": pattern["state"],
                    "timestamp": datetime.now(),
                    "trigger": processed_input.content[:50]
                })
    
    async def _store_interaction_memory(self,
                                      processed_input: ProcessedInput,
                                      context: Dict[str, Any]):
        """Store interaction in episodic memory"""
        # Prepare memory context
        memory_context = {
            "source": "context_persistence",
            "thread_id": context["conversation_thread"].thread_id if context["conversation_thread"] else None,
            "activity_id": context["current_activity"].activity_id if context["current_activity"] else None,
            "state": context["system_state"].name,
            "tags": ["interaction", processed_input.input_type]
        }
        
        # Add activity tags
        if context["current_activity"]:
            memory_context["tags"].append(context["current_activity"].activity_type)
        
        # Create interaction data
        interaction = {
            "messages": [{
                "content": processed_input.content,
                "timestamp": datetime.now(),
                "priority": processed_input.priority.value
            }],
            "context": memory_context
        }
        
        # Store in memory
        await self.memory_system.remember_interaction(interaction, memory_context)
    
    async def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context state"""
        active_threads = [t for t in self.conversation_threads.values() if t.is_active]
        active_activities = [
            a for a in self.activity_contexts.values() 
            if (datetime.now() - a.last_action).total_seconds() < 3600
        ]
        
        # Calculate metrics
        if self.interaction_patterns:
            recent_patterns = list(self.interaction_patterns)[-100:]
            self.context_hit_rate = sum(
                1 for p in recent_patterns if p["memory_count"] > 0
            ) / len(recent_patterns)
        
        return {
            "active_conversation_threads": len(active_threads),
            "active_activities": len(active_activities),
            "total_threads": len(self.conversation_threads),
            "total_activities": len(self.activity_contexts),
            "context_switches_today": len([
                s for s in self.context_switches 
                if s["timestamp"].date() == datetime.now().date()
            ]),
            "focus_periods_today": len([
                f for f in self.focus_periods 
                if f["date"] == datetime.now().date()
            ]),
            "context_hit_rate": self.context_hit_rate,
            "user_preferences": {
                "response_length": self.user_preferences.preferred_response_length,
                "formality": self.user_preferences.formality_level,
                "technical_depth": self.user_preferences.technical_depth
            },
            "top_topics": self._get_top_topics(),
            "activity_distribution": self._get_activity_distribution()
        }
    
    def _get_top_topics(self) -> List[Tuple[str, int]]:
        """Get most discussed topics"""
        topic_counts = {}
        for topic, thread_ids in self.topic_index.items():
            active_count = sum(
                1 for tid in thread_ids 
                if tid in self.conversation_threads and 
                self.conversation_threads[tid].is_active
            )
            if active_count > 0:
                topic_counts[topic] = active_count
        
        return sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _get_activity_distribution(self) -> Dict[str, float]:
        """Get distribution of activities"""
        activity_times = defaultdict(float)
        
        for activity in self.activity_contexts.values():
            duration = (activity.last_action - activity.start_time).total_seconds() / 3600
            activity_times[activity.activity_type] += duration
        
        total_time = sum(activity_times.values())
        if total_time > 0:
            return {k: v/total_time for k, v in activity_times.items()}
        return {}
    
    async def _context_maintenance_loop(self):
        """Background task for context maintenance"""
        while self.running:
            try:
                # Archive old threads
                cutoff_time = datetime.now() - timedelta(hours=24)
                for thread in list(self.conversation_threads.values()):
                    if thread.last_update < cutoff_time and thread.is_active:
                        thread.is_active = False
                        await self._archive_thread(thread)
                
                # Clean up stale activities  
                activity_cutoff = datetime.now() - timedelta(hours=2)
                stale_activities = []
                for aid, activity in self.activity_contexts.items():
                    if activity.last_action < activity_cutoff:
                        stale_activities.append(aid)
                
                for aid in stale_activities:
                    await self._archive_activity(self.activity_contexts[aid])
                    del self.activity_contexts[aid]
                
                # Analyze focus periods
                await self._analyze_focus_periods()
                
                # Save state periodically
                await self._save_context_state()
                
            except Exception as e:
                logger.error(f"Context maintenance error: {e}")
            
            await asyncio.sleep(300)  # Run every 5 minutes
    
    async def _pattern_analysis_loop(self):
        """Background task for pattern analysis"""
        while self.running:
            try:
                # Analyze interaction patterns
                if len(self.interaction_patterns) >= 50:
                    patterns = await self._analyze_patterns()
                    
                    # Update user preferences based on patterns
                    await self._update_user_preferences(patterns)
                    
                    # Identify workflow patterns
                    workflows = await self._identify_workflows(patterns)
                    
                    # Store insights
                    await self._store_pattern_insights(patterns, workflows)
                
            except Exception as e:
                logger.error(f"Pattern analysis error: {e}")
            
            await asyncio.sleep(1800)  # Run every 30 minutes
    
    async def _analyze_patterns(self) -> Dict[str, Any]:
        """Analyze interaction patterns"""
        recent_patterns = list(self.interaction_patterns)[-500:]
        
        # Time-based patterns
        hour_distribution = defaultdict(int)
        state_distribution = defaultdict(int)
        
        for pattern in recent_patterns:
            hour = pattern["timestamp"].hour
            hour_distribution[hour] += 1
            state_distribution[pattern["state"]] += 1
        
        # Context effectiveness
        context_effectiveness = {
            "with_thread": [],
            "without_thread": [],
            "with_activity": [],
            "without_activity": []
        }
        
        for pattern in recent_patterns:
            if pattern["has_thread"]:
                context_effectiveness["with_thread"].append(pattern["memory_count"])
            else:
                context_effectiveness["without_thread"].append(pattern["memory_count"])
                
            if pattern["has_activity"]:
                context_effectiveness["with_activity"].append(pattern["memory_count"])
            else:
                context_effectiveness["without_activity"].append(pattern["memory_count"])
        
        return {
            "hour_distribution": dict(hour_distribution),
            "state_distribution": dict(state_distribution),
            "avg_memories_with_thread": np.mean(context_effectiveness["with_thread"]) if context_effectiveness["with_thread"] else 0,
            "avg_memories_without_thread": np.mean(context_effectiveness["without_thread"]) if context_effectiveness["without_thread"] else 0,
            "context_switches_per_hour": len(self.context_switches) / max(len(recent_patterns) / 12, 1)
        }
    
    async def _update_user_preferences(self, patterns: Dict[str, Any]):
        """Update user preferences based on observed patterns"""
        # Update working hours based on activity
        if patterns["hour_distribution"]:
            active_hours = sorted(patterns["hour_distribution"].items(), 
                                key=lambda x: x[1], reverse=True)
            
            # Find continuous blocks
            work_blocks = []
            current_block = None
            
            for hour, count in active_hours:
                if count > 5:  # Threshold for active hour
                    if current_block and hour == current_block[1] + 1:
                        current_block = (current_block[0], hour)
                    else:
                        if current_block:
                            work_blocks.append(current_block)
                        current_block = (hour, hour)
            
            if current_block:
                work_blocks.append(current_block)
            
            if work_blocks:
                self.user_preferences.working_hours = work_blocks[:3]  # Top 3 blocks
    
    async def _identify_workflows(self, patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify common workflows from patterns"""
        workflows = []
        
        # Analyze state transitions
        if self.context_switches:
            # Find common sequences
            sequences = defaultdict(int)
            for i in range(len(self.context_switches) - 2):
                seq = tuple(self.context_switches[i:i+3])
                sequences[seq] += 1
            
            # Extract frequent workflows
            for seq, count in sequences.items():
                if count > 3:  # Threshold for workflow
                    workflows.append({
                        "sequence": [s["to_state"] for s in seq],
                        "frequency": count,
                        "type": "state_transition"
                    })
        
        return workflows
    
    async def _store_pattern_insights(self, 
                                    patterns: Dict[str, Any],
                                    workflows: List[Dict[str, Any]]):
        """Store pattern insights in memory"""
        insights = {
            "timestamp": datetime.now(),
            "patterns": patterns,
            "workflows": workflows,
            "preferences_snapshot": {
                "working_hours": self.user_preferences.working_hours,
                "formality": self.user_preferences.formality_level,
                "technical_depth": self.user_preferences.technical_depth
            }
        }
        
        # Store as memory
        interaction = {
            "messages": [{
                "content": f"Pattern analysis insights: {len(workflows)} workflows identified",
                "metadata": insights
            }],
            "summary": "Automated pattern analysis"
        }
        
        context = {
            "source": "pattern_analysis",
            "tags": ["insights", "patterns", "automation"]
        }
        
        await self.memory_system.remember_interaction(interaction, context)
    
    async def _analyze_focus_periods(self):
        """Analyze and track focus periods"""
        today = datetime.now().date()
        today_activities = [
            a for a in self.activity_contexts.values()
            if a.start_time.date() == today
        ]
        
        for activity in today_activities:
            if activity.focus_score > 0.8 and activity.interruption_count < 2:
                duration = (activity.last_action - activity.start_time).total_seconds() / 3600
                if duration > 0.5:  # At least 30 minutes
                    self.focus_periods.append({
                        "date": today,
                        "activity_type": activity.activity_type,
                        "duration": duration,
                        "focus_score": activity.focus_score,
                        "time_of_day": activity.start_time.hour
                    })
    
    async def _archive_thread(self, thread: ConversationThread):
        """Archive conversation thread"""
        # Create summary
        summary = {
            "thread_id": thread.thread_id,
            "topic": thread.topic,
            "duration": (thread.last_update - thread.start_time).total_seconds() / 3600,
            "message_count": len(thread.context_stack),
            "key_points": thread.key_points,
            "importance": thread.importance_score
        }
        
        # Store in memory
        interaction = {
            "messages": thread.key_points,
            "summary": f"Conversation about {thread.topic}",
            "metadata": summary
        }
        
        context = {
            "source": "thread_archive",
            "tags": ["conversation", thread.topic],
            "thread_id": thread.thread_id
        }
        
        await self.memory_system.remember_interaction(interaction, context)
        
        logger.info(f"Archived thread {thread.thread_id}")
    
    async def _archive_activity(self, activity: ActivityContext):
        """Archive activity context"""
        summary = {
            "activity_id": activity.activity_id,
            "type": activity.activity_type,
            "duration": (activity.last_action - activity.start_time).total_seconds() / 3600,
            "focus_score": activity.focus_score,
            "interruptions": activity.interruption_count,
            "progress_markers": activity.progress_markers
        }
        
        # Store in memory
        interaction = {
            "messages": [{
                "content": f"Completed {activity.activity_type} session",
                "metadata": summary
            }],
            "summary": f"{activity.activity_type} activity"
        }
        
        context = {
            "source": "activity_archive",  
            "tags": ["activity", activity.activity_type]
        }
        
        await self.memory_system.remember_interaction(interaction, context)
    
    async def _save_context_state(self):
        """Save current context state to disk"""
        state = {
            "user_preferences": {
                "preferred_response_length": self.user_preferences.preferred_response_length,
                "formality_level": self.user_preferences.formality_level,
                "technical_depth": self.user_preferences.technical_depth,
                "proactive_suggestions": self.user_preferences.proactive_suggestions,
                "interruption_threshold": self.user_preferences.interruption_threshold,
                "working_hours": self.user_preferences.working_hours
            },
            "metrics": {
                "context_hit_rate": self.context_hit_rate,
                "prediction_accuracy": self.prediction_accuracy,
                "continuity_score": self.continuity_score
            },
            "patterns": {
                "context_switches": len(self.context_switches),
                "focus_periods": len(self.focus_periods)
            }
        }
        
        # Save to file
        state_file = self.persistence_path / "context_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    async def _load_persisted_contexts(self):
        """Load persisted context from disk"""
        state_file = self.persistence_path / "context_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                # Restore metrics
                self.context_hit_rate = state.get("metrics", {}).get("context_hit_rate", 0.0)
                self.prediction_accuracy = state.get("metrics", {}).get("prediction_accuracy", 0.0)
                self.continuity_score = state.get("metrics", {}).get("continuity_score", 0.0)
                
                logger.info("Loaded persisted context state")
                
            except Exception as e:
                logger.error(f"Failed to load context state: {e}")
    
    async def _load_user_preferences(self):
        """Load user preferences"""
        pref_file = self.persistence_path / "user_preferences.json"
        if pref_file.exists():
            try:
                with open(pref_file, 'r') as f:
                    prefs = json.load(f)
                
                self.user_preferences.preferred_response_length = prefs.get("preferred_response_length", "medium")
                self.user_preferences.formality_level = prefs.get("formality_level", 0.5)
                self.user_preferences.technical_depth = prefs.get("technical_depth", 0.7)
                self.user_preferences.proactive_suggestions = prefs.get("proactive_suggestions", True)
                self.user_preferences.interruption_threshold = prefs.get("interruption_threshold", 0.7)
                
                if "working_hours" in prefs:
                    self.user_preferences.working_hours = [
                        tuple(h) for h in prefs["working_hours"]
                    ]
                
                logger.info("Loaded user preferences")
                
            except Exception as e:
                logger.error(f"Failed to load preferences: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown the context manager"""
        logger.info("Shutting down Context Persistence Manager")
        
        self.running = False
        
        # Archive all active contexts
        for thread in self.conversation_threads.values():
            if thread.is_active:
                await self._archive_thread(thread)
        
        for activity in self.activity_contexts.values():
            await self._archive_activity(activity)
        
        # Save final state
        await self._save_context_state()
        
        # Save preferences
        pref_file = self.persistence_path / "user_preferences.json"
        with open(pref_file, 'w') as f:
            json.dump({
                "preferred_response_length": self.user_preferences.preferred_response_length,
                "formality_level": self.user_preferences.formality_level,
                "technical_depth": self.user_preferences.technical_depth,
                "proactive_suggestions": self.user_preferences.proactive_suggestions,
                "interruption_threshold": self.user_preferences.interruption_threshold,
                "working_hours": self.user_preferences.working_hours
            }, f, indent=2)
        
        logger.info("Context Persistence Manager shutdown complete")


# Integration helper
async def create_context_persistence_manager(memory_system: EpisodicMemorySystem,
                                           state_manager: FluidStateManager) -> ContextPersistenceManager:
    """Create and initialize context persistence manager"""
    manager = ContextPersistenceManager(memory_system, state_manager)
    await manager.initialize()
    return manager
