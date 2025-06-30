"""
JARVIS Phase 4: Predictive Intelligence System
==============================================
Advanced predictive capabilities with context persistence and pattern learning.

Features:
- Context persistence across sessions
- Predictive pre-loading of resources
- Pattern-based action prediction
- Smart caching with prediction
- User behavior modeling
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from pathlib import Path
import pickle
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionType(Enum):
    """Types of predictions JARVIS can make"""
    NEXT_ACTION = "next_action"
    RESOURCE_NEED = "resource_need"
    USER_STATE = "user_state"
    TASK_DURATION = "task_duration"
    INTERRUPTION = "interruption"
    PREFERENCE = "preference"


@dataclass
class ContextSnapshot:
    """Snapshot of context at a point in time"""
    timestamp: datetime
    user_state: str
    active_task: Optional[str]
    recent_actions: List[str]
    environmental_factors: Dict[str, Any]
    biometric_data: Optional[Dict[str, float]]
    
    def to_vector(self) -> np.ndarray:
        """Convert context to numerical vector for ML"""
        # Time features
        time_features = [
            self.timestamp.hour / 24,
            self.timestamp.weekday() / 7,
            self.timestamp.day / 31,
            self.timestamp.month / 12
        ]
        
        # State encoding (simplified)
        state_encoding = {
            "focused": [1, 0, 0, 0],
            "relaxed": [0, 1, 0, 0],
            "stressed": [0, 0, 1, 0],
            "tired": [0, 0, 0, 1]
        }.get(self.user_state, [0, 0, 0, 0])
        
        # Recent actions encoding (last 5)
        action_vector = [0] * 20  # Space for 20 action types
        for i, action in enumerate(self.recent_actions[-5:]):
            action_hash = int(hashlib.md5(action.encode()).hexdigest()[:8], 16)
            action_vector[action_hash % 20] = 1
        
        # Combine all features
        return np.array(time_features + state_encoding + action_vector)


@dataclass
class Prediction:
    """A prediction made by JARVIS"""
    prediction_type: PredictionType
    predicted_value: Any
    confidence: float
    reasoning: str
    time_horizon: timedelta
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class PatternMemory:
    """Long-term memory for patterns and sequences"""
    
    def __init__(self, memory_file: Path):
        self.memory_file = memory_file
        self.patterns = defaultdict(lambda: defaultdict(int))
        self.sequences = defaultdict(list)
        self.context_transitions = defaultdict(lambda: defaultdict(int))
        self.load_memory()
    
    def load_memory(self):
        """Load patterns from disk"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'rb') as f:
                    data = pickle.load(f)
                    self.patterns = defaultdict(lambda: defaultdict(int), data.get('patterns', {}))
                    self.sequences = defaultdict(list, data.get('sequences', {}))
                    self.context_transitions = defaultdict(
                        lambda: defaultdict(int), 
                        data.get('transitions', {})
                    )
                logger.info(f"Loaded {len(self.patterns)} patterns from memory")
            except Exception as e:
                logger.error(f"Error loading pattern memory: {e}")
    
    def save_memory(self):
        """Persist patterns to disk"""
        try:
            data = {
                'patterns': dict(self.patterns),
                'sequences': dict(self.sequences),
                'transitions': dict(self.context_transitions)
            }
            with open(self.memory_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info("Pattern memory saved successfully")
        except Exception as e:
            logger.error(f"Error saving pattern memory: {e}")
    
    def record_pattern(self, context: str, action: str, outcome: str):
        """Record a context-action-outcome pattern"""
        pattern_key = f"{context}→{action}"
        self.patterns[pattern_key][outcome] += 1
        
        # Keep sequences
        if context not in self.sequences:
            self.sequences[context] = deque(maxlen=100)
        self.sequences[context].append((action, outcome, datetime.now()))
    
    def record_transition(self, from_context: str, to_context: str):
        """Record context transitions"""
        self.context_transitions[from_context][to_context] += 1
    
    def get_most_likely_outcome(self, context: str, action: str) -> Tuple[str, float]:
        """Predict most likely outcome for context-action pair"""
        pattern_key = f"{context}→{action}"
        if pattern_key not in self.patterns:
            return "unknown", 0.0
        
        outcomes = self.patterns[pattern_key]
        total = sum(outcomes.values())
        if total == 0:
            return "unknown", 0.0
        
        # Get most likely outcome
        best_outcome = max(outcomes.items(), key=lambda x: x[1])
        confidence = best_outcome[1] / total
        
        return best_outcome[0], confidence
    
    def get_next_likely_context(self, current_context: str) -> Tuple[str, float]:
        """Predict next likely context"""
        if current_context not in self.context_transitions:
            return "unknown", 0.0
        
        transitions = self.context_transitions[current_context]
        total = sum(transitions.values())
        if total == 0:
            return "unknown", 0.0
        
        # Get most likely transition
        next_context = max(transitions.items(), key=lambda x: x[1])
        confidence = next_context[1] / total
        
        return next_context[0], confidence


class PredictivePreloader:
    """Preloads resources based on predictions"""
    
    def __init__(self):
        self.preload_cache = {}
        self.preload_history = deque(maxlen=1000)
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def preload_resource(self, resource_type: str, resource_id: str, 
                              prediction: Prediction):
        """Preload a resource based on prediction"""
        if prediction.confidence < 0.7:  # Only preload high-confidence predictions
            return
        
        cache_key = f"{resource_type}:{resource_id}"
        
        # Simulate resource loading
        logger.info(f"Preloading {resource_type} '{resource_id}' "
                   f"(confidence: {prediction.confidence:.2f})")
        
        # In real implementation, this would load actual resources
        resource_data = await self._load_resource(resource_type, resource_id)
        
        self.preload_cache[cache_key] = {
            'data': resource_data,
            'loaded_at': datetime.now(),
            'prediction': prediction,
            'accessed': False
        }
        
        # Schedule cache expiration
        asyncio.create_task(self._expire_cache(cache_key, prediction.time_horizon))
    
    async def _load_resource(self, resource_type: str, resource_id: str) -> Any:
        """Simulate resource loading"""
        await asyncio.sleep(0.1)  # Simulate loading time
        
        # In real implementation, this would load actual resources
        resource_map = {
            'document': lambda: {'content': f'Document {resource_id}', 'size': 1024},
            'application': lambda: {'name': resource_id, 'ready': True},
            'data': lambda: {'values': [1, 2, 3, 4, 5]},
            'model': lambda: {'weights': 'pretrained', 'version': '1.0'}
        }
        
        return resource_map.get(resource_type, lambda: {})()
    
    async def _expire_cache(self, cache_key: str, ttl: timedelta):
        """Expire cache entry after TTL"""
        await asyncio.sleep(ttl.total_seconds())
        if cache_key in self.preload_cache:
            entry = self.preload_cache[cache_key]
            if not entry['accessed']:
                self.cache_misses += 1
                logger.info(f"Preload miss: {cache_key} expired without access")
            del self.preload_cache[cache_key]
    
    def get_resource(self, resource_type: str, resource_id: str) -> Optional[Any]:
        """Get preloaded resource if available"""
        cache_key = f"{resource_type}:{resource_id}"
        if cache_key in self.preload_cache:
            self.cache_hits += 1
            self.preload_cache[cache_key]['accessed'] = True
            logger.info(f"Preload hit: {cache_key}")
            return self.preload_cache[cache_key]['data']
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preloader statistics"""
        total_attempts = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_attempts if total_attempts > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'current_cache_size': len(self.preload_cache)
        }


class PredictiveIntelligence:
    """Main predictive intelligence engine"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)
        
        # Components
        self.pattern_memory = PatternMemory(data_dir / "patterns.pkl")
        self.preloader = PredictivePreloader()
        
        # Context persistence
        self.context_file = data_dir / "context_history.json"
        self.context_history = deque(maxlen=10000)
        self.current_context = None
        self.load_context_history()
        
        # Prediction models (simplified for demo)
        self.action_sequences = defaultdict(lambda: deque(maxlen=50))
        self.time_patterns = defaultdict(list)
        
        # Background tasks
        self.running = False
    
    def load_context_history(self):
        """Load context history from disk"""
        if self.context_file.exists():
            try:
                with open(self.context_file, 'r') as f:
                    data = json.load(f)
                    for item in data[-10000:]:  # Last 10k entries
                        self.context_history.append(
                            ContextSnapshot(**item)
                        )
                logger.info(f"Loaded {len(self.context_history)} context entries")
            except Exception as e:
                logger.error(f"Error loading context history: {e}")
    
    def save_context_history(self):
        """Save context history to disk"""
        try:
            # Convert to serializable format
            data = []
            for ctx in self.context_history:
                ctx_dict = asdict(ctx)
                ctx_dict['timestamp'] = ctx_dict['timestamp'].isoformat()
                data.append(ctx_dict)
            
            with open(self.context_file, 'w') as f:
                json.dump(data, f)
            logger.info("Context history saved")
        except Exception as e:
            logger.error(f"Error saving context history: {e}")
    
    async def update_context(self, context: ContextSnapshot):
        """Update current context and make predictions"""
        # Record transition if context changed
        if self.current_context:
            self.pattern_memory.record_transition(
                self.current_context.user_state,
                context.user_state
            )
        
        # Update current context
        self.current_context = context
        self.context_history.append(context)
        
        # Record action sequences
        if context.active_task:
            self.action_sequences[context.active_task].extend(context.recent_actions)
        
        # Make predictions
        predictions = await self._make_predictions(context)
        
        # Preload resources based on predictions
        for pred in predictions:
            if pred.prediction_type == PredictionType.RESOURCE_NEED:
                resource_type, resource_id = pred.predicted_value
                await self.preloader.preload_resource(resource_type, resource_id, pred)
        
        return predictions
    
    async def _make_predictions(self, context: ContextSnapshot) -> List[Prediction]:
        """Make various predictions based on context"""
        predictions = []
        
        # Predict next action
        next_action_pred = await self._predict_next_action(context)
        if next_action_pred:
            predictions.append(next_action_pred)
        
        # Predict resource needs
        resource_preds = await self._predict_resource_needs(context)
        predictions.extend(resource_preds)
        
        # Predict user state changes
        state_pred = await self._predict_state_change(context)
        if state_pred:
            predictions.append(state_pred)
        
        # Predict task duration
        if context.active_task:
            duration_pred = await self._predict_task_duration(context)
            if duration_pred:
                predictions.append(duration_pred)
        
        return predictions
    
    async def _predict_next_action(self, context: ContextSnapshot) -> Optional[Prediction]:
        """Predict next likely action"""
        if not context.recent_actions:
            return None
        
        # Simple pattern matching (in real implementation, use ML model)
        last_actions = tuple(context.recent_actions[-3:])
        
        # Look for repeated sequences
        action_counts = defaultdict(int)
        for task_actions in self.action_sequences.values():
            actions_list = list(task_actions)
            for i in range(len(actions_list) - len(last_actions)):
                if tuple(actions_list[i:i+len(last_actions)]) == last_actions:
                    if i + len(last_actions) < len(actions_list):
                        next_action = actions_list[i + len(last_actions)]
                        action_counts[next_action] += 1
        
        if action_counts:
            # Get most likely next action
            next_action = max(action_counts.items(), key=lambda x: x[1])
            total = sum(action_counts.values())
            confidence = next_action[1] / total
            
            if confidence > 0.5:
                return Prediction(
                    prediction_type=PredictionType.NEXT_ACTION,
                    predicted_value=next_action[0],
                    confidence=confidence,
                    reasoning=f"Based on {total} similar sequences",
                    time_horizon=timedelta(minutes=5)
                )
        
        return None
    
    async def _predict_resource_needs(self, context: ContextSnapshot) -> List[Prediction]:
        """Predict what resources user will need"""
        predictions = []
        
        # Time-based predictions
        hour = context.timestamp.hour
        day = context.timestamp.weekday()
        
        # Morning email check pattern
        if 8 <= hour <= 9 and 'email' not in context.recent_actions:
            predictions.append(Prediction(
                prediction_type=PredictionType.RESOURCE_NEED,
                predicted_value=('application', 'email_client'),
                confidence=0.85,
                reasoning="Morning email check pattern",
                time_horizon=timedelta(minutes=30)
            ))
        
        # Document predictions based on task
        if context.active_task and 'report' in context.active_task.lower():
            predictions.append(Prediction(
                prediction_type=PredictionType.RESOURCE_NEED,
                predicted_value=('document', 'quarterly_report_template'),
                confidence=0.75,
                reasoning="Report task detected",
                time_horizon=timedelta(minutes=15)
            ))
        
        # Meeting predictions
        if day < 5 and 14 <= hour <= 15:  # Weekday afternoons
            predictions.append(Prediction(
                prediction_type=PredictionType.RESOURCE_NEED,
                predicted_value=('application', 'video_conference'),
                confidence=0.7,
                reasoning="Common meeting time",
                time_horizon=timedelta(minutes=45)
            ))
        
        # Development environment
        if 'code' in ' '.join(context.recent_actions).lower():
            predictions.append(Prediction(
                prediction_type=PredictionType.RESOURCE_NEED,
                predicted_value=('application', 'vscode'),
                confidence=0.9,
                reasoning="Coding activity detected",
                time_horizon=timedelta(minutes=10)
            ))
        
        return predictions
    
    async def _predict_state_change(self, context: ContextSnapshot) -> Optional[Prediction]:
        """Predict user state changes"""
        # Get likely next state from pattern memory
        next_state, confidence = self.pattern_memory.get_next_likely_context(
            context.user_state
        )
        
        if confidence > 0.6:
            # Estimate time to transition
            time_horizon = timedelta(minutes=30)  # Default
            
            # Adjust based on current state
            if context.user_state == "focused":
                time_horizon = timedelta(hours=1)  # Focus sessions last longer
            elif context.user_state == "tired":
                time_horizon = timedelta(minutes=15)  # Fatigue transitions faster
            
            return Prediction(
                prediction_type=PredictionType.USER_STATE,
                predicted_value=next_state,
                confidence=confidence,
                reasoning=f"Historical pattern from {context.user_state}",
                time_horizon=time_horizon
            )
        
        return None
    
    async def _predict_task_duration(self, context: ContextSnapshot) -> Optional[Prediction]:
        """Predict how long current task will take"""
        if not context.active_task:
            return None
        
        # Look for similar tasks in history
        similar_durations = []
        task_keywords = set(context.active_task.lower().split())
        
        current_task_start = None
        for ctx in self.context_history:
            if ctx.active_task:
                ctx_keywords = set(ctx.active_task.lower().split())
                similarity = len(task_keywords & ctx_keywords) / len(task_keywords | ctx_keywords)
                
                if similarity > 0.5:
                    if current_task_start:
                        duration = (ctx.timestamp - current_task_start).total_seconds() / 60
                        similar_durations.append(duration)
                        current_task_start = None
                    else:
                        current_task_start = ctx.timestamp
        
        if similar_durations:
            avg_duration = np.mean(similar_durations)
            std_duration = np.std(similar_durations)
            confidence = 1.0 / (1.0 + std_duration / avg_duration)  # Higher variance = lower confidence
            
            return Prediction(
                prediction_type=PredictionType.TASK_DURATION,
                predicted_value=int(avg_duration),
                confidence=min(confidence, 0.9),
                reasoning=f"Based on {len(similar_durations)} similar tasks",
                time_horizon=timedelta(minutes=int(avg_duration))
            )
        
        return None
    
    async def start(self):
        """Start predictive intelligence system"""
        self.running = True
        logger.info("Predictive Intelligence System started")
        
        # Start background tasks
        asyncio.create_task(self._pattern_learning_loop())
        asyncio.create_task(self._context_persistence_loop())
        asyncio.create_task(self._preload_optimization_loop())
    
    async def stop(self):
        """Stop predictive intelligence system"""
        self.running = False
        
        # Save all data
        self.pattern_memory.save_memory()
        self.save_context_history()
        
        logger.info("Predictive Intelligence System stopped")
    
    async def _pattern_learning_loop(self):
        """Background loop for pattern learning"""
        while self.running:
            try:
                # Analyze recent patterns
                if len(self.context_history) > 100:
                    # Extract patterns from recent history
                    recent_contexts = list(self.context_history)[-100:]
                    
                    # Learn action sequences
                    for i in range(len(recent_contexts) - 1):
                        curr = recent_contexts[i]
                        next_ctx = recent_contexts[i + 1]
                        
                        if curr.recent_actions and next_ctx.recent_actions:
                            # Record action transitions
                            for action in curr.recent_actions:
                                for next_action in next_ctx.recent_actions:
                                    self.pattern_memory.record_pattern(
                                        curr.user_state,
                                        action,
                                        next_action
                                    )
                
            except Exception as e:
                logger.error(f"Error in pattern learning: {e}")
            
            await asyncio.sleep(300)  # Learn every 5 minutes
    
    async def _context_persistence_loop(self):
        """Background loop for persisting context"""
        while self.running:
            try:
                self.save_context_history()
                self.pattern_memory.save_memory()
            except Exception as e:
                logger.error(f"Error in context persistence: {e}")
            
            await asyncio.sleep(60)  # Save every minute
    
    async def _preload_optimization_loop(self):
        """Background loop for optimizing preloading"""
        while self.running:
            try:
                stats = self.preloader.get_stats()
                if stats['hit_rate'] < 0.5 and stats['cache_hits'] + stats['cache_misses'] > 10:
                    logger.warning(f"Low preload hit rate: {stats['hit_rate']:.2%}")
                    # In real implementation, adjust prediction thresholds
            except Exception as e:
                logger.error(f"Error in preload optimization: {e}")
            
            await asyncio.sleep(600)  # Check every 10 minutes
    
    def get_insights(self) -> Dict[str, Any]:
        """Get insights about predictions and patterns"""
        insights = {
            'total_contexts': len(self.context_history),
            'unique_patterns': len(self.pattern_memory.patterns),
            'preloader_stats': self.preloader.get_stats(),
            'top_transitions': {},
            'common_sequences': {}
        }
        
        # Get top state transitions
        all_transitions = []
        for from_state, to_states in self.pattern_memory.context_transitions.items():
            for to_state, count in to_states.items():
                all_transitions.append((f"{from_state}→{to_state}", count))
        
        insights['top_transitions'] = dict(
            sorted(all_transitions, key=lambda x: x[1], reverse=True)[:5]
        )
        
        # Get common action sequences
        sequence_counts = defaultdict(int)
        for actions in self.action_sequences.values():
            if len(actions) >= 3:
                # Get 3-action sequences
                actions_list = list(actions)
                for i in range(len(actions_list) - 2):
                    seq = tuple(actions_list[i:i+3])
                    sequence_counts[seq] += 1
        
        insights['common_sequences'] = {
            '→'.join(seq): count 
            for seq, count in sorted(
                sequence_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        }
        
        return insights


# Example usage and testing
async def demo_predictive_intelligence():
    """Demonstrate predictive intelligence capabilities"""
    
    # Initialize
    data_dir = Path("./jarvis_predictive_data")
    predictor = PredictiveIntelligence(data_dir)
    await predictor.start()
    
    # Simulate user activity
    contexts = [
        ContextSnapshot(
            timestamp=datetime.now(),
            user_state="focused",
            active_task="quarterly_report",
            recent_actions=["open_document", "edit_text", "save_file"],
            environmental_factors={"location": "office", "noise_level": "low"},
            biometric_data={"heart_rate": 65, "stress_level": 0.3}
        ),
        ContextSnapshot(
            timestamp=datetime.now() + timedelta(minutes=5),
            user_state="focused",
            active_task="quarterly_report",
            recent_actions=["edit_text", "save_file", "search_data"],
            environmental_factors={"location": "office", "noise_level": "low"},
            biometric_data={"heart_rate": 68, "stress_level": 0.35}
        ),
        ContextSnapshot(
            timestamp=datetime.now() + timedelta(minutes=10),
            user_state="stressed",
            active_task="quarterly_report",
            recent_actions=["search_data", "open_email", "send_message"],
            environmental_factors={"location": "office", "noise_level": "medium"},
            biometric_data={"heart_rate": 75, "stress_level": 0.7}
        )
    ]
    
    # Process contexts and get predictions
    for ctx in contexts:
        print(f"\n📍 Context Update: {ctx.user_state} - {ctx.active_task}")
        predictions = await predictor.update_context(ctx)
        
        for pred in predictions:
            print(f"   🔮 {pred.prediction_type.value}: {pred.predicted_value}")
            print(f"      Confidence: {pred.confidence:.1%}")
            print(f"      Reasoning: {pred.reasoning}")
            print(f"      Time horizon: {pred.time_horizon}")
        
        await asyncio.sleep(1)
    
    # Check preloaded resources
    print("\n📦 Checking preloaded resources:")
    
    # Try to access a preloaded resource
    resource = predictor.preloader.get_resource('application', 'email_client')
    if resource:
        print("   ✅ Email client was preloaded successfully!")
    
    # Get insights
    print("\n📊 Predictive Intelligence Insights:")
    insights = predictor.get_insights()
    print(f"   Total contexts analyzed: {insights['total_contexts']}")
    print(f"   Unique patterns learned: {insights['unique_patterns']}")
    print(f"   Preloader hit rate: {insights['preloader_stats']['hit_rate']:.1%}")
    
    if insights['top_transitions']:
        print("\n   Top state transitions:")
        for transition, count in insights['top_transitions'].items():
            print(f"      {transition}: {count} times")
    
    if insights['common_sequences']:
        print("\n   Common action sequences:")
        for seq, count in insights['common_sequences'].items():
            print(f"      {seq}: {count} times")
    
    # Stop system
    await predictor.stop()
    print("\n✅ Predictive Intelligence demo completed!")


if __name__ == "__main__":
    print("🔮 JARVIS Phase 4: Predictive Intelligence System")
    print("=" * 50)
    asyncio.run(demo_predictive_intelligence())
