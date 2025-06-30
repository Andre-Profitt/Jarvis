"""
JARVIS Phase 3: Predictive Pre-loading System
=============================================
Advanced predictive system that anticipates user needs and pre-loads
resources, actions, and responses based on learned patterns.

This system provides:
- Pattern-based prediction of next actions
- Resource pre-loading for faster response
- Workflow automation suggestions
- Proactive assistance based on context
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from enum import Enum, auto
import json
import heapq
from pathlib import Path

# Import Phase 1 components
from .unified_input_pipeline import ProcessedInput, InputPriority
from .fluid_state_management import SystemState

# Import context persistence
from .context_persistence_manager import (
    ContextPersistenceManager,
    ConversationThread,
    ActivityContext
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionType(Enum):
    """Types of predictions the system can make"""
    NEXT_ACTION = auto()
    RESOURCE_NEED = auto()
    TASK_COMPLETION = auto()
    USER_QUESTION = auto()
    WORKFLOW_STEP = auto()
    CONTEXT_SWITCH = auto()
    INTERRUPTION = auto()
    INFORMATION_NEED = auto()


@dataclass
class Prediction:
    """Represents a prediction made by the system"""
    prediction_type: PredictionType
    content: Any
    confidence: float  # 0.0 to 1.0
    time_horizon: timedelta  # When this is likely to happen
    context: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    dependencies: List[str] = field(default_factory=list)
    preload_actions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ActionPattern:
    """Represents a learned action pattern"""
    pattern_id: str
    sequence: List[str]  # Action sequence
    frequency: int = 1
    avg_time_between: float = 0.0  # Average seconds between actions
    context_conditions: Dict[str, Any] = field(default_factory=dict)
    outcomes: List[str] = field(default_factory=list)
    confidence: float = 0.5
    last_seen: datetime = field(default_factory=datetime.now)


@dataclass
class ResourceRequirement:
    """Represents a resource that might be needed"""
    resource_type: str  # file, api, memory, computation
    resource_id: str
    likelihood: float
    typical_usage: Dict[str, Any] = field(default_factory=dict)
    preload_priority: int = 5  # 1-10, higher = more important
    dependencies: List[str] = field(default_factory=list)


@dataclass
class WorkflowTemplate:
    """Represents a common workflow"""
    workflow_id: str
    name: str
    steps: List[Dict[str, Any]]
    triggers: List[str]  # What triggers this workflow
    context_requirements: Dict[str, Any]
    success_rate: float = 0.0
    avg_duration: float = 0.0  # seconds
    last_executed: Optional[datetime] = None


class PredictivePreloadingSystem:
    """
    Advanced predictive system for JARVIS Phase 3.
    Learns from patterns to anticipate needs and pre-load resources.
    """
    
    def __init__(self,
                 context_manager: ContextPersistenceManager,
                 cache_size: int = 1000,
                 prediction_window: timedelta = timedelta(minutes=5)):
        self.context_manager = context_manager
        self.cache_size = cache_size
        self.prediction_window = prediction_window
        
        # Pattern storage
        self.action_patterns: Dict[str, ActionPattern] = {}
        self.resource_patterns: Dict[str, List[ResourceRequirement]] = defaultdict(list)
        self.workflow_templates: Dict[str, WorkflowTemplate] = {}
        
        # Prediction tracking
        self.active_predictions: List[Prediction] = []
        self.prediction_history: deque = deque(maxlen=1000)
        self.prediction_accuracy: Dict[PredictionType, float] = defaultdict(float)
        
        # Action tracking
        self.action_history: deque = deque(maxlen=5000)
        self.action_sequences: deque = deque(maxlen=1000)
        self.resource_usage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Pre-loaded resources
        self.preloaded_resources: Dict[str, Any] = {}
        self.preload_cache: Dict[str, Tuple[Any, datetime]] = {}  # with expiry
        
        # Workflow tracking
        self.active_workflows: List[Dict[str, Any]] = []
        self.workflow_history: deque = deque(maxlen=100)
        
        # Performance metrics
        self.hit_rate = 0.0
        self.false_positive_rate = 0.0
        self.avg_prediction_lead_time = 0.0
        
        # Background tasks
        self.running = False
        
    async def initialize(self):
        """Initialize the predictive system"""
        logger.info("🔮 Initializing Predictive Pre-loading System")
        
        # Load learned patterns
        await self._load_patterns()
        
        # Start background tasks
        self.running = True
        asyncio.create_task(self._prediction_loop())
        asyncio.create_task(self._pattern_learning_loop())
        asyncio.create_task(self._resource_management_loop())
        asyncio.create_task(self._accuracy_tracking_loop())
        
        logger.info("✅ Predictive Pre-loading System initialized")
    
    async def process_action(self,
                           action_type: str,
                           content: Any,
                           context: Dict[str, Any]) -> List[Prediction]:
        """
        Process an action and generate predictions.
        Returns list of predictions for what might happen next.
        """
        # Record action
        action = {
            "type": action_type,
            "content": content,
            "context": context,
            "timestamp": datetime.now()
        }
        self.action_history.append(action)
        
        # Update sequences
        self._update_action_sequences(action)
        
        # Generate predictions
        predictions = []
        
        # Predict next actions
        next_action_predictions = await self._predict_next_actions(action, context)
        predictions.extend(next_action_predictions)
        
        # Predict resource needs
        resource_predictions = await self._predict_resource_needs(action, context)
        predictions.extend(resource_predictions)
        
        # Check for workflow matches
        workflow_predictions = await self._predict_workflow_continuation(action, context)
        predictions.extend(workflow_predictions)
        
        # Context switch predictions
        context_predictions = await self._predict_context_switches(action, context)
        predictions.extend(context_predictions)
        
        # Pre-load based on predictions
        await self._execute_preloading(predictions)
        
        # Store predictions for accuracy tracking
        self.active_predictions.extend(predictions)
        
        return predictions
    
    def _update_action_sequences(self, action: Dict[str, Any]):
        """Update action sequence tracking"""
        # Add to recent sequence
        if len(self.action_sequences) > 0:
            last_seq = list(self.action_sequences[-1])
            if len(last_seq) < 10:  # Max sequence length
                last_seq.append(action["type"])
                self.action_sequences[-1] = last_seq
            else:
                self.action_sequences.append([action["type"]])
        else:
            self.action_sequences.append([action["type"]])
        
        # Update pattern frequencies
        self._update_pattern_frequencies()
    
    def _update_pattern_frequencies(self):
        """Update frequencies of observed patterns"""
        if len(self.action_sequences) < 2:
            return
        
        # Look for repeated subsequences
        recent_sequences = list(self.action_sequences)[-20:]
        
        for seq_len in range(2, 6):  # Pattern lengths 2-5
            patterns = defaultdict(int)
            
            for sequence in recent_sequences:
                if len(sequence) >= seq_len:
                    for i in range(len(sequence) - seq_len + 1):
                        pattern = tuple(sequence[i:i + seq_len])
                        patterns[pattern] += 1
            
            # Update action patterns
            for pattern, count in patterns.items():
                if count >= 2:  # Minimum frequency
                    pattern_id = "_".join(pattern)
                    
                    if pattern_id in self.action_patterns:
                        self.action_patterns[pattern_id].frequency = count
                        self.action_patterns[pattern_id].last_seen = datetime.now()
                    else:
                        self.action_patterns[pattern_id] = ActionPattern(
                            pattern_id=pattern_id,
                            sequence=list(pattern),
                            frequency=count
                        )
    
    async def _predict_next_actions(self,
                                  current_action: Dict[str, Any],
                                  context: Dict[str, Any]) -> List[Prediction]:
        """Predict likely next actions based on patterns"""
        predictions = []
        
        # Get recent action sequence
        recent_actions = [a["type"] for a in list(self.action_history)[-5:]]
        
        # Find matching patterns
        for pattern in self.action_patterns.values():
            # Check if recent actions match the beginning of this pattern
            pattern_start = pattern.sequence[:-1]
            
            if len(recent_actions) >= len(pattern_start):
                recent_subset = recent_actions[-len(pattern_start):]
                
                if recent_subset == pattern_start:
                    # We have a match! Predict the next action
                    next_action = pattern.sequence[-1]
                    confidence = min(0.9, pattern.frequency / 10.0)  # Cap at 0.9
                    
                    prediction = Prediction(
                        prediction_type=PredictionType.NEXT_ACTION,
                        content=next_action,
                        confidence=confidence,
                        time_horizon=timedelta(seconds=pattern.avg_time_between or 30),
                        context={
                            "pattern_id": pattern.pattern_id,
                            "pattern_frequency": pattern.frequency,
                            "current_action": current_action["type"]
                        },
                        preload_actions=[{
                            "action": "prepare_for_action",
                            "target": next_action,
                            "priority": int(confidence * 10)
                        }]
                    )
                    
                    predictions.append(prediction)
        
        # Also check context-based predictions
        if context.get("current_activity"):
            activity_predictions = await self._predict_from_activity(
                context["current_activity"],
                current_action
            )
            predictions.extend(activity_predictions)
        
        return predictions[:5]  # Top 5 predictions
    
    async def _predict_from_activity(self,
                                   activity: ActivityContext,
                                   current_action: Dict[str, Any]) -> List[Prediction]:
        """Predict based on current activity type"""
        predictions = []
        
        # Activity-specific predictions
        if activity.activity_type == "coding":
            # Common coding workflow predictions
            if "save" in current_action["type"]:
                predictions.append(Prediction(
                    prediction_type=PredictionType.NEXT_ACTION,
                    content="test_or_run",
                    confidence=0.7,
                    time_horizon=timedelta(seconds=10),
                    context={"activity": "coding", "after": "save"},
                    preload_actions=[{
                        "action": "prepare_test_environment",
                        "priority": 7
                    }]
                ))
            elif "error" in str(current_action.get("content", "")).lower():
                predictions.append(Prediction(
                    prediction_type=PredictionType.NEXT_ACTION,
                    content="debug_or_search",
                    confidence=0.8,
                    time_horizon=timedelta(seconds=5),
                    context={"activity": "coding", "after": "error"},
                    preload_actions=[{
                        "action": "prepare_debug_tools",
                        "priority": 8
                    }]
                ))
        
        elif activity.activity_type == "research":
            # Research workflow predictions
            if "read" in current_action["type"] or "open" in current_action["type"]:
                predictions.append(Prediction(
                    prediction_type=PredictionType.RESOURCE_NEED,
                    content="related_documents",
                    confidence=0.6,
                    time_horizon=timedelta(minutes=2),
                    context={"activity": "research"},
                    preload_actions=[{
                        "action": "find_related_docs",
                        "priority": 6
                    }]
                ))
        
        elif activity.activity_type == "communication":
            # Communication predictions
            if "compose" in current_action["type"]:
                predictions.append(Prediction(
                    prediction_type=PredictionType.RESOURCE_NEED,
                    content="contact_information",
                    confidence=0.5,
                    time_horizon=timedelta(seconds=30),
                    context={"activity": "communication"},
                    preload_actions=[{
                        "action": "load_contacts",
                        "priority": 5
                    }]
                ))
        
        return predictions
    
    async def _predict_resource_needs(self,
                                    action: Dict[str, Any],
                                    context: Dict[str, Any]) -> List[Prediction]:
        """Predict what resources will be needed"""
        predictions = []
        
        # Check historical resource usage patterns
        action_type = action["type"]
        
        if action_type in self.resource_patterns:
            for resource_req in self.resource_patterns[action_type]:
                if resource_req.likelihood > 0.3:
                    prediction = Prediction(
                        prediction_type=PredictionType.RESOURCE_NEED,
                        content=resource_req,
                        confidence=resource_req.likelihood,
                        time_horizon=timedelta(seconds=20),
                        context={
                            "action": action_type,
                            "resource_type": resource_req.resource_type
                        },
                        preload_actions=[{
                            "action": "preload_resource",
                            "resource": resource_req.resource_id,
                            "type": resource_req.resource_type,
                            "priority": resource_req.preload_priority
                        }]
                    )
                    predictions.append(prediction)
        
        # Context-based resource predictions
        if context.get("conversation_thread"):
            thread = context["conversation_thread"]
            
            # Predict need for conversation history
            if len(thread.context_stack) > 3:
                predictions.append(Prediction(
                    prediction_type=PredictionType.RESOURCE_NEED,
                    content="conversation_history",
                    confidence=0.7,
                    time_horizon=timedelta(seconds=10),
                    context={"thread_id": thread.thread_id},
                    preload_actions=[{
                        "action": "load_thread_history",
                        "thread_id": thread.thread_id,
                        "priority": 6
                    }]
                ))
        
        return predictions
    
    async def _predict_workflow_continuation(self,
                                           action: Dict[str, Any],
                                           context: Dict[str, Any]) -> List[Prediction]:
        """Predict workflow continuations"""
        predictions = []
        
        # Check active workflows
        for workflow_state in self.active_workflows:
            workflow = self.workflow_templates.get(workflow_state["workflow_id"])
            if not workflow:
                continue
            
            current_step = workflow_state["current_step"]
            if current_step < len(workflow.steps) - 1:
                next_step = workflow.steps[current_step + 1]
                
                # Check if action matches current step
                if self._action_matches_step(action, workflow.steps[current_step]):
                    prediction = Prediction(
                        prediction_type=PredictionType.WORKFLOW_STEP,
                        content=next_step,
                        confidence=workflow.success_rate,
                        time_horizon=timedelta(seconds=next_step.get("typical_delay", 30)),
                        context={
                            "workflow": workflow.name,
                            "step": current_step + 1,
                            "total_steps": len(workflow.steps)
                        },
                        preload_actions=next_step.get("preload_actions", [])
                    )
                    predictions.append(prediction)
                    
                    # Update workflow state
                    workflow_state["current_step"] = current_step + 1
                    workflow_state["last_action"] = datetime.now()
        
        # Check for workflow triggers
        for workflow in self.workflow_templates.values():
            for trigger in workflow.triggers:
                if trigger in action["type"] or trigger in str(action.get("content", "")):
                    # Check context requirements
                    if self._check_workflow_context(workflow, context):
                        prediction = Prediction(
                            prediction_type=PredictionType.WORKFLOW_STEP,
                            content=workflow.steps[0],
                            confidence=workflow.success_rate * 0.8,  # Lower for new workflow
                            time_horizon=timedelta(seconds=10),
                            context={
                                "workflow": workflow.name,
                                "triggered_by": trigger,
                                "step": 0
                            },
                            preload_actions=workflow.steps[0].get("preload_actions", [])
                        )
                        predictions.append(prediction)
                        
                        # Start tracking this workflow
                        self.active_workflows.append({
                            "workflow_id": workflow.workflow_id,
                            "current_step": 0,
                            "started_at": datetime.now(),
                            "last_action": datetime.now()
                        })
        
        return predictions
    
    def _action_matches_step(self, action: Dict[str, Any], step: Dict[str, Any]) -> bool:
        """Check if an action matches a workflow step"""
        # Simple matching - could be more sophisticated
        step_action = step.get("action", "")
        return (step_action in action["type"] or 
                action["type"] in step_action or
                step_action in str(action.get("content", "")))
    
    def _check_workflow_context(self, 
                              workflow: WorkflowTemplate,
                              context: Dict[str, Any]) -> bool:
        """Check if context meets workflow requirements"""
        for req_key, req_value in workflow.context_requirements.items():
            if req_key not in context:
                return False
            
            if isinstance(req_value, str) and req_value.startswith("has_"):
                # Check existence
                if not context.get(req_key.replace("has_", "")):
                    return False
            elif context.get(req_key) != req_value:
                return False
        
        return True
    
    async def _predict_context_switches(self,
                                      action: Dict[str, Any],
                                      context: Dict[str, Any]) -> List[Prediction]:
        """Predict potential context switches"""
        predictions = []
        
        # Time-based predictions
        current_hour = datetime.now().hour
        
        # Check working hours
        if hasattr(self.context_manager, 'user_preferences'):
            prefs = self.context_manager.user_preferences
            
            # Predict end of work
            for start, end in prefs.working_hours:
                if current_hour == end - 1:  # One hour before end
                    predictions.append(Prediction(
                        prediction_type=PredictionType.CONTEXT_SWITCH,
                        content="work_ending_soon",
                        confidence=0.8,
                        time_horizon=timedelta(hours=1),
                        context={"reason": "working_hours"},
                        preload_actions=[{
                            "action": "prepare_daily_summary",
                            "priority": 7
                        }]
                    ))
        
        # Activity-based predictions
        if context.get("current_activity"):
            activity = context["current_activity"]
            duration = (datetime.now() - activity.start_time).total_seconds() / 3600
            
            # Predict fatigue-based switches
            if duration > 2 and activity.focus_score < 0.5:
                predictions.append(Prediction(
                    prediction_type=PredictionType.CONTEXT_SWITCH,
                    content="activity_fatigue",
                    confidence=0.6,
                    time_horizon=timedelta(minutes=15),
                    context={
                        "activity": activity.activity_type,
                        "duration": duration,
                        "focus_score": activity.focus_score
                    },
                    preload_actions=[{
                        "action": "suggest_break",
                        "priority": 5
                    }]
                ))
        
        # Pattern-based predictions
        recent_switches = [s for s in self.context_manager.context_switches 
                          if (datetime.now() - s["timestamp"]).total_seconds() < 7200]
        
        if len(recent_switches) >= 3:
            # Frequent switching pattern
            avg_switch_interval = np.mean([
                (recent_switches[i+1]["timestamp"] - recent_switches[i]["timestamp"]).total_seconds()
                for i in range(len(recent_switches)-1)
            ])
            
            next_switch_time = recent_switches[-1]["timestamp"] + timedelta(seconds=avg_switch_interval)
            time_until = (next_switch_time - datetime.now()).total_seconds()
            
            if 0 < time_until < 600:  # Within 10 minutes
                predictions.append(Prediction(
                    prediction_type=PredictionType.CONTEXT_SWITCH,
                    content="pattern_based_switch",
                    confidence=0.5,
                    time_horizon=timedelta(seconds=time_until),
                    context={"pattern": "frequent_switching"},
                    preload_actions=[{
                        "action": "prepare_context_transition",
                        "priority": 4
                    }]
                ))
        
        return predictions
    
    async def _execute_preloading(self, predictions: List[Prediction]):
        """Execute pre-loading actions based on predictions"""
        # Sort by priority
        preload_actions = []
        for pred in predictions:
            for action in pred.preload_actions:
                action["confidence"] = pred.confidence
                action["prediction_id"] = id(pred)
                preload_actions.append(action)
        
        # Sort by priority and confidence
        preload_actions.sort(key=lambda x: x["priority"] * x["confidence"], reverse=True)
        
        # Execute top actions
        for action in preload_actions[:10]:  # Limit concurrent preloads
            try:
                await self._execute_single_preload(action)
            except Exception as e:
                logger.error(f"Preload failed: {e}")
    
    async def _execute_single_preload(self, action: Dict[str, Any]):
        """Execute a single preload action"""
        action_type = action["action"]
        
        if action_type == "preload_resource":
            await self._preload_resource(
                action["resource"],
                action["type"],
                action.get("priority", 5)
            )
        
        elif action_type == "prepare_test_environment":
            # Pre-load test frameworks, recent test results, etc.
            self.preloaded_resources["test_env"] = {
                "loaded_at": datetime.now(),
                "data": "test_environment_ready"  # Placeholder
            }
        
        elif action_type == "prepare_debug_tools":
            # Pre-load debugging resources
            self.preloaded_resources["debug_tools"] = {
                "loaded_at": datetime.now(),
                "data": "debug_tools_ready"  # Placeholder
            }
        
        elif action_type == "load_thread_history":
            # Pre-load conversation history
            thread_id = action.get("thread_id")
            if thread_id:
                # Would actually load from memory system
                self.preloaded_resources[f"thread_{thread_id}"] = {
                    "loaded_at": datetime.now(),
                    "data": f"thread_history_{thread_id}"  # Placeholder
                }
        
        elif action_type == "prepare_daily_summary":
            # Pre-generate daily summary
            summary = await self._generate_daily_summary()
            self.preloaded_resources["daily_summary"] = {
                "loaded_at": datetime.now(),
                "data": summary
            }
    
    async def _preload_resource(self, resource_id: str, resource_type: str, priority: int):
        """Pre-load a specific resource"""
        cache_key = f"{resource_type}:{resource_id}"
        
        # Check if already cached
        if cache_key in self.preload_cache:
            cached_data, cached_time = self.preload_cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < 300:  # 5 min cache
                return
        
        # Simulate resource loading based on type
        if resource_type == "file":
            # Would actually read file
            data = f"file_content_{resource_id}"
        elif resource_type == "api":
            # Would actually call API
            data = f"api_response_{resource_id}"
        elif resource_type == "memory":
            # Would query memory system
            data = f"memory_content_{resource_id}"
        else:
            data = f"{resource_type}_data_{resource_id}"
        
        # Cache the resource
        self.preload_cache[cache_key] = (data, datetime.now())
        
        # Track usage
        self.resource_usage[resource_type].append({
            "resource_id": resource_id,
            "timestamp": datetime.now(),
            "preloaded": True
        })
    
    async def _generate_daily_summary(self) -> Dict[str, Any]:
        """Generate a daily summary"""
        # Get today's data from context manager
        context_summary = await self.context_manager.get_context_summary()
        
        # Analyze patterns
        today_actions = [a for a in self.action_history 
                        if a["timestamp"].date() == datetime.now().date()]
        
        action_distribution = defaultdict(int)
        for action in today_actions:
            action_distribution[action["type"]] += 1
        
        # Calculate productivity metrics
        focus_time = sum(f["duration"] for f in context_summary.get("focus_periods_today", []))
        
        return {
            "date": datetime.now().date().isoformat(),
            "total_actions": len(today_actions),
            "action_distribution": dict(action_distribution),
            "focus_time_hours": focus_time,
            "context_switches": context_summary.get("context_switches_today", 0),
            "active_threads": context_summary.get("active_conversation_threads", 0),
            "top_activities": list(action_distribution.items())[:5],
            "prediction_accuracy": self.hit_rate,
            "generated_at": datetime.now().isoformat()
        }
    
    def check_prediction_accuracy(self, action: Dict[str, Any]) -> float:
        """Check if current action was predicted"""
        accurate_predictions = []
        
        # Check active predictions
        current_time = datetime.now()
        for pred in self.active_predictions:
            # Check if prediction matches action
            if pred.prediction_type == PredictionType.NEXT_ACTION:
                if pred.content == action["type"]:
                    # Check if within time window
                    time_diff = (current_time - pred.created_at).total_seconds()
                    if time_diff <= pred.time_horizon.total_seconds():
                        accurate_predictions.append(pred)
        
        # Update accuracy metrics
        if accurate_predictions:
            best_pred = max(accurate_predictions, key=lambda p: p.confidence)
            self._update_accuracy_metrics(best_pred, True)
            return best_pred.confidence
        
        return 0.0
    
    def _update_accuracy_metrics(self, prediction: Prediction, accurate: bool):
        """Update accuracy tracking"""
        # Move to history
        self.prediction_history.append({
            "prediction": prediction,
            "accurate": accurate,
            "timestamp": datetime.now()
        })
        
        # Update type-specific accuracy
        pred_type = prediction.prediction_type
        recent_predictions = [p for p in self.prediction_history 
                            if p["prediction"].prediction_type == pred_type][-100:]
        
        if recent_predictions:
            accuracy = sum(1 for p in recent_predictions if p["accurate"]) / len(recent_predictions)
            self.prediction_accuracy[pred_type] = accuracy
    
    async def get_active_predictions(self) -> List[Dict[str, Any]]:
        """Get currently active predictions"""
        current_time = datetime.now()
        active = []
        
        for pred in self.active_predictions:
            time_remaining = (pred.created_at + pred.time_horizon - current_time).total_seconds()
            if time_remaining > 0:
                active.append({
                    "type": pred.prediction_type.name,
                    "content": pred.content,
                    "confidence": pred.confidence,
                    "time_remaining": time_remaining,
                    "context": pred.context
                })
        
        return active
    
    async def get_preloaded_resources(self) -> Dict[str, Any]:
        """Get currently pre-loaded resources"""
        current_time = datetime.now()
        active_resources = {}
        
        # Check cache
        for key, (data, load_time) in self.preload_cache.items():
            age = (current_time - load_time).total_seconds()
            if age < 300:  # 5 minute expiry
                active_resources[key] = {
                    "age_seconds": age,
                    "size": len(str(data))  # Simplified size
                }
        
        return active_resources
    
    async def _prediction_loop(self):
        """Background task for continuous prediction"""
        while self.running:
            try:
                # Clean expired predictions
                current_time = datetime.now()
                self.active_predictions = [
                    p for p in self.active_predictions
                    if (current_time - p.created_at).total_seconds() < p.time_horizon.total_seconds() * 2
                ]
                
                # Generate periodic predictions
                if len(self.action_history) > 0:
                    last_action = self.action_history[-1]
                    
                    # Get current context
                    context_summary = await self.context_manager.get_context_summary()
                    
                    # Generate time-based predictions
                    time_predictions = await self._generate_time_based_predictions(context_summary)
                    self.active_predictions.extend(time_predictions)
                
            except Exception as e:
                logger.error(f"Prediction loop error: {e}")
            
            await asyncio.sleep(30)  # Run every 30 seconds
    
    async def _generate_time_based_predictions(self, context: Dict[str, Any]) -> List[Prediction]:
        """Generate predictions based on time patterns"""
        predictions = []
        current_time = datetime.now()
        current_hour = current_time.hour
        current_minute = current_time.minute
        
        # Common time-based patterns
        if current_hour == 9 and current_minute < 30:
            predictions.append(Prediction(
                prediction_type=PredictionType.NEXT_ACTION,
                content="morning_routine",
                confidence=0.7,
                time_horizon=timedelta(minutes=30),
                context={"time_pattern": "morning_start"},
                preload_actions=[{
                    "action": "load_daily_agenda",
                    "priority": 8
                }]
            ))
        
        elif current_hour == 12 and current_minute > 30:
            predictions.append(Prediction(
                prediction_type=PredictionType.CONTEXT_SWITCH,
                content="lunch_break",
                confidence=0.6,
                time_horizon=timedelta(minutes=30),
                context={"time_pattern": "lunch_time"},
                preload_actions=[{
                    "action": "save_work_state",
                    "priority": 6
                }]
            ))
        
        return predictions
    
    async def _pattern_learning_loop(self):
        """Background task for pattern learning"""
        while self.running:
            try:
                # Learn from recent actions
                if len(self.action_history) >= 100:
                    await self._learn_patterns()
                
                # Learn workflows
                if len(self.workflow_history) >= 10:
                    await self._learn_workflows()
                
                # Update resource patterns
                await self._update_resource_patterns()
                
            except Exception as e:
                logger.error(f"Pattern learning error: {e}")
            
            await asyncio.sleep(300)  # Run every 5 minutes
    
    async def _learn_patterns(self):
        """Learn patterns from action history"""
        # Extract frequent sequences
        sequences = defaultdict(int)
        
        actions = list(self.action_history)[-500:]  # Last 500 actions
        
        for i in range(len(actions) - 5):
            for length in range(2, 6):
                if i + length <= len(actions):
                    seq = tuple(a["type"] for a in actions[i:i+length])
                    sequences[seq] += 1
        
        # Create patterns from frequent sequences
        for seq, count in sequences.items():
            if count >= 3:  # Minimum frequency
                pattern_id = "_".join(seq)
                
                if pattern_id not in self.action_patterns:
                    # Calculate average time between actions
                    time_diffs = []
                    for i in range(len(actions) - len(seq)):
                        if tuple(a["type"] for a in actions[i:i+len(seq)]) == seq:
                            for j in range(len(seq) - 1):
                                diff = (actions[i+j+1]["timestamp"] - actions[i+j]["timestamp"]).total_seconds()
                                time_diffs.append(diff)
                    
                    avg_time = np.mean(time_diffs) if time_diffs else 30.0
                    
                    self.action_patterns[pattern_id] = ActionPattern(
                        pattern_id=pattern_id,
                        sequence=list(seq),
                        frequency=count,
                        avg_time_between=avg_time,
                        confidence=min(0.9, count / 10.0)
                    )
    
    async def _learn_workflows(self):
        """Learn workflows from successful sequences"""
        # Group similar workflows
        workflow_groups = defaultdict(list)
        
        for workflow in self.workflow_history:
            key = tuple(workflow["steps"][:3])  # Group by first 3 steps
            workflow_groups[key].append(workflow)
        
        # Create templates from groups
        for key, workflows in workflow_groups.items():
            if len(workflows) >= 3:  # Minimum instances
                # Calculate average duration and success rate
                durations = [w["duration"] for w in workflows]
                successes = [w["successful"] for w in workflows]
                
                avg_duration = np.mean(durations)
                success_rate = sum(successes) / len(successes)
                
                if success_rate > 0.5:  # Only keep successful workflows
                    workflow_id = f"workflow_{hash(key)}"
                    
                    if workflow_id not in self.workflow_templates:
                        # Extract common steps
                        all_steps = [w["steps"] for w in workflows]
                        common_steps = self._find_common_steps(all_steps)
                        
                        self.workflow_templates[workflow_id] = WorkflowTemplate(
                            workflow_id=workflow_id,
                            name=f"Workflow {len(self.workflow_templates) + 1}",
                            steps=common_steps,
                            triggers=self._extract_workflow_triggers(workflows),
                            context_requirements={},
                            success_rate=success_rate,
                            avg_duration=avg_duration
                        )
    
    def _find_common_steps(self, step_lists: List[List[str]]) -> List[Dict[str, Any]]:
        """Find common steps across multiple workflow instances"""
        if not step_lists:
            return []
        
        # Simple approach: use the most common sequence
        common_steps = []
        min_length = min(len(steps) for steps in step_lists)
        
        for i in range(min_length):
            step_variations = [steps[i] for steps in step_lists]
            most_common = max(set(step_variations), key=step_variations.count)
            common_steps.append({
                "action": most_common,
                "position": i,
                "variations": len(set(step_variations))
            })
        
        return common_steps
    
    def _extract_workflow_triggers(self, workflows: List[Dict[str, Any]]) -> List[str]:
        """Extract common triggers for workflows"""
        triggers = []
        
        # Look at first actions
        first_actions = [w["steps"][0] if w["steps"] else "" for w in workflows]
        
        # Find common triggers
        action_counts = defaultdict(int)
        for action in first_actions:
            action_counts[action] += 1
        
        # Return most common triggers
        sorted_triggers = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        return [trigger for trigger, count in sorted_triggers[:3] if count >= 2]
    
    async def _update_resource_patterns(self):
        """Update resource usage patterns"""
        # Analyze resource usage
        for resource_type, usage_history in self.resource_usage.items():
            if len(usage_history) < 10:
                continue
            
            # Find patterns in resource usage
            resource_actions = defaultdict(list)
            
            # Match resources to actions
            for usage in usage_history:
                # Find the action that preceded this resource usage
                usage_time = usage["timestamp"]
                preceding_actions = [
                    a for a in self.action_history
                    if (usage_time - a["timestamp"]).total_seconds() < 60
                    and (usage_time - a["timestamp"]).total_seconds() > 0
                ]
                
                if preceding_actions:
                    last_action = max(preceding_actions, key=lambda a: a["timestamp"])
                    resource_actions[last_action["type"]].append(usage["resource_id"])
            
            # Create resource requirements
            for action_type, resources in resource_actions.items():
                resource_counts = defaultdict(int)
                for res in resources:
                    resource_counts[res] += 1
                
                # Create requirements for frequently used resources
                for resource_id, count in resource_counts.items():
                    likelihood = count / len(resources) if resources else 0
                    
                    if likelihood > 0.3:
                        req = ResourceRequirement(
                            resource_type=resource_type,
                            resource_id=resource_id,
                            likelihood=likelihood,
                            preload_priority=int(likelihood * 10)
                        )
                        
                        # Add or update in patterns
                        existing = False
                        for existing_req in self.resource_patterns[action_type]:
                            if existing_req.resource_id == resource_id:
                                existing_req.likelihood = likelihood
                                existing = True
                                break
                        
                        if not existing:
                            self.resource_patterns[action_type].append(req)
    
    async def _resource_management_loop(self):
        """Background task for resource cache management"""
        while self.running:
            try:
                # Clean expired cache entries
                current_time = datetime.now()
                expired_keys = []
                
                for key, (data, load_time) in self.preload_cache.items():
                    if (current_time - load_time).total_seconds() > 300:  # 5 min expiry
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.preload_cache[key]
                
                # Clean old preloaded resources
                expired_resources = []
                for key, resource in self.preloaded_resources.items():
                    if (current_time - resource["loaded_at"]).total_seconds() > 600:  # 10 min
                        expired_resources.append(key)
                
                for key in expired_resources:
                    del self.preloaded_resources[key]
                
                # Log cache stats
                if len(self.preload_cache) > 0:
                    logger.debug(f"Cache size: {len(self.preload_cache)} items")
                
            except Exception as e:
                logger.error(f"Resource management error: {e}")
            
            await asyncio.sleep(60)  # Run every minute
    
    async def _accuracy_tracking_loop(self):
        """Background task for tracking prediction accuracy"""
        while self.running:
            try:
                # Calculate metrics
                if len(self.prediction_history) >= 20:
                    recent = list(self.prediction_history)[-100:]
                    
                    # Overall hit rate
                    hits = sum(1 for p in recent if p["accurate"])
                    self.hit_rate = hits / len(recent)
                    
                    # False positive rate
                    action_predictions = [p for p in recent 
                                        if p["prediction"].prediction_type == PredictionType.NEXT_ACTION]
                    
                    if action_predictions:
                        false_positives = sum(1 for p in action_predictions 
                                            if not p["accurate"] and p["prediction"].confidence > 0.7)
                        self.false_positive_rate = false_positives / len(action_predictions)
                    
                    # Average lead time for accurate predictions
                    accurate_predictions = [p for p in recent if p["accurate"]]
                    if accurate_predictions:
                        lead_times = [p["prediction"].time_horizon.total_seconds() 
                                     for p in accurate_predictions]
                        self.avg_prediction_lead_time = np.mean(lead_times)
                
            except Exception as e:
                logger.error(f"Accuracy tracking error: {e}")
            
            await asyncio.sleep(120)  # Run every 2 minutes
    
    async def _load_patterns(self):
        """Load learned patterns from disk"""
        patterns_dir = Path("./predictive_patterns")
        if not patterns_dir.exists():
            patterns_dir.mkdir()
            return
        
        # Load action patterns
        patterns_file = patterns_dir / "action_patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                
                for pattern_id, data in patterns_data.items():
                    self.action_patterns[pattern_id] = ActionPattern(
                        pattern_id=pattern_id,
                        sequence=data["sequence"],
                        frequency=data["frequency"],
                        avg_time_between=data.get("avg_time_between", 30.0),
                        confidence=data.get("confidence", 0.5),
                        last_seen=datetime.fromisoformat(data.get("last_seen", datetime.now().isoformat()))
                    )
                
                logger.info(f"Loaded {len(self.action_patterns)} action patterns")
                
            except Exception as e:
                logger.error(f"Failed to load patterns: {e}")
        
        # Load workflow templates
        workflows_file = patterns_dir / "workflows.json"
        if workflows_file.exists():
            try:
                with open(workflows_file, 'r') as f:
                    workflows_data = json.load(f)
                
                for workflow_id, data in workflows_data.items():
                    self.workflow_templates[workflow_id] = WorkflowTemplate(
                        workflow_id=workflow_id,
                        name=data["name"],
                        steps=data["steps"],
                        triggers=data["triggers"],
                        context_requirements=data.get("context_requirements", {}),
                        success_rate=data.get("success_rate", 0.0),
                        avg_duration=data.get("avg_duration", 0.0)
                    )
                
                logger.info(f"Loaded {len(self.workflow_templates)} workflow templates")
                
            except Exception as e:
                logger.error(f"Failed to load workflows: {e}")
    
    async def save_patterns(self):
        """Save learned patterns to disk"""
        patterns_dir = Path("./predictive_patterns")
        patterns_dir.mkdir(exist_ok=True)
        
        # Save action patterns
        patterns_data = {}
        for pattern_id, pattern in self.action_patterns.items():
            patterns_data[pattern_id] = {
                "sequence": pattern.sequence,
                "frequency": pattern.frequency,
                "avg_time_between": pattern.avg_time_between,
                "confidence": pattern.confidence,
                "last_seen": pattern.last_seen.isoformat()
            }
        
        with open(patterns_dir / "action_patterns.json", 'w') as f:
            json.dump(patterns_data, f, indent=2)
        
        # Save workflows
        workflows_data = {}
        for workflow_id, workflow in self.workflow_templates.items():
            workflows_data[workflow_id] = {
                "name": workflow.name,
                "steps": workflow.steps,
                "triggers": workflow.triggers,
                "context_requirements": workflow.context_requirements,
                "success_rate": workflow.success_rate,
                "avg_duration": workflow.avg_duration
            }
        
        with open(patterns_dir / "workflows.json", 'w') as f:
            json.dump(workflows_data, f, indent=2)
        
        logger.info("Saved patterns and workflows")
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return {
            "active_predictions": len(self.active_predictions),
            "learned_patterns": len(self.action_patterns),
            "workflow_templates": len(self.workflow_templates),
            "cache_size": len(self.preload_cache),
            "preloaded_resources": len(self.preloaded_resources),
            "hit_rate": self.hit_rate,
            "false_positive_rate": self.false_positive_rate,
            "avg_lead_time_seconds": self.avg_prediction_lead_time,
            "prediction_accuracy_by_type": dict(self.prediction_accuracy),
            "active_workflows": len(self.active_workflows),
            "resource_patterns": sum(len(patterns) for patterns in self.resource_patterns.values())
        }
    
    async def shutdown(self):
        """Gracefully shutdown the predictive system"""
        logger.info("Shutting down Predictive Pre-loading System")
        
        self.running = False
        
        # Save learned patterns
        await self.save_patterns()
        
        # Clear caches
        self.preload_cache.clear()
        self.preloaded_resources.clear()
        
        logger.info("Predictive Pre-loading System shutdown complete")


# Integration helper
async def create_predictive_system(context_manager: ContextPersistenceManager) -> PredictivePreloadingSystem:
    """Create and initialize predictive system"""
    system = PredictivePreloadingSystem(context_manager)
    await system.initialize()
    return system
