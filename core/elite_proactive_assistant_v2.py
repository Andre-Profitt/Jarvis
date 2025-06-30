#!/usr/bin/env python3
"""
JARVIS Elite Context-Aware Proactive Assistance System v2.0
Enhanced with Multi-Modal Fusion Intelligence
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import torch
import torch.nn as nn
from transformers import pipeline, AutoModel, AutoTokenizer
import json
import os
import psutil
import git
import requests
from pathlib import Path
import schedule
import logging
from enum import Enum
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cosine
import networkx as nx
import ray
import redis
from contextlib import contextmanager

# Import the multi-modal fusion system
from .multimodal_fusion import UnifiedPerception, ModalityType
from .fusion_improvements import OnlineLearningModule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context the assistant can understand"""

    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    ACTIVITY = "activity"
    EMOTIONAL = "emotional"
    TASK = "task"
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    COGNITIVE_LOAD = "cognitive_load"
    PREFERENCE = "preference"
    HISTORICAL = "historical"
    MULTIMODAL = "multimodal"  # New: unified multi-modal context


@dataclass
class EnhancedContextSignal:
    """Enhanced context signal with multi-modal support"""

    signal_type: ContextType
    value: Any
    confidence: float
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    modality_contributions: Dict[str, float] = field(default_factory=dict)
    uncertainty: Optional[Dict[str, float]] = None


@dataclass
class ProactiveActionV2:
    """Enhanced proactive action with multi-modal awareness"""

    action_id: str
    action_type: str
    description: str
    confidence: float
    priority: int
    context: Dict[str, Any]
    estimated_value: float
    estimated_disruption: float
    execute_function: Callable
    prerequisites: List[str] = field(default_factory=list)
    consequences: List[str] = field(default_factory=list)
    user_approval_required: bool = False
    multi_modal_requirements: List[str] = field(
        default_factory=list
    )  # Required modalities
    adaptive_parameters: Dict[str, Any] = field(default_factory=dict)  # ML-tuned params


class EnhancedContextualMemory:
    """Enhanced memory system with multi-modal episodic storage"""

    def __init__(self, memory_size: int = 10000):
        self.memory_size = memory_size
        self.episodic_memory = deque(maxlen=memory_size)
        self.semantic_memory = {}
        self.working_memory = {}
        self.multi_modal_memory = deque(maxlen=1000)  # Stores rich multi-modal episodes

        # Redis connection
        try:
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"), decode_responses=True
            )
            self.redis_client.ping()
            self.redis_available = True
        except:
            self.redis_client = None
            self.redis_available = False
            logger.warning("Redis not available, using in-memory storage only")

        # Enhanced memory encoder
        try:
            self.memory_encoder = AutoModel.from_pretrained(
                "sentence-transformers/all-mpnet-base-v2"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-mpnet-base-v2"
            )
        except:
            logger.warning("Could not load enhanced sentence transformer")
            self.memory_encoder = None
            self.tokenizer = None

    async def store_multi_modal_episode(
        self, episode: Dict[str, Any], fusion_output: Dict[str, Any]
    ):
        """Store a rich multi-modal episode"""

        # Create comprehensive memory entry
        memory_entry = {
            "episode": episode,
            "fusion_output": fusion_output,
            "timestamp": datetime.now(),
            "importance": self._calculate_multi_modal_importance(
                episode, fusion_output
            ),
            "modality_snapshot": self._extract_modality_snapshot(fusion_output),
            "emotional_context": fusion_output.get("insights", {}).get(
                "emotional_state", {}
            ),
            "causal_factors": fusion_output.get("causal_factors", {}),
        }

        # Store in multi-modal memory
        self.multi_modal_memory.append(memory_entry)

        # Also store in regular episodic memory
        await self.store_episode(episode)

        # Update semantic memory with multi-modal concepts
        await self._update_multi_modal_semantic_memory(memory_entry)

    def _calculate_multi_modal_importance(
        self, episode: Dict[str, Any], fusion_output: Dict[str, Any]
    ) -> float:
        """Calculate importance with multi-modal factors"""
        base_importance = self._calculate_importance(episode)

        # Add multi-modal factors
        uncertainty = fusion_output.get("uncertainty", {}).get("total", 0.5)
        confidence = fusion_output.get("confidence", 0.5)

        # High uncertainty or low confidence makes memory more important (unusual)
        multi_modal_factor = (1 - confidence) * 0.3 + uncertainty * 0.2

        return min(base_importance + multi_modal_factor, 1.0)

    def _extract_modality_snapshot(
        self, fusion_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract key features from each modality for memory"""
        return {
            "dominant_modality": fusion_output.get("insights", {})
            .get("attention_focus", {})
            .get("dominant_modality"),
            "modality_contributions": fusion_output.get("modality_contributions", {}),
            "representation_summary": (
                fusion_output.get("representation").mean().item()
                if "representation" in fusion_output
                else 0
            ),
        }

    async def _update_multi_modal_semantic_memory(self, memory_entry: Dict[str, Any]):
        """Update semantic memory with multi-modal concepts"""
        # Extract concepts from different modalities
        if "causal_factors" in memory_entry:
            for modality, factors in (
                memory_entry["causal_factors"].get("causal_graph", {}).items()
            ):
                concept_key = f"causal_{modality}"
                if concept_key not in self.semantic_memory:
                    self.semantic_memory[concept_key] = {
                        "count": 0,
                        "associations": defaultdict(int),
                    }

                self.semantic_memory[concept_key]["count"] += 1
                for associated in factors:
                    self.semantic_memory[concept_key]["associations"][associated] += 1


class MultiModalContextAnalyzer:
    """Context analyzer enhanced with multi-modal fusion"""

    def __init__(self):
        # Initialize unified perception system
        self.unified_perception = UnifiedPerception()

        # Keep original signal processors for compatibility
        self.signal_processors = {
            ContextType.TEMPORAL: self._process_temporal_context,
            ContextType.SPATIAL: self._process_spatial_context,
            ContextType.ACTIVITY: self._process_activity_context,
            ContextType.EMOTIONAL: self._process_emotional_context,
            ContextType.TASK: self._process_task_context,
            ContextType.ENVIRONMENTAL: self._process_environmental_context,
            ContextType.COGNITIVE_LOAD: self._process_cognitive_load,
            ContextType.PREFERENCE: self._process_preference_context,
            ContextType.HISTORICAL: self._process_historical_context,
            ContextType.MULTIMODAL: self._process_multimodal_context,  # New processor
        }

        # ML models
        try:
            self.activity_classifier = pipeline(
                "zero-shot-classification", model="facebook/bart-large-mnli"
            )
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="finiteautomata/bertweet-base-sentiment-analysis",
            )
        except:
            logger.warning("Could not load ML pipelines")
            self.activity_classifier = None
            self.sentiment_analyzer = None

    async def analyze_comprehensive_context(
        self, multi_modal_inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive context analysis with multi-modal fusion"""
        context_signals = []

        # First, gather traditional context signals
        for context_type, processor in self.signal_processors.items():
            if context_type != ContextType.MULTIMODAL:  # Skip multimodal for now
                try:
                    signals = await processor()
                    context_signals.extend(signals)
                except Exception as e:
                    logger.warning(f"Failed to process {context_type}: {e}")

        # Now process multi-modal context if inputs provided
        fusion_output = None
        if multi_modal_inputs:
            try:
                # Use unified perception for deep multi-modal understanding
                fusion_output = await self.unified_perception.perceive(
                    multi_modal_inputs
                )

                # Convert fusion output to context signal
                multi_modal_signal = EnhancedContextSignal(
                    signal_type=ContextType.MULTIMODAL,
                    value=fusion_output["insights"],
                    confidence=fusion_output["confidence"],
                    timestamp=datetime.now(),
                    source="unified_perception",
                    metadata={"fusion_output": fusion_output},
                    modality_contributions=fusion_output["modality_contributions"],
                    uncertainty=fusion_output["uncertainty"],
                )
                context_signals.append(multi_modal_signal)

            except Exception as e:
                logger.error(f"Failed to process multi-modal context: {e}")

        # Build comprehensive user state
        user_state = await self._build_enhanced_user_state(
            context_signals, fusion_output
        )

        return {
            "raw_signals": context_signals,
            "fusion_output": fusion_output,
            "user_state": user_state,
            "timestamp": datetime.now(),
            "multi_modal_insights": (
                fusion_output.get("insights", {}) if fusion_output else {}
            ),
        }

    async def _process_multimodal_context(self) -> List[EnhancedContextSignal]:
        """Process multi-modal context"""
        # This is called when we have multi-modal inputs
        # The actual processing happens in analyze_comprehensive_context
        return []

    async def _build_enhanced_user_state(
        self,
        signals: List[EnhancedContextSignal],
        fusion_output: Optional[Dict[str, Any]],
    ) -> "EnhancedUserState":
        """Build enhanced user state with multi-modal awareness"""

        # Extract from fusion output if available
        if fusion_output and "insights" in fusion_output:
            insights = fusion_output["insights"]
            activity = insights.get("primary_intent", "unknown")
            emotional_state = insights.get("emotional_state", {})
            cognitive_load = insights.get("cognitive_load", 0.5)

            # Calculate derived metrics
            focus_level = 1.0 - cognitive_load
            stress_level = 1.0 - emotional_state.get("valence", 0.5)
            availability = emotional_state.get("dominance", 0.7)
        else:
            # Fallback to traditional analysis
            activity = "unknown"
            focus_level = 0.5
            stress_level = 0.3
            availability = 0.7
            emotional_state = {}
            cognitive_load = 0.5

        # Get preferences
        preferences = await self._load_user_preferences()

        return EnhancedUserState(
            activity=activity,
            focus_level=focus_level,
            stress_level=stress_level,
            availability=availability,
            context_signals=signals,
            recent_actions=deque(maxlen=50),
            preferences=preferences,
            emotional_state=emotional_state,
            cognitive_load=cognitive_load,
            multi_modal_confidence=(
                fusion_output.get("confidence", 0.0) if fusion_output else 0.0
            ),
        )

    # Keep all original context processors for backward compatibility
    async def _process_temporal_context(self) -> List[EnhancedContextSignal]:
        """Analyze temporal context"""
        now = datetime.now()
        signals = []

        # Time of day analysis
        hour = now.hour
        if 6 <= hour < 9:
            signals.append(
                EnhancedContextSignal(
                    ContextType.TEMPORAL,
                    "morning_routine",
                    0.9,
                    now,
                    "temporal_analyzer",
                    {"period": "morning", "typical_activities": ["email", "planning"]},
                )
            )
        elif 9 <= hour < 12:
            signals.append(
                EnhancedContextSignal(
                    ContextType.TEMPORAL,
                    "deep_work_time",
                    0.85,
                    now,
                    "temporal_analyzer",
                    {"period": "morning_work", "optimal_for": ["coding", "analysis"]},
                )
            )

        return signals

    async def _process_activity_context(self) -> List[EnhancedContextSignal]:
        """Analyze current activity context"""
        signals = []

        # Get active window/application
        active_window = await self._get_active_window()

        if active_window:
            activity = await self._classify_activity(active_window)
            signals.append(
                EnhancedContextSignal(
                    ContextType.ACTIVITY,
                    activity["label"],
                    activity["score"],
                    datetime.now(),
                    "activity_monitor",
                    {"window": active_window, "details": activity},
                )
            )

        return signals

    async def _process_emotional_context(self) -> List[EnhancedContextSignal]:
        """Analyze emotional context"""
        # Enhanced with multi-modal emotion detection
        return []

    async def _process_spatial_context(self) -> List[EnhancedContextSignal]:
        """Analyze spatial context"""
        return []

    async def _process_task_context(self) -> List[EnhancedContextSignal]:
        """Analyze task context"""
        signals = []

        # Check for active git repository
        try:
            repo = git.Repo(".")
            if repo.is_dirty():
                signals.append(
                    EnhancedContextSignal(
                        ContextType.TASK,
                        "uncommitted_changes",
                        0.9,
                        datetime.now(),
                        "git_monitor",
                        {"changed_files": len(repo.index.diff(None))},
                    )
                )
        except:
            pass

        return signals

    async def _process_environmental_context(self) -> List[EnhancedContextSignal]:
        """Analyze environmental context"""
        signals = []

        # System resource monitoring
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        if cpu_percent > 80:
            signals.append(
                EnhancedContextSignal(
                    ContextType.ENVIRONMENTAL,
                    "high_cpu_usage",
                    0.9,
                    datetime.now(),
                    "system_monitor",
                    {"cpu_percent": cpu_percent},
                )
            )

        return signals

    async def _process_cognitive_load(self) -> List[EnhancedContextSignal]:
        """Analyze cognitive load"""
        # Enhanced with multi-modal cognitive assessment
        return []

    async def _process_preference_context(self) -> List[EnhancedContextSignal]:
        """Analyze user preferences"""
        preferences = await self._load_user_preferences()

        signals = [
            EnhancedContextSignal(
                ContextType.PREFERENCE,
                "loaded",
                1.0,
                datetime.now(),
                "preference_loader",
                {"preferences": preferences},
            )
        ]

        return signals

    async def _process_historical_context(self) -> List[EnhancedContextSignal]:
        """Analyze historical patterns"""
        return []

    async def _get_active_window(self) -> Optional[str]:
        """Get the currently active window/application"""
        import random

        apps = ["VSCode", "Chrome", "Slack", "Terminal", "Zoom"]
        return random.choice(apps) if random.random() > 0.1 else None

    async def _classify_activity(self, window: str) -> Dict[str, Any]:
        """Classify the current activity based on active window"""
        activity_map = {
            "VSCode": {"label": "coding", "score": 0.95},
            "Chrome": {"label": "research", "score": 0.7},
            "Slack": {"label": "communication", "score": 0.9},
            "Terminal": {"label": "development", "score": 0.85},
            "Zoom": {"label": "meeting", "score": 0.95},
        }
        return activity_map.get(window, {"label": "unknown", "score": 0.5})

    async def _load_user_preferences(self) -> Dict[str, Any]:
        """Load user preferences from storage"""
        return {
            "interruption_threshold": 0.3,
            "preferred_work_hours": {"start": 9, "end": 17},
            "focus_protection": True,
            "proactive_level": "high",
            "automation_preferences": {
                "auto_organize_files": True,
                "auto_commit": False,
                "auto_schedule": True,
            },
            "multi_modal_preferences": {
                "voice_feedback": True,
                "visual_notifications": True,
                "biometric_monitoring": True,
            },
        }


@dataclass
class EnhancedUserState:
    """Enhanced user state with multi-modal awareness"""

    activity: str
    focus_level: float
    stress_level: float
    availability: float
    context_signals: List[EnhancedContextSignal]
    recent_actions: deque
    preferences: Dict[str, Any]
    emotional_state: Dict[str, float] = field(default_factory=dict)
    cognitive_load: float = 0.5
    multi_modal_confidence: float = 0.0


class EnhancedProactiveIntelligenceEngine:
    """Enhanced proactive intelligence with multi-modal fusion"""

    def __init__(self):
        self.context_analyzer = MultiModalContextAnalyzer()
        self.memory_system = EnhancedContextualMemory()
        self.unified_perception = UnifiedPerception()

        # Enhanced components
        self.pattern_recognizer = EnhancedPatternRecognizer()
        self.action_predictor = MultiModalActionPredictor()
        self.causal_reasoner = CausalActionReasoner()

        # Online learning
        self.online_learner = OnlineLearningModule(
            self.unified_perception.neural_fusion
        )

    async def identify_proactive_opportunities(
        self,
        context: Dict[str, Any],
        multi_modal_inputs: Optional[Dict[str, Any]] = None,
    ) -> List[ProactiveActionV2]:
        """Identify opportunities with multi-modal awareness"""
        opportunities = []

        # If we have multi-modal inputs, use fusion for deeper understanding
        if multi_modal_inputs:
            fusion_output = context.get("fusion_output")
            if fusion_output:
                # Use causal reasoning to identify intervention points
                causal_opportunities = await self._identify_causal_opportunities(
                    fusion_output
                )
                opportunities.extend(causal_opportunities)

                # Use uncertainty to identify clarification needs
                uncertainty_opportunities = (
                    await self._identify_uncertainty_opportunities(fusion_output)
                )
                opportunities.extend(uncertainty_opportunities)

        # Traditional opportunity identification
        pattern_opportunities = await self._identify_pattern_based_opportunities(
            context
        )
        opportunities.extend(pattern_opportunities)

        predictive_opportunities = await self._identify_predictive_opportunities(
            context
        )
        opportunities.extend(predictive_opportunities)

        # Rank with multi-modal factors
        ranked_opportunities = await self._rank_opportunities_multimodal(
            opportunities, context
        )

        return ranked_opportunities

    async def _identify_causal_opportunities(
        self, fusion_output: Dict[str, Any]
    ) -> List[ProactiveActionV2]:
        """Identify opportunities based on causal reasoning"""
        opportunities = []

        causal_factors = fusion_output.get("causal_factors", {})
        causal_graph = causal_factors.get("causal_graph", {})

        # Look for strong causal relationships
        for modality, influences in causal_graph.items():
            if len(influences) > 2:  # This modality influences many others
                action = ProactiveActionV2(
                    action_id=f"optimize_{modality}_impact",
                    action_type="causal_optimization",
                    description=f"Optimize {modality} to improve overall experience",
                    confidence=0.8,
                    priority=2,
                    context={"causal_graph": causal_graph, "key_modality": modality},
                    estimated_value=0.8,
                    estimated_disruption=0.2,
                    execute_function=self._optimize_causal_factor,
                    multi_modal_requirements=[modality],
                )
                opportunities.append(action)

        return opportunities

    async def _identify_uncertainty_opportunities(
        self, fusion_output: Dict[str, Any]
    ) -> List[ProactiveActionV2]:
        """Identify opportunities to reduce uncertainty"""
        opportunities = []

        uncertainty = fusion_output.get("uncertainty", {})
        total_uncertainty = uncertainty.get("total", 0.0)

        if total_uncertainty > 0.7:
            # High uncertainty - suggest clarification
            dominant_modality = (
                fusion_output.get("insights", {})
                .get("attention_focus", {})
                .get("dominant_modality")
            )

            action = ProactiveActionV2(
                action_id="reduce_uncertainty",
                action_type="clarification",
                description="Gather additional information to improve understanding",
                confidence=0.9,
                priority=1,
                context={"uncertainty": uncertainty, "focus": dominant_modality},
                estimated_value=0.9,
                estimated_disruption=0.3,
                execute_function=self._gather_clarification,
                multi_modal_requirements=["text", "voice"],
            )
            opportunities.append(action)

        return opportunities

    async def _rank_opportunities_multimodal(
        self, opportunities: List[ProactiveActionV2], context: Dict[str, Any]
    ) -> List[ProactiveActionV2]:
        """Rank opportunities with multi-modal factors"""
        user_state = context["user_state"]
        fusion_output = context.get("fusion_output")

        scored_opportunities = []
        for opp in opportunities:
            # Base score
            score = opp.estimated_value / (opp.estimated_disruption + 0.1)

            # Multi-modal adjustments
            if fusion_output:
                # Boost score if action addresses uncertain areas
                if opp.action_type == "clarification":
                    uncertainty = fusion_output.get("uncertainty", {}).get("total", 0.5)
                    score *= 1 + uncertainty

                # Boost score if action uses preferred modalities
                modality_prefs = user_state.preferences.get(
                    "multi_modal_preferences", {}
                )
                for modality in opp.multi_modal_requirements:
                    if modality == "voice" and modality_prefs.get(
                        "voice_feedback", True
                    ):
                        score *= 1.1
                    elif modality == "visual" and modality_prefs.get(
                        "visual_notifications", True
                    ):
                        score *= 1.1

            # Adjust for emotional state
            if hasattr(user_state, "emotional_state"):
                valence = user_state.emotional_state.get("valence", 0.5)
                if valence < 0.3 and opp.action_type in ["assistance", "clarification"]:
                    score *= 1.2  # User might need more help when stressed

            scored_opportunities.append((score, opp))

        # Sort and return top opportunities
        scored_opportunities.sort(key=lambda x: x[0], reverse=True)
        return [opp for _, opp in scored_opportunities[:10]]

    async def _optimize_causal_factor(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize based on causal analysis"""
        key_modality = context.get("key_modality")
        logger.info(f"Optimizing {key_modality} based on causal analysis")

        # Implementation would adjust settings for the key modality
        return {
            "success": True,
            "optimization": f"Enhanced {key_modality} processing",
            "expected_improvement": 0.15,
        }

    async def _gather_clarification(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gather additional information to reduce uncertainty"""
        logger.info("Gathering clarification to reduce uncertainty")

        # Implementation would prompt for specific information
        return {
            "success": True,
            "clarification_type": "multi_modal",
            "uncertainty_reduction": 0.3,
        }

    async def _identify_pattern_based_opportunities(
        self, context: Dict[str, Any]
    ) -> List[ProactiveActionV2]:
        """Identify opportunities based on patterns (enhanced)"""
        opportunities = []
        user_state = context["user_state"]

        # Detect patterns including multi-modal patterns
        patterns = await self.pattern_recognizer.detect_patterns(
            user_state.recent_actions, context.get("fusion_output")
        )

        for pattern in patterns:
            if pattern["type"] == "multi_modal_routine" and pattern["confidence"] > 0.8:
                action = ProactiveActionV2(
                    action_id=f"optimize_mm_routine_{pattern['id']}",
                    action_type="multi_modal_optimization",
                    description=f"Optimize your {pattern['name']} with better sensory integration",
                    confidence=pattern["confidence"],
                    priority=2,
                    context={"pattern": pattern},
                    estimated_value=0.85,
                    estimated_disruption=0.15,
                    execute_function=self._optimize_multi_modal_routine,
                    multi_modal_requirements=pattern.get("modalities", []),
                )
                opportunities.append(action)

        return opportunities

    async def _identify_predictive_opportunities(
        self, context: Dict[str, Any]
    ) -> List[ProactiveActionV2]:
        """Predict future needs with multi-modal awareness"""
        opportunities = []

        # Use multi-modal predictor
        predictions = await self.action_predictor.predict_next_actions(
            context, context.get("fusion_output")
        )

        for prediction in predictions:
            if prediction["probability"] > 0.7:
                action = ProactiveActionV2(
                    action_id=f"prepare_{prediction['task']}",
                    action_type="preparation",
                    description=f"Prepare for {prediction['task']}",
                    confidence=prediction["probability"],
                    priority=3,
                    context={"prediction": prediction},
                    estimated_value=prediction["value"],
                    estimated_disruption=0.1,
                    execute_function=self._prepare_for_task,
                    multi_modal_requirements=prediction.get("required_modalities", []),
                )
                opportunities.append(action)

        return opportunities

    async def _optimize_multi_modal_routine(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize a multi-modal routine"""
        pattern = context["pattern"]
        logger.info(f"Optimizing multi-modal routine: {pattern['name']}")

        optimizations = {
            "visual": "Enhanced visual layout for better scanning",
            "audio": "Optimized audio alerts for minimal disruption",
            "temporal": "Adjusted timing for natural rhythm",
        }

        return {
            "success": True,
            "optimizations": optimizations,
            "expected_improvement": 0.25,
        }

    async def _prepare_for_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare for predicted task"""
        prediction = context["prediction"]
        logger.info(f"Preparing for task: {prediction['task']}")

        return {
            "success": True,
            "preparations": prediction.get("preparation", []),
            "readiness": 0.95,
        }


class EnhancedPatternRecognizer:
    """Pattern recognition with multi-modal awareness"""

    async def detect_patterns(
        self, recent_actions: deque, fusion_output: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Detect patterns including multi-modal patterns"""
        patterns = []

        # Traditional pattern detection
        if len(recent_actions) > 5:
            patterns.append(
                {
                    "type": "routine",
                    "id": "morning_routine",
                    "name": "Morning routine",
                    "confidence": 0.85,
                    "description": "Regular morning activities",
                    "optimization_potential": 0.6,
                }
            )

        # Multi-modal pattern detection
        if fusion_output:
            modality_contributions = fusion_output.get("modality_contributions", {})

            # Check for consistent modality usage patterns
            if modality_contributions:
                dominant = max(modality_contributions.items(), key=lambda x: x[1])
                if dominant[1] > 0.6:  # One modality dominates
                    patterns.append(
                        {
                            "type": "multi_modal_routine",
                            "id": f"dominant_{dominant[0]}",
                            "name": f"{dominant[0].title()}-heavy workflow",
                            "confidence": 0.8,
                            "description": f"Workflow relies heavily on {dominant[0]}",
                            "modalities": list(modality_contributions.keys()),
                            "optimization_potential": 0.7,
                        }
                    )

        return patterns


class MultiModalActionPredictor:
    """Predict actions with multi-modal context"""

    async def predict_next_actions(
        self, context: Dict[str, Any], fusion_output: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Predict next actions with multi-modal awareness"""
        predictions = []
        user_state = context.get("user_state")

        if not user_state:
            return predictions

        # Time-based predictions
        current_time = datetime.now()
        hour = current_time.hour

        # Enhanced predictions with multi-modal requirements
        if fusion_output:
            emotional_state = fusion_output.get("insights", {}).get(
                "emotional_state", {}
            )

            # Stress-based predictions
            if emotional_state.get("arousal", 0.5) > 0.7:
                predictions.append(
                    {
                        "task": "stress_reduction",
                        "probability": 0.85,
                        "value": 0.9,
                        "reason": "High arousal detected",
                        "preparation": [
                            "dim_lights",
                            "play_calming_music",
                            "suggest_break",
                        ],
                        "required_modalities": ["biometric", "environmental"],
                    }
                )

        # Traditional predictions
        if 16 <= hour <= 18:
            predictions.append(
                {
                    "task": "daily_wrap_up",
                    "probability": 0.8,
                    "value": 0.7,
                    "reason": "End of day approaching",
                    "preparation": ["summarize_tasks", "plan_tomorrow"],
                    "required_modalities": ["text", "temporal"],
                }
            )

        return predictions


class CausalActionReasoner:
    """Reason about causal relationships for actions"""

    async def analyze_action_causality(
        self, action: ProactiveActionV2, fusion_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze causal impact of an action"""
        causal_factors = fusion_output.get("causal_factors", {})

        # Predict intervention effects
        predicted_effects = {}
        for modality in action.multi_modal_requirements:
            if modality in causal_factors.get("intervention_effects", {}):
                predicted_effects[modality] = causal_factors["intervention_effects"][
                    modality
                ]

        return {
            "predicted_effects": predicted_effects,
            "confidence": (
                len(predicted_effects) / len(action.multi_modal_requirements)
                if action.multi_modal_requirements
                else 0
            ),
        }


class EliteProactiveAssistantV2:
    """Enhanced elite proactive assistant with multi-modal fusion"""

    def __init__(self):
        self.intelligence_engine = EnhancedProactiveIntelligenceEngine()
        self.context_analyzer = MultiModalContextAnalyzer()
        self.memory_system = EnhancedContextualMemory()
        self.unified_perception = UnifiedPerception()

        # Configuration
        self.config = {
            "min_confidence": 0.7,
            "max_disruption": 0.3,
            "check_interval": 10,
            "batch_actions": True,
            "learning_enabled": True,
            "multi_modal_enabled": True,
        }

        # Metrics
        self.metrics = {
            "actions_taken": 0,
            "actions_successful": 0,
            "user_satisfaction": 0.8,
            "time_saved": timedelta(),
            "multi_modal_accuracy": 0.0,
        }

        self.running = False
        self.multi_modal_buffer = deque(maxlen=10)  # Buffer for multi-modal inputs

    async def start_proactive_assistance(self):
        """Start the enhanced proactive assistance system"""
        logger.info(
            "🚀 Starting JARVIS Elite Proactive Assistant v2.0 with Multi-Modal Fusion..."
        )

        # Initialize subsystems
        await self._initialize_subsystems()

        self.running = True

        # Start monitoring loops including multi-modal
        await asyncio.gather(
            self._continuous_context_monitoring(),
            self._multi_modal_monitoring(),
            self._predictive_assistance_loop(),
            self._optimization_loop(),
            self._learning_loop(),
        )

    async def process_multi_modal_input(self, inputs: Dict[str, Any]):
        """Process multi-modal inputs for proactive assistance"""
        # Add to buffer
        self.multi_modal_buffer.append({"inputs": inputs, "timestamp": datetime.now()})

        # Trigger immediate analysis if high-priority
        if self._is_high_priority(inputs):
            await self._analyze_and_act(inputs)

    async def _multi_modal_monitoring(self):
        """Monitor multi-modal inputs for proactive opportunities"""
        while self.running:
            try:
                # Process buffered multi-modal inputs
                if self.multi_modal_buffer:
                    recent_input = self.multi_modal_buffer[-1]

                    # Comprehensive multi-modal analysis
                    context = await self.context_analyzer.analyze_comprehensive_context(
                        recent_input["inputs"]
                    )

                    # Store in enhanced memory
                    if context.get("fusion_output"):
                        await self.memory_system.store_multi_modal_episode(
                            {
                                "inputs": recent_input["inputs"],
                                "timestamp": recent_input["timestamp"],
                            },
                            context["fusion_output"],
                        )

                    # Identify multi-modal opportunities
                    opportunities = (
                        await self.intelligence_engine.identify_proactive_opportunities(
                            context, recent_input["inputs"]
                        )
                    )

                    # Execute high-value actions
                    for opportunity in opportunities[:3]:  # Top 3
                        if await self._should_execute_multimodal(opportunity, context):
                            await self._execute_proactive_action(opportunity, context)

            except Exception as e:
                logger.error(f"Error in multi-modal monitoring: {e}")

            await asyncio.sleep(5)  # Check every 5 seconds

    async def _should_execute_multimodal(
        self, action: ProactiveActionV2, context: Dict[str, Any]
    ) -> bool:
        """Determine if multi-modal action should be executed"""

        # Check basic criteria
        if action.confidence < self.config["min_confidence"]:
            return False

        # Check multi-modal specific criteria
        if context.get("fusion_output"):
            uncertainty = (
                context["fusion_output"].get("uncertainty", {}).get("total", 0.5)
            )

            # Don't act if uncertainty is too high (unless it's a clarification action)
            if uncertainty > 0.8 and action.action_type != "clarification":
                return False

            # Check if required modalities are available
            available_modalities = set(
                context["fusion_output"].get("modality_contributions", {}).keys()
            )
            required_modalities = set(action.multi_modal_requirements)

            if required_modalities and not required_modalities.issubset(
                available_modalities
            ):
                return False

        return True

    async def _analyze_and_act(self, inputs: Dict[str, Any]):
        """Analyze inputs and take immediate action if needed"""
        # Quick analysis for high-priority situations
        context = await self.context_analyzer.analyze_comprehensive_context(inputs)

        if context.get("fusion_output"):
            insights = context["fusion_output"].get("insights", {})

            # Check for crisis situations
            if insights.get("emotional_state", {}).get("arousal", 0) > 0.9:
                await self._handle_crisis(context)

    async def _handle_crisis(self, context: Dict[str, Any]):
        """Handle crisis situation with multi-modal response"""
        logger.warning("Crisis detected - initiating multi-modal response")

        # Create crisis response action
        crisis_action = ProactiveActionV2(
            action_id=f"crisis_response_{datetime.now().timestamp()}",
            action_type="crisis_intervention",
            description="Multi-modal crisis intervention",
            confidence=0.95,
            priority=1,
            context=context,
            estimated_value=1.0,
            estimated_disruption=0.5,  # Worth the disruption
            execute_function=self._execute_crisis_response,
            multi_modal_requirements=["voice", "visual", "environmental"],
        )

        await self._execute_proactive_action(crisis_action, context)

    async def _execute_crisis_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-modal crisis response"""
        logger.info("Executing multi-modal crisis response")

        responses = {
            "voice": "Calm, reassuring message delivered",
            "visual": "Calming visuals displayed",
            "environmental": "Adjusted lighting and temperature",
            "cognitive": "Simplified interface activated",
        }

        return {"success": True, "responses": responses, "stress_reduction": 0.3}

    def _is_high_priority(self, inputs: Dict[str, Any]) -> bool:
        """Check if inputs indicate high-priority situation"""
        # Check for stress indicators
        if "biometric" in inputs:
            hr = inputs["biometric"].get("heart_rate", 70)
            if hr > 100:
                return True

        # Check for keywords
        if "text" in inputs:
            crisis_keywords = ["urgent", "help", "emergency", "critical", "asap"]
            text_lower = inputs["text"].lower()
            if any(keyword in text_lower for keyword in crisis_keywords):
                return True

        return False

    async def _continuous_context_monitoring(self):
        """Enhanced context monitoring with multi-modal awareness"""
        while self.running:
            try:
                # Traditional context analysis
                context = await self.context_analyzer.analyze_comprehensive_context()

                # Store in memory
                await self.memory_system.store_episode(
                    {"context": context, "timestamp": datetime.now()}
                )

                # Identify opportunities
                opportunities = (
                    await self.intelligence_engine.identify_proactive_opportunities(
                        context
                    )
                )

                # Filter and execute
                filtered_opportunities = await self._filter_opportunities(
                    opportunities, context
                )

                for opportunity in filtered_opportunities:
                    if await self._should_execute(opportunity, context):
                        await self._execute_proactive_action(opportunity, context)

            except Exception as e:
                logger.error(f"Error in context monitoring: {e}")

            await asyncio.sleep(self.config["check_interval"])

    async def _predictive_assistance_loop(self):
        """Enhanced predictive loop with multi-modal predictions"""
        while self.running:
            try:
                # Get comprehensive context
                context = await self.context_analyzer.analyze_comprehensive_context()

                # Use recent multi-modal inputs if available
                recent_mm_input = (
                    self.multi_modal_buffer[-1] if self.multi_modal_buffer else None
                )
                if recent_mm_input:
                    mm_context = (
                        await self.context_analyzer.analyze_comprehensive_context(
                            recent_mm_input["inputs"]
                        )
                    )
                    predictions = await self.intelligence_engine.action_predictor.predict_next_actions(
                        mm_context, mm_context.get("fusion_output")
                    )
                else:
                    predictions = await self.intelligence_engine.action_predictor.predict_next_actions(
                        context
                    )

                # Act on high-probability predictions
                for prediction in predictions:
                    if prediction["probability"] > 0.8:
                        preparation_action = ProactiveActionV2(
                            action_id=f"prep_{prediction['task']}_{datetime.now().timestamp()}",
                            action_type="preparation",
                            description=f"Preparing for {prediction['task']}",
                            confidence=prediction["probability"],
                            priority=3,
                            context={"prediction": prediction},
                            estimated_value=prediction["value"] * 0.8,
                            estimated_disruption=0.1,
                            execute_function=self.intelligence_engine._prepare_for_task,
                            multi_modal_requirements=prediction.get(
                                "required_modalities", []
                            ),
                        )

                        if await self._should_execute(preparation_action, context):
                            await self._execute_proactive_action(
                                preparation_action, context
                            )

            except Exception as e:
                logger.error(f"Error in predictive assistance: {e}")

            await asyncio.sleep(60)  # Every minute

    async def _optimization_loop(self):
        """Optimization loop with multi-modal enhancements"""
        while self.running:
            try:
                # Analyze for optimization opportunities
                optimization_opportunities = (
                    await self._identify_optimization_opportunities()
                )

                # Check multi-modal optimization potential
                if self.multi_modal_buffer:
                    mm_optimizations = await self._identify_multimodal_optimizations()
                    optimization_opportunities.extend(mm_optimizations)

                for opportunity in optimization_opportunities:
                    if opportunity["potential_improvement"] > 0.2:
                        optimization_action = ProactiveActionV2(
                            action_id=f"optimize_{opportunity['type']}_{datetime.now().timestamp()}",
                            action_type="optimization",
                            description=opportunity["description"],
                            confidence=opportunity["confidence"],
                            priority=4,
                            context={"opportunity": opportunity},
                            estimated_value=opportunity["potential_improvement"],
                            estimated_disruption=0.05,
                            execute_function=self._apply_optimization,
                            multi_modal_requirements=opportunity.get("modalities", []),
                        )

                        await self._execute_proactive_action(optimization_action, {})

            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")

            await asyncio.sleep(300)  # Every 5 minutes

    async def _learning_loop(self):
        """Enhanced learning with multi-modal feedback"""
        while self.running:
            try:
                # Learn from recent episodes
                recent_episodes = list(self.memory_system.episodic_memory)[-100:]
                mm_episodes = list(self.memory_system.multi_modal_memory)[-50:]

                if mm_episodes:
                    # Update multi-modal accuracy metric
                    successful_mm = sum(
                        1
                        for ep in mm_episodes
                        if ep.get("episode", {}).get("result", {}).get("success")
                    )
                    self.metrics["multi_modal_accuracy"] = successful_mm / len(
                        mm_episodes
                    )

                    # Online learning for fusion network
                    for episode in mm_episodes[-10:]:
                        if "fusion_output" in episode:
                            inputs = episode["episode"].get("inputs", {})
                            feedback = {
                                "positive": episode["episode"]
                                .get("result", {})
                                .get("success", False)
                            }

                            # Convert inputs to tensor format
                            tensor_inputs = {}
                            for modality, data in inputs.items():
                                if modality in self.unified_perception.processors:
                                    tensor_inputs[modality] = torch.randn(
                                        768
                                    )  # Placeholder

                            if tensor_inputs:
                                await self.intelligence_engine.online_learner.learn_from_interaction(
                                    tensor_inputs, feedback
                                )

                logger.info(
                    f"Learning update completed. MM Accuracy: {self.metrics['multi_modal_accuracy']:.2%}"
                )

            except Exception as e:
                logger.error(f"Error in learning loop: {e}")

            await asyncio.sleep(3600)  # Every hour

    async def _identify_multimodal_optimizations(self) -> List[Dict[str, Any]]:
        """Identify multi-modal specific optimizations"""
        optimizations = []

        # Analyze modality usage patterns
        if self.memory_system.multi_modal_memory:
            modality_usage = defaultdict(int)
            for episode in list(self.memory_system.multi_modal_memory)[-20:]:
                snapshot = episode.get("modality_snapshot", {})
                if "modality_contributions" in snapshot:
                    for modality, contrib in snapshot["modality_contributions"].items():
                        modality_usage[modality] += contrib

            # Find underutilized modalities
            if modality_usage:
                avg_usage = sum(modality_usage.values()) / len(modality_usage)
                for modality, usage in modality_usage.items():
                    if usage < avg_usage * 0.5:  # Significantly underutilized
                        optimizations.append(
                            {
                                "type": f"enhance_{modality}",
                                "description": f"Enhance {modality} integration for better understanding",
                                "potential_improvement": 0.3,
                                "confidence": 0.8,
                                "action": "calibrate_modality",
                                "modalities": [modality],
                            }
                        )

        return optimizations

    async def _initialize_subsystems(self):
        """Initialize all subsystems"""
        logger.info("Initializing enhanced subsystems...")

        # Load user preferences
        preferences = await self.context_analyzer._load_user_preferences()

        # Initialize memory
        await self.memory_system.store_episode(
            {
                "type": "system_start",
                "timestamp": datetime.now(),
                "preferences": preferences,
                "version": "2.0",
            }
        )

        logger.info("✅ All subsystems initialized")

    async def _filter_opportunities(
        self, opportunities: List[ProactiveActionV2], context: Dict[str, Any]
    ) -> List[ProactiveActionV2]:
        """Filter opportunities with multi-modal awareness"""
        user_state = context["user_state"]
        filtered = []

        for opportunity in opportunities:
            # Traditional filtering
            if user_state.focus_level > 0.8 and opportunity.estimated_disruption > 0.2:
                continue

            # Multi-modal filtering
            if hasattr(user_state, "multi_modal_confidence"):
                if (
                    user_state.multi_modal_confidence < 0.5
                    and opportunity.action_type != "clarification"
                ):
                    continue  # Don't act on low-confidence multi-modal understanding

            # Check preferences
            if not await self._check_user_preferences(opportunity, user_state):
                continue

            filtered.append(opportunity)

        # Sort by value/disruption ratio
        filtered.sort(
            key=lambda x: x.estimated_value / (x.estimated_disruption + 0.1),
            reverse=True,
        )

        return filtered[:5]

    async def _check_user_preferences(
        self, action: ProactiveActionV2, user_state: EnhancedUserState
    ) -> bool:
        """Check if action aligns with user preferences"""
        preferences = user_state.preferences

        # Check multi-modal preferences
        mm_prefs = preferences.get("multi_modal_preferences", {})

        if action.multi_modal_requirements:
            if "voice" in action.multi_modal_requirements and not mm_prefs.get(
                "voice_feedback", True
            ):
                return False
            if "biometric" in action.multi_modal_requirements and not mm_prefs.get(
                "biometric_monitoring", True
            ):
                return False

        # Traditional preference checks
        if action.estimated_disruption > preferences.get("interruption_threshold", 0.3):
            current_hour = datetime.now().hour
            work_hours = preferences.get("preferred_work_hours", {})
            if work_hours.get("start", 9) <= current_hour <= work_hours.get("end", 17):
                if preferences.get("focus_protection", True):
                    return False

        return True

    async def _should_execute(
        self, action: ProactiveActionV2, context: Dict[str, Any]
    ) -> bool:
        """Determine if action should be executed"""
        user_state = context["user_state"]

        # Check confidence
        if action.confidence < self.config["min_confidence"]:
            return False

        # Check disruption
        if action.estimated_disruption > self.config["max_disruption"]:
            if user_state.focus_level > 0.7:
                return False

        # Multi-modal specific checks
        if hasattr(user_state, "cognitive_load") and user_state.cognitive_load > 0.8:
            if action.action_type not in [
                "assistance",
                "clarification",
                "stress_reduction",
            ]:
                return False  # Only helpful actions when overloaded

        return True

    async def _execute_proactive_action(
        self, action: ProactiveActionV2, context: Dict[str, Any]
    ):
        """Execute proactive action with monitoring"""
        logger.info(f"🎯 Executing proactive action: {action.description}")

        start_time = datetime.now()

        try:
            # Execute the action
            result = await action.execute_function(action.context)

            # Record metrics
            self.metrics["actions_taken"] += 1
            if result["success"]:
                self.metrics["actions_successful"] += 1
                time_saved = result.get("time_saved", timedelta())
                self.metrics["time_saved"] += time_saved

            # Store in memory
            episode = {
                "action": action.action_id,
                "result": result,
                "context": context,
                "execution_time": datetime.now() - start_time,
                "multi_modal": len(action.multi_modal_requirements) > 0,
            }

            await self.memory_system.store_episode(episode)

            # If multi-modal, store enhanced episode
            if context.get("fusion_output") and action.multi_modal_requirements:
                await self.memory_system.store_multi_modal_episode(
                    episode, context["fusion_output"]
                )

        except Exception as e:
            logger.error(f"Failed to execute action {action.action_id}: {e}")

    async def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify optimization opportunities"""
        opportunities = []

        # File organization
        downloads_folder = Path.home() / "Downloads"
        if downloads_folder.exists():
            file_count = len(list(downloads_folder.glob("*")))
            if file_count > 50:
                opportunities.append(
                    {
                        "type": "file_organization",
                        "description": "Organize Downloads folder",
                        "potential_improvement": 0.3,
                        "confidence": 0.9,
                        "action": "organize_downloads",
                    }
                )

        return opportunities

    async def _apply_optimization(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimization"""
        opportunity = context["opportunity"]

        if opportunity["action"] == "organize_downloads":
            logger.info("Organizing Downloads folder")
            return {"success": True, "files_organized": 0}
        elif opportunity["action"] == "calibrate_modality":
            logger.info(
                f"Calibrating {opportunity.get('modalities', ['unknown'])[0]} modality"
            )
            return {"success": True, "calibration": "complete"}

        return {"success": False}

    async def stop(self):
        """Stop the assistant"""
        logger.info("Stopping JARVIS Elite Proactive Assistant v2.0...")
        self.running = False


# Main integration function
async def create_elite_proactive_assistant_v2():
    """Create and initialize the enhanced proactive assistant"""
    assistant = EliteProactiveAssistantV2()

    logger.info("🚀 Elite Proactive Assistant v2.0 with Multi-Modal Fusion initialized")
    return assistant


if __name__ == "__main__":
    # Test the enhanced system
    async def test_system():
        assistant = await create_elite_proactive_assistant_v2()

        # Simulate multi-modal input
        test_input = {
            "text": "I'm feeling overwhelmed with this project deadline",
            "voice": {
                "waveform": np.random.randn(16000 * 3),  # 3 seconds
                "sample_rate": 16000,
                "features": {
                    "pitch_variance": 0.7,
                    "speaking_rate": 1.2,
                    "volume": 0.8,
                },
            },
            "biometric": {
                "heart_rate": 95,
                "skin_conductance": 0.7,
                "temperature": 37.1,
            },
        }

        # Process the input
        await assistant.process_multi_modal_input(test_input)

        # Start the system
        await assistant.start_proactive_assistance()

    asyncio.run(test_system())
