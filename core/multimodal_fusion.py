#!/usr/bin/env python3
"""
Elite Multi-Modal Fusion Intelligence for JARVIS
Next-generation unified perception across all sensory modalities
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import logging
from dataclasses import dataclass
from collections import deque
import time
from transformers import AutoModel, AutoTokenizer, pipeline
import cv2
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
import librosa
import spacy
import redis
from pathlib import Path
import os

# Import improvements from the enhanced version
from .fusion_improvements import (
    ImprovedCrossModalAttention,
    AdaptiveNeuralFusionNetwork,
    UncertaintyQuantification,
    CausalReasoningModule,
    MetaLearningAdapter,
    RobustModalityProcessor,
    OnlineLearningModule,
    FederatedUnifiedPerception,
    DeploymentOptimizedPerception,
)

logger = logging.getLogger(__name__)


@dataclass
class ModalityInput:
    """Structured input for a single modality"""

    modality_type: str
    data: Any
    metadata: Dict[str, Any]
    timestamp: float
    confidence: float = 1.0


class ModalityType:
    """Supported modality types"""

    TEXT = "text"
    VOICE = "voice"
    VISION = "vision"
    BIOMETRIC = "biometric"
    TEMPORAL = "temporal"
    ENVIRONMENTAL = "environmental"
    SCREEN = "screen"
    GESTURE = "gesture"


class UnifiedPerception:
    """Elite Multi-Modal Fusion Intelligence System"""

    def __init__(
        self,
        fusion_dim: int = 1024,
        output_dim: int = 512,
        context_window: int = 100,
        device: str = None,
    ):
        """Initialize the unified perception system"""

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing UnifiedPerception on {self.device}")

        self.fusion_dim = fusion_dim
        self.output_dim = output_dim
        self.context_window = context_window

        # Initialize modality processors
        self._initialize_processors()

        # Initialize neural fusion network with improvements
        self.neural_fusion = AdaptiveNeuralFusionNetwork(
            input_dims=self._get_input_dims(),
            fusion_dim=fusion_dim,
            output_dim=output_dim,
            use_flash_attention=torch.cuda.is_available(),
        ).to(self.device)

        # Initialize enhanced components
        self.uncertainty_module = UncertaintyQuantification(output_dim)
        self.causal_reasoner = CausalReasoningModule(fusion_dim)
        self.meta_learner = MetaLearningAdapter(fusion_dim)
        self.online_learner = OnlineLearningModule(self.neural_fusion)

        # Context memory
        self.context_memory = deque(maxlen=context_window)
        self.temporal_fusion = TemporalContextIntegrator(fusion_dim)

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()

        # Distributed processing support
        self.distributed_processor = DistributedModalityProcessor()

        # Robust error handling
        self.robust_processor = RobustModalityProcessor()

        # Redis for distributed state
        try:
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"), decode_responses=True
            )
            self.redis_client.ping()
            self.redis_available = True
        except:
            self.redis_client = None
            self.redis_available = False
            logger.warning("Redis not available, using local memory only")

        logger.info("✅ UnifiedPerception initialized successfully")

    def _initialize_processors(self):
        """Initialize modality-specific processors"""
        self.processors = {}

        # Text processor
        try:
            self.processors[ModalityType.TEXT] = TextModalityProcessor()
        except Exception as e:
            logger.warning(f"Could not initialize text processor: {e}")
            self.processors[ModalityType.TEXT] = None

        # Voice processor
        try:
            self.processors[ModalityType.VOICE] = VoiceModalityProcessor()
        except Exception as e:
            logger.warning(f"Could not initialize voice processor: {e}")
            self.processors[ModalityType.VOICE] = None

        # Vision processor
        try:
            self.processors[ModalityType.VISION] = VisionModalityProcessor()
        except Exception as e:
            logger.warning(f"Could not initialize vision processor: {e}")
            self.processors[ModalityType.VISION] = None

        # Biometric processor
        self.processors[ModalityType.BIOMETRIC] = BiometricModalityProcessor()

        # Temporal processor
        self.processors[ModalityType.TEMPORAL] = TemporalModalityProcessor()

        # Environmental processor
        self.processors[ModalityType.ENVIRONMENTAL] = EnvironmentalModalityProcessor()

    def _get_input_dims(self) -> Dict[str, int]:
        """Get input dimensions for each modality"""
        return {
            ModalityType.TEXT: 768,  # BERT embedding size
            ModalityType.VOICE: 512,  # Audio feature size
            ModalityType.VISION: 2048,  # Vision feature size
            ModalityType.BIOMETRIC: 128,  # Biometric feature size
            ModalityType.TEMPORAL: 64,  # Temporal feature size
            ModalityType.ENVIRONMENTAL: 32,  # Environmental feature size
            ModalityType.SCREEN: 1024,  # Screen content feature size
            ModalityType.GESTURE: 256,  # Gesture feature size
        }

    async def perceive(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process multi-modal inputs and generate unified understanding"""

        start_time = time.time()

        try:
            # Process each modality with robust error handling
            processed_inputs = await self._process_modalities(inputs)

            # Build context from history
            context = self._build_context()

            # Neural fusion with uncertainty and causal reasoning
            fusion_output = await self._neural_fusion(processed_inputs, context)

            # Temporal integration
            temporal_features = await self.temporal_fusion.integrate(
                fusion_output["representation"], self.context_memory
            )

            # Generate insights
            insights = await self._generate_insights(fusion_output, temporal_features)

            # Update context memory
            self.context_memory.append(
                {
                    "timestamp": time.time(),
                    "inputs": inputs,
                    "output": fusion_output,
                    "insights": insights,
                }
            )

            # Online learning if feedback provided
            if "feedback" in inputs:
                await self.online_learner.learn_from_interaction(
                    processed_inputs, inputs["feedback"]
                )

            # Record performance metrics
            processing_time = time.time() - start_time
            self.performance_monitor.record_processing_time(processing_time)

            return {
                "unified_understanding": fusion_output["representation"],
                "confidence": fusion_output["uncertainty"]["confidence"].item(),
                "uncertainty": fusion_output["uncertainty"],
                "causal_analysis": fusion_output["causal_factors"],
                "insights": insights,
                "processing_time": processing_time,
                "modality_contributions": fusion_output["routing_weights"],
                "context_relevance": await self._compute_context_relevance(
                    fusion_output
                ),
            }

        except Exception as e:
            logger.error(f"Error in perception: {e}")
            return self._generate_error_response(str(e))

    async def _process_modalities(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Process each modality with appropriate processor"""
        processed = {}

        # Parallel processing of modalities
        tasks = []
        for modality, data in inputs.items():
            if modality in self.processors and self.processors[modality]:
                tasks.append(self._process_single_modality(modality, data))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, tuple) and len(result) == 2:
                    modality, features = result
                    if features is not None:
                        processed[modality] = features
                elif isinstance(result, Exception):
                    logger.error(f"Error processing modality: {result}")

        return processed

    async def _process_single_modality(
        self, modality: str, data: Any
    ) -> Tuple[str, Optional[torch.Tensor]]:
        """Process a single modality"""
        try:
            processor = self.processors[modality]
            features = await processor.process(data)
            return modality, features.to(self.device) if features is not None else None
        except Exception as e:
            logger.error(f"Error processing {modality}: {e}")
            # Use robust processor as fallback
            features = await self.robust_processor.process(modality, data)
            return modality, features.to(self.device) if features is not None else None

    async def _neural_fusion(
        self, inputs: Dict[str, torch.Tensor], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform neural fusion of modalities"""

        # Adapt model based on context
        adapted_fusion = self.meta_learner.adapt(self.neural_fusion, context)

        # Forward pass through adapted fusion network
        with torch.cuda.amp.autocast(enabled=self.device == "cuda"):
            fusion_output = adapted_fusion(inputs, context)

        return fusion_output

    def _build_context(self) -> Dict[str, Any]:
        """Build context from memory"""
        if not self.context_memory:
            return {}

        # Get recent context
        recent_context = list(self.context_memory)[-10:]

        # Extract patterns
        patterns = self._extract_patterns(recent_context)

        # Get previous state
        previous_state = (
            recent_context[-1]["output"]["representation"] if recent_context else None
        )

        return {
            "previous_state": previous_state,
            "patterns": patterns,
            "history_length": len(self.context_memory),
            "timestamp": time.time(),
        }

    def _extract_patterns(
        self, context_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract patterns from context history"""
        patterns = {"temporal": [], "emotional": [], "activity": []}

        # Simple pattern extraction - could be more sophisticated
        for context in context_history:
            if "insights" in context:
                insights = context["insights"]
                if "emotional_state" in insights:
                    patterns["emotional"].append(insights["emotional_state"])
                if "primary_intent" in insights:
                    patterns["activity"].append(insights["primary_intent"])

        return patterns

    async def _generate_insights(
        self, fusion_output: Dict[str, Any], temporal_features: torch.Tensor
    ) -> Dict[str, Any]:
        """Generate high-level insights from fusion"""

        insights = {}

        # Extract unified representation
        unified = fusion_output["representation"]

        # Determine primary intent/activity
        insights["primary_intent"] = await self._classify_intent(unified)

        # Analyze emotional state
        insights["emotional_state"] = await self._analyze_emotion(unified)

        # Assess cognitive load
        insights["cognitive_load"] = await self._assess_cognitive_load(unified)

        # Identify attention focus
        insights["attention_focus"] = await self._identify_attention_focus(
            fusion_output
        )

        # Generate contextual recommendations
        insights["suggested_actions"] = await self._generate_recommendations(
            unified, insights, fusion_output["uncertainty"]
        )

        # Add causal insights
        if "causal_factors" in fusion_output:
            insights["causal_relationships"] = fusion_output["causal_factors"][
                "causal_graph"
            ]

        return insights

    async def _classify_intent(self, features: torch.Tensor) -> str:
        """Classify user intent from features"""
        # Simplified intent classification
        intents = ["query", "command", "conversation", "analysis", "creation"]

        # In production, use a trained classifier
        # For now, return most likely intent
        return "analysis"

    async def _analyze_emotion(self, features: torch.Tensor) -> Dict[str, float]:
        """Analyze emotional state from features"""
        # VAD (Valence, Arousal, Dominance) model
        return {
            "valence": 0.7,  # Positive
            "arousal": 0.5,  # Moderate energy
            "dominance": 0.6,  # In control
            "confidence": 0.85,
        }

    async def _assess_cognitive_load(self, features: torch.Tensor) -> float:
        """Assess cognitive load from features"""
        # Simplified assessment - in production use trained model
        return 0.4  # Moderate cognitive load

    async def _identify_attention_focus(
        self, fusion_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identify where attention is focused"""

        # Analyze modality contributions
        contributions = fusion_output.get("routing_weights", {})

        # Find dominant modality
        if contributions:
            dominant_modality = max(contributions.items(), key=lambda x: x[1])
            return {
                "dominant_modality": dominant_modality[0],
                "focus_strength": dominant_modality[1],
                "distribution": contributions,
            }

        return {"dominant_modality": "unknown", "focus_strength": 0.0}

    async def _generate_recommendations(
        self,
        features: torch.Tensor,
        insights: Dict[str, Any],
        uncertainty: Dict[str, torch.Tensor],
    ) -> List[Dict[str, Any]]:
        """Generate contextual recommendations"""
        recommendations = []

        # Based on cognitive load
        if insights["cognitive_load"] > 0.7:
            recommendations.append(
                {
                    "action": "simplify_interface",
                    "priority": "high",
                    "reason": "High cognitive load detected",
                }
            )

        # Based on emotional state
        if insights["emotional_state"]["arousal"] > 0.8:
            recommendations.append(
                {
                    "action": "provide_calming_response",
                    "priority": "medium",
                    "reason": "High arousal detected",
                }
            )

        # Based on uncertainty
        if uncertainty["total"] > 0.7:
            recommendations.append(
                {
                    "action": "request_clarification",
                    "priority": "high",
                    "reason": "High uncertainty in understanding",
                }
            )

        return recommendations

    async def _compute_context_relevance(self, fusion_output: Dict[str, Any]) -> float:
        """Compute relevance to historical context"""
        if not self.context_memory:
            return 1.0

        # Compare with recent context
        recent_outputs = [
            ctx["output"]["representation"]
            for ctx in list(self.context_memory)[-5:]
            if "output" in ctx and "representation" in ctx["output"]
        ]

        if not recent_outputs:
            return 1.0

        current = fusion_output["representation"]

        # Compute similarity
        similarities = []
        for past in recent_outputs:
            if past.shape == current.shape:
                similarity = F.cosine_similarity(
                    current.unsqueeze(0), past.unsqueeze(0)
                )
                similarities.append(similarity.item())

        return np.mean(similarities) if similarities else 1.0

    def _generate_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Generate response for error cases"""
        return {
            "unified_understanding": torch.zeros(self.output_dim),
            "confidence": 0.0,
            "insights": {"error": error_msg},
            "processing_time": 0.0,
            "modality_contributions": {},
            "context_relevance": 0.0,
        }

    async def optimize_for_deployment(self, target: str = "edge") -> Any:
        """Get deployment-optimized version"""
        optimizer = DeploymentOptimizedPerception(self.neural_fusion)
        return optimizer.optimize_for_deployment(target)


# Modality-specific processors


class TextModalityProcessor:
    """Process text inputs with semantic understanding"""

    def __init__(self):
        try:
            self.model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("Could not load text models, using fallback")
            self.model = None
            self.tokenizer = None
            self.nlp = None

    async def process(self, text: Union[str, Dict[str, Any]]) -> torch.Tensor:
        """Process text input"""
        if isinstance(text, dict):
            text = text.get("text", "")

        if not text:
            return torch.zeros(768)

        if self.model and self.tokenizer:
            # Encode with transformer
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1).squeeze()
        else:
            # Fallback to simple encoding
            features = torch.randn(768)

        return features


class VoiceModalityProcessor:
    """Process voice/audio inputs with emotion and prosody"""

    def __init__(self):
        self.sample_rate = 16000
        self.emotion_model = None  # Would load emotion detection model

    async def process(self, voice_data: Dict[str, Any]) -> torch.Tensor:
        """Process voice input"""

        if isinstance(voice_data, dict):
            waveform = voice_data.get("waveform", None)
            sample_rate = voice_data.get("sample_rate", self.sample_rate)
        else:
            waveform = voice_data
            sample_rate = self.sample_rate

        if waveform is None:
            return torch.zeros(512)

        # Extract features
        features_list = []

        # 1. MFCC features
        if isinstance(waveform, np.ndarray):
            mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            features_list.append(mfcc_mean)

        # 2. Prosodic features (pitch, energy)
        if isinstance(waveform, np.ndarray) and len(waveform) > 0:
            pitch = self._extract_pitch(waveform, sample_rate)
            energy = np.sqrt(np.mean(waveform**2))
            features_list.extend([pitch, energy])

        # 3. Emotion features (if available in metadata)
        if isinstance(voice_data, dict) and "features" in voice_data:
            emotion_features = [
                voice_data["features"].get("pitch_variance", 0.5),
                voice_data["features"].get("speaking_rate", 1.0),
                voice_data["features"].get("volume", 0.5),
            ]
            features_list.extend(emotion_features)

        # Combine features
        if features_list:
            combined = np.concatenate([np.atleast_1d(f) for f in features_list])
            # Pad or truncate to expected size
            if len(combined) < 512:
                combined = np.pad(combined, (0, 512 - len(combined)))
            else:
                combined = combined[:512]

            return torch.tensor(combined, dtype=torch.float32)

        return torch.zeros(512)

    def _extract_pitch(self, waveform: np.ndarray, sample_rate: int) -> float:
        """Extract pitch from waveform"""
        try:
            pitches = librosa.yin(waveform, fmin=50, fmax=400, sr=sample_rate)
            return (
                np.mean(pitches[pitches > 0])
                if len(pitches[pitches > 0]) > 0
                else 100.0
            )
        except:
            return 100.0


class VisionModalityProcessor:
    """Process visual inputs including screens and images"""

    def __init__(self):
        self.feature_extractor = None  # Would use CLIP or similar
        self.target_size = (224, 224)

    async def process(
        self, vision_data: Union[np.ndarray, Dict[str, Any]]
    ) -> torch.Tensor:
        """Process vision input"""

        if isinstance(vision_data, dict):
            image = vision_data.get("image", vision_data.get("screen_content", None))
        else:
            image = vision_data

        if image is None:
            return torch.zeros(2048)

        # Process image
        if isinstance(image, np.ndarray):
            # Resize if needed
            if image.shape[:2] != self.target_size:
                image = cv2.resize(image, self.target_size)

            # Extract features (simplified - would use pretrained model)
            features = self._extract_visual_features(image)

            return torch.tensor(features, dtype=torch.float32)

        return torch.zeros(2048)

    def _extract_visual_features(self, image: np.ndarray) -> np.ndarray:
        """Extract visual features from image"""
        # Simplified feature extraction
        # In production, use CLIP or similar

        features = []

        # Color histogram
        for channel in range(3):
            hist = cv2.calcHist([image], [channel], None, [32], [0, 256])
            features.extend(hist.flatten())

        # Edge features
        edges = cv2.Canny(image, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)

        # Pad to expected size
        features = np.array(features)
        if len(features) < 2048:
            features = np.pad(features, (0, 2048 - len(features)))
        else:
            features = features[:2048]

        return features


class BiometricModalityProcessor:
    """Process biometric signals"""

    def __init__(self):
        self.scaler = StandardScaler()

    async def process(self, biometric_data: Dict[str, Any]) -> torch.Tensor:
        """Process biometric input"""

        features = []

        # Extract available biometric signals
        hr = biometric_data.get("heart_rate", 70)
        hrv = biometric_data.get("hrv", 50)
        skin_conductance = biometric_data.get("skin_conductance", 0.5)
        temperature = biometric_data.get("temperature", 36.6)
        breathing_rate = biometric_data.get("breathing_rate", 16)

        # Normalize features
        features = [
            (hr - 70) / 20,  # Normalize around typical HR
            (hrv - 50) / 30,  # Normalize HRV
            skin_conductance,  # Already 0-1
            (temperature - 36.6) / 1.0,  # Normalize temperature
            (breathing_rate - 16) / 8,  # Normalize breathing
        ]

        # Add derived features
        stress_indicator = (hr - 70) / 20 + (1 - hrv / 80) + skin_conductance
        features.append(stress_indicator / 3)  # Average stress

        # Pad to expected size
        features_array = np.array(features)
        if len(features_array) < 128:
            features_array = np.pad(features_array, (0, 128 - len(features_array)))

        return torch.tensor(features_array, dtype=torch.float32)


class TemporalModalityProcessor:
    """Process temporal context and patterns"""

    def __init__(self):
        self.time_encoder = TimeEncoder()

    async def process(self, temporal_data: Dict[str, Any]) -> torch.Tensor:
        """Process temporal input"""

        features = []

        # Current time features
        current_time = temporal_data.get("current_time", time.time())
        dt = datetime.fromtimestamp(current_time)

        # Time of day encoding (cyclic)
        hour_sin = np.sin(2 * np.pi * dt.hour / 24)
        hour_cos = np.cos(2 * np.pi * dt.hour / 24)

        # Day of week encoding (cyclic)
        dow_sin = np.sin(2 * np.pi * dt.weekday() / 7)
        dow_cos = np.cos(2 * np.pi * dt.weekday() / 7)

        features.extend([hour_sin, hour_cos, dow_sin, dow_cos])

        # Activity history encoding
        if "activity_history" in temporal_data:
            history = temporal_data["activity_history"]
            # Simple encoding of recent activities
            activity_features = self._encode_activity_history(history)
            features.extend(activity_features)

        # Pad to expected size
        features_array = np.array(features)
        if len(features_array) < 64:
            features_array = np.pad(features_array, (0, 64 - len(features_array)))

        return torch.tensor(features_array, dtype=torch.float32)

    def _encode_activity_history(self, history: List[Dict[str, Any]]) -> List[float]:
        """Encode activity history"""
        # Simple encoding - count activity types
        activity_counts = {}
        for activity in history[-10:]:  # Last 10 activities
            activity_type = activity.get("type", "unknown")
            activity_counts[activity_type] = activity_counts.get(activity_type, 0) + 1

        # Convert to fixed-size feature vector
        common_activities = ["work", "break", "meeting", "communication", "learning"]
        features = [activity_counts.get(act, 0) / 10 for act in common_activities]

        return features


class EnvironmentalModalityProcessor:
    """Process environmental context"""

    async def process(self, environmental_data: Dict[str, Any]) -> torch.Tensor:
        """Process environmental input"""

        features = []

        # System resource features
        cpu_usage = environmental_data.get("cpu_usage", 50) / 100
        memory_usage = environmental_data.get("memory_usage", 50) / 100
        disk_usage = environmental_data.get("disk_usage", 50) / 100

        features.extend([cpu_usage, memory_usage, disk_usage])

        # Network features
        network_latency = (
            environmental_data.get("network_latency", 50) / 1000
        )  # ms to seconds
        bandwidth = environmental_data.get("bandwidth", 100) / 1000  # Mbps to Gbps

        features.extend([network_latency, bandwidth])

        # Application context
        active_apps = environmental_data.get("active_applications", [])
        app_features = self._encode_applications(active_apps)
        features.extend(app_features)

        # Pad to expected size
        features_array = np.array(features)
        if len(features_array) < 32:
            features_array = np.pad(features_array, (0, 32 - len(features_array)))

        return torch.tensor(features_array, dtype=torch.float32)

    def _encode_applications(self, apps: List[str]) -> List[float]:
        """Encode active applications"""
        common_apps = ["browser", "ide", "terminal", "communication", "media"]
        features = []

        for app_category in common_apps:
            # Check if any app matches category
            has_app = any(app_category in app.lower() for app in apps)
            features.append(1.0 if has_app else 0.0)

        return features


class TemporalContextIntegrator:
    """Integrate temporal context across time"""

    def __init__(self, hidden_dim: int = 512):
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )

    async def integrate(
        self, current_features: torch.Tensor, context_memory: deque
    ) -> torch.Tensor:
        """Integrate current features with temporal context"""

        if not context_memory:
            return current_features

        # Extract recent features
        recent_features = []
        for ctx in list(context_memory)[-10:]:  # Last 10 timesteps
            if "output" in ctx and "representation" in ctx["output"]:
                recent_features.append(ctx["output"]["representation"])

        if not recent_features:
            return current_features

        # Stack features
        temporal_sequence = torch.stack(recent_features + [current_features])
        temporal_sequence = temporal_sequence.unsqueeze(0)  # Add batch dimension

        # LSTM processing
        lstm_out, _ = self.lstm(temporal_sequence)

        # Self-attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Return last timestep
        return attended[0, -1, :]


class TimeEncoder:
    """Encode time information"""

    def encode_cyclic(self, value: float, max_value: float) -> Tuple[float, float]:
        """Encode value as sine/cosine for cyclic features"""
        normalized = value / max_value
        return np.sin(2 * np.pi * normalized), np.cos(2 * np.pi * normalized)


class PerformanceMonitor:
    """Monitor system performance"""

    def __init__(self):
        self.processing_times = deque(maxlen=1000)
        self.modality_errors = defaultdict(int)

    def record_processing_time(self, time_seconds: float):
        """Record processing time"""
        self.processing_times.append(time_seconds)

    def record_modality_error(self, modality: str):
        """Record modality processing error"""
        self.modality_errors[modality] += 1

    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.processing_times:
            return {}

        times = list(self.processing_times)
        return {
            "avg_processing_time": np.mean(times),
            "p95_processing_time": np.percentile(times, 95),
            "p99_processing_time": np.percentile(times, 99),
            "total_errors": sum(self.modality_errors.values()),
        }


class DistributedModalityProcessor:
    """Process modalities in distributed fashion"""

    def __init__(self):
        self.ray_initialized = False
        try:
            import ray

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            self.ray_initialized = True
        except:
            logger.warning("Ray not available for distributed processing")

    async def process_distributed(
        self, modalities: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Process modalities in parallel using Ray"""
        if not self.ray_initialized:
            # Fallback to sequential processing
            return modalities

        # Distributed processing implementation
        # Would use Ray remote functions
        return modalities


# Integration with JARVIS
async def create_unified_perception():
    """Create and initialize unified perception system"""
    perception = UnifiedPerception(
        fusion_dim=1024,
        context_window=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    logger.info("🧠 Unified Perception System initialized")
    return perception
