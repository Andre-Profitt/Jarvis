"""
JARVIS Phase 5: Emotional Continuity System
Maintains emotional understanding and empathy across interactions
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
import numpy as np

class EmotionType(Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"

@dataclass
class EmotionalState:
    """Represents current emotional state"""
    primary_emotion: EmotionType
    secondary_emotion: Optional[EmotionType]
    intensity: float  # 0.0 to 1.0
    valence: float   # -1.0 (negative) to 1.0 (positive)
    arousal: float   # 0.0 (calm) to 1.0 (excited)
    confidence: float
    timestamp: datetime

class EmotionalContinuity:
    """Maintains emotional understanding across conversations"""
    
    def __init__(self):
        # Emotional history
        self.emotional_history = []
        self.emotion_transitions = defaultdict(int)
        
        # Current state
        self.current_state = EmotionalState(
            primary_emotion=EmotionType.NEUTRAL,
            secondary_emotion=None,
            intensity=0.0,
            valence=0.0,
            arousal=0.5,
            confidence=1.0,
            timestamp=datetime.now()
        )
        
        # User emotion model
        self.user_baseline = {
            "typical_valence": 0.0,
            "typical_arousal": 0.5,
            "emotional_range": 0.5,
            "recovery_rate": 0.8
        }
        
        # Emotion patterns
        self.emotion_patterns = defaultdict(list)
        self.trigger_words = self._initialize_trigger_words()
        
        # Empathy model
        self.empathy_responses = self._initialize_empathy_responses()
    
    async def update_emotional_state(self, 
                                   inputs: Dict[str, Any],
                                   context: Dict[str, Any]) -> EmotionalState:
        """Update emotional state based on multimodal inputs"""
        
        # Extract emotional indicators
        indicators = await self._extract_emotional_indicators(inputs)
        
        # Combine with historical context
        historical_weight = 0.3
        current_weight = 0.7
        
        # Calculate new emotional state
        new_emotion = await self._determine_emotion(indicators)
        new_intensity = indicators.get("intensity", 0.5)
        new_valence = indicators.get("valence", 0.0)
        new_arousal = indicators.get("arousal", 0.5)
        
        # Apply continuity - smooth transitions
        if self.current_state.timestamp > datetime.now() - timedelta(minutes=5):
            # Recent interaction - apply smoothing
            new_intensity = (self.current_state.intensity * historical_weight + 
                           new_intensity * current_weight)
            new_valence = (self.current_state.valence * historical_weight + 
                         new_valence * current_weight)
            new_arousal = (self.current_state.arousal * historical_weight + 
                         new_arousal * current_weight)
        
        # Create new state
        new_state = EmotionalState(
            primary_emotion=new_emotion["primary"],
            secondary_emotion=new_emotion.get("secondary"),
            intensity=new_intensity,
            valence=new_valence,
            arousal=new_arousal,
            confidence=indicators.get("confidence", 0.8),
            timestamp=datetime.now()
        )
        
        # Update tracking
        await self._update_emotional_tracking(new_state)
        
        self.current_state = new_state
        return new_state
    
    async def get_empathetic_response(self, 
                                     user_state: EmotionalState,
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate empathetic response based on user's emotional state"""
        
        response_strategy = {
            "tone": self._determine_response_tone(user_state),
            "energy_level": self._determine_energy_level(user_state),
            "approach": self._determine_approach(user_state),
            "specific_actions": []
        }
        
        # Add specific actions based on emotion
        if user_state.primary_emotion == EmotionType.SADNESS:
            response_strategy["specific_actions"].extend([
                "acknowledge_feelings",
                "offer_support",
                "suggest_gentle_activities"
            ])
        elif user_state.primary_emotion == EmotionType.ANGER:
            response_strategy["specific_actions"].extend([
                "remain_calm",
                "validate_frustration",
                "help_problem_solve"
            ])
        elif user_state.primary_emotion == EmotionType.JOY:
            response_strategy["specific_actions"].extend([
                "share_enthusiasm",
                "celebrate_success",
                "maintain_energy"
            ])
        elif user_state.primary_emotion == EmotionType.FEAR:
            response_strategy["specific_actions"].extend([
                "provide_reassurance",
                "offer_information",
                "suggest_calming_techniques"
            ])
        
        # Get response templates
        templates = self.empathy_responses.get(user_state.primary_emotion, [])
        
        return {
            "strategy": response_strategy,
            "suggested_phrases": templates,
            "emotional_mirroring": self._calculate_mirroring(user_state),
            "intervention_level": self._determine_intervention_level(user_state)
        }
    
    async def predict_emotional_trajectory(self, 
                                         current_state: EmotionalState,
                                         time_horizon: int = 10) -> List[EmotionalState]:
        """Predict likely emotional trajectory over next N minutes"""
        
        predictions = []
        state = current_state
        
        for minute in range(time_horizon):
            # Calculate natural decay/recovery
            recovery_factor = self.user_baseline["recovery_rate"]
            
            # Predict next state
            next_valence = state.valence * recovery_factor + \
                          self.user_baseline["typical_valence"] * (1 - recovery_factor)
            
            next_arousal = state.arousal * recovery_factor + \
                          self.user_baseline["typical_arousal"] * (1 - recovery_factor)
            
            next_intensity = state.intensity * (0.9 ** minute)  # Natural decay
            
            # Determine likely emotion
            if next_intensity < 0.3:
                next_emotion = EmotionType.NEUTRAL
            else:
                next_emotion = state.primary_emotion
            
            predicted_state = EmotionalState(
                primary_emotion=next_emotion,
                secondary_emotion=None,
                intensity=next_intensity,
                valence=next_valence,
                arousal=next_arousal,
                confidence=0.8 - (minute * 0.05),  # Decreasing confidence
                timestamp=datetime.now() + timedelta(minutes=minute)
            )
            
            predictions.append(predicted_state)
            state = predicted_state
        
        return predictions
    
    async def detect_emotional_patterns(self) -> List[Dict[str, Any]]:
        """Detect recurring emotional patterns"""
        
        patterns = []
        
        # Time-based patterns
        hourly_emotions = defaultdict(list)
        daily_emotions = defaultdict(list)
        
        for state in self.emotional_history:
            hour = state.timestamp.hour
            day = state.timestamp.weekday()
            
            hourly_emotions[hour].append(state.primary_emotion)
            daily_emotions[day].append(state.primary_emotion)
        
        # Find dominant patterns
        for hour, emotions in hourly_emotions.items():
            if len(emotions) > 5:
                dominant = max(set(emotions), key=emotions.count)
                if emotions.count(dominant) / len(emotions) > 0.6:
                    patterns.append({
                        "type": "hourly",
                        "time": hour,
                        "emotion": dominant,
                        "confidence": emotions.count(dominant) / len(emotions)
                    })
        
        # Trigger patterns
        for trigger, emotions in self.emotion_patterns.items():
            if len(emotions) > 3:
                dominant = max(set(emotions), key=emotions.count)
                patterns.append({
                    "type": "trigger",
                    "trigger": trigger,
                    "emotion": dominant,
                    "frequency": len(emotions)
                })
        
        return patterns
    
    async def _extract_emotional_indicators(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract emotional indicators from multimodal inputs"""
        
        indicators = {
            "intensity": 0.5,
            "valence": 0.0,
            "arousal": 0.5,
            "confidence": 0.8
        }
        
        # Voice analysis
        if "voice" in inputs:
            voice_features = inputs["voice"].get("features", {})
            
            # Pitch variance indicates emotional intensity
            indicators["intensity"] = min(1.0, voice_features.get("pitch_variance", 0.5))
            
            # Speaking rate indicates arousal
            rate = voice_features.get("speaking_rate", 1.0)
            indicators["arousal"] = min(1.0, abs(rate - 1.0) + 0.5)
            
            # Volume indicates confidence/intensity
            volume = voice_features.get("volume", 0.5)
            indicators["intensity"] = (indicators["intensity"] + volume) / 2
        
        # Text sentiment
        if "text" in inputs:
            text_analysis = await self._analyze_text_emotion(inputs["text"])
            indicators["valence"] = text_analysis["valence"]
            
            # Check for emotion words
            for word, emotion_data in self.trigger_words.items():
                if word in inputs["text"].lower():
                    indicators["valence"] = emotion_data["valence"]
                    indicators["arousal"] = emotion_data["arousal"]
        
        # Biometric data
        if "biometric" in inputs:
            bio = inputs["biometric"]
            
            # Heart rate indicates arousal
            hr = bio.get("heart_rate", 70)
            if hr > 100:
                indicators["arousal"] = min(1.0, 0.7 + (hr - 100) / 100)
            elif hr < 60:
                indicators["arousal"] = max(0.0, 0.3 - (60 - hr) / 60)
            
            # Skin conductance indicates emotional intensity
            sc = bio.get("skin_conductance", 0.5)
            indicators["intensity"] = (indicators["intensity"] + sc) / 2
        
        return indicators
    
    async def _determine_emotion(self, indicators: Dict[str, Any]) -> Dict[str, EmotionType]:
        """Determine emotion from indicators"""
        
        valence = indicators["valence"]
        arousal = indicators["arousal"]
        intensity = indicators["intensity"]
        
        # Emotion mapping based on circumplex model
        if valence > 0.3 and arousal > 0.6:
            primary = EmotionType.JOY
            secondary = EmotionType.SURPRISE if arousal > 0.8 else None
        elif valence > 0.3 and arousal < 0.4:
            primary = EmotionType.TRUST
            secondary = None
        elif valence < -0.3 and arousal > 0.6:
            primary = EmotionType.ANGER
            secondary = EmotionType.FEAR if intensity > 0.7 else None
        elif valence < -0.3 and arousal < 0.4:
            primary = EmotionType.SADNESS
            secondary = None
        elif arousal > 0.8:
            primary = EmotionType.SURPRISE
            secondary = EmotionType.FEAR if valence < 0 else EmotionType.JOY
        else:
            primary = EmotionType.NEUTRAL
            secondary = None
        
        return {
            "primary": primary,
            "secondary": secondary
        }
    
    async def _update_emotional_tracking(self, state: EmotionalState):
        """Update emotional history and patterns"""
        
        # Add to history
        self.emotional_history.append(state)
        
        # Limit history size
        if len(self.emotional_history) > 1000:
            self.emotional_history = self.emotional_history[-1000:]
        
        # Track transitions
        if len(self.emotional_history) > 1:
            prev_emotion = self.emotional_history[-2].primary_emotion
            curr_emotion = state.primary_emotion
            self.emotion_transitions[(prev_emotion, curr_emotion)] += 1
        
        # Update baseline
        await self._update_baseline(state)
    
    async def _update_baseline(self, state: EmotionalState):
        """Update user's emotional baseline"""
        
        # Exponential moving average
        alpha = 0.05  # Learning rate
        
        self.user_baseline["typical_valence"] = (
            (1 - alpha) * self.user_baseline["typical_valence"] + 
            alpha * state.valence
        )
        
        self.user_baseline["typical_arousal"] = (
            (1 - alpha) * self.user_baseline["typical_arousal"] + 
            alpha * state.arousal
        )
        
        # Update emotional range
        recent_states = self.emotional_history[-20:]
        if recent_states:
            valences = [s.valence for s in recent_states]
            self.user_baseline["emotional_range"] = np.std(valences)
    
    def _determine_response_tone(self, user_state: EmotionalState) -> str:
        """Determine appropriate response tone"""
        
        if user_state.intensity < 0.3:
            return "neutral"
        
        tone_map = {
            EmotionType.JOY: "enthusiastic",
            EmotionType.SADNESS: "gentle",
            EmotionType.ANGER: "calm",
            EmotionType.FEAR: "reassuring",
            EmotionType.SURPRISE: "explanatory",
            EmotionType.TRUST: "warm",
            EmotionType.NEUTRAL: "conversational"
        }
        
        return tone_map.get(user_state.primary_emotion, "neutral")
    
    def _determine_energy_level(self, user_state: EmotionalState) -> str:
        """Determine appropriate energy level for response"""
        
        if user_state.arousal > 0.7:
            return "high" if user_state.valence > 0 else "controlled"
        elif user_state.arousal < 0.3:
            return "low"
        else:
            return "moderate"
    
    def _determine_approach(self, user_state: EmotionalState) -> str:
        """Determine interaction approach"""
        
        if user_state.valence < -0.5:
            return "supportive"
        elif user_state.valence > 0.5:
            return "collaborative"
        elif user_state.arousal > 0.7:
            return "focusing"
        else:
            return "exploratory"
    
    def _calculate_mirroring(self, user_state: EmotionalState) -> Dict[str, float]:
        """Calculate emotional mirroring parameters"""
        
        # Mirror with slight reduction to avoid amplification
        return {
            "valence": user_state.valence * 0.7,
            "arousal": user_state.arousal * 0.8,
            "intensity": min(0.7, user_state.intensity * 0.9)
        }
    
    def _determine_intervention_level(self, user_state: EmotionalState) -> str:
        """Determine if intervention is needed"""
        
        if user_state.intensity > 0.8 and user_state.valence < -0.5:
            return "high"  # User in distress
        elif user_state.intensity > 0.6 and user_state.primary_emotion == EmotionType.ANGER:
            return "moderate"  # User frustrated
        elif user_state.arousal < 0.2 and user_state.valence < 0:
            return "moderate"  # User disengaged
        else:
            return "low"  # Normal interaction
    
    async def _analyze_text_emotion(self, text: str) -> Dict[str, float]:
        """Analyze emotional content of text"""
        
        # Simple sentiment analysis
        positive_words = ["good", "great", "happy", "love", "excellent", "wonderful"]
        negative_words = ["bad", "hate", "angry", "sad", "terrible", "awful"]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        if total > 0:
            valence = (positive_count - negative_count) / total
        else:
            valence = 0.0
        
        return {"valence": valence}
    
    def _initialize_trigger_words(self) -> Dict[str, Dict[str, float]]:
        """Initialize emotion trigger words"""
        
        return {
            # Positive triggers
            "amazing": {"valence": 0.9, "arousal": 0.8},
            "wonderful": {"valence": 0.8, "arousal": 0.6},
            "excited": {"valence": 0.7, "arousal": 0.9},
            "happy": {"valence": 0.8, "arousal": 0.6},
            "love": {"valence": 0.9, "arousal": 0.7},
            
            # Negative triggers
            "frustrated": {"valence": -0.6, "arousal": 0.8},
            "angry": {"valence": -0.8, "arousal": 0.9},
            "sad": {"valence": -0.7, "arousal": 0.3},
            "worried": {"valence": -0.5, "arousal": 0.7},
            "scared": {"valence": -0.8, "arousal": 0.8},
            
            # Neutral triggers
            "confused": {"valence": -0.2, "arousal": 0.6},
            "tired": {"valence": -0.3, "arousal": 0.2},
            "okay": {"valence": 0.1, "arousal": 0.5}
        }
    
    def _initialize_empathy_responses(self) -> Dict[EmotionType, List[str]]:
        """Initialize empathetic response templates"""
        
        return {
            EmotionType.JOY: [
                "That's wonderful to hear!",
                "I'm so glad you're feeling good!",
                "Your enthusiasm is contagious!"
            ],
            EmotionType.SADNESS: [
                "I understand this is difficult.",
                "I'm here if you need support.",
                "Take all the time you need."
            ],
            EmotionType.ANGER: [
                "I can see why that would be frustrating.",
                "Let's work through this together.",
                "Your feelings are completely valid."
            ],
            EmotionType.FEAR: [
                "It's okay to feel uncertain.",
                "We can take this step by step.",
                "You're not alone in this."
            ],
            EmotionType.SURPRISE: [
                "That is unexpected!",
                "Let me help you process this.",
                "Quite a development!"
            ]
        }
