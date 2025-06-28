#!/usr/bin/env python3
"""
JARVIS Personal Emotional Intelligence System
Simplified for single-user personal assistant use
"""

import asyncio
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import torch
import torch.nn as nn
from collections import deque
import time
import json
from datetime import datetime
from transformers import pipeline
import structlog
from pathlib import Path

from .monitoring import monitor_performance, monitoring_service
from .config_manager import config_manager

logger = structlog.get_logger()


class EmotionType(Enum):
    """Core emotions JARVIS can detect"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    STRESSED = "stressed"
    EXCITED = "excited"
    FRUSTRATED = "frustrated"
    CONTENT = "content"
    ANXIOUS = "anxious"
    FOCUSED = "focused"
    TIRED = "tired"


@dataclass
class EmotionalState:
    """Your current emotional state"""
    primary_emotion: EmotionType
    intensity: float  # 0.0 to 1.0
    confidence: float
    trigger: Optional[str] = None  # What caused this emotion
    duration_minutes: float = 0.0
    suggestions: List[str] = None
    
    def __str__(self):
        return f"{self.primary_emotion.value.title()} (intensity: {self.intensity:.1%})"


@dataclass
class UserContext:
    """Context about your current situation"""
    time_of_day: float
    work_duration_hours: float = 0.0
    last_break_minutes_ago: float = 0.0
    calendar_next: Optional[str] = None  # Next event
    recent_activities: List[str] = None
    location: str = "home"  # home, office, traveling
    
    @property
    def is_late_night(self) -> bool:
        return self.time_of_day >= 22 or self.time_of_day <= 5
    
    @property
    def needs_break(self) -> bool:
        return self.work_duration_hours > 2 and self.last_break_minutes_ago > 90


class SimpleEmotionDetector:
    """Simplified emotion detection using pre-trained models"""
    
    def __init__(self):
        try:
            # Use HuggingFace sentiment pipeline for text
            self.text_classifier = pipeline(
                "sentiment-analysis",
                model="j-hartmann/emotion-english-distilroberta-base"
            )
            self.model_available = True
        except Exception as e:
            logger.warning(f"Could not load emotion model: {e}")
            self.model_available = False
        
        # Simple patterns for additional context
        self.stress_indicators = [
            "can't", "impossible", "frustrated", "stuck", "deadline",
            "overwhelmed", "too much", "exhausted", "stressed"
        ]
        
        self.happiness_indicators = [
            "excited", "great", "awesome", "perfect", "wonderful",
            "amazing", "fantastic", "accomplished", "success"
        ]
    
    def analyze_text(self, text: str) -> Dict[EmotionType, float]:
        """Analyze emotion from text"""
        emotions = {}
        
        if self.model_available:
            try:
                # Get emotions from model
                results = self.text_classifier(text)
                
                # Convert to our emotion types
                emotion_map = {
                    'anger': EmotionType.ANGRY,
                    'disgust': EmotionType.FRUSTRATED,
                    'fear': EmotionType.ANXIOUS,
                    'joy': EmotionType.HAPPY,
                    'neutral': EmotionType.NEUTRAL,
                    'sadness': EmotionType.SAD,
                    'surprise': EmotionType.EXCITED
                }
                
                for result in results[:1]:  # Take top result
                    emotion_type = emotion_map.get(result['label'].lower(), EmotionType.NEUTRAL)
                    emotions[emotion_type] = result['score']
            except Exception as e:
                logger.error(f"Model inference failed: {e}")
                emotions[EmotionType.NEUTRAL] = 0.5
        else:
            # Fallback to keyword analysis
            emotions[EmotionType.NEUTRAL] = 0.5
        
        # Add stress detection
        text_lower = text.lower()
        stress_count = sum(1 for word in self.stress_indicators if word in text_lower)
        if stress_count > 0:
            emotions[EmotionType.STRESSED] = min(0.3 + stress_count * 0.2, 0.9)
        
        # Add happiness detection
        happy_count = sum(1 for word in self.happiness_indicators if word in text_lower)
        if happy_count > 0:
            emotions[EmotionType.HAPPY] = min(0.3 + happy_count * 0.2, 0.9)
        
        return emotions


class PersonalEmotionalIntelligence:
    """JARVIS's emotional intelligence system for personal use"""
    
    def __init__(self):
        self.detector = SimpleEmotionDetector()
        self.emotion_history = deque(maxlen=50)  # Last 50 emotional states
        self.daily_patterns = {}  # Track your patterns
        
        # Personal thresholds (can be customized)
        self.stress_threshold = config_manager.get("emotional_intelligence.stress_threshold", 0.6)
        self.intervention_threshold = config_manager.get("emotional_intelligence.intervention_threshold", 0.7)
        
        # Your personal preferences
        self.preferences = {
            "break_reminder_style": config_manager.get("emotional_intelligence.reminder_style", "gentle"),
            "intervention_types": config_manager.get("emotional_intelligence.intervention_types", 
                                                   ["breathing", "music", "walk", "gaming"]),
            "work_session_length": config_manager.get("emotional_intelligence.work_session_minutes", 90),
        }
        
        # Load emotion history if exists
        self._load_history()
        
        logger.info("JARVIS Emotional Intelligence initialized")
    
    def _load_history(self):
        """Load emotion history from storage"""
        history_file = Path(config_manager.get("paths.storage", "./storage")) / "emotion_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    # Convert back to EmotionalState objects
                    for entry in data[-50:]:  # Last 50 entries
                        state = EmotionalState(
                            primary_emotion=EmotionType(entry['primary_emotion']),
                            intensity=entry['intensity'],
                            confidence=entry['confidence'],
                            trigger=entry.get('trigger'),
                            duration_minutes=entry.get('duration_minutes', 0),
                            suggestions=entry.get('suggestions', [])
                        )
                        self.emotion_history.append((entry['timestamp'], state))
            except Exception as e:
                logger.error(f"Failed to load emotion history: {e}")
    
    def _save_history(self):
        """Save emotion history to storage"""
        history_file = Path(config_manager.get("paths.storage", "./storage")) / "emotion_history.json"
        history_file.parent.mkdir(exist_ok=True)
        
        try:
            data = []
            for timestamp, state in self.emotion_history:
                data.append({
                    'timestamp': timestamp,
                    'primary_emotion': state.primary_emotion.value,
                    'intensity': state.intensity,
                    'confidence': state.confidence,
                    'trigger': state.trigger,
                    'duration_minutes': state.duration_minutes,
                    'suggestions': state.suggestions
                })
            
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save emotion history: {e}")
    
    @monitor_performance("emotional_intelligence")
    async def analyze_emotion(
        self, 
        text: Optional[str] = None,
        context: Optional[UserContext] = None,
        voice_tone: Optional[Dict[str, float]] = None
    ) -> EmotionalState:
        """Analyze your current emotional state"""
        
        emotions = {}
        
        # Analyze text if provided
        if text:
            emotions = self.detector.analyze_text(text)
        
        # Add voice analysis if available
        if voice_tone:
            if voice_tone.get('pitch_variance', 0) > 50:
                emotions[EmotionType.STRESSED] = emotions.get(EmotionType.STRESSED, 0) + 0.3
            if voice_tone.get('energy', 0) < 0.3:
                emotions[EmotionType.TIRED] = emotions.get(EmotionType.TIRED, 0) + 0.4
        
        # Context-based adjustments
        if context:
            # Late night work detection
            if context.is_late_night and context.work_duration_hours > 0:
                emotions[EmotionType.TIRED] = emotions.get(EmotionType.TIRED, 0) + 0.4
            
            # Long work session detection
            if context.needs_break:
                emotions[EmotionType.STRESSED] = emotions.get(EmotionType.STRESSED, 0) + 0.2
                emotions[EmotionType.TIRED] = emotions.get(EmotionType.TIRED, 0) + 0.3
        
        # Determine primary emotion
        if not emotions:
            emotions = {EmotionType.NEUTRAL: 0.8}
        
        primary_emotion = max(emotions, key=emotions.get)
        intensity = emotions[primary_emotion]
        
        # Calculate confidence (simple version)
        confidence = 0.9 if text else 0.6
        if voice_tone:
            confidence = min(confidence + 0.2, 0.95)
        
        # Create emotional state
        state = EmotionalState(
            primary_emotion=primary_emotion,
            intensity=intensity,
            confidence=confidence,
            trigger=self._identify_trigger(text, context),
            duration_minutes=self._calculate_duration(primary_emotion),
            suggestions=self._generate_suggestions(primary_emotion, intensity, context)
        )
        
        # Update history
        self.emotion_history.append((time.time(), state))
        self._save_history()
        
        # Update metrics
        monitoring_service.metrics_collector.record_event({
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "emotion_detected",
            "component": "emotional_intelligence",
            "metrics": {
                "emotion": primary_emotion.value,
                "intensity": intensity,
                "confidence": confidence
            }
        })
        
        # Log significant emotions
        if intensity > self.intervention_threshold:
            logger.info(f"High intensity {primary_emotion.value}: {intensity:.1%}")
        
        return state
    
    def _identify_trigger(self, text: Optional[str], context: Optional[UserContext]) -> Optional[str]:
        """Identify what might have triggered the emotion"""
        if text:
            # Simple keyword matching
            text_lower = text.lower()
            if "deadline" in text_lower:
                return "upcoming deadline"
            elif "meeting" in text_lower:
                return "meeting-related"
            elif "bug" in text_lower or "error" in text_lower:
                return "technical issue"
            elif "success" in text_lower or "fixed" in text_lower:
                return "achievement"
            elif "family" in text_lower or "brother" in text_lower:
                return "family-related"
        
        if context and context.work_duration_hours > 3:
            return "extended work session"
        
        return None
    
    def _calculate_duration(self, emotion: EmotionType) -> float:
        """Calculate how long you've been in this emotional state"""
        if not self.emotion_history:
            return 0.0
        
        duration = 0.0
        current_time = time.time()
        
        for timestamp, state in reversed(self.emotion_history):
            if state.primary_emotion == emotion:
                duration = (current_time - timestamp) / 60  # minutes
            else:
                break
        
        return duration
    
    def _generate_suggestions(
        self, 
        emotion: EmotionType, 
        intensity: float,
        context: Optional[UserContext]
    ) -> List[str]:
        """Generate personalized suggestions based on your state"""
        suggestions = []
        
        if emotion == EmotionType.STRESSED and intensity > self.stress_threshold:
            suggestions.extend([
                "Take 5 deep breaths (4-7-8 technique)",
                "Quick 5-minute walk outside",
                "Play your 'Calm' Spotify playlist",
                "Try the 2-minute meditation"
            ])
            
            if context and context.needs_break:
                suggestions.insert(0, "You've been working for a while - time for a proper break!")
        
        elif emotion == EmotionType.TIRED:
            suggestions.extend([
                "20-minute power nap",
                "Get some fresh air",
                "Hydrate - grab some water",
                "Light stretching routine"
            ])
            
            if context and context.is_late_night:
                suggestions.insert(0, "Consider wrapping up for the night")
        
        elif emotion == EmotionType.FRUSTRATED:
            suggestions.extend([
                "Step away from the problem for 10 minutes",
                "Try explaining the issue out loud (rubber duck debugging)",
                "Switch to a different task temporarily",
                "Quick gaming session to reset?"
            ])
        
        elif emotion == EmotionType.HAPPY or emotion == EmotionType.EXCITED:
            suggestions.extend([
                "Great job! Ride this momentum",
                "Perfect time to tackle that challenging task",
                "Share your success with someone"
            ])
        
        elif emotion == EmotionType.ANXIOUS:
            suggestions.extend([
                "Ground yourself: 5 things you see, 4 you hear, 3 you touch",
                "Write down what's worrying you",
                "Progressive muscle relaxation",
                "Call a friend or family member"
            ])
        
        return suggestions[:3]  # Top 3 suggestions
    
    def get_emotional_summary(self) -> Dict[str, Any]:
        """Get a summary of your recent emotional patterns"""
        if not self.emotion_history:
            return {"message": "No emotional data yet"}
        
        # Count emotions in last 2 hours
        two_hours_ago = time.time() - 7200
        recent_emotions = {}
        
        for timestamp, state in self.emotion_history:
            if timestamp > two_hours_ago:
                emotion = state.primary_emotion
                recent_emotions[emotion] = recent_emotions.get(emotion, 0) + 1
        
        # Find dominant emotion
        if recent_emotions:
            dominant = max(recent_emotions, key=recent_emotions.get)
            total = sum(recent_emotions.values())
            
            return {
                "dominant_emotion": dominant.value,
                "percentage": recent_emotions[dominant] / total,
                "variety": len(recent_emotions),
                "states_tracked": total,
                "recommendation": self._get_overall_recommendation(dominant, recent_emotions)
            }
        
        return {"message": "Not enough recent data"}
    
    def _get_overall_recommendation(self, dominant: EmotionType, emotion_counts: Dict) -> str:
        """Get overall recommendation based on patterns"""
        
        stress_emotions = [EmotionType.STRESSED, EmotionType.FRUSTRATED, EmotionType.ANXIOUS]
        stress_count = sum(emotion_counts.get(e, 0) for e in stress_emotions)
        total_count = sum(emotion_counts.values())
        
        if stress_count / total_count > 0.6:
            return "High stress detected. Consider taking a longer break or switching activities."
        elif dominant == EmotionType.TIRED:
            return "Fatigue is dominant. Prioritize rest and consider postponing complex tasks."
        elif dominant in [EmotionType.HAPPY, EmotionType.EXCITED, EmotionType.FOCUSED]:
            return "You're in a great state! Perfect time for challenging or creative work."
        else:
            return "Emotional state is balanced. Maintain regular breaks and stay hydrated."
    
    async def check_intervention_needed(self, state: EmotionalState, context: UserContext) -> Optional[str]:
        """Check if JARVIS should intervene with a suggestion"""
        
        # High stress/frustration
        if state.intensity > self.intervention_threshold:
            if state.primary_emotion in [EmotionType.STRESSED, EmotionType.FRUSTRATED]:
                return f"I notice you're quite {state.primary_emotion.value}. Would you like me to help with a quick break activity?"
        
        # Long work session
        if context.needs_break and state.primary_emotion != EmotionType.FOCUSED:
            return "You've been working for a while. How about a quick break? I can set a timer."
        
        # Late night fatigue
        if context.is_late_night and state.primary_emotion == EmotionType.TIRED:
            return "It's getting late and you seem tired. Should we plan to wrap up soon?"
        
        # Extended negative emotion
        if state.duration_minutes > 30 and state.primary_emotion in [EmotionType.SAD, EmotionType.ANXIOUS]:
            return "You've been feeling this way for a while. Want to talk about it or try something different?"
        
        return None
    
    def get_family_aware_response(self, emotion: EmotionType) -> str:
        """Get a family-aware response based on emotion"""
        family_responses = {
            EmotionType.HAPPY: "That's wonderful! Your family would be proud of you.",
            EmotionType.STRESSED: "Remember, your family is here for you. Maybe take a break and spend some time with them?",
            EmotionType.SAD: "I'm here for you. Would you like to talk to your family? Sometimes that helps.",
            EmotionType.TIRED: "You've been working hard. Your family would want you to rest.",
            EmotionType.EXCITED: "Your excitement is contagious! This would be great to share with your family."
        }
        
        return family_responses.get(emotion, "Remember, I'm here to help you, just like family.")


# Global emotional intelligence instance
emotional_intelligence = PersonalEmotionalIntelligence()


# Convenience functions for JARVIS integration
async def analyze_user_emotion(text: str = None, context: Dict[str, Any] = None) -> EmotionalState:
    """Analyze user's emotional state"""
    
    # Convert dict context to UserContext if needed
    if context and not isinstance(context, UserContext):
        user_context = UserContext(
            time_of_day=context.get('time_of_day', datetime.now().hour),
            work_duration_hours=context.get('work_duration_hours', 0),
            last_break_minutes_ago=context.get('last_break_minutes_ago', 0),
            calendar_next=context.get('calendar_next'),
            recent_activities=context.get('recent_activities', []),
            location=context.get('location', 'home')
        )
    else:
        user_context = context
    
    return await emotional_intelligence.analyze_emotion(text, user_context)


def get_emotional_support(emotion: EmotionType) -> str:
    """Get emotional support message"""
    return emotional_intelligence.get_family_aware_response(emotion)