"""
JARVIS Phase 6: Emotional Intelligence Engine
===========================================
Advanced emotional understanding and response adaptation
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field
import json
import math

@dataclass
class EmotionalState:
    """Comprehensive emotional state representation"""
    valence: float = 0.0  # -1 (negative) to 1 (positive)
    arousal: float = 0.0  # -1 (calm) to 1 (excited)
    dominance: float = 0.0  # -1 (submissive) to 1 (dominant)
    
    @property
    def quadrant(self) -> str:
        """Get emotional quadrant based on valence and arousal"""
        if self.valence > 0 and self.arousal > 0:
            return "excited_happy"  # Joy, excitement
        elif self.valence > 0 and self.arousal <= 0:
            return "calm_happy"     # Content, peaceful
        elif self.valence <= 0 and self.arousal > 0:
            return "excited_unhappy"  # Angry, frustrated
        else:
            return "calm_unhappy"   # Sad, depressed
            
    def distance_to(self, other: 'EmotionalState') -> float:
        """Calculate emotional distance to another state"""
        return math.sqrt(
            (self.valence - other.valence) ** 2 +
            (self.arousal - other.arousal) ** 2 +
            (self.dominance - other.dominance) ** 2
        )

@dataclass
class EmotionalMemory:
    """Long-term emotional memory and patterns"""
    state: EmotionalState
    timestamp: datetime
    context: Dict
    triggers: List[str] = field(default_factory=list)
    response_effectiveness: float = 0.5

class EmotionalIntelligence:
    """Advanced emotional understanding and adaptation"""
    
    def __init__(self):
        self.current_state = EmotionalState()
        self.target_state = EmotionalState(valence=0.5, arousal=0.0)  # Calm positive
        self.emotional_history = deque(maxlen=100)
        self.pattern_memory = deque(maxlen=500)
        self.emotional_triggers = self._initialize_triggers()
        self.response_strategies = self._initialize_strategies()
        self.empathy_model = self._initialize_empathy_model()
        
    def _initialize_triggers(self) -> Dict:
        """Initialize emotional trigger patterns"""
        return {
            "stress_indicators": {
                "words": ["overwhelmed", "stressed", "anxious", "worried", "panic"],
                "patterns": ["too much", "can't handle", "falling behind", "no time"],
                "biometric": {"heart_rate": 90, "breathing_rate": 20}
            },
            "joy_indicators": {
                "words": ["excited", "happy", "great", "wonderful", "amazing"],
                "patterns": ["can't wait", "looking forward", "best day"],
                "biometric": {"heart_rate": 75, "vocal_pitch": 1.1}
            },
            "anger_indicators": {
                "words": ["angry", "furious", "pissed", "frustrated", "annoyed"],
                "patterns": ["sick of", "had enough", "can't stand"],
                "biometric": {"heart_rate": 95, "vocal_volume": 1.3}
            },
            "sadness_indicators": {
                "words": ["sad", "depressed", "down", "lonely", "hurt"],
                "patterns": ["miss", "wish", "used to", "never be"],
                "biometric": {"heart_rate": 65, "vocal_pitch": 0.9}
            }
        }
        
    def _initialize_strategies(self) -> Dict:
        """Initialize response strategies for different emotional states"""
        return {
            "excited_happy": {
                "mirror": "Match their energy and enthusiasm",
                "actions": ["celebrate", "encourage", "amplify"],
                "tone_adjustment": {"energy": 1.2, "positivity": 1.3},
                "phrases": ["That's fantastic!", "I'm excited for you!", "This is amazing!"]
            },
            "calm_happy": {
                "maintain": "Keep the peaceful, content atmosphere",
                "actions": ["support", "appreciate", "flow"],
                "tone_adjustment": {"energy": 0.8, "positivity": 1.1},
                "phrases": ["That's wonderful", "I'm glad to hear that", "Sounds peaceful"]
            },
            "excited_unhappy": {
                "de-escalate": "Calm and redirect the energy",
                "actions": ["validate", "breathe", "problem-solve"],
                "tone_adjustment": {"energy": 0.6, "calmness": 1.4},
                "phrases": ["I hear your frustration", "Let's work through this", "Take a breath"]
            },
            "calm_unhappy": {
                "uplift": "Gently increase energy and positivity",
                "actions": ["empathize", "comfort", "hope"],
                "tone_adjustment": {"warmth": 1.3, "gentleness": 1.2},
                "phrases": ["I'm here with you", "It's okay to feel this way", "We'll get through this"]
            }
        }
        
    def _initialize_empathy_model(self) -> Dict:
        """Initialize empathy and understanding patterns"""
        return {
            "validation_templates": [
                "Your feelings about {situation} are completely valid",
                "It makes sense that you'd feel {emotion} given {context}",
                "Anyone would feel {emotion} in this situation"
            ],
            "reflection_templates": [
                "What I'm hearing is that you're feeling {emotion} because {reason}",
                "It sounds like {situation} is really affecting you",
                "You're experiencing {emotion} and that's weighing on you"
            ],
            "support_templates": [
                "I'm here to support you through this",
                "Let's work together on {goal}",
                "You don't have to handle this alone"
            ]
        }
        
    async def analyze_emotional_content(self, 
                                      text: str, 
                                      voice_features: Optional[Dict] = None,
                                      biometrics: Optional[Dict] = None,
                                      context: Optional[Dict] = None) -> Dict:
        """Comprehensive emotional analysis"""
        
        # Text-based emotion detection
        text_emotions = self._analyze_text_emotions(text)
        
        # Voice-based emotion detection
        voice_emotions = self._analyze_voice_emotions(voice_features) if voice_features else None
        
        # Biometric-based emotion detection
        bio_emotions = self._analyze_biometric_emotions(biometrics) if biometrics else None
        
        # Combine all signals
        combined_state = self._combine_emotional_signals(text_emotions, voice_emotions, bio_emotions)
        
        # Detect emotional patterns
        patterns = self._detect_emotional_patterns(combined_state, context)
        
        # Calculate emotional trajectory
        trajectory = self._calculate_trajectory(combined_state)
        
        # Generate empathetic understanding
        understanding = self._generate_empathetic_understanding(combined_state, text, context)
        
        # Update internal state
        self._update_emotional_state(combined_state, context)
        
        return {
            "current_state": {
                "valence": combined_state.valence,
                "arousal": combined_state.arousal,
                "dominance": combined_state.dominance,
                "quadrant": combined_state.quadrant
            },
            "trajectory": trajectory,
            "patterns": patterns,
            "understanding": understanding,
            "recommended_response": self._recommend_response_strategy(combined_state, trajectory)
        }
        
    def _analyze_text_emotions(self, text: str) -> EmotionalState:
        """Analyze emotions from text content"""
        state = EmotionalState()
        text_lower = text.lower()
        
        # Check each trigger category
        for emotion, indicators in self.emotional_triggers.items():
            word_score = sum(1 for word in indicators["words"] if word in text_lower)
            pattern_score = sum(1 for pattern in indicators["patterns"] if pattern in text_lower)
            
            total_score = word_score + pattern_score * 2  # Patterns weighted more
            
            # Map to emotional dimensions
            if "stress" in emotion or "anger" in emotion:
                state.valence -= total_score * 0.2
                state.arousal += total_score * 0.3
            elif "joy" in emotion:
                state.valence += total_score * 0.3
                state.arousal += total_score * 0.2
            elif "sad" in emotion:
                state.valence -= total_score * 0.3
                state.arousal -= total_score * 0.2
                
        # Normalize to [-1, 1]
        state.valence = max(-1, min(1, state.valence))
        state.arousal = max(-1, min(1, state.arousal))
        
        # Dominance from language patterns
        if any(word in text_lower for word in ["i need", "must", "have to", "demanding"]):
            state.dominance += 0.3
        elif any(word in text_lower for word in ["please", "could you", "would you mind"]):
            state.dominance -= 0.2
            
        state.dominance = max(-1, min(1, state.dominance))
        
        return state
        
    def _analyze_voice_emotions(self, voice_features: Dict) -> EmotionalState:
        """Analyze emotions from voice features"""
        state = EmotionalState()
        
        # Pitch indicates arousal and valence
        pitch_ratio = voice_features.get("pitch_ratio", 1.0)
        if pitch_ratio > 1.1:
            state.arousal += 0.3
            state.valence += 0.1  # Higher pitch often positive
        elif pitch_ratio < 0.9:
            state.arousal -= 0.2
            state.valence -= 0.1
            
        # Volume indicates arousal and dominance
        volume_ratio = voice_features.get("volume_ratio", 1.0)
        if volume_ratio > 1.2:
            state.arousal += 0.3
            state.dominance += 0.2
        elif volume_ratio < 0.8:
            state.arousal -= 0.2
            state.dominance -= 0.3
            
        # Speaking rate indicates arousal
        rate_ratio = voice_features.get("rate_ratio", 1.0)
        if rate_ratio > 1.2:
            state.arousal += 0.4
        elif rate_ratio < 0.8:
            state.arousal -= 0.3
            
        # Voice quality
        if voice_features.get("tremor", 0) > 0.5:
            state.valence -= 0.3  # Tremor indicates distress
            
        return state
        
    def _analyze_biometric_emotions(self, biometrics: Dict) -> EmotionalState:
        """Analyze emotions from biometric data"""
        state = EmotionalState()
        
        # Heart rate
        hr = biometrics.get("heart_rate", 70)
        if hr > 100:
            state.arousal += 0.5
            # Could be excitement or stress
            if biometrics.get("hrv", 50) < 30:
                state.valence -= 0.3  # Low HRV suggests stress
        elif hr < 60:
            state.arousal -= 0.3
            
        # Skin conductance (if available)
        gsr = biometrics.get("skin_conductance", 1.0)
        if gsr > 1.5:
            state.arousal += 0.4
            
        # Breathing rate
        br = biometrics.get("breathing_rate", 15)
        if br > 20:
            state.arousal += 0.3
            state.valence -= 0.2  # Fast breathing often negative
        elif br < 12:
            state.arousal -= 0.2
            state.valence += 0.1  # Slow breathing often positive
            
        return state
        
    def _combine_emotional_signals(self, 
                                 text: Optional[EmotionalState],
                                 voice: Optional[EmotionalState],
                                 bio: Optional[EmotionalState]) -> EmotionalState:
        """Combine multiple emotional signals with weighted averaging"""
        weights = {"text": 0.4, "voice": 0.35, "bio": 0.25}
        combined = EmotionalState()
        
        total_weight = 0
        
        if text:
            combined.valence += text.valence * weights["text"]
            combined.arousal += text.arousal * weights["text"]
            combined.dominance += text.dominance * weights["text"]
            total_weight += weights["text"]
            
        if voice:
            combined.valence += voice.valence * weights["voice"]
            combined.arousal += voice.arousal * weights["voice"]
            combined.dominance += voice.dominance * weights["voice"]
            total_weight += weights["voice"]
            
        if bio:
            combined.valence += bio.valence * weights["bio"]
            combined.arousal += bio.arousal * weights["bio"]
            combined.dominance += bio.dominance * weights["bio"]
            total_weight += weights["bio"]
            
        # Normalize by actual weights used
        if total_weight > 0:
            combined.valence /= total_weight
            combined.arousal /= total_weight
            combined.dominance /= total_weight
            
        return combined
        
    def _detect_emotional_patterns(self, state: EmotionalState, context: Dict) -> List[str]:
        """Detect emotional patterns and cycles"""
        patterns = []
        
        # Check recent history
        recent_states = [em.state for em in list(self.emotional_history)[-10:]]
        
        if len(recent_states) >= 3:
            # Escalation pattern
            arousal_trend = [s.arousal for s in recent_states[-3:]]
            if all(arousal_trend[i] < arousal_trend[i+1] for i in range(2)):
                patterns.append("escalating_arousal")
                
            # Mood swing pattern
            valence_values = [s.valence for s in recent_states[-5:]]
            if max(valence_values) - min(valence_values) > 1.2:
                patterns.append("mood_swings")
                
            # Stuck pattern (low variation)
            if all(abs(s.valence - state.valence) < 0.2 for s in recent_states[-5:]):
                if state.valence < -0.3:
                    patterns.append("stuck_negative")
                elif state.valence > 0.3:
                    patterns.append("stable_positive")
                    
        # Context-based patterns
        if context:
            time_of_day = datetime.now().hour
            if time_of_day >= 22 or time_of_day <= 6:
                if state.arousal > 0.5:
                    patterns.append("late_night_activation")
                    
            # Work hours stress
            if 9 <= time_of_day <= 17 and state.valence < -0.3:
                patterns.append("work_hours_stress")
                
        return patterns
        
    def _calculate_trajectory(self, current_state: EmotionalState) -> Dict:
        """Calculate emotional trajectory and predictions"""
        if len(self.emotional_history) < 2:
            return {"direction": "stable", "velocity": 0.0, "prediction": "maintaining"}
            
        recent = list(self.emotional_history)[-5:]
        
        # Calculate derivatives
        valence_delta = current_state.valence - recent[-1].state.valence
        arousal_delta = current_state.arousal - recent[-1].state.arousal
        
        # Velocity of change
        velocity = math.sqrt(valence_delta**2 + arousal_delta**2)
        
        # Direction
        if velocity < 0.1:
            direction = "stable"
        elif valence_delta > 0.2:
            direction = "improving"
        elif valence_delta < -0.2:
            direction = "declining"
        elif arousal_delta > 0.3:
            direction = "activating"
        else:
            direction = "calming"
            
        # Prediction based on trajectory
        if direction == "declining" and velocity > 0.3:
            prediction = "needs_intervention"
        elif direction == "improving":
            prediction = "positive_momentum"
        elif direction == "stable" and current_state.valence < -0.5:
            prediction = "stuck_negative"
        else:
            prediction = "maintaining"
            
        return {
            "direction": direction,
            "velocity": velocity,
            "prediction": prediction,
            "target_distance": current_state.distance_to(self.target_state)
        }
        
    def _generate_empathetic_understanding(self, 
                                         state: EmotionalState,
                                         text: str,
                                         context: Dict) -> Dict:
        """Generate deep empathetic understanding"""
        understanding = {
            "primary_emotion": self._identify_primary_emotion(state),
            "underlying_needs": self._identify_needs(state, text, context),
            "validation": self._generate_validation(state, text),
            "perspective": self._generate_perspective(state, context)
        }
        
        return understanding
        
    def _identify_primary_emotion(self, state: EmotionalState) -> str:
        """Identify the primary emotion from the emotional state"""
        emotions = {
            "excited_happy": ["joy", "excitement", "enthusiasm"],
            "calm_happy": ["contentment", "peace", "satisfaction"],
            "excited_unhappy": ["anger", "frustration", "anxiety"],
            "calm_unhappy": ["sadness", "disappointment", "loneliness"]
        }
        
        quadrant_emotions = emotions.get(state.quadrant, ["neutral"])
        
        # Refine based on specific values
        if state.quadrant == "excited_unhappy":
            if state.dominance > 0.5:
                return "anger"
            else:
                return "anxiety"
        elif state.quadrant == "calm_unhappy":
            if state.dominance < -0.3:
                return "helplessness"
            else:
                return "sadness"
                
        return quadrant_emotions[0]
        
    def _identify_needs(self, state: EmotionalState, text: str, context: Dict) -> List[str]:
        """Identify underlying emotional needs"""
        needs = []
        
        # Based on emotional state
        if state.valence < -0.3:
            needs.append("support")
            if state.arousal > 0.3:
                needs.append("resolution")
            else:
                needs.append("comfort")
                
        if state.arousal > 0.5:
            needs.append("grounding")
            
        if state.dominance < -0.5:
            needs.append("empowerment")
            
        # Based on text content
        text_lower = text.lower()
        if "help" in text_lower:
            needs.append("assistance")
        if "listen" in text_lower or "hear" in text_lower:
            needs.append("validation")
        if "alone" in text_lower or "nobody" in text_lower:
            needs.append("connection")
            
        return list(set(needs))  # Remove duplicates
        
    def _generate_validation(self, state: EmotionalState, text: str) -> str:
        """Generate appropriate validation statement"""
        emotion = self._identify_primary_emotion(state)
        
        # Extract situation from text (simplified)
        situation = "this situation"
        if "work" in text.lower():
            situation = "your work situation"
        elif "family" in text.lower():
            situation = "your family situation"
            
        template = np.random.choice(self.empathy_model["validation_templates"])
        return template.format(emotion=emotion, situation=situation, context=situation)
        
    def _generate_perspective(self, state: EmotionalState, context: Dict) -> str:
        """Generate perspective-taking statement"""
        if state.quadrant == "excited_unhappy":
            return "This seems really frustrating and urgent for you"
        elif state.quadrant == "calm_unhappy":
            return "This has been weighing on you for a while"
        elif state.quadrant == "excited_happy":
            return "You have wonderful energy about this"
        else:
            return "You seem at peace with where things are"
            
    def _recommend_response_strategy(self, state: EmotionalState, trajectory: Dict) -> Dict:
        """Recommend response strategy based on emotional analysis"""
        strategy = self.response_strategies.get(state.quadrant, {})
        
        # Adjust based on trajectory
        if trajectory["prediction"] == "needs_intervention":
            strategy["urgency"] = "high"
            strategy["primary_action"] = "stabilize"
        elif trajectory["prediction"] == "positive_momentum":
            strategy["primary_action"] = "reinforce"
            
        # Add specific recommendations
        recommendations = {
            "tone": strategy.get("tone_adjustment", {}),
            "actions": strategy.get("actions", []),
            "phrases": strategy.get("phrases", []),
            "approach": self._determine_approach(state, trajectory),
            "intensity": self._calculate_response_intensity(state, trajectory)
        }
        
        return recommendations
        
    def _determine_approach(self, state: EmotionalState, trajectory: Dict) -> str:
        """Determine the best approach for response"""
        if state.valence < -0.5 and trajectory["velocity"] > 0.5:
            return "crisis_support"
        elif state.valence < -0.3:
            return "gentle_support"
        elif state.valence > 0.3 and state.arousal > 0.3:
            return "enthusiastic_engagement"
        elif trajectory["direction"] == "stable":
            return "maintain_connection"
        else:
            return "adaptive_guidance"
            
    def _calculate_response_intensity(self, state: EmotionalState, trajectory: Dict) -> float:
        """Calculate appropriate response intensity (0-1)"""
        # Higher intensity for extreme states
        intensity = abs(state.valence) * 0.3 + abs(state.arousal) * 0.3
        
        # Adjust for trajectory
        if trajectory["velocity"] > 0.5:
            intensity += 0.2
            
        # Cap at reasonable levels
        return min(1.0, intensity)
        
    def _update_emotional_state(self, state: EmotionalState, context: Dict):
        """Update internal emotional tracking"""
        memory = EmotionalMemory(
            state=state,
            timestamp=datetime.now(),
            context=context or {},
            triggers=[],  # Would be filled based on analysis
            response_effectiveness=0.5  # Would be updated based on feedback
        )
        
        self.emotional_history.append(memory)
        self.current_state = state
        
        # Learn patterns
        if len(self.emotional_history) > 10:
            self._learn_emotional_patterns()
            
    def _learn_emotional_patterns(self):
        """Learn from emotional patterns over time"""
        # Simplified pattern learning
        recent = list(self.emotional_history)[-20:]
        
        # Identify recurring triggers
        trigger_counts = {}
        for memory in recent:
            for trigger in memory.triggers:
                trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
                
        # Store significant patterns
        for trigger, count in trigger_counts.items():
            if count > 3:
                self.pattern_memory.append({
                    "pattern": trigger,
                    "frequency": count,
                    "typical_response": self._get_typical_response(trigger, recent)
                })
                
    def _get_typical_response(self, trigger: str, memories: List[EmotionalMemory]) -> str:
        """Get typical emotional response to a trigger"""
        states = [m.state for m in memories if trigger in m.triggers]
        if not states:
            return "neutral"
            
        avg_valence = sum(s.valence for s in states) / len(states)
        avg_arousal = sum(s.arousal for s in states) / len(states)
        
        if avg_valence < -0.3:
            return "negative"
        elif avg_valence > 0.3:
            return "positive"
        else:
            return "neutral"
            
    def generate_emotionally_aware_response(self, 
                                          base_response: str,
                                          emotional_analysis: Dict) -> str:
        """Enhance response with emotional awareness"""
        strategy = emotional_analysis["recommended_response"]
        
        # Apply tone adjustments
        response = self._apply_tone_adjustments(base_response, strategy["tone"])
        
        # Add empathetic elements
        if emotional_analysis["understanding"]["underlying_needs"]:
            needs = emotional_analysis["understanding"]["underlying_needs"]
            if "support" in needs:
                response = emotional_analysis["understanding"]["validation"] + " " + response
                
        # Add appropriate closure
        approach = strategy["approach"]
        if approach == "crisis_support":
            response += " I'm here with you through this."
        elif approach == "gentle_support":
            response += " Take your time."
        elif approach == "enthusiastic_engagement":
            response += " Let's make this happen!"
            
        return response
        
    def _apply_tone_adjustments(self, text: str, tone_adjustments: Dict) -> str:
        """Apply tone adjustments to text"""
        # This would ideally modify the actual language
        # For now, we'll just note the adjustments
        if tone_adjustments.get("warmth", 1.0) > 1.2:
            # Add warmer language
            text = text.replace("Okay", "Of course")
            text = text.replace("I'll", "I'd be happy to")
            
        if tone_adjustments.get("energy", 1.0) > 1.2:
            # Add more energetic language
            if not text.endswith("!"):
                text += "!"
                
        return text
        
    def get_emotional_summary(self) -> Dict:
        """Get summary of emotional patterns and state"""
        if not self.emotional_history:
            return {"status": "no_data"}
            
        recent = list(self.emotional_history)[-20:]
        
        return {
            "current_state": {
                "primary_emotion": self._identify_primary_emotion(self.current_state),
                "quadrant": self.current_state.quadrant,
                "intensity": abs(self.current_state.valence) + abs(self.current_state.arousal)
            },
            "patterns": {
                "average_valence": sum(m.state.valence for m in recent) / len(recent),
                "average_arousal": sum(m.state.arousal for m in recent) / len(recent),
                "volatility": np.std([m.state.valence for m in recent]),
                "dominant_quadrant": max(
                    set(m.state.quadrant for m in recent),
                    key=lambda x: sum(1 for m in recent if m.state.quadrant == x)
                )
            },
            "trajectory": self._calculate_trajectory(self.current_state),
            "recommendations": {
                "interaction_style": self._recommend_interaction_style(),
                "topics_to_avoid": self._identify_sensitive_topics(),
                "supportive_actions": self._recommend_supportive_actions()
            }
        }
        
    def _recommend_interaction_style(self) -> str:
        """Recommend interaction style based on patterns"""
        if self.current_state.valence < -0.5:
            return "gentle_supportive"
        elif self.current_state.arousal > 0.7:
            return "calm_grounding"
        elif self.current_state.valence > 0.5:
            return "enthusiastic_collaborative"
        else:
            return "balanced_responsive"
            
    def _identify_sensitive_topics(self) -> List[str]:
        """Identify topics to approach carefully"""
        sensitive = []
        
        # Check pattern memory for negative associations
        for pattern in self.pattern_memory:
            if pattern.get("typical_response") == "negative":
                sensitive.append(pattern["pattern"])
                
        return sensitive
        
    def _recommend_supportive_actions(self) -> List[str]:
        """Recommend supportive actions based on state"""
        actions = []
        
        if self.current_state.arousal > 0.7:
            actions.append("suggest_breathing_exercise")
            actions.append("offer_break")
            
        if self.current_state.valence < -0.3:
            actions.append("increase_check_ins")
            actions.append("offer_support_resources")
            
        if self._detect_burnout_risk():
            actions.append("recommend_rest")
            actions.append("reduce_notifications")
            
        return actions
        
    def _detect_burnout_risk(self) -> bool:
        """Detect risk of emotional burnout"""
        if len(self.emotional_history) < 50:
            return False
            
        recent = list(self.emotional_history)[-50:]
        
        # Check for prolonged negative state
        negative_count = sum(1 for m in recent if m.state.valence < -0.3)
        high_arousal_count = sum(1 for m in recent if m.state.arousal > 0.6)
        
        return negative_count > 30 or high_arousal_count > 35
