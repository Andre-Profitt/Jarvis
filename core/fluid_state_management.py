# Fluid State Management System for JARVIS
# Phase 1 Implementation - Smooth transitions and intelligent state tracking

import asyncio
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import math
import json

# ============================================
# STATE DEFINITIONS & CURVES
# ============================================

class StateType(Enum):
    """Core states that JARVIS tracks"""
    STRESS = auto()
    FOCUS = auto()
    ENERGY = auto()
    MOOD = auto()
    CREATIVITY = auto()
    PRODUCTIVITY = auto()
    SOCIAL = auto()
    HEALTH = auto()

class ResponseMode(Enum):
    """JARVIS response modes based on state"""
    PROTECTIVE = auto()      # High stress, low energy
    SUPPORTIVE = auto()      # Medium stress, medium energy
    COLLABORATIVE = auto()   # Low stress, high focus
    BACKGROUND = auto()      # Flow state, minimal intervention
    PROACTIVE = auto()       # Normal state, helpful suggestions
    EMERGENCY = auto()       # Critical state, immediate action

@dataclass
class StateVector:
    """Multi-dimensional state representation"""
    values: Dict[StateType, float]
    timestamp: datetime
    confidence: float = 1.0
    source_weights: Dict[str, float] = field(default_factory=dict)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for calculations"""
        return np.array([self.values[st] for st in StateType])
    
    def distance_to(self, other: 'StateVector') -> float:
        """Calculate distance to another state"""
        return np.linalg.norm(self.to_array() - other.to_array())

class SmoothCurve:
    """Advanced smoothing with multiple algorithms"""
    
    def __init__(self, 
                 smoothing: float = 0.3,
                 algorithm: str = 'exponential',
                 min_val: float = 0.0,
                 max_val: float = 1.0,
                 memory_size: int = 100):
        self.smoothing = smoothing
        self.algorithm = algorithm
        self.min_val = min_val
        self.max_val = max_val
        self.current = (min_val + max_val) / 2
        self.history = deque(maxlen=memory_size)
        self.velocity = 0.0
        self.acceleration = 0.0
        
    def apply(self, target: float, dt: float = 0.1) -> float:
        """Apply smoothing based on selected algorithm"""
        # Clamp target
        target = max(self.min_val, min(self.max_val, target))
        
        if self.algorithm == 'exponential':
            result = self._exponential_smoothing(target)
        elif self.algorithm == 'kalman':
            result = self._kalman_filter(target, dt)
        elif self.algorithm == 'physics':
            result = self._physics_based(target, dt)
        elif self.algorithm == 'adaptive':
            result = self._adaptive_smoothing(target, dt)
        else:
            result = target
            
        self.history.append(result)
        self.current = result
        return result
    
    def _exponential_smoothing(self, target: float) -> float:
        """Classic exponential smoothing"""
        return self.current * (1 - self.smoothing) + target * self.smoothing
    
    def _kalman_filter(self, target: float, dt: float) -> float:
        """Simplified Kalman filter for 1D"""
        # Process noise
        q = 0.01
        # Measurement noise
        r = 0.1
        
        # Predict
        predicted = self.current + self.velocity * dt
        
        # Update
        gain = q / (q + r)
        self.current = predicted + gain * (target - predicted)
        self.velocity += gain * (target - predicted) / dt
        
        return self.current
    
    def _physics_based(self, target: float, dt: float) -> float:
        """Spring-damper system for natural motion"""
        # Spring constant
        k = 10.0
        # Damping coefficient
        c = 2 * math.sqrt(k) * self.smoothing
        
        # Calculate forces
        spring_force = k * (target - self.current)
        damping_force = -c * self.velocity
        
        # Update physics
        self.acceleration = spring_force + damping_force
        self.velocity += self.acceleration * dt
        self.current += self.velocity * dt
        
        return max(self.min_val, min(self.max_val, self.current))
    
    def _adaptive_smoothing(self, target: float, dt: float) -> float:
        """Adaptive smoothing based on rate of change"""
        if len(self.history) < 2:
            return self._exponential_smoothing(target)
            
        # Calculate recent volatility
        recent = list(self.history)[-10:]
        volatility = np.std(recent) if len(recent) > 1 else 0
        
        # Adjust smoothing based on volatility
        adaptive_smoothing = self.smoothing * (1 + volatility * 2)
        adaptive_smoothing = min(0.9, adaptive_smoothing)  # Cap at 0.9
        
        return self.current * (1 - adaptive_smoothing) + target * adaptive_smoothing
    
    def get_trend(self) -> str:
        """Get current trend direction"""
        if len(self.history) < 3:
            return "stable"
            
        recent = list(self.history)[-5:]
        trend = recent[-1] - recent[0]
        
        if abs(trend) < 0.05:
            return "stable"
        elif trend > 0:
            return "increasing"
        else:
            return "decreasing"

# ============================================
# STATE CALCULATORS
# ============================================

class StateCalculator:
    """Calculate individual states from raw inputs"""
    
    def __init__(self):
        self.weights = self._initialize_weights()
        
    def _initialize_weights(self) -> Dict[str, Dict[str, float]]:
        """Initialize input weights for each state"""
        return {
            'stress': {
                'heart_rate': 0.3,
                'heart_rate_variability': -0.3,  # Negative correlation
                'skin_conductance': 0.25,
                'voice_pitch_variance': 0.2,
                'breathing_rate': 0.15,
                'facial_tension': 0.1
            },
            'focus': {
                'eye_movement_stability': 0.3,
                'task_switching_frequency': -0.25,
                'response_time_consistency': 0.2,
                'distraction_events': -0.15,
                'flow_duration': 0.1
            },
            'energy': {
                'movement_level': 0.25,
                'voice_energy': 0.2,
                'typing_speed': 0.15,
                'interaction_frequency': 0.15,
                'posture_quality': 0.15,
                'time_since_break': -0.1
            },
            'mood': {
                'facial_expression': 0.3,
                'voice_sentiment': 0.25,
                'word_choice_positivity': 0.2,
                'social_interaction_quality': 0.15,
                'music_choice': 0.1
            }
        }
    
    def calculate_stress(self, inputs: Dict[str, Any]) -> float:
        """Calculate stress level from multiple inputs"""
        stress_factors = []
        weights_sum = 0
        
        # Biometric factors
        if 'biometric' in inputs:
            bio = inputs['biometric']
            
            # Heart rate
            if 'heart_rate' in bio:
                hr = bio['heart_rate']
                hr_stress = self._normalize_heart_rate(hr)
                weight = self.weights['stress']['heart_rate']
                stress_factors.append(hr_stress * weight)
                weights_sum += weight
                
            # HRV (inverse relationship)
            if 'hrv' in bio:
                hrv = bio['hrv']
                hrv_stress = 1 - self._normalize_hrv(hrv)
                weight = abs(self.weights['stress']['heart_rate_variability'])
                stress_factors.append(hrv_stress * weight)
                weights_sum += weight
                
            # Skin conductance
            if 'skin_conductance' in bio:
                sc = bio['skin_conductance']
                weight = self.weights['stress']['skin_conductance']
                stress_factors.append(sc * weight)
                weights_sum += weight
        
        # Voice factors
        if 'voice' in inputs:
            voice = inputs['voice']
            features = voice.get('features', {})
            
            if 'pitch_variance' in features:
                pv = features['pitch_variance']
                weight = self.weights['stress']['voice_pitch_variance']
                stress_factors.append(pv * weight)
                weights_sum += weight
        
        # Environmental factors
        if 'environment' in inputs:
            env = inputs['environment']
            
            # Deadline pressure
            if 'deadline_minutes' in env:
                deadline = env['deadline_minutes']
                pressure = max(0, 1 - deadline / 120)  # Max pressure at 0 min, none at 2h+
                stress_factors.append(pressure * 0.2)
                weights_sum += 0.2
        
        # Calculate weighted average
        if weights_sum > 0:
            return sum(stress_factors) / weights_sum
        else:
            return 0.5  # Default neutral
    
    def calculate_focus(self, inputs: Dict[str, Any]) -> float:
        """Calculate focus level"""
        focus_factors = []
        
        # Task consistency
        if 'activity' in inputs:
            activity = inputs['activity']
            
            # Check task switching
            switches = activity.get('task_switches_per_hour', 0)
            focus_from_switches = max(0, 1 - switches / 10)  # 10+ switches = no focus
            focus_factors.append(focus_from_switches)
            
            # Flow state duration
            if 'flow_duration_minutes' in activity:
                flow = activity['flow_duration_minutes']
                focus_from_flow = min(1, flow / 30)  # 30+ min = max focus
                focus_factors.append(focus_from_flow)
        
        # Eye tracking
        if 'eye_tracking' in inputs:
            eye = inputs['eye_tracking']
            stability = eye.get('gaze_stability', 0.5)
            focus_factors.append(stability)
        
        # Distraction events
        if 'distractions' in inputs:
            distractions = inputs['distractions']
            count = distractions.get('count_per_hour', 0)
            focus_from_distractions = max(0, 1 - count / 5)  # 5+ distractions = no focus
            focus_factors.append(focus_from_distractions)
        
        return np.mean(focus_factors) if focus_factors else 0.7
    
    def calculate_energy(self, inputs: Dict[str, Any]) -> float:
        """Calculate energy level"""
        energy_factors = []
        
        # Physical activity
        if 'movement' in inputs:
            movement = inputs['movement']
            activity_level = movement.get('activity_level', 0.5)
            energy_factors.append(activity_level)
        
        # Voice energy
        if 'voice' in inputs:
            voice = inputs['voice']
            features = voice.get('features', {})
            voice_energy = features.get('energy', 0.5)
            energy_factors.append(voice_energy)
        
        # Time of day factor
        if 'temporal' in inputs:
            temporal = inputs['temporal']
            hour = temporal.get('hour', 12)
            # Simple circadian rhythm model
            circadian = 0.5 + 0.5 * math.sin((hour - 6) * math.pi / 12)
            energy_factors.append(circadian)
        
        # Fatigue factors
        if 'activity' in inputs:
            activity = inputs['activity']
            hours_worked = activity.get('continuous_work_hours', 0)
            fatigue = max(0, 1 - hours_worked / 8)  # 8+ hours = exhausted
            energy_factors.append(fatigue)
        
        return np.mean(energy_factors) if energy_factors else 0.6
    
    def calculate_mood(self, inputs: Dict[str, Any]) -> float:
        """Calculate mood level"""
        mood_factors = []
        
        # Facial expression
        if 'vision' in inputs:
            vision = inputs['vision']
            expression = vision.get('facial_expression', {})
            positivity = expression.get('positivity', 0.5)
            mood_factors.append(positivity)
        
        # Voice sentiment
        if 'voice' in inputs:
            voice = inputs['voice']
            sentiment = voice.get('sentiment', 0.5)
            mood_factors.append(sentiment)
        
        # Text sentiment
        if 'text' in inputs:
            text = inputs['text']
            text_sentiment = text.get('sentiment', 0.5)
            mood_factors.append(text_sentiment)
        
        # Social interactions
        if 'social' in inputs:
            social = inputs['social']
            interaction_quality = social.get('interaction_quality', 0.5)
            mood_factors.append(interaction_quality)
        
        return np.mean(mood_factors) if mood_factors else 0.7
    
    def _normalize_heart_rate(self, hr: float) -> float:
        """Normalize heart rate to 0-1 scale"""
        # Assuming 60-100 is normal range
        if hr < 60:
            return 0.3  # Low HR might indicate other issues
        elif hr > 100:
            return min(1.0, (hr - 60) / 80)  # Stress increases above 100
        else:
            return 0.3 + (hr - 60) / 40 * 0.2  # Slight increase in normal range
    
    def _normalize_hrv(self, hrv: float) -> float:
        """Normalize HRV to 0-1 scale (higher is better)"""
        # Assuming 20-80 ms is typical range
        return max(0, min(1, (hrv - 20) / 60))

# ============================================
# FLUID STATE MANAGER
# ============================================

class FluidStateManager:
    """Main state management system with smooth transitions"""
    
    def __init__(self):
        self.state_curves = self._initialize_curves()
        self.state_calculator = StateCalculator()
        self.state_history = deque(maxlen=1000)
        self.response_curve = self._initialize_response_curve()
        
        # State patterns for different scenarios
        self.state_patterns = self._load_state_patterns()
        
        # Anomaly detection
        self.baseline_states = {}
        self.anomaly_threshold = 2.0  # Standard deviations
        
    def _initialize_curves(self) -> Dict[StateType, SmoothCurve]:
        """Initialize smoothing curves for each state"""
        return {
            StateType.STRESS: SmoothCurve(smoothing=0.3, algorithm='adaptive'),
            StateType.FOCUS: SmoothCurve(smoothing=0.5, algorithm='physics'),
            StateType.ENERGY: SmoothCurve(smoothing=0.4, algorithm='kalman'),
            StateType.MOOD: SmoothCurve(smoothing=0.6, algorithm='exponential'),
            StateType.CREATIVITY: SmoothCurve(smoothing=0.7, algorithm='physics'),
            StateType.PRODUCTIVITY: SmoothCurve(smoothing=0.4, algorithm='adaptive'),
            StateType.SOCIAL: SmoothCurve(smoothing=0.5, algorithm='exponential'),
            StateType.HEALTH: SmoothCurve(smoothing=0.8, algorithm='kalman')
        }
    
    def _initialize_response_curve(self):
        """Initialize response intensity mapping"""
        return {
            'stress_weight': 0.4,
            'focus_weight': 0.3,
            'energy_weight': 0.2,
            'mood_weight': 0.1,
            'threshold_emergency': 0.85,
            'threshold_proactive': 0.6,
            'threshold_supportive': 0.4,
            'threshold_background': 0.2
        }
    
    def _load_state_patterns(self) -> Dict[str, StateVector]:
        """Load known state patterns"""
        return {
            'flow_state': StateVector(
                values={
                    StateType.STRESS: 0.2,
                    StateType.FOCUS: 0.95,
                    StateType.ENERGY: 0.7,
                    StateType.MOOD: 0.8,
                    StateType.CREATIVITY: 0.85,
                    StateType.PRODUCTIVITY: 0.9,
                    StateType.SOCIAL: 0.3,
                    StateType.HEALTH: 0.7
                },
                timestamp=datetime.now()
            ),
            'burnout': StateVector(
                values={
                    StateType.STRESS: 0.9,
                    StateType.FOCUS: 0.2,
                    StateType.ENERGY: 0.1,
                    StateType.MOOD: 0.2,
                    StateType.CREATIVITY: 0.1,
                    StateType.PRODUCTIVITY: 0.1,
                    StateType.SOCIAL: 0.1,
                    StateType.HEALTH: 0.3
                },
                timestamp=datetime.now()
            ),
            'creative_burst': StateVector(
                values={
                    StateType.STRESS: 0.3,
                    StateType.FOCUS: 0.7,
                    StateType.ENERGY: 0.8,
                    StateType.MOOD: 0.85,
                    StateType.CREATIVITY: 0.95,
                    StateType.PRODUCTIVITY: 0.6,
                    StateType.SOCIAL: 0.5,
                    StateType.HEALTH: 0.7
                },
                timestamp=datetime.now()
            )
        }
    
    async def update_state(self, inputs: Dict[str, Any]) -> StateVector:
        """Update all states with smooth transitions"""
        # Calculate raw state values
        raw_states = {
            StateType.STRESS: self.state_calculator.calculate_stress(inputs),
            StateType.FOCUS: self.state_calculator.calculate_focus(inputs),
            StateType.ENERGY: self.state_calculator.calculate_energy(inputs),
            StateType.MOOD: self.state_calculator.calculate_mood(inputs)
        }
        
        # Calculate derived states
        raw_states[StateType.CREATIVITY] = self._calculate_creativity(raw_states)
        raw_states[StateType.PRODUCTIVITY] = self._calculate_productivity(raw_states)
        raw_states[StateType.SOCIAL] = self._calculate_social(inputs)
        raw_states[StateType.HEALTH] = self._calculate_health(raw_states, inputs)
        
        # Apply smoothing curves
        smoothed_states = {}
        dt = 0.1  # 100ms update interval
        
        for state_type, raw_value in raw_states.items():
            curve = self.state_curves[state_type]
            smoothed_value = curve.apply(raw_value, dt)
            smoothed_states[state_type] = smoothed_value
        
        # Create state vector
        state_vector = StateVector(
            values=smoothed_states,
            timestamp=datetime.now(),
            confidence=self._calculate_confidence(inputs)
        )
        
        # Store in history
        self.state_history.append(state_vector)
        
        # Check for anomalies
        if self._is_anomalous(state_vector):
            await self._handle_anomaly(state_vector)
        
        return state_vector
    
    def get_response_mode(self, state: StateVector) -> ResponseMode:
        """Determine appropriate response mode from state"""
        # Calculate weighted intensity
        intensity = self._calculate_response_intensity(state)
        
        # Check for emergency conditions first
        if state.values[StateType.STRESS] > 0.9 or state.values[StateType.HEALTH] < 0.2:
            return ResponseMode.EMERGENCY
            
        # Flow state detection
        if self._is_flow_state(state):
            return ResponseMode.BACKGROUND
            
        # Map intensity to response mode
        if intensity > self.response_curve['threshold_emergency']:
            return ResponseMode.EMERGENCY
        elif intensity > self.response_curve['threshold_proactive']:
            return ResponseMode.PROACTIVE
        elif intensity > self.response_curve['threshold_supportive']:
            return ResponseMode.SUPPORTIVE
        elif intensity < self.response_curve['threshold_background']:
            return ResponseMode.BACKGROUND
        else:
            return ResponseMode.COLLABORATIVE
    
    def _calculate_response_intensity(self, state: StateVector) -> float:
        """Calculate response intensity from state"""
        weights = self.response_curve
        
        # High stress increases intensity
        stress_component = state.values[StateType.STRESS] * weights['stress_weight']
        
        # Low focus increases intensity (user might need help)
        focus_component = (1 - state.values[StateType.FOCUS]) * weights['focus_weight']
        
        # Low energy increases intensity
        energy_component = (1 - state.values[StateType.ENERGY]) * weights['energy_weight']
        
        # Low mood increases intensity
        mood_component = (1 - state.values[StateType.MOOD]) * weights['mood_weight']
        
        total = stress_component + focus_component + energy_component + mood_component
        return min(1.0, total)
    
    def _calculate_creativity(self, states: Dict[StateType, float]) -> float:
        """Calculate creativity from other states"""
        # Creativity peaks at moderate stress, high energy, good mood
        stress_factor = 1 - abs(states[StateType.STRESS] - 0.3) / 0.7  # Peak at 0.3
        energy_factor = states[StateType.ENERGY]
        mood_factor = states[StateType.MOOD]
        focus_factor = states[StateType.FOCUS] * 0.7  # Some focus helps
        
        return (stress_factor * 0.3 + energy_factor * 0.3 + 
                mood_factor * 0.2 + focus_factor * 0.2)
    
    def _calculate_productivity(self, states: Dict[StateType, float]) -> float:
        """Calculate productivity from other states"""
        # Productivity needs focus, energy, and low stress
        focus_factor = states[StateType.FOCUS]
        energy_factor = states[StateType.ENERGY]
        stress_penalty = max(0, 1 - states[StateType.STRESS] * 1.5)  # High stress kills productivity
        mood_bonus = states[StateType.MOOD] * 0.2  # Good mood helps a bit
        
        return (focus_factor * 0.4 + energy_factor * 0.3 + 
                stress_penalty * 0.3) * (1 + mood_bonus)
    
    def _calculate_social(self, inputs: Dict[str, Any]) -> float:
        """Calculate social state from inputs"""
        social_factors = []
        
        if 'social' in inputs:
            social = inputs['social']
            interaction_count = social.get('interactions_per_hour', 0)
            social_factors.append(min(1, interaction_count / 5))  # 5+ interactions = socially active
            
            quality = social.get('interaction_quality', 0.5)
            social_factors.append(quality)
        
        if 'communication' in inputs:
            comm = inputs['communication']
            response_time = comm.get('avg_response_time_minutes', 30)
            responsiveness = max(0, 1 - response_time / 60)  # Fast responses = socially engaged
            social_factors.append(responsiveness)
        
        return np.mean(social_factors) if social_factors else 0.5
    
    def _calculate_health(self, states: Dict[StateType, float], inputs: Dict[str, Any]) -> float:
        """Calculate overall health state"""
        health_factors = []
        
        # Stress impact on health
        stress_health = 1 - states[StateType.STRESS] * 0.5
        health_factors.append(stress_health)
        
        # Energy as health indicator
        health_factors.append(states[StateType.ENERGY])
        
        # Physical health indicators
        if 'biometric' in inputs:
            bio = inputs['biometric']
            # Various health metrics normalized to 0-1
            if 'resting_heart_rate' in bio:
                rhr = bio['resting_heart_rate']
                rhr_health = 1 - abs(rhr - 65) / 35  # 65 is ideal
                health_factors.append(max(0, rhr_health))
                
            if 'sleep_quality' in bio:
                health_factors.append(bio['sleep_quality'])
                
            if 'hydration' in bio:
                health_factors.append(bio['hydration'])
        
        return np.mean(health_factors) if health_factors else 0.7
    
    def _calculate_confidence(self, inputs: Dict[str, Any]) -> float:
        """Calculate confidence in state assessment"""
        # More input sources = higher confidence
        source_count = len(inputs)
        source_confidence = min(1, source_count / 5)  # 5+ sources = max confidence
        
        # Data quality factors
        quality_factors = []
        
        if 'biometric' in inputs and 'signal_quality' in inputs['biometric']:
            quality_factors.append(inputs['biometric']['signal_quality'])
            
        if 'voice' in inputs and 'confidence' in inputs['voice']:
            quality_factors.append(inputs['voice']['confidence'])
            
        quality_confidence = np.mean(quality_factors) if quality_factors else 0.8
        
        return source_confidence * 0.5 + quality_confidence * 0.5
    
    def _is_flow_state(self, state: StateVector) -> bool:
        """Detect if user is in flow state"""
        flow_pattern = self.state_patterns['flow_state']
        distance = state.distance_to(flow_pattern)
        return distance < 0.3  # Close enough to flow pattern
    
    def _is_anomalous(self, state: StateVector) -> bool:
        """Detect anomalous states"""
        if len(self.state_history) < 100:
            return False  # Not enough history
            
        # Calculate baseline statistics
        recent_states = list(self.state_history)[-100:-1]  # Last 100 except current
        
        for state_type in StateType:
            values = [s.values[state_type] for s in recent_states]
            mean = np.mean(values)
            std = np.std(values)
            
            current_value = state.values[state_type]
            z_score = abs(current_value - mean) / (std + 0.01)  # Avoid division by zero
            
            if z_score > self.anomaly_threshold:
                return True
                
        return False
    
    async def _handle_anomaly(self, state: StateVector):
        """Handle anomalous state detection"""
        # Log anomaly
        print(f"⚠️ Anomalous state detected at {state.timestamp}")
        
        # Trigger appropriate response based on anomaly type
        if state.values[StateType.STRESS] > 0.9:
            print("  → Critical stress level detected")
        elif state.values[StateType.ENERGY] < 0.1:
            print("  → Dangerously low energy detected")
        elif state.values[StateType.HEALTH] < 0.3:
            print("  → Health concerns detected")
    
    def get_state_trends(self) -> Dict[StateType, str]:
        """Get current trends for all states"""
        trends = {}
        for state_type, curve in self.state_curves.items():
            trends[state_type] = curve.get_trend()
        return trends
    
    def get_state_prediction(self, minutes_ahead: int = 30) -> StateVector:
        """Predict future state based on trends"""
        if len(self.state_history) < 10:
            return self.state_history[-1] if self.state_history else None
            
        # Simple linear extrapolation
        recent_states = list(self.state_history)[-10:]
        predicted_values = {}
        
        for state_type in StateType:
            values = [s.values[state_type] for s in recent_states]
            # Calculate trend
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            # Extrapolate
            future_x = len(values) + (minutes_ahead / 6)  # Assuming 6 min between samples
            predicted = np.polyval(coeffs, future_x)
            # Clamp to valid range
            predicted_values[state_type] = max(0, min(1, predicted))
        
        return StateVector(
            values=predicted_values,
            timestamp=datetime.now() + timedelta(minutes=minutes_ahead),
            confidence=0.5  # Lower confidence for predictions
        )

# Export main components
__all__ = [
    'FluidStateManager',
    'StateVector',
    'ResponseMode',
    'StateType',
    'SmoothCurve',
    'StateCalculator'
]