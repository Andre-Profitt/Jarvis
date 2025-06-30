#!/usr/bin/env python3
"""
JARVIS Phase 5: Graduated Interventions System
==============================================

Implements sophisticated intervention strategies that start subtle
and gradually increase based on user response and context.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import json
from collections import deque

logger = logging.getLogger(__name__)


class InterventionLevel(Enum):
    """Levels of intervention from subtle to direct"""
    AMBIENT = auto()      # Environmental cues (lighting, music)
    SUBTLE = auto()       # Gentle hints (notifications, badges)
    SUGGESTIVE = auto()   # Clear suggestions ("Maybe take a break?")
    ASSERTIVE = auto()    # Strong recommendations ("You should rest")
    PROTECTIVE = auto()   # Active intervention (blocking distractions)
    EMERGENCY = auto()    # Crisis response (immediate action)


class InterventionType(Enum):
    """Types of interventions JARVIS can perform"""
    BREAK_REMINDER = auto()
    STRESS_RELIEF = auto()
    FOCUS_PROTECTION = auto()
    HEALTH_CHECK = auto()
    PRODUCTIVITY_BOOST = auto()
    EMOTIONAL_SUPPORT = auto()
    CRISIS_RESPONSE = auto()
    CELEBRATION = auto()
    LEARNING_OPPORTUNITY = auto()


@dataclass
class InterventionContext:
    """Context for determining intervention approach"""
    user_state: str
    stress_level: float
    focus_level: float
    time_since_break: float  # minutes
    current_activity: str
    response_history: List[str] = field(default_factory=list)
    ignored_count: int = 0
    accepted_count: int = 0
    emotional_state: str = "neutral"
    urgency: float = 0.0


@dataclass
class Intervention:
    """A specific intervention action"""
    level: InterventionLevel
    type: InterventionType
    message: Optional[str]
    actions: List[Dict[str, Any]]
    context: InterventionContext
    timestamp: datetime = field(default_factory=datetime.now)
    escalation_timer: Optional[float] = None  # minutes until escalation


class GraduatedInterventionSystem:
    """
    Sophisticated intervention system that adapts to user responses
    and gradually escalates interventions when necessary
    """
    
    def __init__(self):
        self.intervention_history = deque(maxlen=100)
        self.active_interventions: Dict[str, Intervention] = {}
        self.user_preferences = self._load_preferences()
        self.response_patterns = {}
        self.escalation_rules = self._initialize_escalation_rules()
        
    def _load_preferences(self) -> Dict[str, Any]:
        """Load user preferences for interventions"""
        return {
            "intervention_style": "gentle",  # gentle, moderate, assertive
            "max_level": InterventionLevel.ASSERTIVE,
            "preferred_break_duration": 15,  # minutes
            "focus_protection_threshold": 0.8,
            "stress_intervention_threshold": 0.7,
            "notification_preferences": {
                "sound": True,
                "visual": True,
                "haptic": False
            }
        }
    
    def _initialize_escalation_rules(self) -> Dict[InterventionType, List[Tuple[InterventionLevel, float]]]:
        """Define escalation paths for each intervention type"""
        return {
            InterventionType.BREAK_REMINDER: [
                (InterventionLevel.AMBIENT, 5.0),
                (InterventionLevel.SUBTLE, 10.0),
                (InterventionLevel.SUGGESTIVE, 15.0),
                (InterventionLevel.ASSERTIVE, 30.0),
                (InterventionLevel.PROTECTIVE, 45.0)
            ],
            InterventionType.STRESS_RELIEF: [
                (InterventionLevel.SUBTLE, 5.0),
                (InterventionLevel.SUGGESTIVE, 10.0),
                (InterventionLevel.ASSERTIVE, 20.0),
                (InterventionLevel.PROTECTIVE, 30.0),
                (InterventionLevel.EMERGENCY, 45.0)
            ],
            InterventionType.FOCUS_PROTECTION: [
                (InterventionLevel.AMBIENT, 0.0),
                (InterventionLevel.PROTECTIVE, 0.0)  # Immediate protection
            ],
            InterventionType.CRISIS_RESPONSE: [
                (InterventionLevel.EMERGENCY, 0.0)  # Immediate response
            ]
        }
    
    async def evaluate_intervention_need(self, context: InterventionContext) -> Optional[Intervention]:
        """Evaluate if intervention is needed and at what level"""
        
        # Check each intervention type
        interventions_needed = []
        
        # Break reminder logic
        if context.time_since_break > 90:
            urgency = min((context.time_since_break - 90) / 60, 1.0)
            interventions_needed.append((InterventionType.BREAK_REMINDER, urgency))
        
        # Stress relief logic
        if context.stress_level > 0.7:
            urgency = (context.stress_level - 0.7) / 0.3
            interventions_needed.append((InterventionType.STRESS_RELIEF, urgency))
        
        # Focus protection logic
        if context.focus_level > 0.8 and context.current_activity == "deep_work":
            interventions_needed.append((InterventionType.FOCUS_PROTECTION, 0.9))
        
        # Crisis detection
        if context.stress_level > 0.9 or "panic" in context.emotional_state:
            interventions_needed.append((InterventionType.CRISIS_RESPONSE, 1.0))
        
        if not interventions_needed:
            return None
        
        # Select highest priority intervention
        intervention_type, urgency = max(interventions_needed, key=lambda x: x[1])
        
        # Determine appropriate level based on history and context
        level = await self._determine_intervention_level(intervention_type, context, urgency)
        
        # Create intervention
        return await self._create_intervention(level, intervention_type, context)
    
    async def _determine_intervention_level(
        self, 
        intervention_type: InterventionType, 
        context: InterventionContext,
        urgency: float
    ) -> InterventionLevel:
        """Determine appropriate intervention level"""
        
        # Check if there's an active intervention of this type
        active_key = f"{intervention_type.name}_{context.user_state}"
        if active_key in self.active_interventions:
            # Escalate from current level
            current = self.active_interventions[active_key]
            return await self._get_next_level(current.level, intervention_type)
        
        # Start with base level adjusted by urgency
        escalation_path = self.escalation_rules.get(intervention_type, [])
        if not escalation_path:
            return InterventionLevel.SUBTLE
        
        # Higher urgency starts at higher level
        if urgency > 0.8:
            start_index = min(2, len(escalation_path) - 1)
        elif urgency > 0.5:
            start_index = 1
        else:
            start_index = 0
        
        # Adjust based on user response history
        if context.ignored_count > 3:
            start_index = min(start_index + 1, len(escalation_path) - 1)
        
        return escalation_path[start_index][0]
    
    async def _get_next_level(
        self, 
        current_level: InterventionLevel, 
        intervention_type: InterventionType
    ) -> InterventionLevel:
        """Get next escalation level"""
        escalation_path = self.escalation_rules.get(intervention_type, [])
        
        # Find current position
        current_index = -1
        for i, (level, _) in enumerate(escalation_path):
            if level == current_level:
                current_index = i
                break
        
        # Move to next level
        if current_index >= 0 and current_index < len(escalation_path) - 1:
            return escalation_path[current_index + 1][0]
        
        return current_level  # Stay at max level
    
    async def _create_intervention(
        self,
        level: InterventionLevel,
        intervention_type: InterventionType,
        context: InterventionContext
    ) -> Intervention:
        """Create appropriate intervention based on level and type"""
        
        # Generate intervention content
        message, actions = await self._generate_intervention_content(level, intervention_type, context)
        
        # Set escalation timer
        escalation_timer = self._get_escalation_timer(level, intervention_type)
        
        intervention = Intervention(
            level=level,
            type=intervention_type,
            message=message,
            actions=actions,
            context=context,
            escalation_timer=escalation_timer
        )
        
        # Track active intervention
        active_key = f"{intervention_type.name}_{context.user_state}"
        self.active_interventions[active_key] = intervention
        
        return intervention
    
    async def _generate_intervention_content(
        self,
        level: InterventionLevel,
        intervention_type: InterventionType,
        context: InterventionContext
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Generate appropriate message and actions for intervention"""
        
        # Break reminder interventions
        if intervention_type == InterventionType.BREAK_REMINDER:
            if level == InterventionLevel.AMBIENT:
                return None, [
                    {"type": "adjust_lighting", "brightness": 0.8, "warmth": 0.7},
                    {"type": "play_sound", "sound": "gentle_chime", "volume": 0.3}
                ]
            elif level == InterventionLevel.SUBTLE:
                return "🌟", [
                    {"type": "show_notification", "style": "minimal", "duration": 3}
                ]
            elif level == InterventionLevel.SUGGESTIVE:
                return "You've been focused for a while. How about a quick stretch?", [
                    {"type": "show_notification", "style": "suggestion", "duration": 10},
                    {"type": "offer_timer", "duration": 5}
                ]
            elif level == InterventionLevel.ASSERTIVE:
                return f"You've been working for {context.time_since_break:.0f} minutes. You really should take a break.", [
                    {"type": "show_notification", "style": "important", "duration": 30},
                    {"type": "suggest_activity", "activities": ["walk", "water", "breathing"]},
                    {"type": "start_timer", "duration": 15}
                ]
            elif level == InterventionLevel.PROTECTIVE:
                return "Taking a mandatory break for your wellbeing.", [
                    {"type": "dim_screen", "level": 0.5},
                    {"type": "pause_notifications", "duration": 15},
                    {"type": "play_relaxation", "track": "nature_sounds"}
                ]
        
        # Stress relief interventions
        elif intervention_type == InterventionType.STRESS_RELIEF:
            if level == InterventionLevel.SUBTLE:
                return None, [
                    {"type": "adjust_music", "genre": "calming", "volume": 0.4},
                    {"type": "reduce_notification_frequency", "factor": 0.5}
                ]
            elif level == InterventionLevel.SUGGESTIVE:
                return "I notice you might be feeling stressed. Would you like to try a breathing exercise?", [
                    {"type": "offer_breathing_exercise", "duration": 3},
                    {"type": "suggest_playlist", "mood": "relaxing"}
                ]
            elif level == InterventionLevel.ASSERTIVE:
                return "Your stress levels are high. Let's take a moment to reset.", [
                    {"type": "start_breathing_guide", "pattern": "4-7-8"},
                    {"type": "block_stressful_notifications", "duration": 30}
                ]
            elif level == InterventionLevel.EMERGENCY:
                return "I'm here to help. Let's work through this together.", [
                    {"type": "activate_crisis_protocol"},
                    {"type": "offer_support_resources"},
                    {"type": "suggest_human_contact", "contacts": ["trusted_friend", "therapist"]}
                ]
        
        # Focus protection interventions
        elif intervention_type == InterventionType.FOCUS_PROTECTION:
            if level == InterventionLevel.AMBIENT:
                return None, [
                    {"type": "enable_focus_mode", "strength": "light"},
                    {"type": "play_focus_sounds", "type": "white_noise", "volume": 0.2}
                ]
            elif level == InterventionLevel.PROTECTIVE:
                return "Protecting your focus time", [
                    {"type": "block_all_notifications"},
                    {"type": "close_distracting_apps"},
                    {"type": "set_status", "message": "In deep focus - unavailable"}
                ]
        
        # Default fallback
        return f"Intervention needed: {intervention_type.name}", [
            {"type": "log_intervention", "level": level.name}
        ]
    
    def _get_escalation_timer(self, level: InterventionLevel, intervention_type: InterventionType) -> Optional[float]:
        """Get timer for escalation to next level"""
        escalation_path = self.escalation_rules.get(intervention_type, [])
        
        for i, (path_level, timer) in enumerate(escalation_path):
            if path_level == level and i < len(escalation_path) - 1:
                return timer
        
        return None
    
    async def execute_intervention(self, intervention: Intervention) -> Dict[str, Any]:
        """Execute the intervention actions"""
        results = {
            "intervention_id": f"{intervention.type.name}_{intervention.timestamp.isoformat()}",
            "level": intervention.level.name,
            "message_shown": False,
            "actions_executed": []
        }
        
        # Show message if provided
        if intervention.message:
            results["message_shown"] = True
            logger.info(f"Intervention message: {intervention.message}")
        
        # Execute each action
        for action in intervention.actions:
            try:
                result = await self._execute_action(action)
                results["actions_executed"].append({
                    "action": action["type"],
                    "success": result.get("success", True),
                    "details": result
                })
            except Exception as e:
                logger.error(f"Failed to execute action {action['type']}: {e}")
                results["actions_executed"].append({
                    "action": action["type"],
                    "success": False,
                    "error": str(e)
                })
        
        # Record in history
        self.intervention_history.append({
            "intervention": intervention,
            "results": results,
            "timestamp": datetime.now()
        })
        
        return results
    
    async def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific intervention action"""
        action_type = action.get("type")
        
        # Simulate different action types
        if action_type == "adjust_lighting":
            return {
                "success": True,
                "brightness_set": action.get("brightness"),
                "warmth_set": action.get("warmth")
            }
        elif action_type == "show_notification":
            return {
                "success": True,
                "notification_id": f"notif_{datetime.now().timestamp()}",
                "duration": action.get("duration")
            }
        elif action_type == "play_sound":
            return {
                "success": True,
                "sound": action.get("sound"),
                "volume": action.get("volume")
            }
        elif action_type == "start_breathing_guide":
            return {
                "success": True,
                "pattern": action.get("pattern"),
                "guide_started": True
            }
        elif action_type == "block_all_notifications":
            return {
                "success": True,
                "notifications_blocked": True,
                "previous_state_saved": True
            }
        else:
            return {
                "success": True,
                "action_type": action_type,
                "simulated": True
            }
    
    async def record_user_response(self, intervention_id: str, response: str) -> None:
        """Record how user responded to intervention"""
        # Update response patterns
        for key, intervention in self.active_interventions.items():
            if f"{intervention.type.name}_{intervention.timestamp.isoformat()}" == intervention_id:
                if response == "accepted":
                    intervention.context.accepted_count += 1
                elif response == "ignored":
                    intervention.context.ignored_count += 1
                elif response == "dismissed":
                    intervention.context.ignored_count += 1
                
                intervention.context.response_history.append(response)
                break
        
        # Learn from response
        await self._update_intervention_model(intervention_id, response)
    
    async def _update_intervention_model(self, intervention_id: str, response: str) -> None:
        """Update intervention model based on user response"""
        # This would connect to the learning system
        logger.info(f"Learning from response: {intervention_id} -> {response}")
    
    async def check_escalation_needed(self) -> List[Intervention]:
        """Check if any active interventions need escalation"""
        escalations_needed = []
        current_time = datetime.now()
        
        for key, intervention in list(self.active_interventions.items()):
            if intervention.escalation_timer:
                time_elapsed = (current_time - intervention.timestamp).total_seconds() / 60
                
                if time_elapsed >= intervention.escalation_timer:
                    # Create escalated intervention
                    new_level = await self._get_next_level(intervention.level, intervention.type)
                    
                    if new_level != intervention.level:
                        escalated = await self._create_intervention(
                            new_level,
                            intervention.type,
                            intervention.context
                        )
                        escalations_needed.append(escalated)
        
        return escalations_needed
    
    def get_intervention_analytics(self) -> Dict[str, Any]:
        """Get analytics on intervention effectiveness"""
        total_interventions = len(self.intervention_history)
        if total_interventions == 0:
            return {"message": "No interventions yet"}
        
        # Calculate metrics
        type_counts = {}
        level_counts = {}
        response_rates = {"accepted": 0, "ignored": 0, "dismissed": 0}
        
        for record in self.intervention_history:
            intervention = record["intervention"]
            type_counts[intervention.type.name] = type_counts.get(intervention.type.name, 0) + 1
            level_counts[intervention.level.name] = level_counts.get(intervention.level.name, 0) + 1
        
        return {
            "total_interventions": total_interventions,
            "by_type": type_counts,
            "by_level": level_counts,
            "response_rates": response_rates,
            "most_common_type": max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None,
            "average_escalation_level": sum(list(InterventionLevel).index(record["intervention"].level) 
                                          for record in self.intervention_history) / total_interventions
        }


# Testing functions
async def test_graduated_interventions():
    """Test the graduated intervention system"""
    system = GraduatedInterventionSystem()
    
    # Test 1: Break reminder escalation
    print("Test 1: Break Reminder Escalation")
    print("-" * 50)
    
    contexts = [
        InterventionContext(
            user_state="working",
            stress_level=0.4,
            focus_level=0.7,
            time_since_break=95,
            current_activity="coding",
            ignored_count=0
        ),
        InterventionContext(
            user_state="working",
            stress_level=0.4,
            focus_level=0.7,
            time_since_break=105,
            current_activity="coding",
            ignored_count=1
        ),
        InterventionContext(
            user_state="working",
            stress_level=0.4,
            focus_level=0.7,
            time_since_break=120,
            current_activity="coding",
            ignored_count=2
        )
    ]
    
    for i, context in enumerate(contexts):
        intervention = await system.evaluate_intervention_need(context)
        if intervention:
            print(f"\nIteration {i+1}:")
            print(f"Level: {intervention.level.name}")
            print(f"Message: {intervention.message}")
            result = await system.execute_intervention(intervention)
            print(f"Actions executed: {len(result['actions_executed'])}")
            
            # Simulate user ignoring
            await system.record_user_response(result["intervention_id"], "ignored")
    
    # Test 2: Stress intervention
    print("\n\nTest 2: Stress Intervention")
    print("-" * 50)
    
    stress_context = InterventionContext(
        user_state="working",
        stress_level=0.85,
        focus_level=0.3,
        time_since_break=45,
        current_activity="debugging",
        emotional_state="frustrated"
    )
    
    intervention = await system.evaluate_intervention_need(stress_context)
    if intervention:
        print(f"Level: {intervention.level.name}")
        print(f"Message: {intervention.message}")
        result = await system.execute_intervention(intervention)
        print(f"Actions: {[a['action'] for a in result['actions_executed']]}")
    
    # Test 3: Focus protection
    print("\n\nTest 3: Focus Protection")
    print("-" * 50)
    
    focus_context = InterventionContext(
        user_state="flow_state",
        stress_level=0.2,
        focus_level=0.9,
        time_since_break=30,
        current_activity="deep_work"
    )
    
    intervention = await system.evaluate_intervention_need(focus_context)
    if intervention:
        print(f"Level: {intervention.level.name}")
        print(f"Message: {intervention.message}")
        result = await system.execute_intervention(intervention)
        print(f"Actions: {[a['action'] for a in result['actions_executed']]}")
    
    # Show analytics
    print("\n\nIntervention Analytics")
    print("-" * 50)
    analytics = system.get_intervention_analytics()
    print(json.dumps(analytics, indent=2))


if __name__ == "__main__":
    asyncio.run(test_graduated_interventions())
