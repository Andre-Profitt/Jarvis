"""
JARVIS Phase 8: Intervention Preview System
==========================================
Shows users what JARVIS is about to do before it happens
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

class InterventionType(Enum):
    NOTIFICATION_BLOCK = "notification_block"
    BREAK_REMINDER = "break_reminder"
    FOCUS_MODE = "focus_mode"
    EMERGENCY_ACTION = "emergency_action"
    SUGGESTION = "suggestion"
    AUTOMATION = "automation"
    HEALTH_INTERVENTION = "health_intervention"
    PRODUCTIVITY_BOOST = "productivity_boost"

@dataclass
class InterventionPreview:
    """Preview of an upcoming intervention"""
    id: str
    type: InterventionType
    title: str
    description: str
    reason: str
    impact_level: str  # minimal, moderate, significant
    countdown_seconds: int
    can_cancel: bool = True
    can_delay: bool = True
    can_modify: bool = False
    visual_preview: Dict[str, Any] = field(default_factory=dict)
    alternative_actions: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.9
    user_preference_score: float = 0.8

class InterventionPreviewSystem:
    """Intelligent intervention preview system"""
    
    def __init__(self):
        self.pending_interventions = {}
        self.intervention_history = []
        self.user_responses = {}  # Track user reactions
        self.preview_settings = {
            'default_countdown': 5,
            'urgent_countdown': 2,
            'allow_silent_mode': True,
            'smart_timing': True
        }
        
    async def preview_intervention(self, intervention: Dict[str, Any]) -> InterventionPreview:
        """Create and show intervention preview"""
        
        # Create preview object
        preview = InterventionPreview(
            id=f"int_{datetime.now().timestamp()}",
            type=InterventionType(intervention.get('type', 'suggestion')),
            title=intervention.get('title', 'JARVIS Action'),
            description=intervention.get('description', ''),
            reason=self._generate_reason(intervention),
            impact_level=self._assess_impact(intervention),
            countdown_seconds=self._calculate_countdown(intervention),
            can_cancel=intervention.get('can_cancel', True),
            can_delay=intervention.get('can_delay', True),
            can_modify=intervention.get('can_modify', False),
            visual_preview=self._create_visual_preview(intervention),
            alternative_actions=self._suggest_alternatives(intervention),
            confidence=intervention.get('confidence', 0.9),
            user_preference_score=self._calculate_preference_score(intervention)
        )
        
        # Store pending intervention
        self.pending_interventions[preview.id] = {
            'preview': preview,
            'intervention': intervention,
            'created_at': datetime.now(),
            'status': 'pending'
        }
        
        # Start countdown
        asyncio.create_task(self._countdown_handler(preview.id))
        
        return preview
        
    def _generate_reason(self, intervention: Dict[str, Any]) -> str:
        """Generate human-friendly reason for intervention"""
        reasons = {
            'notification_block': "I noticed you're in deep focus. Blocking distractions to protect your flow state.",
            'break_reminder': "You've been working intensely for {duration}. A short break will boost your productivity.",
            'emergency_action': "Critical situation detected that requires immediate attention.",
            'health_intervention': "Your biometric data suggests you need {action} for optimal wellbeing.",
            'productivity_boost': "I can optimize your workflow by {optimization}.",
            'suggestion': "Based on your patterns, this might help: {suggestion}"
        }
        
        reason_template = reasons.get(
            intervention.get('type', 'suggestion'),
            "This action will help improve your {goal}."
        )
        
        # Fill in placeholders
        return reason_template.format(**intervention.get('context', {}))
        
    def _assess_impact(self, intervention: Dict[str, Any]) -> str:
        """Assess the impact level of intervention"""
        intervention_type = intervention.get('type', 'suggestion')
        urgency = intervention.get('urgency', 0.5)
        
        if intervention_type == 'emergency_action' or urgency > 0.9:
            return 'significant'
        elif intervention_type in ['notification_block', 'break_reminder'] or urgency > 0.6:
            return 'moderate'
        else:
            return 'minimal'
            
    def _calculate_countdown(self, intervention: Dict[str, Any]) -> int:
        """Calculate appropriate countdown duration"""
        impact = self._assess_impact(intervention)
        urgency = intervention.get('urgency', 0.5)
        
        if impact == 'significant' or urgency > 0.9:
            return self.preview_settings['urgent_countdown']
        elif impact == 'moderate':
            return self.preview_settings['default_countdown']
        else:
            return self.preview_settings['default_countdown'] + 2
            
    def _create_visual_preview(self, intervention: Dict[str, Any]) -> Dict[str, Any]:
        """Create visual representation of the intervention"""
        intervention_type = intervention.get('type', 'suggestion')
        
        visuals = {
            'notification_block': {
                'animation': 'shield-deploy',
                'icon': '🛡️',
                'color': '#3366ff',
                'preview_image': 'notification_blocking.png',
                'effect': 'fade-out-distractions'
            },
            'break_reminder': {
                'animation': 'gentle-wave',
                'icon': '☕',
                'color': '#00ff88',
                'preview_image': 'break_suggestion.png',
                'effect': 'soft-transition'
            },
            'emergency_action': {
                'animation': 'urgent-alert',
                'icon': '🚨',
                'color': '#ff3366',
                'preview_image': 'emergency_action.png',
                'effect': 'immediate-attention'
            },
            'health_intervention': {
                'animation': 'health-pulse',
                'icon': '❤️',
                'color': '#ff6699',
                'preview_image': 'health_check.png',
                'effect': 'wellness-glow'
            },
            'productivity_boost': {
                'animation': 'speed-lines',
                'icon': '⚡',
                'color': '#ffaa33',
                'preview_image': 'productivity_boost.png',
                'effect': 'acceleration'
            },
            'suggestion': {
                'animation': 'lightbulb-glow',
                'icon': '💡',
                'color': '#ffff33',
                'preview_image': 'suggestion.png',
                'effect': 'soft-highlight'
            }
        }
        
        return visuals.get(intervention_type, visuals['suggestion'])
        
    def _suggest_alternatives(self, intervention: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest alternative actions user can choose"""
        intervention_type = intervention.get('type', 'suggestion')
        
        alternatives = {
            'notification_block': [
                {'action': 'delay_15min', 'label': 'Delay 15 minutes'},
                {'action': 'allow_urgent_only', 'label': 'Allow urgent only'},
                {'action': 'customize_filters', 'label': 'Customize filters'}
            ],
            'break_reminder': [
                {'action': 'snooze_5min', 'label': 'Snooze 5 minutes'},
                {'action': 'quick_stretch', 'label': 'Quick stretch instead'},
                {'action': 'after_task', 'label': 'After current task'}
            ],
            'health_intervention': [
                {'action': 'remind_later', 'label': 'Remind in 30 min'},
                {'action': 'quick_version', 'label': 'Quick version'},
                {'action': 'track_only', 'label': 'Just track for now'}
            ]
        }
        
        return alternatives.get(intervention_type, [
            {'action': 'accept', 'label': 'Accept'},
            {'action': 'modify', 'label': 'Modify'},
            {'action': 'dismiss', 'label': 'Dismiss'}
        ])
        
    def _calculate_preference_score(self, intervention: Dict[str, Any]) -> float:
        """Calculate how well this aligns with user preferences"""
        intervention_type = intervention.get('type', 'suggestion')
        
        # Check historical responses
        type_history = [h for h in self.intervention_history 
                       if h.get('type') == intervention_type]
        
        if not type_history:
            return 0.8  # Default score for new types
            
        # Calculate acceptance rate
        accepted = sum(1 for h in type_history if h.get('accepted', False))
        acceptance_rate = accepted / len(type_history) if type_history else 0.8
        
        # Factor in timing preferences
        current_hour = datetime.now().hour
        timing_score = self._get_timing_preference_score(intervention_type, current_hour)
        
        # Combine scores
        return (acceptance_rate * 0.7) + (timing_score * 0.3)
        
    def _get_timing_preference_score(self, intervention_type: str, hour: int) -> float:
        """Get timing preference score based on hour"""
        # User-specific timing preferences (learned over time)
        timing_prefs = {
            'break_reminder': {
                'preferred_hours': [10, 14, 16],  # Mid-morning, after lunch, mid-afternoon
                'avoid_hours': [12, 13, 17, 18]    # Lunch, end of day
            },
            'notification_block': {
                'preferred_hours': list(range(9, 12)) + list(range(14, 17)),  # Work hours
                'avoid_hours': [8, 12, 13, 18, 19]  # Start/end of day, lunch
            }
        }
        
        prefs = timing_prefs.get(intervention_type, {})
        if hour in prefs.get('preferred_hours', []):
            return 1.0
        elif hour in prefs.get('avoid_hours', []):
            return 0.3
        else:
            return 0.7
            
    async def _countdown_handler(self, preview_id: str):
        """Handle countdown and execute intervention"""
        if preview_id not in self.pending_interventions:
            return
            
        pending = self.pending_interventions[preview_id]
        preview = pending['preview']
        
        # Countdown
        for remaining in range(preview.countdown_seconds, 0, -1):
            if pending['status'] != 'pending':
                break
                
            await self._broadcast_countdown_update(preview_id, remaining)
            await asyncio.sleep(1)
            
        # Execute if still pending
        if pending['status'] == 'pending':
            await self._execute_intervention(preview_id)
            
    async def _broadcast_countdown_update(self, preview_id: str, remaining: int):
        """Broadcast countdown update"""
        update = {
            'preview_id': preview_id,
            'remaining_seconds': remaining,
            'timestamp': datetime.now().isoformat()
        }
        print(f"Countdown: {remaining}s remaining for {preview_id}")
        
    async def _execute_intervention(self, preview_id: str):
        """Execute the intervention"""
        if preview_id not in self.pending_interventions:
            return
            
        pending = self.pending_interventions[preview_id]
        intervention = pending['intervention']
        
        # Mark as executed
        pending['status'] = 'executed'
        pending['executed_at'] = datetime.now()
        
        # Add to history
        self.intervention_history.append({
            'preview_id': preview_id,
            'type': intervention.get('type'),
            'executed_at': datetime.now(),
            'accepted': True,  # Auto-accepted after countdown
            'user_response': 'timeout'
        })
        
        print(f"Executing intervention: {intervention.get('title')}")
        
        # Execute actual intervention
        if 'execute_function' in intervention:
            await intervention['execute_function'](intervention.get('context', {}))
            
    async def cancel_intervention(self, preview_id: str):
        """Cancel a pending intervention"""
        if preview_id in self.pending_interventions:
            self.pending_interventions[preview_id]['status'] = 'cancelled'
            
            # Track cancellation
            self.intervention_history.append({
                'preview_id': preview_id,
                'type': self.pending_interventions[preview_id]['intervention'].get('type'),
                'cancelled_at': datetime.now(),
                'accepted': False,
                'user_response': 'cancelled'
            })
            
    async def delay_intervention(self, preview_id: str, delay_minutes: int = 15):
        """Delay an intervention"""
        if preview_id in self.pending_interventions:
            pending = self.pending_interventions[preview_id]
            pending['status'] = 'delayed'
            pending['resume_at'] = datetime.now() + timedelta(minutes=delay_minutes)
            
            # Schedule resumption
            asyncio.create_task(self._resume_after_delay(preview_id, delay_minutes))
            
    async def _resume_after_delay(self, preview_id: str, delay_minutes: int):
        """Resume intervention after delay"""
        await asyncio.sleep(delay_minutes * 60)
        
        if preview_id in self.pending_interventions:
            pending = self.pending_interventions[preview_id]
            if pending['status'] == 'delayed':
                # Re-preview the intervention
                await self.preview_intervention(pending['intervention'])
