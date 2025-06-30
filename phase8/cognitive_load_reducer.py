"""
JARVIS Phase 8: Cognitive Load Reducer
======================================
Dynamically adjusts interface complexity based on user state
"""

import asyncio
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from enum import Enum

class CognitiveLoadLevel(Enum):
    MINIMAL = "minimal"      # User is relaxed, can handle full interface
    LOW = "low"             # Normal state
    MODERATE = "moderate"   # Some stress/fatigue
    HIGH = "high"          # Significant cognitive load
    OVERLOAD = "overload"  # Critical - minimize everything

@dataclass
class InterfaceElement:
    """Represents a UI element that can be shown/hidden"""
    id: str
    name: str
    category: str
    priority: int  # 1-10, 10 being most important
    cognitive_cost: float  # 0-1, how much mental effort it requires
    current_state: str = "visible"  # visible, hidden, minimized
    dependencies: List[str] = None

class CognitiveLoadReducer:
    """Intelligently reduces cognitive load through UI adaptation"""
    
    def __init__(self):
        self.current_load_level = CognitiveLoadLevel.LOW
        self.interface_elements = self._initialize_interface_elements()
        self.user_preferences = {}
        self.adaptation_history = []
        self.smart_summarizer = SmartSummarizer()
        self.progressive_disclosure = ProgressiveDisclosure()
        
    def _initialize_interface_elements(self) -> Dict[str, InterfaceElement]:
        """Initialize all interface elements"""
        return {
            # Core elements (always visible)
            'main_display': InterfaceElement(
                id='main_display',
                name='Main Display',
                category='core',
                priority=10,
                cognitive_cost=0.1
            ),
            'critical_alerts': InterfaceElement(
                id='critical_alerts',
                name='Critical Alerts',
                category='core',
                priority=10,
                cognitive_cost=0.2
            ),
            
            # Status elements
            'detailed_metrics': InterfaceElement(
                id='detailed_metrics',
                name='Detailed Metrics',
                category='status',
                priority=5,
                cognitive_cost=0.6
            ),
            'activity_timeline': InterfaceElement(
                id='activity_timeline',
                name='Activity Timeline',
                category='status',
                priority=4,
                cognitive_cost=0.5
            ),
            'sensor_readings': InterfaceElement(
                id='sensor_readings',
                name='Sensor Readings',
                category='status',
                priority=3,
                cognitive_cost=0.4
            ),
            
            # Advanced features
            'predictions_panel': InterfaceElement(
                id='predictions_panel',
                name='Predictions Panel',
                category='advanced',
                priority=6,
                cognitive_cost=0.7
            ),
            'optimization_suggestions': InterfaceElement(
                id='optimization_suggestions',
                name='Optimization Suggestions',
                category='advanced',
                priority=5,
                cognitive_cost=0.6
            ),
            'pattern_analysis': InterfaceElement(
                id='pattern_analysis',
                name='Pattern Analysis',
                category='advanced',
                priority=4,
                cognitive_cost=0.8
            ),
            
            # Ambient elements
            'background_animations': InterfaceElement(
                id='background_animations',
                name='Background Animations',
                category='ambient',
                priority=2,
                cognitive_cost=0.3
            ),
            'decorative_elements': InterfaceElement(
                id='decorative_elements',
                name='Decorative Elements',
                category='ambient',
                priority=1,
                cognitive_cost=0.2
            )
        }
        
    async def assess_cognitive_load(self, user_state: Dict[str, Any]) -> CognitiveLoadLevel:
        """Assess current cognitive load from user state"""
        
        # Extract relevant metrics
        stress_level = user_state.get('stress_level', 0.3)
        focus_level = user_state.get('focus_level', 0.7)
        fatigue_level = user_state.get('fatigue_level', 0.3)
        task_complexity = user_state.get('task_complexity', 0.5)
        multitasking_level = user_state.get('multitasking_level', 0.3)
        
        # Calculate composite cognitive load
        cognitive_load = (
            stress_level * 0.3 +
            (1 - focus_level) * 0.2 +
            fatigue_level * 0.2 +
            task_complexity * 0.2 +
            multitasking_level * 0.1
        )
        
        # Map to load level
        if cognitive_load < 0.2:
            return CognitiveLoadLevel.MINIMAL
        elif cognitive_load < 0.4:
            return CognitiveLoadLevel.LOW
        elif cognitive_load < 0.6:
            return CognitiveLoadLevel.MODERATE
        elif cognitive_load < 0.8:
            return CognitiveLoadLevel.HIGH
        else:
            return CognitiveLoadLevel.OVERLOAD
            
    async def adapt_interface(self, cognitive_load: CognitiveLoadLevel) -> Dict[str, Any]:
        """Adapt interface based on cognitive load"""
        
        self.current_load_level = cognitive_load
        
        # Define visibility rules for each load level
        visibility_rules = {
            CognitiveLoadLevel.MINIMAL: {
                'show_all': True,
                'enhancements': ['rich_visualizations', 'advanced_analytics']
            },
            CognitiveLoadLevel.LOW: {
                'hide_categories': [],
                'minimize_categories': [],
                'cognitive_budget': 0.9
            },
            CognitiveLoadLevel.MODERATE: {
                'hide_categories': ['ambient'],
                'minimize_categories': ['advanced'],
                'cognitive_budget': 0.6
            },
            CognitiveLoadLevel.HIGH: {
                'hide_categories': ['ambient', 'advanced'],
                'minimize_categories': ['status'],
                'cognitive_budget': 0.3
            },
            CognitiveLoadLevel.OVERLOAD: {
                'hide_categories': ['ambient', 'advanced', 'status'],
                'minimize_categories': [],
                'cognitive_budget': 0.1,
                'emergency_mode': True
            }
        }
        
        rules = visibility_rules[cognitive_load]
        adaptations = []
        
        # Apply visibility rules
        if rules.get('show_all'):
            for element in self.interface_elements.values():
                element.current_state = 'visible'
                adaptations.append({
                    'element': element.id,
                    'action': 'show',
                    'reason': 'Low cognitive load - showing all elements'
                })
        else:
            # Hide or minimize based on categories
            for element in self.interface_elements.values():
                if element.category in rules.get('hide_categories', []):
                    element.current_state = 'hidden'
                    adaptations.append({
                        'element': element.id,
                        'action': 'hide',
                        'reason': f'Reducing cognitive load - hiding {element.category}'
                    })
                elif element.category in rules.get('minimize_categories', []):
                    element.current_state = 'minimized'
                    adaptations.append({
                        'element': element.id,
                        'action': 'minimize',
                        'reason': f'Reducing cognitive load - minimizing {element.category}'
                    })
                    
        # Apply cognitive budget optimization
        if 'cognitive_budget' in rules:
            adaptations.extend(
                await self._optimize_within_budget(rules['cognitive_budget'])
            )
            
        # Emergency mode
        if rules.get('emergency_mode'):
            adaptations.extend(await self._activate_emergency_mode())
            
        # Record adaptation
        self.adaptation_history.append({
            'timestamp': datetime.now(),
            'cognitive_load': cognitive_load,
            'adaptations': adaptations
        })
        
        return {
            'cognitive_load': cognitive_load,
            'adaptations': adaptations,
            'visible_elements': [e.id for e in self.interface_elements.values() 
                               if e.current_state == 'visible'],
            'interface_mode': self._get_interface_mode()
        }
        
    async def _optimize_within_budget(self, budget: float) -> List[Dict[str, Any]]:
        """Optimize interface within cognitive budget"""
        adaptations = []
        current_cost = sum(
            e.cognitive_cost for e in self.interface_elements.values()
            if e.current_state == 'visible'
        )
        
        # If over budget, hide lowest priority elements
        if current_cost > budget:
            # Sort by priority (ascending) to hide low priority first
            sorted_elements = sorted(
                [e for e in self.interface_elements.values() if e.current_state == 'visible'],
                key=lambda x: x.priority
            )
            
            for element in sorted_elements:
                if current_cost <= budget:
                    break
                    
                element.current_state = 'hidden'
                current_cost -= element.cognitive_cost
                adaptations.append({
                    'element': element.id,
                    'action': 'hide',
                    'reason': f'Optimizing cognitive budget (saved {element.cognitive_cost:.2f})'
                })
                
        return adaptations
        
    async def _activate_emergency_mode(self) -> List[Dict[str, Any]]:
        """Activate emergency mode - maximum simplification"""
        adaptations = []
        
        # Keep only critical elements
        for element in self.interface_elements.values():
            if element.priority < 10:
                element.current_state = 'hidden'
                adaptations.append({
                    'element': element.id,
                    'action': 'hide',
                    'reason': 'Emergency mode - showing only critical elements'
                })
                
        # Additional emergency adaptations
        adaptations.extend([
            {
                'element': 'color_scheme',
                'action': 'simplify',
                'reason': 'Emergency mode - high contrast colors only'
            },
            {
                'element': 'animations',
                'action': 'disable',
                'reason': 'Emergency mode - disabling all animations'
            },
            {
                'element': 'notifications',
                'action': 'critical_only',
                'reason': 'Emergency mode - critical notifications only'
            }
        ])
        
        return adaptations
        
    def _get_interface_mode(self) -> str:
        """Get current interface mode description"""
        modes = {
            CognitiveLoadLevel.MINIMAL: "Full Experience",
            CognitiveLoadLevel.LOW: "Standard",
            CognitiveLoadLevel.MODERATE: "Simplified",
            CognitiveLoadLevel.HIGH: "Minimal",
            CognitiveLoadLevel.OVERLOAD: "Emergency"
        }
        return modes.get(self.current_load_level, "Unknown")


class SmartSummarizer:
    """Intelligently summarizes information based on cognitive load"""
    
    async def summarize(self, data: Dict[str, Any], load_level: CognitiveLoadLevel) -> Dict[str, Any]:
        """Summarize data appropriately for cognitive load level"""
        
        if load_level == CognitiveLoadLevel.MINIMAL:
            # Full detail
            return data
            
        elif load_level == CognitiveLoadLevel.LOW:
            # Slight summarization
            return self._moderate_summary(data)
            
        elif load_level == CognitiveLoadLevel.MODERATE:
            # Key points only
            return self._key_points_summary(data)
            
        elif load_level == CognitiveLoadLevel.HIGH:
            # Essential info only
            return self._essential_summary(data)
            
        else:  # OVERLOAD
            # Absolute minimum
            return self._crisis_summary(data)
            
    def _moderate_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Moderate summarization"""
        return {
            'summary': data.get('summary', 'No summary available'),
            'key_metrics': data.get('key_metrics', {}),
            'recommendations': data.get('recommendations', [])[:3]
        }
        
    def _key_points_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Key points only"""
        return {
            'main_point': data.get('summary', 'No summary available')[:100],
            'action_needed': data.get('action_required', False),
            'top_metric': self._get_top_metric(data.get('key_metrics', {}))
        }
        
    def _essential_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Essential information only"""
        return {
            'status': 'OK' if not data.get('action_required') else 'ACTION NEEDED',
            'detail': data.get('summary', '')[:50]
        }
        
    def _crisis_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Crisis mode - absolute minimum"""
        if data.get('critical'):
            return {'ALERT': data.get('critical_message', 'CHECK SYSTEM')}
        return {'status': 'monitoring'}
        
    def _get_top_metric(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Get most important metric"""
        if not metrics:
            return {}
        # Logic to determine most important metric
        return {list(metrics.keys())[0]: list(metrics.values())[0]}


class ProgressiveDisclosure:
    """Manages progressive disclosure of information"""
    
    def __init__(self):
        self.disclosure_levels = {
            'basic': 1,
            'intermediate': 2,
            'advanced': 3,
            'expert': 4
        }
        
    async def get_appropriate_level(self, user_expertise: float, 
                                  cognitive_load: CognitiveLoadLevel) -> str:
        """Determine appropriate disclosure level"""
        
        # Adjust expertise based on cognitive load
        effective_expertise = user_expertise
        
        if cognitive_load == CognitiveLoadLevel.HIGH:
            effective_expertise *= 0.5
        elif cognitive_load == CognitiveLoadLevel.OVERLOAD:
            effective_expertise *= 0.2
            
        # Map to disclosure level
        if effective_expertise < 0.25:
            return 'basic'
        elif effective_expertise < 0.5:
            return 'intermediate'
        elif effective_expertise < 0.75:
            return 'advanced'
        else:
            return 'expert'
