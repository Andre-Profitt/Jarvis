"""
JARVIS Phase 8: Visual State Indicator System
============================================
Beautiful, intuitive visual indicators for system state
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

@dataclass
class VisualIndicator:
    """Visual indicator configuration"""
    name: str
    icon: str
    color: str
    animation: str
    position: Tuple[str, str]  # (vertical, horizontal)
    size: str
    opacity: float = 1.0
    pulse: bool = False
    glow: bool = False

class VisualStateIndicatorSystem:
    """Advanced visual state indicator system"""
    
    def __init__(self):
        self.active_indicators = {}
        self.state_colors = {
            'flow': '#00ff88',      # Vibrant green
            'crisis': '#ff3366',    # Alert red
            'focus': '#3366ff',     # Deep blue
            'rest': '#9966ff',      # Soft purple
            'social': '#ffaa33',    # Warm orange
            'creative': '#ff66ff',  # Creative pink
            'learning': '#33ffff',  # Cyan
            'planning': '#ffff33'   # Yellow
        }
        
        self.mode_indicators = {
            'adaptive': VisualIndicator(
                name="Adaptive Mode",
                icon="🎯",
                color="#00ff88",
                animation="smooth-pulse",
                position=("top", "right"),
                size="medium",
                pulse=True
            ),
            'focused': VisualIndicator(
                name="Focus Mode",
                icon="🧘",
                color="#3366ff",
                animation="gentle-glow",
                position=("top", "right"),
                size="medium",
                glow=True
            ),
            'supportive': VisualIndicator(
                name="Support Mode",
                icon="🤝",
                color="#ffaa33",
                animation="warm-pulse",
                position=("top", "right"),
                size="medium",
                pulse=True
            ),
            'emergency': VisualIndicator(
                name="Emergency Mode",
                icon="🚨",
                color="#ff3366",
                animation="urgent-flash",
                position=("top", "right"),
                size="large",
                pulse=True,
                glow=True
            ),
            'creative': VisualIndicator(
                name="Creative Mode",
                icon="🎨",
                color="#ff66ff",
                animation="rainbow-shift",
                position=("top", "right"),
                size="medium",
                glow=True
            ),
            'minimal': VisualIndicator(
                name="Minimal Mode",
                icon="🔇",
                color="#666666",
                animation="subtle-fade",
                position=("top", "right"),
                size="small",
                opacity=0.5
            )
        }
        
        self.status_indicators = {
            'monitoring': {
                'voice': {'icon': '🎤', 'active': False},
                'biometric': {'icon': '❤️', 'active': False},
                'vision': {'icon': '👁️', 'active': False},
                'environment': {'icon': '🌐', 'active': False},
                'temporal': {'icon': '⏰', 'active': False}
            }
        }
        
    async def update_state_indicator(self, state: str, intensity: float = 1.0):
        """Update the main state indicator"""
        color = self._interpolate_color(self.state_colors.get(state, '#ffffff'), intensity)
        
        indicator = {
            'state': state,
            'color': color,
            'intensity': intensity,
            'timestamp': datetime.now(),
            'animation': self._get_state_animation(state, intensity)
        }
        
        self.active_indicators['main_state'] = indicator
        await self._broadcast_update('state', indicator)
        
    async def show_mode_indicator(self, mode: str):
        """Display mode indicator"""
        if mode in self.mode_indicators:
            indicator = self.mode_indicators[mode]
            self.active_indicators['current_mode'] = indicator
            await self._broadcast_update('mode', indicator)
            
    async def update_monitoring_status(self, modality: str, active: bool):
        """Update monitoring status indicators"""
        if modality in self.status_indicators['monitoring']:
            self.status_indicators['monitoring'][modality]['active'] = active
            await self._broadcast_update('monitoring', self.status_indicators['monitoring'])
            
    async def show_intervention_preview(self, intervention: Dict[str, Any]):
        """Show preview of upcoming intervention"""
        preview = {
            'type': intervention.get('type', 'unknown'),
            'description': intervention.get('description', ''),
            'intensity': intervention.get('intensity', 0.5),
            'countdown': intervention.get('countdown', 5),
            'can_cancel': intervention.get('can_cancel', True),
            'visual_hint': self._get_intervention_visual(intervention)
        }
        
        await self._broadcast_update('intervention_preview', preview)
        
    def _interpolate_color(self, base_color: str, intensity: float) -> str:
        """Interpolate color based on intensity"""
        # Convert hex to RGB
        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)
        
        # Adjust brightness based on intensity
        r = int(r * intensity)
        g = int(g * intensity)
        b = int(b * intensity)
        
        # Convert back to hex
        return f"#{r:02x}{g:02x}{b:02x}"
        
    def _get_state_animation(self, state: str, intensity: float) -> str:
        """Get appropriate animation for state"""
        animations = {
            'flow': 'smooth-wave',
            'crisis': 'urgent-pulse',
            'focus': 'steady-glow',
            'rest': 'gentle-fade',
            'social': 'warm-bounce',
            'creative': 'color-shift',
            'learning': 'expand-contract',
            'planning': 'rotate-slow'
        }
        
        base_animation = animations.get(state, 'default-pulse')
        speed = 'fast' if intensity > 0.8 else 'normal' if intensity > 0.5 else 'slow'
        
        return f"{base_animation}-{speed}"
        
    def _get_intervention_visual(self, intervention: Dict[str, Any]) -> Dict[str, Any]:
        """Get visual representation for intervention"""
        intervention_type = intervention.get('type', 'unknown')
        
        visuals = {
            'notification_block': {
                'icon': '🔕',
                'color': '#3366ff',
                'animation': 'fade-in'
            },
            'break_reminder': {
                'icon': '☕',
                'color': '#00ff88',
                'animation': 'gentle-bounce'
            },
            'emergency_action': {
                'icon': '🚨',
                'color': '#ff3366',
                'animation': 'urgent-flash'
            },
            'suggestion': {
                'icon': '💡',
                'color': '#ffaa33',
                'animation': 'soft-glow'
            }
        }
        
        return visuals.get(intervention_type, {
            'icon': '❓',
            'color': '#888888',
            'animation': 'default'
        })
        
    async def _broadcast_update(self, update_type: str, data: Any):
        """Broadcast visual update to all displays"""
        update_message = {
            'type': update_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        # In real implementation, this would send to WebSocket clients
        # For now, we'll just log it
        print(f"Visual Update: {update_type} - {data}")
        
    def get_current_display_state(self) -> Dict[str, Any]:
        """Get complete current display state"""
        return {
            'indicators': self.active_indicators,
            'monitoring': self.status_indicators,
            'theme': self._get_current_theme(),
            'layout': self._get_optimal_layout()
        }
        
    def _get_current_theme(self) -> Dict[str, Any]:
        """Get current visual theme based on state"""
        main_state = self.active_indicators.get('main_state', {})
        state_name = main_state.get('state', 'default')
        
        themes = {
            'flow': {
                'primary': '#00ff88',
                'secondary': '#00cc66',
                'background': '#001a0d',
                'accent': '#33ffaa'
            },
            'crisis': {
                'primary': '#ff3366',
                'secondary': '#cc0033',
                'background': '#1a0009',
                'accent': '#ff6699'
            },
            'focus': {
                'primary': '#3366ff',
                'secondary': '#0033cc',
                'background': '#000d1a',
                'accent': '#6699ff'
            },
            'default': {
                'primary': '#00ff88',
                'secondary': '#0099cc',
                'background': '#0a0a0a',
                'accent': '#00ffcc'
            }
        }
        
        return themes.get(state_name, themes['default'])
        
    def _get_optimal_layout(self) -> Dict[str, Any]:
        """Get optimal layout based on active indicators"""
        active_count = len(self.active_indicators)
        
        if active_count <= 2:
            return {'style': 'minimal', 'density': 'sparse'}
        elif active_count <= 5:
            return {'style': 'balanced', 'density': 'medium'}
        else:
            return {'style': 'dashboard', 'density': 'dense'}
