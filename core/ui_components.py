"""
JARVIS Phase 7: Visual UI Components
===================================
Modern, responsive UI components for JARVIS visual feedback
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class UITheme(Enum):
    """UI Theme variants"""
    DARK = "dark"
    LIGHT = "light"
    AUTO = "auto"
    HIGH_CONTRAST = "high_contrast"

class StatusIndicator(Enum):
    """Status indicator states"""
    ACTIVE = "active"
    IDLE = "idle"
    PROCESSING = "processing"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"

class JARVISUIComponents:
    """UI component generator for JARVIS visual feedback"""
    
    def __init__(self, theme: UITheme = UITheme.DARK):
        self.theme = theme
        self.color_schemes = self._initialize_color_schemes()
        self.icon_library = self._initialize_icons()
        self.animation_presets = self._initialize_animations()
        
    def _initialize_color_schemes(self) -> Dict:
        """Initialize color schemes for different themes"""
        return {
            UITheme.DARK: {
                "primary": "#4fc3f7",
                "secondary": "#667eea",
                "accent": "#f093fb",
                "success": "#4ade80",
                "warning": "#fbbf24",
                "error": "#f87171",
                "background": "#0a0e27",
                "surface": "rgba(255, 255, 255, 0.05)",
                "text": "#e0e0e0",
                "text_secondary": "#9ca3af"
            },
            UITheme.LIGHT: {
                "primary": "#2196f3",
                "secondary": "#5e35b1",
                "accent": "#e91e63",
                "success": "#4caf50",
                "warning": "#ff9800",
                "error": "#f44336",
                "background": "#f5f5f5",
                "surface": "rgba(0, 0, 0, 0.05)",
                "text": "#212121",
                "text_secondary": "#757575"
            }
        }
        
    def _initialize_icons(self) -> Dict:
        """Initialize SVG icons for different sensors and states"""
        return {
            "voice": '<svg viewBox="0 0 24 24"><path d="M12 15c1.66 0 3-1.34 3-3V6c0-1.66-1.34-3-3-3S9 4.34 9 6v6c0 1.66 1.34 3 3 3zm-1-9c0-.55.45-1 1-1s1 .45 1 1v6c0 .55-.45 1-1 1s-1-.45-1-1V6zm6 6c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>',
            
            "biometric": '<svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/></svg>',
            
            "vision": '<svg viewBox="0 0 24 24"><path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/></svg>',
            
            "context": '<svg viewBox="0 0 24 24"><path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/></svg>',
            
            "emotional": '<svg viewBox="0 0 24 24"><path d="M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm3.5-9c.83 0 1.5-.67 1.5-1.5S16.33 8 15.5 8 14 8.67 14 9.5s.67 1.5 1.5 1.5zm-7 0c.83 0 1.5-.67 1.5-1.5S9.33 8 8.5 8 7 8.67 7 9.5 7.67 11 8.5 11zm3.5 6.5c2.33 0 4.31-1.46 5.11-3.5H6.89c.8 2.04 2.78 3.5 5.11 3.5z"/></svg>',
            
            "flow_state": '<svg viewBox="0 0 24 24"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/></svg>',
            
            "notification": '<svg viewBox="0 0 24 24"><path d="M12 22c1.1 0 2-.9 2-2h-4c0 1.1.9 2 2 2zm6-6v-5c0-3.07-1.64-5.64-4.5-6.32V4c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5v.68C7.63 5.36 6 7.92 6 11v5l-2 2v1h16v-1l-2-2z"/></svg>',
            
            "warning": '<svg viewBox="0 0 24 24"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>',
            
            "success": '<svg viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>',
            
            "processing": '<svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>'
        }
        
    def _initialize_animations(self) -> Dict:
        """Initialize CSS animation presets"""
        return {
            "pulse": """
                @keyframes pulse {
                    0% { transform: scale(1); opacity: 1; }
                    50% { transform: scale(1.1); opacity: 0.8; }
                    100% { transform: scale(1); opacity: 1; }
                }
            """,
            "spin": """
                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
            """,
            "fade_in": """
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
            """,
            "slide_in": """
                @keyframes slideIn {
                    from { transform: translateX(-100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
            """,
            "bounce": """
                @keyframes bounce {
                    0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
                    40% { transform: translateY(-10px); }
                    60% { transform: translateY(-5px); }
                }
            """,
            "ripple": """
                @keyframes ripple {
                    0% { transform: scale(0); opacity: 1; }
                    100% { transform: scale(4); opacity: 0; }
                }
            """
        }
        
    def generate_status_indicator(self, 
                                sensor_type: str,
                                status: StatusIndicator,
                                label: Optional[str] = None) -> str:
        """Generate a status indicator component"""
        colors = self.color_schemes[self.theme]
        icon = self.icon_library.get(sensor_type, self.icon_library["processing"])
        
        status_colors = {
            StatusIndicator.ACTIVE: colors["success"],
            StatusIndicator.IDLE: colors["text_secondary"],
            StatusIndicator.PROCESSING: colors["primary"],
            StatusIndicator.WARNING: colors["warning"],
            StatusIndicator.ERROR: colors["error"],
            StatusIndicator.SUCCESS: colors["success"]
        }
        
        animation = ""
        if status == StatusIndicator.PROCESSING:
            animation = "animation: spin 2s linear infinite;"
        elif status == StatusIndicator.ACTIVE:
            animation = "animation: pulse 2s ease-in-out infinite;"
            
        return f"""
        <div class="status-indicator" data-sensor="{sensor_type}" data-status="{status.value}">
            <div class="status-icon" style="color: {status_colors[status]}; {animation}">
                {icon}
            </div>
            {f'<span class="status-label">{label}</span>' if label else ''}
        </div>
        """
        
    def generate_intervention_preview(self,
                                    action: str,
                                    description: str,
                                    countdown: int = 3,
                                    can_cancel: bool = True) -> str:
        """Generate an intervention preview notification"""
        colors = self.color_schemes[self.theme]
        
        return f"""
        <div class="intervention-preview" data-action="{action}">
            <div class="intervention-header">
                <div class="intervention-icon">
                    {self.icon_library["notification"]}
                </div>
                <div class="intervention-title">JARVIS Action</div>
                {f'<button class="intervention-cancel" onclick="cancelIntervention(\'{action}\')">×</button>' if can_cancel else ''}
            </div>
            <div class="intervention-content">
                <p>{description}</p>
                <div class="intervention-countdown" data-countdown="{countdown}">
                    Starting in <span class="countdown-value">{countdown}</span>s...
                </div>
            </div>
            <div class="intervention-progress">
                <div class="progress-bar" style="animation: progress {countdown}s linear;"></div>
            </div>
        </div>
        """
        
    def generate_mode_indicator(self,
                              current_mode: str,
                              state_info: Dict) -> str:
        """Generate a mode indicator showing current JARVIS state"""
        colors = self.color_schemes[self.theme]
        
        mode_configs = {
            "flow": {
                "color": colors["success"],
                "icon": "flow_state",
                "label": "Flow State",
                "description": "Deep focus mode - minimal interruptions"
            },
            "crisis": {
                "color": colors["error"],
                "icon": "warning",
                "label": "Crisis Support",
                "description": "Emergency support mode active"
            },
            "normal": {
                "color": colors["primary"],
                "icon": "processing",
                "label": "Normal",
                "description": "Standard operation mode"
            },
            "rest": {
                "color": colors["text_secondary"],
                "icon": "emotional",
                "label": "Rest Mode",
                "description": "Low activity - encouraging rest"
            }
        }
        
        config = mode_configs.get(current_mode, mode_configs["normal"])
        
        return f"""
        <div class="mode-indicator" data-mode="{current_mode}">
            <div class="mode-icon" style="color: {config['color']}">
                {self.icon_library[config['icon']]}
            </div>
            <div class="mode-info">
                <div class="mode-label">{config['label']}</div>
                <div class="mode-description">{config['description']}</div>
            </div>
            <div class="mode-metrics">
                <span class="metric">Stress: {state_info.get('stress_level', 0):.1%}</span>
                <span class="metric">Focus: {state_info.get('focus_level', 0):.1%}</span>
            </div>
        </div>
        """
        
    def generate_sensor_dashboard(self, active_sensors: List[Dict]) -> str:
        """Generate a complete sensor status dashboard"""
        colors = self.color_schemes[self.theme]
        
        sensor_html = ""
        for sensor in active_sensors:
            sensor_html += self.generate_status_indicator(
                sensor["type"],
                StatusIndicator(sensor["status"]),
                sensor.get("label")
            )
            
        return f"""
        <div class="sensor-dashboard">
            <div class="dashboard-header">
                <h3>Active Monitoring</h3>
                <div class="dashboard-status">
                    {len([s for s in active_sensors if s['status'] == 'active'])} active
                </div>
            </div>
            <div class="sensor-grid">
                {sensor_html}
            </div>
        </div>
        """
        
    def generate_emotional_state_visualizer(self, emotional_state: Dict) -> str:
        """Generate emotional state visualization"""
        colors = self.color_schemes[self.theme]
        
        valence = emotional_state.get("valence", 0)
        arousal = emotional_state.get("arousal", 0)
        
        # Map to position (convert -1 to 1 range to 0 to 100%)
        x_pos = (valence + 1) * 50
        y_pos = (1 - arousal) * 50  # Invert Y axis
        
        return f"""
        <div class="emotional-visualizer">
            <div class="emotion-graph">
                <div class="emotion-axes">
                    <span class="axis-label top">High Arousal</span>
                    <span class="axis-label right">Positive</span>
                    <span class="axis-label bottom">Low Arousal</span>
                    <span class="axis-label left">Negative</span>
                </div>
                <div class="emotion-point" style="left: {x_pos}%; top: {y_pos}%;">
                    <div class="point-ripple"></div>
                </div>
            </div>
            <div class="emotion-label">{emotional_state.get('quadrant', 'neutral')}</div>
        </div>
        """
        
    def generate_notification_toast(self,
                                  message: str,
                                  type: str = "info",
                                  duration: int = 5000,
                                  action: Optional[Dict] = None) -> str:
        """Generate a notification toast"""
        colors = self.color_schemes[self.theme]
        
        type_configs = {
            "info": {"color": colors["primary"], "icon": "notification"},
            "success": {"color": colors["success"], "icon": "success"},
            "warning": {"color": colors["warning"], "icon": "warning"},
            "error": {"color": colors["error"], "icon": "warning"}
        }
        
        config = type_configs.get(type, type_configs["info"])
        
        action_html = ""
        if action:
            action_html = f"""
            <button class="toast-action" onclick="{action['callback']}">
                {action['label']}
            </button>
            """
            
        return f"""
        <div class="notification-toast toast-{type}" data-duration="{duration}">
            <div class="toast-icon" style="color: {config['color']}">
                {self.icon_library[config['icon']]}
            </div>
            <div class="toast-content">
                <p>{message}</p>
            </div>
            {action_html}
            <button class="toast-close" onclick="dismissToast(this)">×</button>
        </div>
        """
        
    def generate_activity_timeline(self, activities: List[Dict]) -> str:
        """Generate an activity timeline"""
        timeline_html = ""
        
        for activity in activities[-10:]:  # Show last 10 activities
            time = activity.get("timestamp", datetime.now()).strftime("%H:%M")
            timeline_html += f"""
            <div class="timeline-item">
                <div class="timeline-time">{time}</div>
                <div class="timeline-dot" style="background: {self._get_activity_color(activity['type'])}"></div>
                <div class="timeline-content">
                    <div class="timeline-title">{activity['title']}</div>
                    <div class="timeline-description">{activity.get('description', '')}</div>
                </div>
            </div>
            """
            
        return f"""
        <div class="activity-timeline">
            <h3>Recent Activity</h3>
            <div class="timeline-container">
                {timeline_html}
            </div>
        </div>
        """
        
    def _get_activity_color(self, activity_type: str) -> str:
        """Get color for activity type"""
        colors = self.color_schemes[self.theme]
        type_colors = {
            "voice": colors["primary"],
            "biometric": colors["success"],
            "intervention": colors["warning"],
            "emotional": colors["accent"],
            "system": colors["text_secondary"]
        }
        return type_colors.get(activity_type, colors["text_secondary"])
        
    def generate_style_sheet(self) -> str:
        """Generate complete stylesheet for UI components"""
        colors = self.color_schemes[self.theme]
        
        return f"""
        <style>
        /* Base styles */
        .jarvis-ui {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: {colors['text']};
            background: {colors['background']};
        }}
        
        /* Status Indicators */
        .status-indicator {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            background: {colors['surface']};
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }}
        
        .status-indicator:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }}
        
        .status-icon {{
            width: 24px;
            height: 24px;
        }}
        
        .status-icon svg {{
            width: 100%;
            height: 100%;
            fill: currentColor;
        }}
        
        .status-label {{
            font-size: 0.875rem;
            color: {colors['text_secondary']};
        }}
        
        /* Intervention Preview */
        .intervention-preview {{
            position: fixed;
            top: 20px;
            right: 20px;
            width: 320px;
            background: {colors['surface']};
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            overflow: hidden;
            animation: slideIn 0.3s ease-out;
            z-index: 1000;
        }}
        
        .intervention-header {{
            display: flex;
            align-items: center;
            padding: 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .intervention-icon {{
            width: 24px;
            height: 24px;
            color: {colors['warning']};
            margin-right: 12px;
        }}
        
        .intervention-icon svg {{
            width: 100%;
            height: 100%;
            fill: currentColor;
        }}
        
        .intervention-title {{
            flex: 1;
            font-weight: 600;
        }}
        
        .intervention-cancel {{
            background: none;
            border: none;
            color: {colors['text_secondary']};
            font-size: 24px;
            cursor: pointer;
            padding: 0;
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            transition: all 0.2s ease;
        }}
        
        .intervention-cancel:hover {{
            background: rgba(255, 255, 255, 0.1);
            color: {colors['text']};
        }}
        
        .intervention-content {{
            padding: 16px;
        }}
        
        .intervention-content p {{
            margin: 0 0 12px 0;
        }}
        
        .intervention-countdown {{
            font-size: 0.875rem;
            color: {colors['text_secondary']};
        }}
        
        .countdown-value {{
            font-weight: 600;
            color: {colors['warning']};
        }}
        
        .intervention-progress {{
            height: 3px;
            background: rgba(255, 255, 255, 0.1);
            position: relative;
            overflow: hidden;
        }}
        
        .progress-bar {{
            position: absolute;
            left: 0;
            top: 0;
            height: 100%;
            background: {colors['warning']};
            width: 0;
        }}
        
        @keyframes progress {{
            from {{ width: 0; }}
            to {{ width: 100%; }}
        }}
        
        /* Mode Indicator */
        .mode-indicator {{
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 16px;
            background: {colors['surface']};
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }}
        
        .mode-icon {{
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.05);
        }}
        
        .mode-icon svg {{
            width: 28px;
            height: 28px;
            fill: currentColor;
        }}
        
        .mode-info {{
            flex: 1;
        }}
        
        .mode-label {{
            font-size: 1.125rem;
            font-weight: 600;
            margin-bottom: 4px;
        }}
        
        .mode-description {{
            font-size: 0.875rem;
            color: {colors['text_secondary']};
        }}
        
        .mode-metrics {{
            display: flex;
            flex-direction: column;
            gap: 4px;
            font-size: 0.75rem;
            color: {colors['text_secondary']};
        }}
        
        /* Sensor Dashboard */
        .sensor-dashboard {{
            background: {colors['surface']};
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .dashboard-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}
        
        .dashboard-header h3 {{
            margin: 0;
            font-size: 1.25rem;
        }}
        
        .dashboard-status {{
            font-size: 0.875rem;
            color: {colors['success']};
        }}
        
        .sensor-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
            gap: 12px;
        }}
        
        /* Emotional Visualizer */
        .emotional-visualizer {{
            background: {colors['surface']};
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .emotion-graph {{
            position: relative;
            width: 200px;
            height: 200px;
            margin: 0 auto 16px;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.03), transparent);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }}
        
        .emotion-axes {{
            position: absolute;
            inset: 0;
        }}
        
        .axis-label {{
            position: absolute;
            font-size: 0.75rem;
            color: {colors['text_secondary']};
        }}
        
        .axis-label.top {{ top: -20px; left: 50%; transform: translateX(-50%); }}
        .axis-label.right {{ right: -60px; top: 50%; transform: translateY(-50%); }}
        .axis-label.bottom {{ bottom: -20px; left: 50%; transform: translateX(-50%); }}
        .axis-label.left {{ left: -60px; top: 50%; transform: translateY(-50%); }}
        
        .emotion-point {{
            position: absolute;
            width: 12px;
            height: 12px;
            background: {colors['accent']};
            border-radius: 50%;
            transform: translate(-50%, -50%);
            box-shadow: 0 0 20px {colors['accent']};
        }}
        
        .point-ripple {{
            position: absolute;
            inset: 0;
            background: {colors['accent']};
            border-radius: 50%;
            animation: ripple 2s ease-out infinite;
        }}
        
        .emotion-label {{
            text-align: center;
            font-weight: 600;
            color: {colors['accent']};
        }}
        
        /* Notification Toast */
        .notification-toast {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 16px;
            background: {colors['surface']};
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            animation: slideIn 0.3s ease-out;
            margin-bottom: 12px;
        }}
        
        .toast-icon {{
            width: 24px;
            height: 24px;
        }}
        
        .toast-icon svg {{
            width: 100%;
            height: 100%;
            fill: currentColor;
        }}
        
        .toast-content {{
            flex: 1;
        }}
        
        .toast-content p {{
            margin: 0;
        }}
        
        .toast-action {{
            background: none;
            border: 1px solid currentColor;
            color: inherit;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 0.875rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .toast-action:hover {{
            background: rgba(255, 255, 255, 0.1);
        }}
        
        .toast-close {{
            background: none;
            border: none;
            color: {colors['text_secondary']};
            font-size: 20px;
            cursor: pointer;
            padding: 0;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            transition: all 0.2s ease;
        }}
        
        .toast-close:hover {{
            background: rgba(255, 255, 255, 0.1);
            color: {colors['text']};
        }}
        
        /* Activity Timeline */
        .activity-timeline {{
            background: {colors['surface']};
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .activity-timeline h3 {{
            margin: 0 0 16px 0;
            font-size: 1.125rem;
        }}
        
        .timeline-container {{
            position: relative;
            padding-left: 40px;
        }}
        
        .timeline-container::before {{
            content: '';
            position: absolute;
            left: 15px;
            top: 0;
            bottom: 0;
            width: 2px;
            background: rgba(255, 255, 255, 0.1);
        }}
        
        .timeline-item {{
            position: relative;
            margin-bottom: 16px;
            animation: fadeIn 0.3s ease-out;
        }}
        
        .timeline-time {{
            position: absolute;
            left: -40px;
            font-size: 0.75rem;
            color: {colors['text_secondary']};
        }}
        
        .timeline-dot {{
            position: absolute;
            left: -25px;
            top: 6px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            border: 2px solid {colors['background']};
        }}
        
        .timeline-content {{
            background: rgba(255, 255, 255, 0.03);
            padding: 12px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        .timeline-title {{
            font-weight: 600;
            margin-bottom: 4px;
        }}
        
        .timeline-description {{
            font-size: 0.875rem;
            color: {colors['text_secondary']};
        }}
        
        /* Animations */
        {self.animation_presets['pulse']}
        {self.animation_presets['spin']}
        {self.animation_presets['fade_in']}
        {self.animation_presets['slide_in']}
        {self.animation_presets['bounce']}
        {self.animation_presets['ripple']}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .intervention-preview {{
                width: calc(100vw - 40px);
                right: 20px;
            }}
            
            .sensor-grid {{
                grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            }}
        }}
        </style>
        """
        
    def generate_complete_ui(self,
                           state: Dict,
                           active_sensors: List[Dict],
                           recent_activities: List[Dict]) -> str:
        """Generate complete UI with all components"""
        return f"""
        {self.generate_style_sheet()}
        <div class="jarvis-ui">
            <div class="ui-header">
                {self.generate_mode_indicator(state.get('mode', 'normal'), state)}
            </div>
            
            <div class="ui-main">
                <div class="ui-column">
                    {self.generate_sensor_dashboard(active_sensors)}
                    {self.generate_emotional_state_visualizer(state.get('emotional_state', {}))}
                </div>
                
                <div class="ui-column">
                    {self.generate_activity_timeline(recent_activities)}
                </div>
            </div>
        </div>
        """
