"""
JARVIS Phase 7: Visual Feedback System
=====================================
Real-time visual feedback and notification management
"""

import asyncio
import json
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import random

from .ui_components import JARVISUIComponents, StatusIndicator, UITheme

class FeedbackType(Enum):
    """Types of visual feedback"""
    STATUS_UPDATE = "status_update"
    INTERVENTION = "intervention"
    MODE_CHANGE = "mode_change"
    NOTIFICATION = "notification"
    PROGRESS = "progress"
    ALERT = "alert"

class InterventionType(Enum):
    """Types of interventions"""
    BLOCK_NOTIFICATIONS = "block_notifications"
    SUGGEST_BREAK = "suggest_break"
    BREATHING_EXERCISE = "breathing_exercise"
    FOCUS_MODE = "focus_mode"
    EMERGENCY_CONTACT = "emergency_contact"
    SYSTEM_ACTION = "system_action"

class VisualFeedbackSystem:
    """Manages visual feedback and notifications for JARVIS"""
    
    def __init__(self, theme: UITheme = UITheme.DARK):
        self.ui_components = JARVISUIComponents(theme)
        self.active_sensors = {}
        self.notification_queue = deque(maxlen=50)
        self.intervention_queue = deque(maxlen=10)
        self.activity_log = deque(maxlen=100)
        self.current_mode = "normal"
        self.current_state = {}
        self.callbacks = {}
        self.intervention_timers = {}
        self.update_listeners = []
        
    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for specific events"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
        
    def add_update_listener(self, listener: Callable):
        """Add listener for UI updates"""
        self.update_listeners.append(listener)
        
    async def _notify_listeners(self, update_type: str, data: Dict):
        """Notify all update listeners"""
        for listener in self.update_listeners:
            try:
                await listener(update_type, data)
            except Exception as e:
                print(f"Error notifying listener: {e}")
                
    async def update_sensor_status(self, 
                                 sensor_type: str,
                                 status: str,
                                 data: Optional[Dict] = None):
        """Update sensor status with visual feedback"""
        self.active_sensors[sensor_type] = {
            "type": sensor_type,
            "status": status,
            "label": self._get_sensor_label(sensor_type, data),
            "last_update": datetime.now(),
            "data": data
        }
        
        # Add to activity log
        self._log_activity({
            "type": "sensor",
            "title": f"{sensor_type.title()} Sensor",
            "description": f"Status: {status}",
            "timestamp": datetime.now()
        })
        
        # Notify listeners
        await self._notify_listeners(FeedbackType.STATUS_UPDATE.value, {
            "sensor": sensor_type,
            "status": status,
            "html": self._generate_sensor_update_html()
        })
        
    def _get_sensor_label(self, sensor_type: str, data: Optional[Dict]) -> str:
        """Generate descriptive label for sensor"""
        labels = {
            "voice": "Voice Active" if data and data.get("active") else "Voice Ready",
            "biometric": f"HR: {data.get('heart_rate', 'N/A')}" if data else "Biometrics",
            "vision": "Camera Active" if data and data.get("active") else "Vision Ready",
            "context": f"{data.get('context_items', 0)} items" if data else "Context",
            "emotional": f"{data.get('state', 'neutral')}" if data else "Emotional"
        }
        return labels.get(sensor_type, sensor_type.title())
        
    async def preview_intervention(self,
                                 intervention_type: InterventionType,
                                 description: str,
                                 countdown: int = 3,
                                 can_cancel: bool = True,
                                 callback: Optional[Callable] = None):
        """Show intervention preview with countdown"""
        intervention_id = f"{intervention_type.value}_{datetime.now().timestamp()}"
        
        # Create intervention data
        intervention = {
            "id": intervention_id,
            "type": intervention_type,
            "description": description,
            "countdown": countdown,
            "can_cancel": can_cancel,
            "callback": callback,
            "status": "pending"
        }
        
        self.intervention_queue.append(intervention)
        
        # Generate preview HTML
        preview_html = self.ui_components.generate_intervention_preview(
            intervention_id,
            description,
            countdown,
            can_cancel
        )
        
        # Notify listeners
        await self._notify_listeners(FeedbackType.INTERVENTION.value, {
            "id": intervention_id,
            "type": intervention_type.value,
            "html": preview_html,
            "action": "show"
        })
        
        # Start countdown timer
        if countdown > 0:
            timer = asyncio.create_task(
                self._intervention_countdown(intervention_id, countdown, callback)
            )
            self.intervention_timers[intervention_id] = timer
            
        return intervention_id
        
    async def _intervention_countdown(self, 
                                    intervention_id: str,
                                    countdown: int,
                                    callback: Optional[Callable]):
        """Handle intervention countdown"""
        try:
            for remaining in range(countdown, 0, -1):
                await asyncio.sleep(1)
                
                # Update countdown display
                await self._notify_listeners(FeedbackType.INTERVENTION.value, {
                    "id": intervention_id,
                    "action": "update_countdown",
                    "remaining": remaining - 1
                })
                
            # Execute intervention
            intervention = next(
                (i for i in self.intervention_queue if i["id"] == intervention_id),
                None
            )
            
            if intervention and intervention["status"] == "pending":
                intervention["status"] = "executed"
                
                # Execute callback
                if callback:
                    await callback()
                    
                # Log activity
                self._log_activity({
                    "type": "intervention",
                    "title": "Intervention Executed",
                    "description": intervention["description"],
                    "timestamp": datetime.now()
                })
                
                # Notify completion
                await self._notify_listeners(FeedbackType.INTERVENTION.value, {
                    "id": intervention_id,
                    "action": "complete"
                })
                
        except asyncio.CancelledError:
            # Intervention was cancelled
            intervention = next(
                (i for i in self.intervention_queue if i["id"] == intervention_id),
                None
            )
            if intervention:
                intervention["status"] = "cancelled"
                
    async def cancel_intervention(self, intervention_id: str):
        """Cancel a pending intervention"""
        # Cancel timer
        if intervention_id in self.intervention_timers:
            self.intervention_timers[intervention_id].cancel()
            del self.intervention_timers[intervention_id]
            
        # Update status
        intervention = next(
            (i for i in self.intervention_queue if i["id"] == intervention_id),
            None
        )
        if intervention:
            intervention["status"] = "cancelled"
            
        # Notify listeners
        await self._notify_listeners(FeedbackType.INTERVENTION.value, {
            "id": intervention_id,
            "action": "cancel"
        })
        
    async def update_mode(self, 
                        new_mode: str,
                        state_info: Dict,
                        reason: Optional[str] = None):
        """Update current mode with visual feedback"""
        old_mode = self.current_mode
        self.current_mode = new_mode
        self.current_state = state_info
        
        # Generate mode indicator
        mode_html = self.ui_components.generate_mode_indicator(new_mode, state_info)
        
        # Create notification for mode change
        if old_mode != new_mode:
            mode_descriptions = {
                "flow": "Entering flow state - minimizing distractions",
                "crisis": "Crisis support activated - here to help",
                "rest": "Rest mode enabled - encouraging relaxation",
                "normal": "Returning to normal operation"
            }
            
            await self.show_notification(
                mode_descriptions.get(new_mode, f"Mode changed to {new_mode}"),
                "info" if new_mode != "crisis" else "warning",
                reason=reason
            )
            
        # Log activity
        self._log_activity({
            "type": "system",
            "title": "Mode Change",
            "description": f"{old_mode} → {new_mode}" + (f": {reason}" if reason else ""),
            "timestamp": datetime.now()
        })
        
        # Notify listeners
        await self._notify_listeners(FeedbackType.MODE_CHANGE.value, {
            "old_mode": old_mode,
            "new_mode": new_mode,
            "html": mode_html
        })
        
    async def show_notification(self,
                              message: str,
                              notification_type: str = "info",
                              duration: int = 5000,
                              action: Optional[Dict] = None,
                              reason: Optional[str] = None):
        """Show a notification toast"""
        notification_id = f"notif_{datetime.now().timestamp()}"
        
        # Add to queue
        self.notification_queue.append({
            "id": notification_id,
            "message": message,
            "type": notification_type,
            "timestamp": datetime.now(),
            "reason": reason
        })
        
        # Generate notification HTML
        notif_html = self.ui_components.generate_notification_toast(
            message,
            notification_type,
            duration,
            action
        )
        
        # Notify listeners
        await self._notify_listeners(FeedbackType.NOTIFICATION.value, {
            "id": notification_id,
            "html": notif_html,
            "duration": duration
        })
        
        return notification_id
        
    async def update_progress(self,
                            task_id: str,
                            progress: float,
                            message: Optional[str] = None):
        """Update progress indicator"""
        await self._notify_listeners(FeedbackType.PROGRESS.value, {
            "task_id": task_id,
            "progress": progress,
            "message": message
        })
        
    def _log_activity(self, activity: Dict):
        """Log activity for timeline"""
        self.activity_log.append(activity)
        
    def _generate_sensor_update_html(self) -> str:
        """Generate updated sensor dashboard HTML"""
        sensor_list = list(self.active_sensors.values())
        return self.ui_components.generate_sensor_dashboard(sensor_list)
        
    def get_current_ui_state(self) -> Dict:
        """Get current UI state for rendering"""
        return {
            "mode": self.current_mode,
            "state": self.current_state,
            "sensors": list(self.active_sensors.values()),
            "activities": list(self.activity_log)[-20:],  # Last 20 activities
            "notifications": list(self.notification_queue)[-10:],  # Last 10 notifications
            "interventions": [i for i in self.intervention_queue if i["status"] == "pending"]
        }
        
    def generate_full_ui(self) -> str:
        """Generate complete UI HTML"""
        state = self.get_current_ui_state()
        return self.ui_components.generate_complete_ui(
            {"mode": state["mode"], **state["state"]},
            state["sensors"],
            state["activities"]
        )
        
    async def simulate_interventions(self):
        """Simulate various interventions for demo"""
        interventions = [
            {
                "type": InterventionType.BLOCK_NOTIFICATIONS,
                "description": "Blocking all non-urgent notifications to maintain your flow state",
                "condition": lambda s: s.get("mode") == "flow"
            },
            {
                "type": InterventionType.SUGGEST_BREAK,
                "description": "You've been focused for 90 minutes. Time for a short break?",
                "condition": lambda s: s.get("focus_duration", 0) > 90
            },
            {
                "type": InterventionType.BREATHING_EXERCISE,
                "description": "High stress detected. Would you like to try a breathing exercise?",
                "condition": lambda s: s.get("stress_level", 0) > 0.8
            }
        ]
        
        for intervention in interventions:
            if intervention["condition"](self.current_state):
                await self.preview_intervention(
                    intervention["type"],
                    intervention["description"],
                    countdown=5,
                    can_cancel=True,
                    callback=lambda: self.show_notification(
                        "Intervention completed successfully",
                        "success"
                    )
                )
                break
                
    async def demonstrate_visual_feedback(self):
        """Demonstrate visual feedback capabilities"""
        print("\n🎨 Visual Feedback System Demo")
        print("="*50)
        
        # Update sensors
        sensors = [
            ("voice", "active", {"active": True}),
            ("biometric", "active", {"heart_rate": 75}),
            ("emotional", "processing", {"state": "calm"}),
            ("context", "active", {"context_items": 5})
        ]
        
        for sensor_type, status, data in sensors:
            await self.update_sensor_status(sensor_type, status, data)
            await asyncio.sleep(0.5)
            
        # Show mode changes
        modes = [
            ("normal", {"stress_level": 0.3, "focus_level": 0.5}, "Starting normal operation"),
            ("flow", {"stress_level": 0.2, "focus_level": 0.9}, "Deep focus detected"),
            ("crisis", {"stress_level": 0.9, "focus_level": 0.3}, "High stress detected")
        ]
        
        for mode, state, reason in modes:
            await self.update_mode(mode, state, reason)
            await asyncio.sleep(2)
            
        # Show interventions
        await self.preview_intervention(
            InterventionType.BLOCK_NOTIFICATIONS,
            "Blocking distractions to protect your flow state",
            countdown=3,
            can_cancel=True
        )
        
        await asyncio.sleep(5)
        
        # Show notifications
        notifications = [
            ("Task completed successfully", "success"),
            ("New priority email received", "info"),
            ("System optimization complete", "success")
        ]
        
        for message, notif_type in notifications:
            await self.show_notification(message, notif_type)
            await asyncio.sleep(1)


class UIBridge:
    """Bridge between JARVIS core and visual feedback system"""
    
    def __init__(self, feedback_system: VisualFeedbackSystem):
        self.feedback_system = feedback_system
        self.sensor_mappings = {
            "voice_input": "voice",
            "biometric_data": "biometric",
            "camera_feed": "vision",
            "context_analysis": "context",
            "emotional_state": "emotional"
        }
        
    async def process_jarvis_update(self, update_type: str, data: Dict):
        """Process updates from JARVIS core"""
        if update_type == "sensor_update":
            sensor_type = self.sensor_mappings.get(data["sensor"], data["sensor"])
            await self.feedback_system.update_sensor_status(
                sensor_type,
                data["status"],
                data.get("data")
            )
            
        elif update_type == "state_change":
            await self.feedback_system.update_mode(
                data["new_state"],
                data["state_info"],
                data.get("reason")
            )
            
        elif update_type == "intervention_needed":
            intervention_map = {
                "notification_block": InterventionType.BLOCK_NOTIFICATIONS,
                "break_suggestion": InterventionType.SUGGEST_BREAK,
                "breathing": InterventionType.BREATHING_EXERCISE,
                "focus_mode": InterventionType.FOCUS_MODE,
                "emergency": InterventionType.EMERGENCY_CONTACT
            }
            
            intervention_type = intervention_map.get(
                data["intervention_type"],
                InterventionType.SYSTEM_ACTION
            )
            
            await self.feedback_system.preview_intervention(
                intervention_type,
                data["description"],
                data.get("countdown", 3),
                data.get("can_cancel", True),
                data.get("callback")
            )
            
        elif update_type == "notification":
            await self.feedback_system.show_notification(
                data["message"],
                data.get("type", "info"),
                data.get("duration", 5000),
                data.get("action")
            )
