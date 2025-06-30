"""
JARVIS Proactive Monitoring Layer
Connects to existing JARVIS infrastructure to provide Nordic Fintech Week-style interruptions
"""

import os
import sys
import time
import threading
from datetime import datetime, timedelta
import subprocess
import Quartz
import AppKit
from typing import Dict, List, Optional

# Import existing JARVIS components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from jarvis_integration import JARVISCore, launch_jarvis_v10

class ActivityMonitor:
    """Monitors user's current activity on macOS"""
    
    def __init__(self, jarvis_core: JARVISCore):
        self.jarvis = jarvis_core
        self.current_activity = None
        self.activity_start_time = None
        self.monitoring = True
        
    def get_active_application(self) -> Dict[str, str]:
        """Get currently active application"""
        workspace = AppKit.NSWorkspace.sharedWorkspace()
        active_app = workspace.activeApplication()
        return {
            'name': active_app['NSApplicationName'],
            'bundle': active_app['NSApplicationBundleIdentifier'],
            'path': active_app['NSApplicationPath']
        }
        
    def detect_activity_type(self, app_info: Dict[str, str]) -> str:
        """Detect what type of activity user is doing"""
        app_name = app_info['name'].lower()
        bundle = app_info['bundle'].lower()
        
        # Video/Screen recording
        if any(x in app_name for x in ['quicktime', 'obs', 'screenflow', 'camtasia']):
            return 'video_recording'
        elif 'com.apple.screencaptureui' in bundle:
            return 'screen_recording'
            
        # Communication
        elif any(x in app_name for x in ['zoom', 'teams', 'slack', 'discord']):
            return 'meeting'
        elif 'mail' in app_name:
            return 'email'
            
        # Development
        elif any(x in app_name for x in ['code', 'xcode', 'pycharm', 'intellij']):
            return 'coding'
            
        # Content creation
        elif any(x in app_name for x in ['final cut', 'premiere', 'davinci']):
            return 'video_editing'
        elif any(x in app_name for x in ['logic', 'garageband', 'ableton']):
            return 'audio_editing'
            
        # Browsing
        elif any(x in app_name for x in ['safari', 'chrome', 'firefox']):
            return 'browsing'
            
        return 'general'
        
    def monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Get current activity
                app_info = self.get_active_application()
                new_activity = self.detect_activity_type(app_info)
                
                # Activity changed
                if new_activity != self.current_activity:
                    self.handle_activity_change(new_activity, app_info)
                    self.current_activity = new_activity
                    self.activity_start_time = datetime.now()
                    
                # Check duration-based interventions
                if self.activity_start_time:
                    duration = (datetime.now() - self.activity_start_time).seconds
                    self.check_activity_interventions(duration)
                    
            except Exception as e:
                print(f"Monitor error: {e}")
                
            time.sleep(1)  # Check every second
            
    def handle_activity_change(self, new_activity: str, app_info: Dict):
        """Handle when user changes activity"""
        consciousness = self.jarvis.component_manager.get_component('consciousness')
        if consciousness:
            consciousness.think_about(f"User switched to {new_activity} ({app_info['name']})", "activity")
            
        # Activity-specific setup
        if new_activity == 'video_recording':
            self.jarvis.bus.publish('activity_started', {
                'type': 'video_recording',
                'recommendations': [
                    "Videos under 60 seconds get 34% more engagement",
                    "Maintain eye contact with camera",
                    "Speak at 150 words per minute"
                ]
            })
            
    def check_activity_interventions(self, duration: int):
        """Check if we should intervene based on activity duration"""
        if self.current_activity == 'video_recording' and duration == 45:
            self.intervene("You're at 45 seconds. Remember, videos under 60 seconds get 34% more engagement.")
        elif self.current_activity == 'coding' and duration % 3600 == 0:  # Every hour
            self.intervene("You've been coding for an hour. Time for a quick stretch break.")
        elif self.current_activity == 'meeting' and duration % 1800 == 0:  # Every 30 min
            self.intervene("Meeting tip: Check if everyone has had a chance to contribute.")
            
    def intervene(self, message: str):
        """Send intervention through JARVIS"""
        self.jarvis.bus.publish('proactive_intervention', {
            'message': message,
            'activity': self.current_activity,
            'timestamp': datetime.now().isoformat()
        })


class ProactiveJARVIS:
    """Main proactive JARVIS system"""
    
    def __init__(self):
        # Launch existing JARVIS v10
        self.core = launch_jarvis_v10()
        
        # Add proactive components
        self.activity_monitor = ActivityMonitor(self.core)
        self.schedule_monitor = ScheduleMonitor(self.core)
        self.health_monitor = HealthMonitor(self.core)
        
        # Set up intervention handlers
        self.setup_intervention_handlers()
        
    def setup_intervention_handlers(self):
        """Set up handlers for proactive interventions"""
        self.core.bus.subscribe('proactive_intervention', self.handle_intervention)
        self.core.bus.subscribe('schedule_reminder', self.handle_intervention)
        self.core.bus.subscribe('health_alert', self.handle_intervention)
        
    def handle_intervention(self, data: Dict):
        """Handle proactive interventions"""
        message = data.get('message', '')
        
        # Speak through voice system
        voice = self.core.component_manager.get_component('voice')
        if voice:
            voice.speak(message)
            
        # Show notification
        self.show_notification("JARVIS", message)
        
        # Log to consciousness
        consciousness = self.core.component_manager.get_component('consciousness')
        if consciousness:
            consciousness.think_about(f"Intervention delivered: {message}", "proactive")
            
    def show_notification(self, title: str, message: str):
        """Show macOS notification"""
        script = f'''
        display notification "{message}" with title "{title}"
        '''
        subprocess.run(['osascript', '-e', script])
        
    def start(self):
        """Start all proactive monitoring"""
        # Start activity monitoring
        threading.Thread(target=self.activity_monitor.monitor_loop, daemon=True).start()
        
        # Start schedule monitoring
        threading.Thread(target=self.schedule_monitor.monitor_loop, daemon=True).start()
        
        # Start health monitoring
        threading.Thread(target=self.health_monitor.monitor_loop, daemon=True).start()
        
        print("🚀 Proactive JARVIS is now monitoring your activities!")
        

class ScheduleMonitor:
    """Monitor calendar and provide proactive reminders"""
    
    def __init__(self, jarvis_core: JARVISCore):
        self.jarvis = jarvis_core
        self.events = self.load_calendar_events()
        
    def load_calendar_events(self) -> List[Dict]:
        """Load calendar events (would integrate with actual calendar)"""
        # For demo, return sample events
        now = datetime.now()
        return [
            {
                'title': 'Paddle training',
                'time': now + timedelta(minutes=42),
                'reminded': False
            },
            {
                'title': 'Team meeting',
                'time': now + timedelta(hours=2),
                'reminded': False
            }
        ]
        
    def monitor_loop(self):
        """Check for upcoming events"""
        while True:
            now = datetime.now()
            
            for event in self.events:
                if event['reminded']:
                    continue
                    
                time_until = (event['time'] - now).total_seconds() / 60
                
                # Contextual reminders based on current activity
                activity_monitor = self.jarvis.component_manager.get_component('activity_monitor')
                current_activity = activity_monitor.current_activity if activity_monitor else None
                
                if 40 < time_until < 45:
                    if current_activity == 'video_recording':
                        message = f"Speaking of anticipation, you're due for {event['title']} in {int(time_until)} minutes."
                    else:
                        message = f"Reminder: {event['title']} in {int(time_until)} minutes."
                        
                    self.jarvis.bus.publish('schedule_reminder', {
                        'message': message,
                        'event': event['title']
                    })
                    event['reminded'] = True
                    
            time.sleep(30)  # Check every 30 seconds


class HealthMonitor:
    """Monitor health-related activities"""
    
    def __init__(self, jarvis_core: JARVISCore):
        self.jarvis = jarvis_core
        self.user_profile = {
            'lactose_intolerant': True,
            'allergies': [],
            'water_goal': 8,  # glasses per day
            'break_interval': 90  # minutes
        }
        self.last_break = datetime.now()
        self.water_count = 0
        
    def monitor_loop(self):
        """Health monitoring loop"""
        while True:
            # Break reminders
            time_since_break = (datetime.now() - self.last_break).seconds / 60
            if time_since_break > self.user_profile['break_interval']:
                self.jarvis.bus.publish('health_alert', {
                    'message': "You've been working for 90 minutes. Time for a quick break!",
                    'type': 'break_reminder'
                })
                self.last_break = datetime.now()
                
            # Water reminders (every 2 hours)
            hour = datetime.now().hour
            if hour % 2 == 0 and datetime.now().minute == 0:
                self.jarvis.bus.publish('health_alert', {
                    'message': f"Hydration check! You've had {self.water_count} glasses today. Goal: {self.user_profile['water_goal']}",
                    'type': 'water_reminder'
                })
                
            time.sleep(60)  # Check every minute


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║          JARVIS PROACTIVE MONITORING ACTIVATED               ║
    ║                                                              ║
    ║  Now monitoring:                                             ║
    ║  • Your current activities                                   ║
    ║  • Video recording performance                               ║
    ║  • Meeting participation                                     ║
    ║  • Coding sessions                                           ║
    ║  • Schedule and reminders                                    ║
    ║  • Health and wellness                                       ║
    ║                                                              ║
    ║  JARVIS will proactively help without being asked!          ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Launch proactive JARVIS
    proactive_jarvis = ProactiveJARVIS()
    proactive_jarvis.start()
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Proactive monitoring stopped.")
