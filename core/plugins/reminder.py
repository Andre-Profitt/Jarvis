#!/usr/bin/env python3
"""
Reminder Plugin for JARVIS
Manages reminders, timers, and scheduled notifications
"""

from typing import Dict, Any, Tuple, Optional, List
import re
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
import threading
from core.plugin_system import JARVISPlugin, PluginMetadata, PluginCommand


class ReminderPlugin(JARVISPlugin):
    """Reminder and timer management plugin"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Reminder",
            version="1.0.0",
            author="JARVIS Team",
            description="Set reminders, timers, and scheduled notifications",
            category="productivity",
            keywords=["reminder", "timer", "alarm", "notification", "schedule"],
            requirements=[],
            permissions=["file_access", "notifications"],
            config_schema={
                "storage_path": {"type": "string", "default": "~/.jarvis/reminders.json"},
                "notification_sound": {"type": "boolean", "default": True},
                "max_reminders": {"type": "integer", "default": 100},
                "default_snooze": {"type": "integer", "default": 5, "description": "Default snooze time in minutes"}
            }
        )
        
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the reminder plugin"""
        self.config = config
        self.storage_path = Path(config.get("storage_path", "~/.jarvis/reminders.json")).expanduser()
        self.notification_sound = config.get("notification_sound", True)
        self.max_reminders = config.get("max_reminders", 100)
        self.default_snooze = config.get("default_snooze", 5)
        
        # Storage for reminders
        self.reminders: List[Dict[str, Any]] = []
        self.timers: Dict[str, asyncio.Task] = {}
        self.reminder_id_counter = 0
        
        # Create storage directory
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing reminders
        self._load_reminders()
        
        # Register commands
        self.register_command(PluginCommand(
            name="set_reminder",
            patterns=[
                r"remind me (?:to |about )?(.+?)(?:\s+(?:at|in|on)\s+(.+))",
                r"set (?:a )?reminder (?:to |for |about )?(.+?)(?:\s+(?:at|in|on)\s+(.+))",
                r"(?:create|add) (?:a )?reminder (?:to |for |about )?(.+?)(?:\s+(?:at|in|on)\s+(.+))"
            ],
            description="Set a new reminder",
            parameters={
                "message": {"type": "string", "description": "What to be reminded about"},
                "time": {"type": "string", "description": "When to remind (e.g., 'in 5 minutes', 'at 3pm', 'tomorrow at 9am')"}
            },
            examples=[
                "remind me to call mom in 30 minutes",
                "set reminder for meeting at 3pm",
                "remind me about lunch at noon",
                "create reminder to take medicine in 2 hours"
            ],
            handler=self.handle_set_reminder
        ))
        
        self.register_command(PluginCommand(
            name="set_timer",
            patterns=[
                r"(?:set |start )?(?:a )?timer (?:for )?(\d+)\s*(seconds?|minutes?|hours?)",
                r"(?:set |start )?(?:a )?(\d+)\s*(second|minute|hour) timer",
                r"timer (\d+)\s*(s|m|min|mins|h|hr|hrs|hour|hours)"
            ],
            description="Set a timer",
            parameters={
                "duration": {"type": "integer", "description": "Timer duration"},
                "unit": {"type": "string", "description": "Time unit (seconds, minutes, hours)"}
            },
            examples=[
                "set timer for 5 minutes",
                "timer 30 seconds",
                "start a 2 hour timer",
                "10 minute timer"
            ],
            handler=self.handle_set_timer
        ))
        
        self.register_command(PluginCommand(
            name="list_reminders",
            patterns=[
                r"(?:show|list|what are) (?:my )?reminders?",
                r"(?:do i have any |what) reminders?",
                r"reminder list"
            ],
            description="List all active reminders",
            parameters={},
            examples=[
                "show my reminders",
                "list reminders",
                "what reminders do I have"
            ],
            handler=self.handle_list_reminders
        ))
        
        self.register_command(PluginCommand(
            name="cancel_reminder",
            patterns=[
                r"(?:cancel|delete|remove) reminder (?:#)?(\d+)",
                r"(?:cancel|delete|remove) (?:the )?(.+?) reminder"
            ],
            description="Cancel a reminder",
            parameters={
                "identifier": {"type": "string", "description": "Reminder ID or description"}
            },
            examples=[
                "cancel reminder #3",
                "delete reminder 5",
                "remove the meeting reminder"
            ],
            handler=self.handle_cancel_reminder
        ))
        
        self.register_command(PluginCommand(
            name="snooze_reminder",
            patterns=[
                r"snooze (?:reminder )?(?:#)?(\d+)(?:\s+for\s+(\d+)\s*(minutes?|hours?))?",
                r"snooze (?:for )?(\d+)\s*(minutes?|hours?)"
            ],
            description="Snooze a reminder",
            parameters={
                "reminder_id": {"type": "integer", "description": "Reminder ID", "optional": True},
                "duration": {"type": "integer", "description": "Snooze duration", "optional": True},
                "unit": {"type": "string", "description": "Time unit", "optional": True}
            },
            examples=[
                "snooze reminder #2 for 10 minutes",
                "snooze for 5 minutes",
                "snooze 15 minutes"
            ],
            handler=self.handle_snooze_reminder
        ))
        
        # Restart any active reminders
        await self._restart_active_reminders()
        
        self.logger.info("Reminder plugin initialized")
        return True
        
    async def shutdown(self):
        """Clean up resources"""
        # Cancel all active timers
        for timer_id, task in self.timers.items():
            task.cancel()
            
        # Save reminders
        self._save_reminders()
        
        self.logger.info("Reminder plugin shutting down")
        
    async def handle_set_reminder(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Handle setting a new reminder"""
        try:
            message = match.group(1).strip()
            time_str = match.group(2).strip() if match.lastindex >= 2 else ""
            
            if not time_str:
                return False, "Please specify when you want to be reminded."
                
            # Parse time
            reminder_time = self._parse_time(time_str)
            
            if not reminder_time:
                return False, f"I couldn't understand the time '{time_str}'. Try something like 'in 30 minutes' or 'at 3pm'."
                
            # Check if time is in the past
            if reminder_time <= datetime.now():
                return False, "That time is in the past. Please specify a future time."
                
            # Create reminder
            reminder = {
                "id": self.reminder_id_counter,
                "message": message,
                "time": reminder_time.isoformat(),
                "created": datetime.now().isoformat(),
                "status": "active",
                "recurring": False
            }
            
            self.reminder_id_counter += 1
            self.reminders.append(reminder)
            
            # Save reminders
            self._save_reminders()
            
            # Schedule the reminder
            await self._schedule_reminder(reminder)
            
            # Format confirmation
            time_until = reminder_time - datetime.now()
            if time_until.total_seconds() < 3600:  # Less than an hour
                time_desc = f"{int(time_until.total_seconds() / 60)} minutes"
            elif time_until.total_seconds() < 86400:  # Less than a day
                time_desc = f"{int(time_until.total_seconds() / 3600)} hours"
            else:
                time_desc = reminder_time.strftime("%B %d at %I:%M %p")
                
            return True, f"✅ Reminder set: '{message}' in {time_desc} (ID: #{reminder['id']})"
            
        except Exception as e:
            self.logger.error(f"Error setting reminder: {e}")
            return False, f"Sorry, I couldn't set the reminder: {str(e)}"
            
    async def handle_set_timer(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Handle setting a timer"""
        try:
            # Extract duration and unit
            duration = int(match.group(1))
            unit = match.group(2).lower()
            
            # Convert to seconds
            if unit.startswith('s'):
                total_seconds = duration
                unit_name = "second" if duration == 1 else "seconds"
            elif unit.startswith('m'):
                total_seconds = duration * 60
                unit_name = "minute" if duration == 1 else "minutes"
            elif unit.startswith('h'):
                total_seconds = duration * 3600
                unit_name = "hour" if duration == 1 else "hours"
            else:
                return False, f"Unknown time unit: {unit}"
                
            # Create timer reminder
            timer_time = datetime.now() + timedelta(seconds=total_seconds)
            reminder = {
                "id": self.reminder_id_counter,
                "message": f"Timer: {duration} {unit_name} timer finished!",
                "time": timer_time.isoformat(),
                "created": datetime.now().isoformat(),
                "status": "active",
                "type": "timer",
                "duration": f"{duration} {unit_name}"
            }
            
            self.reminder_id_counter += 1
            self.reminders.append(reminder)
            
            # Save reminders
            self._save_reminders()
            
            # Schedule the timer
            await self._schedule_reminder(reminder)
            
            return True, f"⏰ Timer set for {duration} {unit_name} (ID: #{reminder['id']})"
            
        except Exception as e:
            self.logger.error(f"Error setting timer: {e}")
            return False, f"Sorry, I couldn't set the timer: {str(e)}"
            
    async def handle_list_reminders(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Handle listing reminders"""
        try:
            active_reminders = [r for r in self.reminders if r["status"] == "active"]
            
            if not active_reminders:
                return True, "You don't have any active reminders."
                
            # Sort by time
            active_reminders.sort(key=lambda r: r["time"])
            
            response = "📋 Your active reminders:\n\n"
            
            for i, reminder in enumerate(active_reminders, 1):
                reminder_time = datetime.fromisoformat(reminder["time"])
                time_until = reminder_time - datetime.now()
                
                # Format time description
                if time_until.total_seconds() < 0:
                    time_desc = "overdue"
                elif time_until.total_seconds() < 60:
                    time_desc = "less than a minute"
                elif time_until.total_seconds() < 3600:
                    minutes = int(time_until.total_seconds() / 60)
                    time_desc = f"{minutes} minute{'s' if minutes != 1 else ''}"
                elif time_until.total_seconds() < 86400:
                    hours = int(time_until.total_seconds() / 3600)
                    time_desc = f"{hours} hour{'s' if hours != 1 else ''}"
                else:
                    time_desc = reminder_time.strftime("%B %d at %I:%M %p")
                    
                response += f"{i}. #{reminder['id']} - {reminder['message']}\n"
                response += f"   ⏰ In {time_desc}\n\n"
                
            return True, response.strip()
            
        except Exception as e:
            self.logger.error(f"Error listing reminders: {e}")
            return False, f"Sorry, I couldn't list the reminders: {str(e)}"
            
    async def handle_cancel_reminder(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Handle canceling a reminder"""
        try:
            identifier = match.group(1).strip()
            
            # Find reminder
            reminder = None
            if identifier.isdigit():
                # Search by ID
                reminder_id = int(identifier)
                reminder = next((r for r in self.reminders if r["id"] == reminder_id and r["status"] == "active"), None)
            else:
                # Search by description
                active_reminders = [r for r in self.reminders if r["status"] == "active"]
                for r in active_reminders:
                    if identifier.lower() in r["message"].lower():
                        reminder = r
                        break
                        
            if not reminder:
                return False, f"I couldn't find an active reminder matching '{identifier}'."
                
            # Cancel the reminder
            reminder["status"] = "cancelled"
            
            # Cancel the timer task if it exists
            timer_key = f"reminder_{reminder['id']}"
            if timer_key in self.timers:
                self.timers[timer_key].cancel()
                del self.timers[timer_key]
                
            # Save reminders
            self._save_reminders()
            
            return True, f"❌ Cancelled reminder: '{reminder['message']}' (ID: #{reminder['id']})"
            
        except Exception as e:
            self.logger.error(f"Error canceling reminder: {e}")
            return False, f"Sorry, I couldn't cancel the reminder: {str(e)}"
            
    async def handle_snooze_reminder(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Handle snoozing a reminder"""
        try:
            # Parse parameters
            groups = match.groups()
            reminder_id = None
            duration = self.default_snooze
            unit = "minutes"
            
            # Determine what was matched
            if groups[0] and groups[0].isdigit():
                if len(groups) >= 3 and groups[1]:
                    # Format: snooze reminder #X for Y units
                    reminder_id = int(groups[0])
                    duration = int(groups[1])
                    unit = groups[2].lower()
                else:
                    # Format: snooze for X units
                    duration = int(groups[0])
                    if len(groups) > 1 and groups[1]:
                        unit = groups[1].lower()
                        
            # Find the most recent active reminder if no ID specified
            if reminder_id is None:
                active_reminders = [r for r in self.reminders if r["status"] == "active"]
                if not active_reminders:
                    return False, "No active reminders to snooze."
                    
                # Get the most recent (soonest) reminder
                active_reminders.sort(key=lambda r: r["time"])
                reminder = active_reminders[0]
            else:
                reminder = next((r for r in self.reminders if r["id"] == reminder_id and r["status"] == "active"), None)
                if not reminder:
                    return False, f"Couldn't find active reminder #{reminder_id}."
                    
            # Calculate new time
            if unit.startswith('h'):
                snooze_delta = timedelta(hours=duration)
                unit_name = "hour" if duration == 1 else "hours"
            else:
                snooze_delta = timedelta(minutes=duration)
                unit_name = "minute" if duration == 1 else "minutes"
                
            old_time = datetime.fromisoformat(reminder["time"])
            new_time = datetime.now() + snooze_delta
            
            # Update reminder
            reminder["time"] = new_time.isoformat()
            reminder["snoozed"] = True
            reminder["snoozed_at"] = datetime.now().isoformat()
            
            # Cancel old timer if exists
            timer_key = f"reminder_{reminder['id']}"
            if timer_key in self.timers:
                self.timers[timer_key].cancel()
                del self.timers[timer_key]
                
            # Schedule new timer
            await self._schedule_reminder(reminder)
            
            # Save reminders
            self._save_reminders()
            
            return True, f"⏰ Snoozed reminder '{reminder['message']}' for {duration} {unit_name}"
            
        except Exception as e:
            self.logger.error(f"Error snoozing reminder: {e}")
            return False, f"Sorry, I couldn't snooze the reminder: {str(e)}"
            
    def _parse_time(self, time_str: str) -> Optional[datetime]:
        """Parse various time formats"""
        time_str = time_str.lower().strip()
        now = datetime.now()
        
        # Relative time patterns
        relative_patterns = [
            (r"in (\d+) seconds?", lambda m: now + timedelta(seconds=int(m.group(1)))),
            (r"in (\d+) minutes?", lambda m: now + timedelta(minutes=int(m.group(1)))),
            (r"in (\d+) hours?", lambda m: now + timedelta(hours=int(m.group(1)))),
            (r"in (\d+) days?", lambda m: now + timedelta(days=int(m.group(1)))),
            (r"tomorrow (?:at )?(\d{1,2}):?(\d{2})?\s*(am|pm)?", self._parse_tomorrow_time),
            (r"tomorrow", lambda m: now + timedelta(days=1)),
        ]
        
        for pattern, parser in relative_patterns:
            match = re.match(pattern, time_str)
            if match:
                return parser(match)
                
        # Absolute time patterns (simplified)
        # Format: at HH:MM [am/pm]
        abs_match = re.match(r"(?:at )?(\d{1,2}):?(\d{2})?\s*(am|pm)?", time_str)
        if abs_match:
            hour = int(abs_match.group(1))
            minute = int(abs_match.group(2)) if abs_match.group(2) else 0
            period = abs_match.group(3)
            
            if period == "pm" and hour < 12:
                hour += 12
            elif period == "am" and hour == 12:
                hour = 0
                
            # Create time for today
            reminder_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # If time has passed, assume tomorrow
            if reminder_time <= now:
                reminder_time += timedelta(days=1)
                
            return reminder_time
            
        return None
        
    def _parse_tomorrow_time(self, match: re.Match) -> datetime:
        """Parse 'tomorrow at X' format"""
        now = datetime.now()
        tomorrow = now + timedelta(days=1)
        
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        period = match.group(3)
        
        if period == "pm" and hour < 12:
            hour += 12
        elif period == "am" and hour == 12:
            hour = 0
            
        return tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
    async def _schedule_reminder(self, reminder: Dict[str, Any]):
        """Schedule a reminder to fire at the specified time"""
        reminder_time = datetime.fromisoformat(reminder["time"])
        time_until = (reminder_time - datetime.now()).total_seconds()
        
        if time_until <= 0:
            # Fire immediately if overdue
            await self._fire_reminder(reminder)
        else:
            # Schedule for future
            timer_key = f"reminder_{reminder['id']}"
            task = asyncio.create_task(self._reminder_timer(reminder, time_until))
            self.timers[timer_key] = task
            
    async def _reminder_timer(self, reminder: Dict[str, Any], delay: float):
        """Timer coroutine for a reminder"""
        try:
            await asyncio.sleep(delay)
            await self._fire_reminder(reminder)
        except asyncio.CancelledError:
            # Timer was cancelled
            pass
            
    async def _fire_reminder(self, reminder: Dict[str, Any]):
        """Fire a reminder notification"""
        try:
            # Update status
            reminder["status"] = "fired"
            reminder["fired_at"] = datetime.now().isoformat()
            
            # Save reminders
            self._save_reminders()
            
            # Create notification
            notification_text = f"🔔 Reminder: {reminder['message']}"
            
            # Emit event for notification
            self.emit_event("notification", {
                "title": "JARVIS Reminder",
                "message": reminder['message'],
                "priority": "high",
                "sound": self.notification_sound
            })
            
            # Also emit for speech
            self.emit_event("speak_announcement", {
                "text": notification_text,
                "priority": "high",
                "interrupt": True
            })
            
            # Clean up timer
            timer_key = f"reminder_{reminder['id']}"
            if timer_key in self.timers:
                del self.timers[timer_key]
                
            self.logger.info(f"Fired reminder: {reminder['message']}")
            
        except Exception as e:
            self.logger.error(f"Error firing reminder: {e}")
            
    async def _restart_active_reminders(self):
        """Restart timers for active reminders after plugin reload"""
        now = datetime.now()
        
        for reminder in self.reminders:
            if reminder["status"] == "active":
                reminder_time = datetime.fromisoformat(reminder["time"])
                
                # Skip if already past
                if reminder_time <= now:
                    reminder["status"] = "overdue"
                else:
                    # Reschedule
                    await self._schedule_reminder(reminder)
                    
    def _load_reminders(self):
        """Load reminders from storage"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.reminders = data.get("reminders", [])
                    self.reminder_id_counter = data.get("id_counter", 0)
                    self.logger.info(f"Loaded {len(self.reminders)} reminders")
        except Exception as e:
            self.logger.error(f"Error loading reminders: {e}")
            self.reminders = []
            self.reminder_id_counter = 0
            
    def _save_reminders(self):
        """Save reminders to storage"""
        try:
            data = {
                "reminders": self.reminders,
                "id_counter": self.reminder_id_counter,
                "last_saved": datetime.now().isoformat()
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving reminders: {e}")