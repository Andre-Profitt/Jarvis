"""
Built-in plugins for common functionality
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
import aiohttp
import subprocess
import json

from .manager import BasePlugin
from ..logger import setup_logger

logger = setup_logger(__name__)


class WeatherPlugin(BasePlugin):
    """Weather information plugin"""
    
    def __init__(self, config):
        super().__init__("weather", "Get weather information")
        self.api_key = config.get("plugins.weather.api_key")
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        
    async def execute(self, params: Dict[str, Any]) -> str:
        """Get weather for location"""
        location = params.get("locations", [""])[0] or "New York"
        
        if not self.api_key:
            return "Weather API key not configured"
            
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}?q={location}&appid={self.api_key}&units=imperial"
            
            try:
                async with session.get(url) as response:
                    data = await response.json()
                    
                    if response.status == 200:
                        temp = data["main"]["temp"]
                        feels_like = data["main"]["feels_like"]
                        description = data["weather"][0]["description"]
                        humidity = data["main"]["humidity"]
                        
                        return (
                            f"Weather in {location}: {description.capitalize()}\n"
                            f"Temperature: {temp}°F (feels like {feels_like}°F)\n"
                            f"Humidity: {humidity}%"
                        )
                    else:
                        return f"Couldn't get weather for {location}"
                        
            except Exception as e:
                logger.error(f"Weather plugin error: {e}")
                return "Failed to get weather information"
                

class TimePlugin(BasePlugin):
    """Time and date plugin"""
    
    def __init__(self, config):
        super().__init__("time", "Get time and date information")
        
    async def execute(self, params: Dict[str, Any]) -> str:
        """Get current time/date"""
        now = datetime.now()
        
        # Check if specific time zone requested
        if locations := params.get("locations"):
            # Would implement timezone conversion here
            pass
            
        return (
            f"Current time: {now.strftime('%I:%M %p')}\n"
            f"Date: {now.strftime('%A, %B %d, %Y')}"
        )
        

class ReminderPlugin(BasePlugin):
    """Reminder management plugin"""
    
    def __init__(self, config):
        super().__init__("reminder", "Set and manage reminders")
        self.reminders = []
        
    async def execute(self, params: Dict[str, Any]) -> str:
        """Handle reminder operations"""
        action = params.get("action", "set")
        
        if action == "set":
            # Extract time and message
            message = params.get("message", "Reminder")
            
            # Parse time from entities
            if times := params.get("times"):
                # Simple implementation - would use proper date parsing
                reminder_time = datetime.now() + timedelta(minutes=30)
            else:
                reminder_time = datetime.now() + timedelta(minutes=30)
                
            self.reminders.append({
                "time": reminder_time,
                "message": message
            })
            
            # Schedule the reminder
            asyncio.create_task(self._schedule_reminder(reminder_time, message))
            
            return f"Reminder set for {reminder_time.strftime('%I:%M %p')}: {message}"
            
        elif action == "list":
            if not self.reminders:
                return "No active reminders"
                
            reminder_list = []
            for r in self.reminders:
                reminder_list.append(
                    f"• {r['time'].strftime('%I:%M %p')}: {r['message']}"
                )
            return "Active reminders:\n" + "\n".join(reminder_list)
            
        return "Reminder action not recognized"
        
    async def _schedule_reminder(self, reminder_time: datetime, message: str):
        """Schedule a reminder"""
        delay = (reminder_time - datetime.now()).total_seconds()
        if delay > 0:
            await asyncio.sleep(delay)
            # Would trigger notification here
            logger.info(f"Reminder triggered: {message}")
            

class SearchPlugin(BasePlugin):
    """Web search plugin"""
    
    def __init__(self, config):
        super().__init__("search", "Search the web")
        self.api_key = config.get("plugins.search.api_key")
        
    async def execute(self, params: Dict[str, Any]) -> str:
        """Perform web search"""
        query = params.get("query", "")
        
        if not query:
            return "What would you like me to search for?"
            
        # Simple implementation - would use actual search API
        return f"Here are the search results for '{query}':\n[Search results would appear here]"
        

class SystemPlugin(BasePlugin):
    """System control plugin"""
    
    def __init__(self, config):
        super().__init__("system", "Control system functions")
        self.allowed_commands = ["volume", "brightness", "sleep", "lock"]
        
    async def execute(self, params: Dict[str, Any]) -> str:
        """Execute system commands"""
        command = params.get("command", "").lower()
        
        if "volume" in command:
            # Get current volume
            try:
                result = subprocess.run(
                    ["osascript", "-e", "output volume of (get volume settings)"],
                    capture_output=True,
                    text=True
                )
                volume = result.stdout.strip()
                
                # Check if setting volume
                if numbers := params.get("numbers"):
                    new_volume = int(numbers[0])
                    subprocess.run([
                        "osascript", "-e", 
                        f"set volume output volume {new_volume}"
                    ])
                    return f"Volume set to {new_volume}%"
                    
                return f"Current volume: {volume}%"
                
            except Exception as e:
                logger.error(f"System plugin error: {e}")
                return "Failed to control volume"
                
        elif "brightness" in command:
            return "Brightness control would be implemented here"
            
        elif "sleep" in command:
            return "System sleep would be triggered here"
            
        elif "lock" in command:
            return "Screen lock would be triggered here"
            
        return "System command not recognized"
        

class SmartHomePlugin(BasePlugin):
    """Smart home control plugin"""
    
    def __init__(self, config):
        super().__init__("smart_home", "Control smart home devices")
        self.devices = {
            "living room light": {"state": "off", "brightness": 100},
            "bedroom light": {"state": "off", "brightness": 100},
            "thermostat": {"state": "auto", "temperature": 72}
        }
        
    async def execute(self, params: Dict[str, Any]) -> str:
        """Control smart home devices"""
        command = params.get("command", "").lower()
        
        # Extract device and action
        device = None
        action = None
        
        for d in self.devices:
            if d in command:
                device = d
                break
                
        if "turn on" in command or "on" in command:
            action = "on"
        elif "turn off" in command or "off" in command:
            action = "off"
        elif "dim" in command or "brightness" in command:
            action = "dim"
            
        if not device:
            return "Which device would you like to control?"
            
        if not action:
            return f"What would you like to do with the {device}?"
            
        # Simulate device control
        if action in ["on", "off"]:
            self.devices[device]["state"] = action
            return f"{device.capitalize()} turned {action}"
        elif action == "dim":
            if numbers := params.get("numbers"):
                brightness = int(numbers[0])
                self.devices[device]["brightness"] = brightness
                return f"{device.capitalize()} brightness set to {brightness}%"
                
        return "Smart home command not recognized"