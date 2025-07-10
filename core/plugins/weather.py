#!/usr/bin/env python3
"""
Weather Plugin for JARVIS
Provides weather information and forecasts using OpenWeatherMap API
"""

from typing import Dict, Any, Tuple, Optional
import re
import aiohttp
import json
from datetime import datetime
from core.plugin_system import JARVISPlugin, PluginMetadata, PluginCommand


class WeatherPlugin(JARVISPlugin):
    """Weather information plugin using OpenWeatherMap API"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Weather",
            version="1.0.0",
            author="JARVIS Team",
            description="Get current weather and forecasts for any location",
            category="utility",
            keywords=["weather", "forecast", "temperature", "climate"],
            requirements=["aiohttp"],
            permissions=["network"],
            config_schema={
                "api_key": {"type": "string", "required": True, "description": "OpenWeatherMap API key"},
                "units": {"type": "string", "default": "metric", "enum": ["metric", "imperial"]},
                "default_location": {"type": "string", "default": "New York", "description": "Default city for weather queries"}
            }
        )
        
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the weather plugin"""
        self.config = config
        self.api_key = config.get("api_key")
        self.units = config.get("units", "metric")
        self.default_location = config.get("default_location", "New York")
        
        if not self.api_key:
            self.logger.warning("No OpenWeatherMap API key configured. Get one at https://openweathermap.org/api")
            # Plugin can still load but will return demo data
        
        # Register commands
        self.register_command(PluginCommand(
            name="current_weather",
            patterns=[
                r"(?:what(?:'s| is) the )?weather(?:\s+(?:in|at|for)\s+(.+?))?(?:\s+today)?",
                r"(?:how(?:'s| is) the )?weather(?:\s+(?:in|at|for)\s+(.+?))?",
                r"temperature(?:\s+(?:in|at|for)\s+(.+?))?",
                r"(?:is it|will it be) (?:hot|cold|warm|cool)(?:\s+(?:in|at|for)\s+(.+?))?"
            ],
            description="Get current weather for a location",
            parameters={
                "location": {"type": "string", "description": "City name or location", "optional": True}
            },
            examples=[
                "what's the weather",
                "weather in London",
                "temperature in Tokyo",
                "how's the weather today"
            ],
            handler=self.handle_current_weather
        ))
        
        self.register_command(PluginCommand(
            name="weather_forecast",
            patterns=[
                r"weather forecast(?:\s+(?:for|in)\s+(.+?))?(?:\s+for\s+(\d+)\s+days?)?",
                r"(?:what(?:'s| is) the )?forecast(?:\s+(?:for|in)\s+(.+?))?",
                r"weather (?:tomorrow|next week)(?:\s+(?:in|at|for)\s+(.+?))?"
            ],
            description="Get weather forecast",
            parameters={
                "location": {"type": "string", "description": "City name or location", "optional": True},
                "days": {"type": "integer", "description": "Number of days", "optional": True, "default": 3}
            },
            examples=[
                "weather forecast",
                "forecast for Paris",
                "weather tomorrow in Berlin",
                "weather forecast for 5 days"
            ],
            handler=self.handle_forecast
        ))
        
        self.logger.info("Weather plugin initialized")
        return True
        
    async def shutdown(self):
        """Clean up resources"""
        self.logger.info("Weather plugin shutting down")
        
    async def handle_current_weather(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Handle current weather requests"""
        try:
            # Extract location from match
            location = None
            for group in match.groups():
                if group:
                    location = group.strip()
                    break
                    
            if not location:
                location = self.default_location
            
            # Get weather data
            weather_data = await self._get_current_weather(location)
            
            if weather_data:
                return True, self._format_current_weather(weather_data, location)
            else:
                return True, f"I couldn't fetch weather data for {location}. Please check the location name or try again later."
                
        except Exception as e:
            self.logger.error(f"Error getting weather: {e}")
            return False, f"Sorry, I encountered an error getting the weather: {str(e)}"
            
    async def handle_forecast(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Handle weather forecast requests"""
        try:
            # Extract parameters
            location = None
            days = 3
            
            groups = match.groups()
            if groups:
                for i, group in enumerate(groups):
                    if group:
                        if group.isdigit():
                            days = min(int(group), 5)  # Max 5 days
                        else:
                            location = group.strip()
                            
            if not location:
                location = self.default_location
                
            # Check for special keywords
            if "tomorrow" in command.lower():
                days = 1
            elif "week" in command.lower():
                days = 7
                
            # Get forecast data
            forecast_data = await self._get_forecast(location, days)
            
            if forecast_data:
                return True, self._format_forecast(forecast_data, location, days)
            else:
                return True, f"I couldn't fetch forecast data for {location}. Please check the location name or try again later."
                
        except Exception as e:
            self.logger.error(f"Error getting forecast: {e}")
            return False, f"Sorry, I encountered an error getting the forecast: {str(e)}"
            
    async def _get_current_weather(self, location: str) -> Optional[Dict[str, Any]]:
        """Fetch current weather from API"""
        if not self.api_key:
            # Return demo data
            return self._get_demo_weather(location)
            
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.openweathermap.org/data/2.5/weather"
                params = {
                    "q": location,
                    "appid": self.api_key,
                    "units": self.units
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        self.logger.error(f"Weather API error: {response.status}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error fetching weather: {e}")
            return self._get_demo_weather(location)
            
    async def _get_forecast(self, location: str, days: int) -> Optional[Dict[str, Any]]:
        """Fetch weather forecast from API"""
        if not self.api_key:
            # Return demo data
            return self._get_demo_forecast(location, days)
            
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.openweathermap.org/data/2.5/forecast"
                params = {
                    "q": location,
                    "appid": self.api_key,
                    "units": self.units,
                    "cnt": days * 8  # 8 data points per day (3-hour intervals)
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        self.logger.error(f"Forecast API error: {response.status}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error fetching forecast: {e}")
            return self._get_demo_forecast(location, days)
            
    def _format_current_weather(self, data: Dict[str, Any], location: str) -> str:
        """Format current weather data for display"""
        if "main" in data:  # Real API data
            temp = data["main"]["temp"]
            feels_like = data["main"]["feels_like"]
            humidity = data["main"]["humidity"]
            description = data["weather"][0]["description"]
            wind_speed = data["wind"]["speed"]
            
            unit_symbol = "°C" if self.units == "metric" else "°F"
            wind_unit = "m/s" if self.units == "metric" else "mph"
            
            response = f"Current weather in {location}:\n"
            response += f"🌡️ Temperature: {temp}{unit_symbol} (feels like {feels_like}{unit_symbol})\n"
            response += f"☁️ Conditions: {description.capitalize()}\n"
            response += f"💧 Humidity: {humidity}%\n"
            response += f"💨 Wind: {wind_speed} {wind_unit}"
            
            # Add weather emoji based on condition
            weather_id = data["weather"][0]["id"]
            if weather_id < 300:  # Thunderstorm
                response = "⛈️ " + response
            elif weather_id < 600:  # Rain
                response = "🌧️ " + response
            elif weather_id < 700:  # Snow
                response = "❄️ " + response
            elif weather_id == 800:  # Clear
                response = "☀️ " + response
            else:  # Clouds
                response = "☁️ " + response
                
        else:  # Demo data
            response = data.get("formatted", f"Demo weather for {location}")
            
        return response
        
    def _format_forecast(self, data: Dict[str, Any], location: str, days: int) -> str:
        """Format forecast data for display"""
        if "list" in data:  # Real API data
            response = f"Weather forecast for {location}:\n\n"
            
            # Group by day
            daily_data = {}
            for item in data["list"]:
                date = datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d")
                if date not in daily_data:
                    daily_data[date] = []
                daily_data[date].append(item)
                
            # Format each day
            unit_symbol = "°C" if self.units == "metric" else "°F"
            for i, (date, items) in enumerate(list(daily_data.items())[:days]):
                if i > 0:
                    response += "\n"
                    
                # Calculate daily summary
                temps = [item["main"]["temp"] for item in items]
                conditions = [item["weather"][0]["main"] for item in items]
                most_common_condition = max(set(conditions), key=conditions.count)
                
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                day_name = date_obj.strftime("%A, %B %d")
                
                response += f"📅 {day_name}:\n"
                response += f"   High: {max(temps):.1f}{unit_symbol}, Low: {min(temps):.1f}{unit_symbol}\n"
                response += f"   Conditions: {most_common_condition}"
                
        else:  # Demo data
            response = data.get("formatted", f"Demo forecast for {location}")
            
        return response
        
    def _get_demo_weather(self, location: str) -> Dict[str, Any]:
        """Return demo weather data when API key is not available"""
        return {
            "formatted": f"🌤️ Demo Weather for {location}:\n"
                        f"🌡️ Temperature: 22°C (feels like 21°C)\n"
                        f"☁️ Conditions: Partly cloudy\n"
                        f"💧 Humidity: 65%\n"
                        f"💨 Wind: 5 m/s\n\n"
                        f"ℹ️ This is demo data. Configure an API key for real weather information."
        }
        
    def _get_demo_forecast(self, location: str, days: int) -> Dict[str, Any]:
        """Return demo forecast data when API key is not available"""
        forecast = f"🌤️ Demo Forecast for {location}:\n\n"
        
        for i in range(min(days, 3)):
            date = datetime.now()
            date = date.replace(day=date.day + i)
            day_name = date.strftime("%A, %B %d")
            
            forecast += f"📅 {day_name}:\n"
            forecast += f"   High: {20 + i*2}°C, Low: {15 + i}°C\n"
            forecast += f"   Conditions: {'Sunny' if i == 0 else 'Partly cloudy' if i == 1 else 'Cloudy'}\n\n"
            
        forecast += "ℹ️ This is demo data. Configure an API key for real weather forecasts."
        
        return {"formatted": forecast}