#!/usr/bin/env python3
"""
JARVIS Smart Home Integration
Control lights, thermostats, smart devices through voice commands.
Supports HomeKit, Philips Hue, Nest, and more.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import requests
from pathlib import Path

# Try to import smart home libraries
try:
    import homeassistant_api
    HOMEASSISTANT_AVAILABLE = True
except ImportError:
    HOMEASSISTANT_AVAILABLE = False

try:
    from phue import Bridge as HueBridge
    HUE_AVAILABLE = True
except ImportError:
    HUE_AVAILABLE = False

try:
    import pyHomeKit
    HOMEKIT_AVAILABLE = True
except ImportError:
    HOMEKIT_AVAILABLE = False

logger = logging.getLogger("jarvis.smarthome")


@dataclass
class SmartDevice:
    """Represents a smart home device"""
    device_id: str
    name: str
    device_type: str  # light, switch, thermostat, sensor, camera
    room: Optional[str]
    capabilities: List[str]
    state: Dict[str, Any]
    brand: str
    is_online: bool = True
    

@dataclass
class Scene:
    """Represents a smart home scene"""
    scene_id: str
    name: str
    description: str
    devices: List[Dict[str, Any]]  # Device states for the scene
    

class SmartHomeProvider(ABC):
    """Abstract base class for smart home providers"""
    
    @abstractmethod
    async def discover_devices(self) -> List[SmartDevice]:
        """Discover available devices"""
        pass
        
    @abstractmethod
    async def get_device_state(self, device_id: str) -> Dict[str, Any]:
        """Get current state of a device"""
        pass
        
    @abstractmethod
    async def set_device_state(self, device_id: str, state: Dict[str, Any]) -> bool:
        """Set device state"""
        pass
        
    @abstractmethod
    async def execute_scene(self, scene_id: str) -> bool:
        """Execute a scene"""
        pass


class HomeKitProvider(SmartHomeProvider):
    """HomeKit integration for Apple devices"""
    
    def __init__(self):
        self.connected = False
        if HOMEKIT_AVAILABLE:
            self.homekit = pyHomeKit.HomeKit()
            
    async def discover_devices(self) -> List[SmartDevice]:
        """Discover HomeKit devices"""
        devices = []
        
        if HOMEKIT_AVAILABLE and self.connected:
            # Discover HomeKit accessories
            accessories = self.homekit.get_accessories()
            
            for accessory in accessories:
                device = SmartDevice(
                    device_id=accessory.aid,
                    name=accessory.name,
                    device_type=self._get_device_type(accessory),
                    room=accessory.room,
                    capabilities=self._get_capabilities(accessory),
                    state=accessory.get_characteristics(),
                    brand="HomeKit"
                )
                devices.append(device)
                
        return devices
        
    def _get_device_type(self, accessory) -> str:
        """Determine device type from accessory"""
        # Simplified type detection
        if "light" in accessory.model.lower():
            return "light"
        elif "thermostat" in accessory.model.lower():
            return "thermostat"
        elif "switch" in accessory.model.lower():
            return "switch"
        else:
            return "unknown"
            
    def _get_capabilities(self, accessory) -> List[str]:
        """Get device capabilities"""
        capabilities = []
        characteristics = accessory.get_service_characteristics()
        
        if "On" in characteristics:
            capabilities.append("on_off")
        if "Brightness" in characteristics:
            capabilities.append("brightness")
        if "Hue" in characteristics:
            capabilities.append("color")
        if "CurrentTemperature" in characteristics:
            capabilities.append("temperature")
            
        return capabilities
        
    async def get_device_state(self, device_id: str) -> Dict[str, Any]:
        """Get HomeKit device state"""
        if HOMEKIT_AVAILABLE and self.connected:
            accessory = self.homekit.get_accessory(device_id)
            if accessory:
                return accessory.get_characteristics()
        return {}
        
    async def set_device_state(self, device_id: str, state: Dict[str, Any]) -> bool:
        """Set HomeKit device state"""
        if HOMEKIT_AVAILABLE and self.connected:
            accessory = self.homekit.get_accessory(device_id)
            if accessory:
                return accessory.set_characteristics(state)
        return False
        
    async def execute_scene(self, scene_id: str) -> bool:
        """Execute HomeKit scene"""
        if HOMEKIT_AVAILABLE and self.connected:
            return self.homekit.trigger_scene(scene_id)
        return False


class HueProvider(SmartHomeProvider):
    """Philips Hue integration"""
    
    def __init__(self, bridge_ip: Optional[str] = None):
        self.bridge_ip = bridge_ip
        self.bridge = None
        self.connected = False
        
        if HUE_AVAILABLE and bridge_ip:
            try:
                self.bridge = HueBridge(bridge_ip)
                self.bridge.connect()
                self.connected = True
            except Exception as e:
                logger.error(f"Failed to connect to Hue bridge: {e}")
                
    async def discover_devices(self) -> List[SmartDevice]:
        """Discover Hue lights"""
        devices = []
        
        if self.connected and self.bridge:
            lights = self.bridge.get_light_objects()
            
            for light in lights:
                device = SmartDevice(
                    device_id=str(light.light_id),
                    name=light.name,
                    device_type="light",
                    room=None,  # Would need room mapping
                    capabilities=self._get_light_capabilities(light),
                    state={
                        "on": light.on,
                        "brightness": light.brightness,
                        "hue": getattr(light, 'hue', None),
                        "saturation": getattr(light, 'saturation', None)
                    },
                    brand="Philips Hue"
                )
                devices.append(device)
                
        return devices
        
    def _get_light_capabilities(self, light) -> List[str]:
        """Get Hue light capabilities"""
        capabilities = ["on_off", "brightness"]
        
        if hasattr(light, 'hue'):
            capabilities.append("color")
        if hasattr(light, 'colortemp'):
            capabilities.append("color_temperature")
            
        return capabilities
        
    async def get_device_state(self, device_id: str) -> Dict[str, Any]:
        """Get Hue light state"""
        if self.connected and self.bridge:
            light = self.bridge.get_light(int(device_id))
            return light
        return {}
        
    async def set_device_state(self, device_id: str, state: Dict[str, Any]) -> bool:
        """Set Hue light state"""
        if self.connected and self.bridge:
            try:
                self.bridge.set_light(int(device_id), state)
                return True
            except Exception as e:
                logger.error(f"Failed to set Hue light state: {e}")
        return False
        
    async def execute_scene(self, scene_id: str) -> bool:
        """Execute Hue scene"""
        if self.connected and self.bridge:
            try:
                self.bridge.activate_scene(scene_id)
                return True
            except Exception as e:
                logger.error(f"Failed to execute Hue scene: {e}")
        return False


class SmartHomeIntegration:
    """Main smart home integration for JARVIS"""
    
    def __init__(self):
        self.providers: List[SmartHomeProvider] = []
        self.devices: Dict[str, SmartDevice] = {}
        self.scenes: Dict[str, Scene] = {}
        self.config_file = Path.home() / ".jarvis" / "smart_home_config.json"
        self.automation_rules = []
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize providers
        self._initialize_providers()
        
        # Discover devices
        asyncio.create_task(self.discover_all_devices())
        
    def _load_config(self) -> Dict[str, Any]:
        """Load smart home configuration"""
        default_config = {
            "providers": {
                "homekit": {"enabled": True},
                "hue": {"enabled": False, "bridge_ip": None},
                "homeassistant": {"enabled": False, "url": None, "token": None}
            },
            "scenes": [],
            "automation_rules": []
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                
        return default_config
        
    def _save_config(self):
        """Save configuration"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            
    def _initialize_providers(self):
        """Initialize enabled providers"""
        config = self.config.get("providers", {})
        
        # HomeKit
        if config.get("homekit", {}).get("enabled") and HOMEKIT_AVAILABLE:
            self.providers.append(HomeKitProvider())
            
        # Philips Hue
        hue_config = config.get("hue", {})
        if hue_config.get("enabled") and HUE_AVAILABLE and hue_config.get("bridge_ip"):
            self.providers.append(HueProvider(hue_config["bridge_ip"]))
            
        # Add more providers as needed
        
    async def discover_all_devices(self):
        """Discover devices from all providers"""
        self.devices.clear()
        
        for provider in self.providers:
            try:
                devices = await provider.discover_devices()
                for device in devices:
                    self.devices[device.device_id] = device
                    logger.info(f"Discovered {device.device_type}: {device.name}")
            except Exception as e:
                logger.error(f"Failed to discover devices from {provider.__class__.__name__}: {e}")
                
        logger.info(f"Total devices discovered: {len(self.devices)}")
        
    def get_devices_by_room(self, room: str) -> List[SmartDevice]:
        """Get all devices in a specific room"""
        return [d for d in self.devices.values() if d.room and d.room.lower() == room.lower()]
        
    def get_devices_by_type(self, device_type: str) -> List[SmartDevice]:
        """Get all devices of a specific type"""
        return [d for d in self.devices.values() if d.device_type == device_type]
        
    def find_device(self, name: str) -> Optional[SmartDevice]:
        """Find device by name (fuzzy matching)"""
        name_lower = name.lower()
        
        # Exact match
        for device in self.devices.values():
            if device.name.lower() == name_lower:
                return device
                
        # Partial match
        for device in self.devices.values():
            if name_lower in device.name.lower():
                return device
                
        return None
        
    async def control_device(self, device_name: str, action: str, value: Any = None) -> Tuple[bool, str]:
        """Control a device by name and action"""
        device = self.find_device(device_name)
        
        if not device:
            return False, f"Device '{device_name}' not found"
            
        # Find the provider for this device
        provider = self._get_provider_for_device(device)
        if not provider:
            return False, f"No provider available for {device.brand} devices"
            
        # Build state update
        state_update = {}
        
        if action == "on":
            state_update["on"] = True
        elif action == "off":
            state_update["on"] = False
        elif action == "brightness" and value is not None:
            state_update["brightness"] = int(value)
        elif action == "color" and value is not None:
            # Value could be color name or RGB
            state_update.update(self._parse_color(value))
        elif action == "temperature" and value is not None:
            state_update["targetTemperature"] = float(value)
        else:
            return False, f"Unknown action '{action}' for device"
            
        # Apply state update
        success = await provider.set_device_state(device.device_id, state_update)
        
        if success:
            # Update local state
            device.state.update(state_update)
            return True, f"{device.name} {action} successful"
        else:
            return False, f"Failed to control {device.name}"
            
    def _get_provider_for_device(self, device: SmartDevice) -> Optional[SmartHomeProvider]:
        """Get the provider for a specific device brand"""
        for provider in self.providers:
            if isinstance(provider, HomeKitProvider) and device.brand == "HomeKit":
                return provider
            elif isinstance(provider, HueProvider) and device.brand == "Philips Hue":
                return provider
        return None
        
    def _parse_color(self, color_value: Any) -> Dict[str, Any]:
        """Parse color value to device format"""
        colors = {
            "red": {"hue": 0, "saturation": 254},
            "blue": {"hue": 46920, "saturation": 254},
            "green": {"hue": 25500, "saturation": 254},
            "yellow": {"hue": 12750, "saturation": 254},
            "purple": {"hue": 50000, "saturation": 254},
            "orange": {"hue": 5000, "saturation": 254},
            "white": {"hue": 0, "saturation": 0}
        }
        
        if isinstance(color_value, str) and color_value.lower() in colors:
            return colors[color_value.lower()]
            
        # Could add RGB parsing here
        return {}
        
    async def control_room(self, room: str, action: str, value: Any = None) -> Tuple[bool, str]:
        """Control all devices in a room"""
        devices = self.get_devices_by_room(room)
        
        if not devices:
            return False, f"No devices found in {room}"
            
        success_count = 0
        for device in devices:
            provider = self._get_provider_for_device(device)
            if provider:
                state_update = {}
                
                if action == "on":
                    state_update["on"] = True
                elif action == "off":
                    state_update["on"] = False
                elif action == "brightness" and value and "brightness" in device.capabilities:
                    state_update["brightness"] = int(value)
                    
                if state_update:
                    if await provider.set_device_state(device.device_id, state_update):
                        success_count += 1
                        
        return success_count > 0, f"Controlled {success_count}/{len(devices)} devices in {room}"
        
    async def execute_scene(self, scene_name: str) -> Tuple[bool, str]:
        """Execute a saved scene"""
        scene = None
        scene_name_lower = scene_name.lower()
        
        # Find scene
        for s in self.scenes.values():
            if s.name.lower() == scene_name_lower:
                scene = s
                break
                
        if not scene:
            return False, f"Scene '{scene_name}' not found"
            
        # Apply device states
        success_count = 0
        for device_state in scene.devices:
            device = self.devices.get(device_state["device_id"])
            if device:
                provider = self._get_provider_for_device(device)
                if provider:
                    if await provider.set_device_state(device.device_id, device_state["state"]):
                        success_count += 1
                        
        return success_count > 0, f"Scene '{scene_name}' activated"
        
    def create_scene(self, name: str, description: str = "") -> Scene:
        """Create a new scene from current device states"""
        scene_devices = []
        
        # Capture current state of all devices
        for device in self.devices.values():
            if device.is_online and device.state.get("on", False):
                scene_devices.append({
                    "device_id": device.device_id,
                    "state": device.state.copy()
                })
                
        scene = Scene(
            scene_id=f"scene_{datetime.now().timestamp()}",
            name=name,
            description=description,
            devices=scene_devices
        )
        
        self.scenes[scene.scene_id] = scene
        
        # Save to config
        self.config["scenes"].append({
            "id": scene.scene_id,
            "name": scene.name,
            "description": scene.description,
            "devices": scene.devices
        })
        self._save_config()
        
        return scene
        
    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of smart home status"""
        summary = {
            "total_devices": len(self.devices),
            "online_devices": sum(1 for d in self.devices.values() if d.is_online),
            "devices_on": sum(1 for d in self.devices.values() if d.state.get("on", False)),
            "by_type": {},
            "by_room": {}
        }
        
        # Count by type
        for device in self.devices.values():
            device_type = device.device_type
            if device_type not in summary["by_type"]:
                summary["by_type"][device_type] = {"total": 0, "on": 0}
            summary["by_type"][device_type]["total"] += 1
            if device.state.get("on", False):
                summary["by_type"][device_type]["on"] += 1
                
        # Count by room
        for device in self.devices.values():
            if device.room:
                if device.room not in summary["by_room"]:
                    summary["by_room"][device.room] = {"total": 0, "on": 0}
                summary["by_room"][device.room]["total"] += 1
                if device.state.get("on", False):
                    summary["by_room"][device.room]["on"] += 1
                    
        return summary
        
    async def suggest_energy_savings(self) -> List[str]:
        """Suggest energy saving opportunities"""
        suggestions = []
        
        # Check for lights left on in empty rooms
        lights_on = [d for d in self.devices.values() 
                    if d.device_type == "light" and d.state.get("on", False)]
        
        if len(lights_on) > 5:
            suggestions.append(f"You have {len(lights_on)} lights on. Consider turning off unused lights.")
            
        # Check for high brightness
        high_brightness = [d for d in lights_on 
                          if d.state.get("brightness", 0) > 200]
        
        if high_brightness:
            suggestions.append(f"{len(high_brightness)} lights are at high brightness. Dimming saves energy.")
            
        # Check thermostat settings
        thermostats = [d for d in self.devices.values() if d.device_type == "thermostat"]
        for thermostat in thermostats:
            target_temp = thermostat.state.get("targetTemperature", 20)
            if target_temp > 24:  # Too hot
                suggestions.append(f"Consider lowering {thermostat.name} temperature for energy savings.")
            elif target_temp < 18:  # Too cold
                suggestions.append(f"Consider raising {thermostat.name} temperature for energy savings.")
                
        return suggestions


# Command processor for JARVIS
class SmartHomeCommandProcessor:
    """Process smart home voice commands"""
    
    def __init__(self, smart_home: SmartHomeIntegration):
        self.smart_home = smart_home
        self.command_patterns = {
            "turn_on": [
                r"turn on (?:the )?(.+)",
                r"switch on (?:the )?(.+)",
                r"lights on(?: in (?:the )?(.+))?"
            ],
            "turn_off": [
                r"turn off (?:the )?(.+)",
                r"switch off (?:the )?(.+)",
                r"lights off(?: in (?:the )?(.+))?"
            ],
            "dim": [
                r"dim (?:the )?(.+) to (\d+)",
                r"set (?:the )?(.+) brightness to (\d+)",
                r"make (?:the )?(.+) dimmer"
            ],
            "color": [
                r"change (?:the )?(.+) to (.+)",
                r"make (?:the )?(.+) (.+)",
                r"set (?:the )?(.+) color to (.+)"
            ],
            "scene": [
                r"(?:activate|set|enable) (.+) scene",
                r"scene (.+)"
            ],
            "temperature": [
                r"set (?:the )?temperature to (\d+)",
                r"make it (?:warmer|cooler)",
                r"(?:increase|decrease) (?:the )?temperature"
            ],
            "status": [
                r"(?:what's|what is) (?:the )?status",
                r"how many (?:lights|devices) are on",
                r"smart home status"
            ]
        }
        
    async def process_command(self, command: str) -> Tuple[bool, str]:
        """Process smart home command"""
        command_lower = command.lower().strip()
        
        # Turn on/off commands
        for pattern in self.command_patterns["turn_on"]:
            match = re.match(pattern, command_lower)
            if match:
                target = match.group(1) if match.lastindex >= 1 else "all lights"
                
                # Check if it's a room
                if any(room_word in target for room_word in ["room", "bedroom", "kitchen", "living", "bathroom"]):
                    return await self.smart_home.control_room(target, "on")
                else:
                    return await self.smart_home.control_device(target, "on")
                    
        for pattern in self.command_patterns["turn_off"]:
            match = re.match(pattern, command_lower)
            if match:
                target = match.group(1) if match.lastindex >= 1 else "all lights"
                
                if any(room_word in target for room_word in ["room", "bedroom", "kitchen", "living", "bathroom"]):
                    return await self.smart_home.control_room(target, "off")
                else:
                    return await self.smart_home.control_device(target, "off")
                    
        # Brightness commands
        for pattern in self.command_patterns["dim"]:
            match = re.match(pattern, command_lower)
            if match:
                device_name = match.group(1)
                brightness = int(match.group(2)) if match.lastindex >= 2 else 50
                return await self.smart_home.control_device(device_name, "brightness", brightness)
                
        # Color commands
        for pattern in self.command_patterns["color"]:
            match = re.match(pattern, command_lower)
            if match:
                device_name = match.group(1)
                color = match.group(2)
                return await self.smart_home.control_device(device_name, "color", color)
                
        # Scene commands
        for pattern in self.command_patterns["scene"]:
            match = re.match(pattern, command_lower)
            if match:
                scene_name = match.group(1)
                return await self.smart_home.execute_scene(scene_name)
                
        # Status command
        if any(pattern in command_lower for pattern in ["status", "how many", "what's on"]):
            summary = self.smart_home.get_status_summary()
            
            response = f"You have {summary['devices_on']} devices on out of {summary['total_devices']} total. "
            
            if summary['by_type'].get('light'):
                lights = summary['by_type']['light']
                response += f"{lights['on']} lights are on. "
                
            return True, response
            
        # Energy suggestions
        if "energy" in command_lower or "save" in command_lower:
            suggestions = await self.smart_home.suggest_energy_savings()
            if suggestions:
                return True, "Here are some energy saving suggestions: " + " ".join(suggestions)
            else:
                return True, "Your energy usage looks good!"
                
        return False, "I didn't understand that smart home command"


# Integration function for JARVIS
def integrate_smart_home_with_jarvis(jarvis_instance):
    """Integrate smart home capabilities with JARVIS"""
    smart_home = SmartHomeIntegration()
    processor = SmartHomeCommandProcessor(smart_home)
    
    # Add to JARVIS command processing
    jarvis_instance.smart_home = smart_home
    jarvis_instance.smart_home_processor = processor
    
    logger.info("Smart home integration initialized")
    
    return smart_home, processor