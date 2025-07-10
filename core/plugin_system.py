#!/usr/bin/env python3
"""
JARVIS Plugin System
Extensible architecture for custom skills and integrations.
"""

import os
import sys
import json
import importlib
import inspect
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod
import asyncio
import yaml
import hashlib
from datetime import datetime

logger = logging.getLogger("jarvis.plugins")


@dataclass
class PluginMetadata:
    """Plugin metadata and configuration"""
    name: str
    version: str
    author: str
    description: str
    category: str  # automation, productivity, entertainment, utility, integration
    keywords: List[str]
    requirements: List[str]
    permissions: List[str]  # file_access, network, system, etc.
    config_schema: Dict[str, Any]
    

@dataclass
class PluginCommand:
    """Represents a plugin command"""
    name: str
    patterns: List[str]  # Regex patterns to match
    description: str
    parameters: Dict[str, Any]
    examples: List[str]
    handler: Callable
    

class JARVISPlugin(ABC):
    """Base class for all JARVIS plugins"""
    
    def __init__(self, jarvis_instance=None):
        self.jarvis = jarvis_instance
        self.config = {}
        self.enabled = True
        self.commands: List[PluginCommand] = []
        self.logger = logging.getLogger(f"jarvis.plugin.{self.get_metadata().name}")
        
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        pass
        
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with configuration"""
        pass
        
    @abstractmethod
    async def shutdown(self):
        """Clean shutdown of the plugin"""
        pass
        
    def register_command(self, command: PluginCommand):
        """Register a command handler"""
        self.commands.append(command)
        
    async def handle_command(self, command: str, match: Any) -> Tuple[bool, str]:
        """Handle a matched command"""
        # Find matching command handler
        for cmd in self.commands:
            for pattern in cmd.patterns:
                if pattern in str(match.re.pattern):
                    return await cmd.handler(command, match)
        return False, "Command not handled"
        
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
        
    def emit_event(self, event_name: str, data: Any):
        """Emit an event that other plugins can listen to"""
        if self.jarvis:
            self.jarvis.plugin_manager.emit_event(event_name, data, self)
            
    def subscribe_event(self, event_name: str, handler: Callable):
        """Subscribe to events from other plugins"""
        if self.jarvis:
            self.jarvis.plugin_manager.subscribe_event(event_name, handler, self)


class PluginManager:
    """Manages loading, execution, and lifecycle of plugins"""
    
    def __init__(self, jarvis_instance=None):
        self.jarvis = jarvis_instance
        self.plugins_dir = Path.home() / ".jarvis" / "plugins"
        self.plugins: Dict[str, JARVISPlugin] = {}
        self.event_subscribers: Dict[str, List[Tuple[Callable, JARVISPlugin]]] = {}
        self.command_registry: Dict[str, List[Tuple[str, JARVISPlugin]]] = {}
        
        # Create plugins directory
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        # Load plugin configurations
        self.config_file = self.plugins_dir / "plugins_config.json"
        self.load_config()
        
    def load_config(self):
        """Load plugin configurations"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load plugin config: {e}")
                self.config = {}
        else:
            self.config = {}
            
    def save_config(self):
        """Save plugin configurations"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save plugin config: {e}")
            
    async def discover_plugins(self) -> List[str]:
        """Discover available plugins"""
        discovered = []
        
        # Check built-in plugins directory
        builtin_dir = Path(__file__).parent / "plugins"
        if builtin_dir.exists():
            for plugin_file in builtin_dir.glob("*.py"):
                if plugin_file.name != "__init__.py":
                    discovered.append(f"builtin.{plugin_file.stem}")
                    
        # Check user plugins directory
        for plugin_file in self.plugins_dir.glob("*.py"):
            if plugin_file.name != "__init__.py":
                discovered.append(f"user.{plugin_file.stem}")
                
        # Check installed packages
        for entry_point in self._get_entry_points():
            discovered.append(f"package.{entry_point}")
            
        return discovered
        
    def _get_entry_points(self) -> List[str]:
        """Get plugins from installed packages"""
        entry_points = []
        
        try:
            import pkg_resources
            for entry_point in pkg_resources.iter_entry_points('jarvis_plugins'):
                entry_points.append(entry_point.name)
        except ImportError:
            pass
            
        return entry_points
        
    async def load_plugin(self, plugin_id: str) -> bool:
        """Load a specific plugin"""
        try:
            # Parse plugin ID
            source, name = plugin_id.split('.', 1)
            
            # Import plugin module
            if source == "builtin":
                module = importlib.import_module(f"core.plugins.{name}")
            elif source == "user":
                # Add user plugins directory to path
                sys.path.insert(0, str(self.plugins_dir))
                module = importlib.import_module(name)
                sys.path.pop(0)
            elif source == "package":
                # Load from entry point
                import pkg_resources
                entry_point = next(
                    ep for ep in pkg_resources.iter_entry_points('jarvis_plugins')
                    if ep.name == name
                )
                plugin_class = entry_point.load()
            else:
                raise ValueError(f"Unknown plugin source: {source}")
                
            # Find plugin class
            if source != "package":
                plugin_class = None
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, JARVISPlugin) and 
                        obj != JARVISPlugin):
                        plugin_class = obj
                        break
                        
            if not plugin_class:
                raise ValueError("No plugin class found")
                
            # Instantiate plugin
            plugin = plugin_class(self.jarvis)
            metadata = plugin.get_metadata()
            
            # Check permissions
            if not self._check_permissions(metadata.permissions):
                logger.warning(f"Plugin {metadata.name} requires permissions: {metadata.permissions}")
                return False
                
            # Load plugin config
            plugin_config = self.config.get(metadata.name, {})
            
            # Initialize plugin
            success = await plugin.initialize(plugin_config)
            
            if success:
                self.plugins[metadata.name] = plugin
                
                # Register commands
                for command in plugin.commands:
                    for pattern in command.patterns:
                        if pattern not in self.command_registry:
                            self.command_registry[pattern] = []
                        self.command_registry[pattern].append((command.name, plugin))
                        
                logger.info(f"Loaded plugin: {metadata.name} v{metadata.version}")
                return True
            else:
                logger.error(f"Failed to initialize plugin: {metadata.name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_id}: {e}")
            return False
            
    def _check_permissions(self, permissions: List[str]) -> bool:
        """Check if plugin permissions are acceptable"""
        # In a production system, this would prompt user for permission
        dangerous_permissions = ["system", "root_access"]
        
        for perm in permissions:
            if perm in dangerous_permissions:
                # Would prompt user here
                return True  # For now, allow all
                
        return True
        
    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        if plugin_name not in self.plugins:
            return False
            
        try:
            plugin = self.plugins[plugin_name]
            
            # Shutdown plugin
            await plugin.shutdown()
            
            # Remove from registries
            del self.plugins[plugin_name]
            
            # Remove commands
            for pattern, handlers in list(self.command_registry.items()):
                self.command_registry[pattern] = [
                    (cmd, p) for cmd, p in handlers if p != plugin
                ]
                if not self.command_registry[pattern]:
                    del self.command_registry[pattern]
                    
            # Remove event subscriptions
            for event, subscribers in list(self.event_subscribers.items()):
                self.event_subscribers[event] = [
                    (h, p) for h, p in subscribers if p != plugin
                ]
                if not self.event_subscribers[event]:
                    del self.event_subscribers[event]
                    
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False
            
    async def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin"""
        await self.unload_plugin(plugin_name)
        
        # Find plugin ID
        discovered = await self.discover_plugins()
        for plugin_id in discovered:
            if plugin_name in plugin_id:
                return await self.load_plugin(plugin_id)
                
        return False
        
    def get_plugin(self, plugin_name: str) -> Optional[JARVISPlugin]:
        """Get a loaded plugin instance"""
        return self.plugins.get(plugin_name)
        
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all loaded plugins"""
        plugins_list = []
        
        for name, plugin in self.plugins.items():
            metadata = plugin.get_metadata()
            plugins_list.append({
                "name": metadata.name,
                "version": metadata.version,
                "author": metadata.author,
                "description": metadata.description,
                "category": metadata.category,
                "enabled": plugin.enabled,
                "commands": len(plugin.commands)
            })
            
        return plugins_list
        
    async def process_command(self, command: str) -> Optional[Tuple[bool, str]]:
        """Process command through plugins"""
        import re
        
        # Check each registered pattern
        for pattern, handlers in self.command_registry.items():
            match = re.match(pattern, command, re.IGNORECASE)
            if match:
                # Try each handler for this pattern
                for cmd_name, plugin in handlers:
                    if plugin.enabled:
                        try:
                            success, response = await plugin.handle_command(command, match)
                            if success:
                                return success, response
                        except Exception as e:
                            logger.error(f"Plugin {plugin.get_metadata().name} error: {e}")
                            
        return None
        
    def emit_event(self, event_name: str, data: Any, source_plugin: JARVISPlugin):
        """Emit an event to subscribers"""
        if event_name in self.event_subscribers:
            for handler, plugin in self.event_subscribers[event_name]:
                if plugin != source_plugin and plugin.enabled:
                    try:
                        # Run handler asynchronously
                        if asyncio.iscoroutinefunction(handler):
                            asyncio.create_task(handler(data))
                        else:
                            handler(data)
                    except Exception as e:
                        logger.error(f"Event handler error in {plugin.get_metadata().name}: {e}")
                        
    def subscribe_event(self, event_name: str, handler: Callable, plugin: JARVISPlugin):
        """Subscribe to an event"""
        if event_name not in self.event_subscribers:
            self.event_subscribers[event_name] = []
            
        self.event_subscribers[event_name].append((handler, plugin))
        

# Plugin development utilities
class PluginDevelopmentKit:
    """Tools for plugin developers"""
    
    @staticmethod
    def create_plugin_template(name: str, category: str = "utility") -> str:
        """Generate plugin template code"""
        template = f'''#!/usr/bin/env python3
"""
{name} Plugin for JARVIS
Description: Your plugin description here
"""

from typing import Dict, Any, Tuple
import re
from core.plugin_system import JARVISPlugin, PluginMetadata, PluginCommand


class {name.replace(" ", "")}Plugin(JARVISPlugin):
    """Your plugin implementation"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="{name}",
            version="1.0.0",
            author="Your Name",
            description="Description of what your plugin does",
            category="{category}",
            keywords=["keyword1", "keyword2"],
            requirements=[],  # Python packages required
            permissions=[],   # Permissions needed (file_access, network, etc.)
            config_schema={{
                "api_key": {{"type": "string", "required": False}},
                "enabled": {{"type": "boolean", "default": True}}
            }}
        )
        
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize your plugin"""
        self.config = config
        
        # Register commands
        self.register_command(PluginCommand(
            name="example_command",
            patterns=[
                r"example\\s+(.+)",
                r"demo\\s+(.+)"
            ],
            description="An example command",
            parameters={{
                "input": {{"type": "string", "description": "Input parameter"}}
            }},
            examples=[
                "example test",
                "demo something"
            ],
            handler=self.handle_example_command
        ))
        
        # Subscribe to events (optional)
        # self.subscribe_event("some_event", self.handle_event)
        
        self.logger.info(f"{{self.get_metadata().name}} initialized")
        return True
        
    async def shutdown(self):
        """Clean up resources"""
        self.logger.info(f"{{self.get_metadata().name}} shutting down")
        
    async def handle_example_command(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Handle the example command"""
        try:
            # Extract parameters from match
            input_text = match.group(1) if match.lastindex >= 1 else ""
            
            # Your plugin logic here
            result = f"Processed: {{input_text}}"
            
            # Emit an event (optional)
            # self.emit_event("example_processed", {{"input": input_text, "result": result}})
            
            return True, result
            
        except Exception as e:
            self.logger.error(f"Error in example command: {{e}}")
            return False, f"Error: {{str(e)}}"
            
    # Add more methods as needed for your plugin functionality
'''
        return template
        
    @staticmethod
    def validate_plugin(plugin_path: str) -> Dict[str, Any]:
        """Validate a plugin file"""
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "metadata": None
        }
        
        try:
            # Import and check plugin
            spec = importlib.util.spec_from_file_location("test_plugin", plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, JARVISPlugin) and 
                    obj != JARVISPlugin):
                    plugin_class = obj
                    break
                    
            if not plugin_class:
                results["valid"] = False
                results["errors"].append("No JARVISPlugin subclass found")
                return results
                
            # Check metadata
            try:
                plugin = plugin_class()
                metadata = plugin.get_metadata()
                results["metadata"] = metadata
                
                # Validate metadata
                if not metadata.name:
                    results["errors"].append("Plugin name is required")
                if not metadata.version:
                    results["errors"].append("Plugin version is required")
                    
            except Exception as e:
                results["errors"].append(f"Failed to get metadata: {e}")
                
            # Check for security issues
            with open(plugin_path, 'r') as f:
                code = f.read()
                
            dangerous_patterns = [
                (r'\beval\s*\(', "Use of eval() is dangerous"),
                (r'\bexec\s*\(', "Use of exec() is dangerous"),
                (r'__import__', "Dynamic imports may be dangerous"),
                (r'subprocess', "Subprocess usage requires 'system' permission"),
                (r'os\.system', "os.system usage requires 'system' permission")
            ]
            
            for pattern, warning in dangerous_patterns:
                if re.search(pattern, code):
                    results["warnings"].append(warning)
                    
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Failed to load plugin: {e}")
            
        results["valid"] = len(results["errors"]) == 0
        return results


# Built-in example plugin
class ExamplePlugin(JARVISPlugin):
    """Example plugin showing capabilities"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Example Plugin",
            version="1.0.0",
            author="JARVIS Team",
            description="Demonstrates plugin capabilities",
            category="utility",
            keywords=["example", "demo"],
            requirements=[],
            permissions=[],
            config_schema={
                "greeting": {"type": "string", "default": "Hello from plugin!"}
            }
        )
        
    async def initialize(self, config: Dict[str, Any]) -> bool:
        self.config = config
        
        self.register_command(PluginCommand(
            name="plugin_demo",
            patterns=[r"plugin demo", r"show plugin"],
            description="Demonstrate plugin functionality",
            parameters={},
            examples=["plugin demo", "show plugin"],
            handler=self.demo_command
        ))
        
        return True
        
    async def shutdown(self):
        pass
        
    async def demo_command(self, command: str, match: Any) -> Tuple[bool, str]:
        greeting = self.config.get("greeting", "Hello from plugin!")
        return True, f"{greeting} This is the example plugin working!"