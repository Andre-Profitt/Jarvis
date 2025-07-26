"""
Plugin Manager - Extensible plugin system for custom functionality
"""
import asyncio
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import importlib
import inspect
from pathlib import Path

from ..logger import setup_logger

logger = setup_logger(__name__)


class BasePlugin(ABC):
    """Base class for all plugins"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.enabled = True
        
    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> Any:
        """Execute plugin functionality"""
        pass
        
    async def initialize(self):
        """Initialize plugin (optional)"""
        pass
        
    async def shutdown(self):
        """Cleanup plugin (optional)"""
        pass
        

class PluginManager:
    """Manages plugin loading and execution"""
    
    def __init__(self, config):
        self.config = config
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_dir = Path(config.get("plugins.directory", "plugins"))
        
    async def initialize(self):
        """Initialize plugin system"""
        logger.info("Initializing plugin manager...")
        
        # Load built-in plugins
        await self._load_builtin_plugins()
        
        # Load custom plugins
        if self.plugin_dir.exists():
            await self._load_custom_plugins()
            
        # Initialize all plugins
        tasks = [
            plugin.initialize() 
            for plugin in self.plugins.values()
        ]
        await asyncio.gather(*tasks)
        
        logger.info(f"Loaded {len(self.plugins)} plugins")
        
    async def _load_builtin_plugins(self):
        """Load built-in plugins"""
        from .builtin import (
            WeatherPlugin,
            TimePlugin,
            ReminderPlugin,
            SearchPlugin,
            SystemPlugin,
            SmartHomePlugin
        )
        
        builtin_plugins = [
            WeatherPlugin(self.config),
            TimePlugin(self.config),
            ReminderPlugin(self.config),
            SearchPlugin(self.config),
            SystemPlugin(self.config),
            SmartHomePlugin(self.config)
        ]
        
        for plugin in builtin_plugins:
            self.register_plugin(plugin)
            
    async def _load_custom_plugins(self):
        """Load custom plugins from directory"""
        for plugin_file in self.plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
                
            try:
                # Import module
                spec = importlib.util.spec_from_file_location(
                    plugin_file.stem,
                    plugin_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find plugin classes
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BasePlugin) and 
                        obj != BasePlugin):
                        
                        # Instantiate plugin
                        plugin = obj(self.config)
                        self.register_plugin(plugin)
                        logger.info(f"Loaded custom plugin: {plugin.name}")
                        
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file}: {e}")
                
    def register_plugin(self, plugin: BasePlugin):
        """Register a plugin"""
        if plugin.name in self.plugins:
            logger.warning(f"Plugin {plugin.name} already registered")
            return
            
        self.plugins[plugin.name] = plugin
        logger.debug(f"Registered plugin: {plugin.name}")
        
    async def execute(
        self, 
        plugin_name: str, 
        params: Dict[str, Any]
    ) -> Any:
        """Execute a plugin"""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin '{plugin_name}' not found")
            
        plugin = self.plugins[plugin_name]
        if not plugin.enabled:
            raise ValueError(f"Plugin '{plugin_name}' is disabled")
            
        try:
            result = await plugin.execute(params)
            return result
        except Exception as e:
            logger.error(f"Plugin {plugin_name} error: {e}")
            raise
            
    def list_plugins(self) -> List[Dict[str, str]]:
        """List all available plugins"""
        return [
            {
                "name": plugin.name,
                "description": plugin.description,
                "enabled": plugin.enabled
            }
            for plugin in self.plugins.values()
        ]
        
    def enable_plugin(self, plugin_name: str):
        """Enable a plugin"""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].enabled = True
            
    def disable_plugin(self, plugin_name: str):
        """Disable a plugin"""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].enabled = False
            
    async def shutdown(self):
        """Shutdown all plugins"""
        logger.info("Shutting down plugins...")
        
        tasks = [
            plugin.shutdown() 
            for plugin in self.plugins.values()
        ]
        await asyncio.gather(*tasks)