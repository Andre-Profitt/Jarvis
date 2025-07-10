#!/usr/bin/env python3
"""
Plugin Management Commands for JARVIS
Provides voice commands to manage plugins
"""

import re
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import asyncio
from core.plugin_system import PluginManager, PluginDevelopmentKit


class PluginCommandHandler:
    """Handles plugin management commands for JARVIS"""
    
    def __init__(self, plugin_manager: Optional[PluginManager] = None):
        self.plugin_manager = plugin_manager or PluginManager()
        self.commands = self._register_commands()
        
    def _register_commands(self) -> Dict[str, Dict[str, Any]]:
        """Register plugin management commands"""
        return {
            "list_plugins": {
                "patterns": [
                    r"(?:list|show|what are) (?:my |the )?(?:installed |loaded |active )?plugins?",
                    r"plugin list",
                    r"what plugins (?:do i have|are there)"
                ],
                "handler": self.handle_list_plugins,
                "description": "List all loaded plugins"
            },
            "enable_plugin": {
                "patterns": [
                    r"(?:enable|activate|turn on) (?:the )?(.+?) plugin",
                    r"plugin enable (.+)"
                ],
                "handler": self.handle_enable_plugin,
                "description": "Enable a specific plugin"
            },
            "disable_plugin": {
                "patterns": [
                    r"(?:disable|deactivate|turn off) (?:the )?(.+?) plugin",
                    r"plugin disable (.+)"
                ],
                "handler": self.handle_disable_plugin,
                "description": "Disable a specific plugin"
            },
            "reload_plugin": {
                "patterns": [
                    r"reload (?:the )?(.+?) plugin",
                    r"plugin reload (.+)"
                ],
                "handler": self.handle_reload_plugin,
                "description": "Reload a specific plugin"
            },
            "plugin_info": {
                "patterns": [
                    r"(?:tell me about|info on|describe) (?:the )?(.+?) plugin",
                    r"plugin info (.+)",
                    r"what does (?:the )?(.+?) plugin do"
                ],
                "handler": self.handle_plugin_info,
                "description": "Get information about a plugin"
            },
            "discover_plugins": {
                "patterns": [
                    r"(?:discover|find|search for) (?:new )?plugins?",
                    r"what plugins are available"
                ],
                "handler": self.handle_discover_plugins,
                "description": "Discover available plugins"
            },
            "install_plugin": {
                "patterns": [
                    r"install (?:the )?(.+?) plugin",
                    r"plugin install (.+)"
                ],
                "handler": self.handle_install_plugin,
                "description": "Install a new plugin"
            },
            "plugin_help": {
                "patterns": [
                    r"(?:help with |how to use )plugins?",
                    r"plugin help"
                ],
                "handler": self.handle_plugin_help,
                "description": "Get help with plugins"
            }
        }
        
    async def process_command(self, command: str) -> Optional[Tuple[bool, str]]:
        """Process a command and return response if handled"""
        command_lower = command.lower().strip()
        
        for cmd_name, cmd_info in self.commands.items():
            for pattern in cmd_info["patterns"]:
                match = re.match(pattern, command_lower)
                if match:
                    handler = cmd_info["handler"]
                    if asyncio.iscoroutinefunction(handler):
                        return await handler(command, match)
                    else:
                        return handler(command, match)
                        
        return None
        
    async def handle_list_plugins(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """List all loaded plugins"""
        try:
            plugins = self.plugin_manager.list_plugins()
            
            if not plugins:
                return True, "No plugins are currently loaded."
                
            response = f"📦 Loaded plugins ({len(plugins)}):\n\n"
            
            # Group by category
            categories = {}
            for plugin in plugins:
                category = plugin["category"]
                if category not in categories:
                    categories[category] = []
                categories[category].append(plugin)
                
            for category, cat_plugins in categories.items():
                response += f"**{category.title()}**:\n"
                for plugin in cat_plugins:
                    status = "✅" if plugin["enabled"] else "❌"
                    response += f"  {status} {plugin['name']} v{plugin['version']}"
                    if plugin["commands"] > 0:
                        response += f" ({plugin['commands']} commands)"
                    response += "\n"
                response += "\n"
                
            return True, response.strip()
            
        except Exception as e:
            return False, f"Error listing plugins: {str(e)}"
            
    async def handle_enable_plugin(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Enable a plugin"""
        try:
            plugin_name = match.group(1).strip()
            
            # Find matching plugin
            plugins = self.plugin_manager.list_plugins()
            matched_plugin = None
            
            for plugin in plugins:
                if plugin_name.lower() in plugin["name"].lower():
                    matched_plugin = plugin
                    break
                    
            if not matched_plugin:
                return False, f"I couldn't find a plugin matching '{plugin_name}'."
                
            # Enable the plugin
            plugin_instance = self.plugin_manager.get_plugin(matched_plugin["name"])
            if plugin_instance:
                plugin_instance.enabled = True
                return True, f"✅ Enabled {matched_plugin['name']} plugin"
            else:
                return False, f"Plugin {matched_plugin['name']} is not loaded."
                
        except Exception as e:
            return False, f"Error enabling plugin: {str(e)}"
            
    async def handle_disable_plugin(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Disable a plugin"""
        try:
            plugin_name = match.group(1).strip()
            
            # Find matching plugin
            plugins = self.plugin_manager.list_plugins()
            matched_plugin = None
            
            for plugin in plugins:
                if plugin_name.lower() in plugin["name"].lower():
                    matched_plugin = plugin
                    break
                    
            if not matched_plugin:
                return False, f"I couldn't find a plugin matching '{plugin_name}'."
                
            # Disable the plugin
            plugin_instance = self.plugin_manager.get_plugin(matched_plugin["name"])
            if plugin_instance:
                plugin_instance.enabled = False
                return True, f"❌ Disabled {matched_plugin['name']} plugin"
            else:
                return False, f"Plugin {matched_plugin['name']} is not loaded."
                
        except Exception as e:
            return False, f"Error disabling plugin: {str(e)}"
            
    async def handle_reload_plugin(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Reload a plugin"""
        try:
            plugin_name = match.group(1).strip()
            
            # Find matching plugin
            plugins = self.plugin_manager.list_plugins()
            matched_plugin = None
            
            for plugin in plugins:
                if plugin_name.lower() in plugin["name"].lower():
                    matched_plugin = plugin
                    break
                    
            if not matched_plugin:
                return False, f"I couldn't find a plugin matching '{plugin_name}'."
                
            # Reload the plugin
            success = await self.plugin_manager.reload_plugin(matched_plugin["name"])
            
            if success:
                return True, f"🔄 Reloaded {matched_plugin['name']} plugin successfully"
            else:
                return False, f"Failed to reload {matched_plugin['name']} plugin"
                
        except Exception as e:
            return False, f"Error reloading plugin: {str(e)}"
            
    async def handle_plugin_info(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Get information about a plugin"""
        try:
            plugin_name = match.group(1).strip()
            
            # Find matching plugin
            plugins = self.plugin_manager.list_plugins()
            matched_plugin = None
            
            for plugin in plugins:
                if plugin_name.lower() in plugin["name"].lower():
                    matched_plugin = plugin
                    break
                    
            if not matched_plugin:
                return False, f"I couldn't find a plugin matching '{plugin_name}'."
                
            # Get detailed info
            plugin_instance = self.plugin_manager.get_plugin(matched_plugin["name"])
            if not plugin_instance:
                return False, f"Plugin {matched_plugin['name']} is not loaded."
                
            metadata = plugin_instance.get_metadata()
            
            response = f"📦 **{metadata.name}** v{metadata.version}\n\n"
            response += f"**Author**: {metadata.author}\n"
            response += f"**Description**: {metadata.description}\n"
            response += f"**Category**: {metadata.category}\n"
            response += f"**Status**: {'✅ Enabled' if plugin_instance.enabled else '❌ Disabled'}\n\n"
            
            if metadata.keywords:
                response += f"**Keywords**: {', '.join(metadata.keywords)}\n"
                
            if plugin_instance.commands:
                response += f"\n**Commands** ({len(plugin_instance.commands)}):\n"
                for cmd in plugin_instance.commands[:5]:  # Show first 5
                    response += f"  • {cmd.description}\n"
                    if cmd.examples:
                        response += f"    Example: \"{cmd.examples[0]}\"\n"
                        
                if len(plugin_instance.commands) > 5:
                    response += f"  ... and {len(plugin_instance.commands) - 5} more\n"
                    
            if metadata.permissions:
                response += f"\n**Permissions**: {', '.join(metadata.permissions)}\n"
                
            return True, response
            
        except Exception as e:
            return False, f"Error getting plugin info: {str(e)}"
            
    async def handle_discover_plugins(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Discover available plugins"""
        try:
            available = await self.plugin_manager.discover_plugins()
            loaded = [p["name"] for p in self.plugin_manager.list_plugins()]
            
            # Filter out already loaded
            not_loaded = [p for p in available if not any(name in p for name in loaded)]
            
            if not not_loaded:
                return True, "All available plugins are already loaded!"
                
            response = f"🔍 Found {len(not_loaded)} available plugins:\n\n"
            
            # Group by source
            sources = {"builtin": [], "user": [], "package": []}
            for plugin_id in not_loaded:
                source = plugin_id.split('.')[0]
                if source in sources:
                    sources[source].append(plugin_id)
                    
            for source, plugins in sources.items():
                if plugins:
                    response += f"**{source.title()} Plugins**:\n"
                    for plugin_id in plugins:
                        name = plugin_id.split('.', 1)[1]
                        response += f"  • {name}\n"
                    response += "\n"
                    
            response += "Use 'install <plugin_name> plugin' to install one."
            
            return True, response
            
        except Exception as e:
            return False, f"Error discovering plugins: {str(e)}"
            
    async def handle_install_plugin(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Install a new plugin"""
        try:
            plugin_name = match.group(1).strip().lower()
            
            # Discover available plugins
            available = await self.plugin_manager.discover_plugins()
            
            # Find matching plugin
            matched = None
            for plugin_id in available:
                if plugin_name in plugin_id.lower():
                    matched = plugin_id
                    break
                    
            if not matched:
                return False, f"I couldn't find a plugin matching '{plugin_name}'. Use 'discover plugins' to see available plugins."
                
            # Load the plugin
            success = await self.plugin_manager.load_plugin(matched)
            
            if success:
                plugin_name = matched.split('.', 1)[1]
                return True, f"✅ Successfully installed and loaded {plugin_name} plugin!"
            else:
                return False, f"Failed to install plugin. Check the logs for details."
                
        except Exception as e:
            return False, f"Error installing plugin: {str(e)}"
            
    def handle_plugin_help(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Get help with plugins"""
        help_text = """
📚 **JARVIS Plugin System Help**

**Basic Commands**:
• "list plugins" - Show all loaded plugins
• "enable/disable <name> plugin" - Turn plugins on/off
• "reload <name> plugin" - Reload a plugin
• "plugin info <name>" - Get details about a plugin
• "discover plugins" - Find available plugins
• "install <name> plugin" - Install a new plugin

**Plugin Categories**:
• **Utility**: Weather, calculator, system tools
• **Productivity**: Reminders, notes, task management
• **Entertainment**: Music, games, fun facts
• **Automation**: Smart home, workflows
• **Integration**: External services, APIs

**Creating Plugins**:
Plugins are Python modules that extend JARVIS. Check the PLUGIN_DEVELOPMENT_GUIDE.md for details.

**Plugin Directory**:
• User plugins: ~/.jarvis/plugins/
• Built-in: core/plugins/

Need more help? Ask about a specific plugin or feature!
"""
        return True, help_text


class PluginIntegration:
    """Integration helper for adding plugin support to JARVIS"""
    
    @staticmethod
    async def integrate_with_jarvis(jarvis_instance):
        """Add plugin support to a JARVIS instance"""
        # Create plugin manager
        plugin_manager = PluginManager(jarvis_instance)
        jarvis_instance.plugin_manager = plugin_manager
        
        # Create command handler
        plugin_commands = PluginCommandHandler(plugin_manager)
        jarvis_instance.plugin_commands = plugin_commands
        
        # Load default plugins
        await PluginIntegration.load_default_plugins(plugin_manager)
        
        # Add command processing to JARVIS
        original_process = jarvis_instance.process_command if hasattr(jarvis_instance, 'process_command') else None
        
        async def enhanced_process_command(command: str):
            # First try plugin commands
            result = await plugin_commands.process_command(command)
            if result:
                return result
                
            # Then try loaded plugins
            plugin_result = await plugin_manager.process_command(command)
            if plugin_result:
                return plugin_result
                
            # Finally, fall back to original processing
            if original_process:
                return await original_process(command)
            else:
                return None
                
        jarvis_instance.process_command = enhanced_process_command
        
        return plugin_manager
        
    @staticmethod
    async def load_default_plugins(plugin_manager: PluginManager):
        """Load default plugins"""
        default_plugins = [
            "builtin.weather",
            "builtin.news", 
            "builtin.reminder",
            "builtin.music"
        ]
        
        for plugin_id in default_plugins:
            try:
                success = await plugin_manager.load_plugin(plugin_id)
                if success:
                    print(f"✅ Loaded {plugin_id}")
                else:
                    print(f"❌ Failed to load {plugin_id}")
            except Exception as e:
                print(f"❌ Error loading {plugin_id}: {e}")


# Example usage in JARVIS
"""
# In your JARVIS class __init__ or setup:

async def setup_plugins(self):
    # Add plugin support
    from core.plugin_commands import PluginIntegration
    self.plugin_manager = await PluginIntegration.integrate_with_jarvis(self)
    
# Then in your command processing:

async def handle_command(self, command):
    # Plugin system will automatically handle plugin commands
    result = await self.process_command(command)
    if result:
        success, response = result
        if success:
            self.speak(response)
        else:
            self.speak(f"Error: {response}")
"""