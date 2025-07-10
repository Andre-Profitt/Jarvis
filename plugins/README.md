# JARVIS User Plugins Directory

This directory is for user-created JARVIS plugins. Place your custom plugin Python files here and JARVIS will automatically discover them.

## Quick Start

1. Create a new Python file in this directory (e.g., `my_plugin.py`)
2. Use the plugin template from the PLUGIN_DEVELOPMENT_GUIDE.md
3. Save the file
4. Use "discover plugins" command in JARVIS
5. Use "install my_plugin plugin" to load it

## Example Plugin Structure

```python
from core.plugin_system import JARVISPlugin, PluginMetadata, PluginCommand

class MyPlugin(JARVISPlugin):
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="My Plugin",
            version="1.0.0",
            author="Your Name",
            description="What it does",
            category="utility",
            keywords=["example"],
            requirements=[],
            permissions=[],
            config_schema={}
        )
    
    async def initialize(self, config):
        # Setup your plugin
        return True
    
    async def shutdown(self):
        # Cleanup
        pass
```

## Available Plugin Categories

- **utility**: General purpose tools
- **productivity**: Task management, notes, reminders
- **entertainment**: Games, music, fun features
- **automation**: Workflow automation, scripting
- **integration**: External service integrations

## Plugin Ideas

- Smart home control (Philips Hue, Nest, etc.)
- Social media integration
- Cryptocurrency tracker
- Workout tracker
- Recipe finder
- Language translator
- File organization
- System monitoring
- Game integration
- Custom workflows

## Need Help?

- Check `/core/plugins/` for example plugins
- Read the PLUGIN_DEVELOPMENT_GUIDE.md
- Use "plugin help" command in JARVIS