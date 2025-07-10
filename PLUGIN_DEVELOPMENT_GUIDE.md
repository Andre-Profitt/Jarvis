# JARVIS Plugin Development Guide

This guide explains how to create custom plugins for JARVIS, extending its capabilities with new features and integrations.

## Table of Contents

1. [Plugin Architecture Overview](#plugin-architecture-overview)
2. [Creating Your First Plugin](#creating-your-first-plugin)
3. [Plugin Structure](#plugin-structure)
4. [Available APIs](#available-apis)
5. [Best Practices](#best-practices)
6. [Example Plugins](#example-plugins)
7. [Testing Your Plugin](#testing-your-plugin)
8. [Publishing Your Plugin](#publishing-your-plugin)

## Plugin Architecture Overview

JARVIS plugins are Python modules that extend JARVIS's functionality by:
- Adding new voice commands
- Integrating with external services
- Processing data and events
- Providing custom responses

### Key Components

1. **JARVISPlugin Base Class**: All plugins inherit from this class
2. **PluginMetadata**: Describes your plugin's properties and requirements
3. **PluginCommand**: Defines voice commands your plugin handles
4. **PluginManager**: Loads, manages, and executes plugins

## Creating Your First Plugin

### 1. Basic Plugin Template

```python
#!/usr/bin/env python3
"""
My Custom Plugin for JARVIS
Description of what your plugin does
"""

from typing import Dict, Any, Tuple
import re
from core.plugin_system import JARVISPlugin, PluginMetadata, PluginCommand


class MyCustomPlugin(JARVISPlugin):
    """Your plugin implementation"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="My Custom Plugin",
            version="1.0.0",
            author="Your Name",
            description="What your plugin does",
            category="utility",  # Options: utility, productivity, entertainment, automation, integration
            keywords=["custom", "example"],
            requirements=[],  # Python packages required
            permissions=[],   # Options: file_access, network, system, notifications
            config_schema={
                "api_key": {"type": "string", "required": False},
                "enabled": {"type": "boolean", "default": True}
            }
        )
        
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize your plugin"""
        self.config = config
        
        # Register commands
        self.register_command(PluginCommand(
            name="my_command",
            patterns=[
                r"do something (.+)",
                r"execute (.+)"
            ],
            description="Does something cool",
            parameters={
                "input": {"type": "string", "description": "Input parameter"}
            },
            examples=[
                "do something awesome",
                "execute task"
            ],
            handler=self.handle_my_command
        ))
        
        self.logger.info(f"{self.get_metadata().name} initialized")
        return True
        
    async def shutdown(self):
        """Clean up resources"""
        self.logger.info(f"{self.get_metadata().name} shutting down")
        
    async def handle_my_command(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Handle your custom command"""
        try:
            # Extract parameters
            input_text = match.group(1) if match.lastindex >= 1 else ""
            
            # Your logic here
            result = f"Processed: {input_text}"
            
            return True, result
            
        except Exception as e:
            self.logger.error(f"Error in command: {e}")
            return False, f"Error: {str(e)}"
```

### 2. Plugin Location

Plugins can be placed in:
- **Built-in**: `/core/plugins/` (for core functionality)
- **User plugins**: `~/.jarvis/plugins/` (for custom plugins)
- **Installed packages**: Via pip with entry points

## Plugin Structure

### Metadata

The `PluginMetadata` class defines:

```python
PluginMetadata(
    name="Plugin Name",              # Display name
    version="1.0.0",                # Semantic version
    author="Author Name",           # Your name
    description="Description",      # What it does
    category="category",           # Plugin category
    keywords=["keyword1"],         # Search keywords
    requirements=["requests"],     # Python dependencies
    permissions=["network"],       # Required permissions
    config_schema={...}           # Configuration schema
)
```

### Commands

Commands are defined with patterns and handlers:

```python
PluginCommand(
    name="command_name",           # Internal name
    patterns=[                     # Regex patterns to match
        r"pattern one (.+)",
        r"alternative pattern (.+)"
    ],
    description="What it does",    # User-friendly description
    parameters={                   # Parameter descriptions
        "param1": {"type": "string", "description": "..."}
    },
    examples=[                     # Usage examples
        "example command usage"
    ],
    handler=self.handler_method    # Async method to handle command
)
```

### Command Patterns

Use regular expressions for flexible matching:

```python
# Basic patterns
r"simple command"              # Matches exact phrase
r"play (.+)"                  # Captures everything after "play"
r"remind me (?:to |about )(.+)"  # Optional words

# Advanced patterns
r"set timer for (\d+) (minutes?|hours?)"  # Capture number and unit
r"weather(?:\s+in\s+(.+))?"              # Optional location
r"(?:show|display|list) (.+)"            # Multiple trigger words
```

## Available APIs

### 1. Logging

```python
self.logger.info("Information message")
self.logger.warning("Warning message")
self.logger.error("Error message")
self.logger.debug("Debug message")
```

### 2. Configuration

```python
# Get config value
api_key = self.get_config_value("api_key", default="")

# Access full config
if self.config.get("enabled", True):
    # Do something
```

### 3. Events

```python
# Emit an event
self.emit_event("my_event", {
    "data": "value",
    "timestamp": datetime.now()
})

# Subscribe to events
self.subscribe_event("other_event", self.handle_event)

async def handle_event(self, data: Any):
    # Process event data
    pass
```

### 4. JARVIS Integration

```python
# Access JARVIS instance (if available)
if self.jarvis:
    # Trigger speech
    self.emit_event("speak_announcement", {
        "text": "Hello from plugin!",
        "priority": "normal"
    })
    
    # Send notification
    self.emit_event("notification", {
        "title": "Plugin Alert",
        "message": "Something happened",
        "priority": "high"
    })
```

## Best Practices

### 1. Error Handling

Always handle exceptions gracefully:

```python
async def handle_command(self, command: str, match: re.Match) -> Tuple[bool, str]:
    try:
        # Your code here
        return True, "Success message"
    except SpecificException as e:
        self.logger.error(f"Specific error: {e}")
        return False, "User-friendly error message"
    except Exception as e:
        self.logger.error(f"Unexpected error: {e}")
        return False, "Sorry, something went wrong"
```

### 2. Async Operations

Use async/await for I/O operations:

```python
import aiohttp

async def fetch_data(self, url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

### 3. Resource Management

Clean up resources in shutdown:

```python
async def shutdown(self):
    # Cancel any running tasks
    if hasattr(self, 'background_task'):
        self.background_task.cancel()
    
    # Close connections
    if hasattr(self, 'connection'):
        await self.connection.close()
```

### 4. Configuration Validation

Validate configuration on initialization:

```python
async def initialize(self, config: Dict[str, Any]) -> bool:
    # Check required config
    if not config.get("api_key"):
        self.logger.warning("No API key configured")
        # Can still return True if plugin can work without it
        
    # Validate config values
    timeout = config.get("timeout", 30)
    if not isinstance(timeout, int) or timeout < 1:
        self.logger.error("Invalid timeout value")
        return False
        
    return True
```

### 5. User Feedback

Provide clear, helpful responses:

```python
# Good
return True, "✅ Reminder set for 3:00 PM: Call mom"

# Bad
return True, "Done"

# Good error message
return False, "I couldn't understand the time. Try saying 'in 5 minutes' or 'at 3pm'"

# Bad error message
return False, "Error"
```

## Example Plugins

### 1. Simple Calculator Plugin

```python
class CalculatorPlugin(JARVISPlugin):
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Calculator",
            version="1.0.0",
            author="JARVIS Team",
            description="Performs basic calculations",
            category="utility",
            keywords=["math", "calculator", "compute"],
            requirements=[],
            permissions=[],
            config_schema={}
        )
        
    async def initialize(self, config: Dict[str, Any]) -> bool:
        self.register_command(PluginCommand(
            name="calculate",
            patterns=[
                r"calculate (.+)",
                r"what is (.+)",
                r"compute (.+)"
            ],
            description="Perform calculations",
            parameters={"expression": {"type": "string"}},
            examples=["calculate 2 + 2", "what is 10 * 5"],
            handler=self.calculate
        ))
        return True
        
    async def calculate(self, command: str, match: re.Match) -> Tuple[bool, str]:
        try:
            expression = match.group(1)
            # Safe evaluation (only math operations)
            result = eval(expression, {"__builtins__": {}}, {})
            return True, f"The answer is: {result}"
        except:
            return False, "I couldn't calculate that expression"
```

### 2. Integration Plugin Example

```python
class SlackPlugin(JARVISPlugin):
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Slack Integration",
            version="1.0.0",
            author="Your Name",
            description="Send messages to Slack",
            category="integration",
            keywords=["slack", "messaging", "communication"],
            requirements=["slack-sdk"],
            permissions=["network"],
            config_schema={
                "token": {"type": "string", "required": True},
                "default_channel": {"type": "string", "default": "general"}
            }
        )
        
    async def initialize(self, config: Dict[str, Any]) -> bool:
        from slack_sdk.web.async_client import AsyncWebClient
        
        self.client = AsyncWebClient(token=config.get("token"))
        self.default_channel = config.get("default_channel", "general")
        
        self.register_command(PluginCommand(
            name="slack_message",
            patterns=[
                r"slack message (.+)",
                r"send to slack (.+)"
            ],
            description="Send message to Slack",
            parameters={"message": {"type": "string"}},
            examples=["slack message Hello team!"],
            handler=self.send_message
        ))
        return True
        
    async def send_message(self, command: str, match: re.Match) -> Tuple[bool, str]:
        try:
            message = match.group(1)
            response = await self.client.chat_postMessage(
                channel=self.default_channel,
                text=message
            )
            return True, f"Message sent to #{self.default_channel}"
        except Exception as e:
            return False, f"Failed to send message: {str(e)}"
```

## Testing Your Plugin

### 1. Unit Testing

Create tests for your plugin:

```python
import pytest
from my_plugin import MyCustomPlugin

@pytest.mark.asyncio
async def test_plugin_initialization():
    plugin = MyCustomPlugin()
    config = {"api_key": "test"}
    
    result = await plugin.initialize(config)
    assert result == True
    assert len(plugin.commands) > 0

@pytest.mark.asyncio
async def test_command_handling():
    plugin = MyCustomPlugin()
    await plugin.initialize({})
    
    # Test command matching
    import re
    pattern = plugin.commands[0].patterns[0]
    match = re.match(pattern, "do something test")
    
    success, response = await plugin.handle_my_command("do something test", match)
    assert success == True
    assert "test" in response
```

### 2. Manual Testing

Test your plugin with JARVIS:

```python
# Create test script
from core.plugin_system import PluginManager
import asyncio

async def test_plugin():
    manager = PluginManager()
    
    # Load your plugin
    success = await manager.load_plugin("user.my_plugin")
    print(f"Plugin loaded: {success}")
    
    # Test commands
    result = await manager.process_command("do something test")
    print(f"Command result: {result}")

asyncio.run(test_plugin())
```

## Publishing Your Plugin

### 1. Package Structure

```
my-jarvis-plugin/
├── setup.py
├── README.md
├── LICENSE
├── my_plugin/
│   ├── __init__.py
│   └── plugin.py
└── tests/
    └── test_plugin.py
```

### 2. Setup.py

```python
from setuptools import setup, find_packages

setup(
    name="jarvis-my-plugin",
    version="1.0.0",
    author="Your Name",
    description="My JARVIS Plugin",
    packages=find_packages(),
    install_requires=[
        "jarvis-assistant",
        # Your dependencies
    ],
    entry_points={
        "jarvis_plugins": [
            "my_plugin = my_plugin.plugin:MyCustomPlugin"
        ]
    }
)
```

### 3. Distribution

1. **GitHub Repository**: Share your plugin code
2. **PyPI**: Publish for pip installation
3. **JARVIS Plugin Registry**: Submit to official registry (coming soon)

## Plugin Ideas

Here are some ideas for plugins you could create:

1. **Home Automation**: Control smart home devices
2. **Task Management**: Integrate with Todoist, Trello, etc.
3. **Health Tracking**: Log water intake, exercise, medication
4. **Development Tools**: Code generation, Git operations
5. **Entertainment**: Game scores, movie recommendations
6. **Finance**: Stock prices, cryptocurrency tracking
7. **Learning**: Flashcards, language practice
8. **Social Media**: Post updates, check notifications
9. **System Monitoring**: CPU usage, disk space alerts
10. **Custom Workflows**: Automate repetitive tasks

## Getting Help

- Check existing plugins in `/core/plugins/` for examples
- Review the `plugin_system.py` source code
- Join the JARVIS community forums
- Submit issues on GitHub

Happy plugin development! 🚀