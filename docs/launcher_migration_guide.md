# JARVIS Launcher Migration Guide

## Overview

The new `LAUNCH-JARVIS-UNIFIED.py` consolidates all previous launchers into a single, modular, best-practices implementation. This guide will help you migrate from the old launchers.

## Migration Steps

### 1. Backup Your Configuration

Before migrating, backup any custom configurations or environment settings you have.

### 2. Update Environment Variables

The new launcher uses a consistent environment variable format:
```bash
# Old format (varies by launcher)
ELEVEN_LABS_API_KEY=xxx
OPENAI_API_KEY=xxx

# New format (still works)
ELEVEN_LABS_API_KEY=xxx
OPENAI_API_KEY=xxx

# Additional override format
JARVIS_SERVICES_VOICE_ENABLED=true
JARVIS_SERVICES_WEBSOCKET_PORT=8080
```

### 3. Create Configuration File

Copy the template configuration:
```bash
cp config/jarvis.yaml config/jarvis.yaml
```

Then customize it for your needs.

### 4. Update Launch Commands

Replace old launch commands with the new unified launcher:

```bash
# Old
python LAUNCH-JARVIS-REAL.py
python LAUNCH-JARVIS-ENHANCED.py
python LAUNCH-JARVIS-FULL.py

# New
python LAUNCH-JARVIS-UNIFIED.py

# With options
python LAUNCH-JARVIS-UNIFIED.py --config myconfig.yaml
python LAUNCH-JARVIS-UNIFIED.py --services websocket multi_ai voice
python LAUNCH-JARVIS-UNIFIED.py --no-interactive
python LAUNCH-JARVIS-UNIFIED.py --log-level DEBUG
```

## Feature Mapping

### From LAUNCH-JARVIS-REAL.py

| Old Feature | New Feature | Configuration |
|------------|-------------|---------------|
| Multi-AI Integration | `multi_ai` service | `services.multi_ai` |
| WebSocket Server | `websocket` service | `services.websocket` |
| ElevenLabs Voice | `voice` service | `services.voice` |
| Neural Resources | `neural_resources` service | `services.neural_resources` |
| Self-Healing | `self_healing` service | `services.self_healing` |
| Consciousness | `consciousness` service | `services.consciousness` |

### From LAUNCH-JARVIS-ENHANCED.py

| Old Feature | New Feature | Configuration |
|------------|-------------|---------------|
| Program Synthesis | Plugin system | `features.plugins` |
| Emotional Intelligence | Plugin system | `features.plugins` |
| Security Sandbox | Built into services | Security handled per-service |
| Resource Management | `neural_resources` service | `services.neural_resources` |

### From LAUNCH-JARVIS-FIXED/FULL/PATCHED.py

The consciousness patch is now automatically applied by the unified launcher when needed.

## New Features

### 1. Service Management

Interactive commands to manage services:
```
JARVIS> status          # Show all service status
JARVIS> health          # Detailed health report
JARVIS> start voice     # Start a specific service
JARVIS> stop voice      # Stop a specific service
JARVIS> restart voice   # Restart a service
```

### 2. Configuration Management

- YAML-based configuration with defaults
- Environment variable overrides
- Runtime configuration changes
- Configuration validation

### 3. Health Monitoring

- Continuous health checks
- Service dependency tracking
- Graceful degradation
- Automatic recovery attempts

### 4. Plugin System

Create custom plugins in the `plugins/` directory:

```python
# plugins/my_plugin.py
class MyPlugin:
    def __init__(self, config):
        self.config = config
    
    async def initialize(self):
        # Setup code
        pass
    
    async def shutdown(self):
        # Cleanup code
        pass
```

### 5. Better Logging

- Structured logging with levels
- Separate log files with rotation
- Per-service log configuration
- Real-time log streaming

## Troubleshooting

### Service Won't Start

1. Check dependencies are met:
   ```
   JARVIS> status
   ```

2. Check logs:
   ```bash
   tail -f logs/jarvis_*.log
   ```

3. Verify configuration:
   ```
   JARVIS> config
   ```

### Missing Features

If a feature from an old launcher is missing:

1. Check if it's available as a plugin
2. Check if it's been renamed/reorganized
3. Consider implementing it as a custom plugin

### Performance Issues

1. Adjust worker threads:
   ```yaml
   performance:
     max_workers: 20
   ```

2. Disable unused services:
   ```yaml
   services:
     unused_service:
       enabled: false
   ```

3. Enable performance monitoring:
   ```yaml
   development:
     debug:
       profiling: true
   ```

## Best Practices

1. **Use Configuration Files**: Don't hardcode settings
2. **Monitor Health**: Regular health checks help catch issues early
3. **Graceful Shutdown**: Always use `exit` command or Ctrl+C
4. **Service Isolation**: Services should be independent
5. **Error Handling**: Services should handle errors gracefully

## Cleanup

After successful migration, you can archive old launchers:

```bash
mkdir archived_launchers
mv LAUNCH-JARVIS-*.py archived_launchers/
# Keep only the unified launcher
mv archived_launchers/LAUNCH-JARVIS-UNIFIED.py ./
```

## Support

For issues or questions:
1. Check the logs in `logs/` directory
2. Review the configuration in `config/jarvis.yaml`
3. Use `--log-level DEBUG` for detailed debugging
4. Check service health with the `health` command