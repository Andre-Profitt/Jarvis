"""
Configuration management
"""
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import os

from .logger import setup_logger

logger = setup_logger(__name__)


class Config:
    """Configuration manager with environment variable support"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.load()
        
    def load(self):
        """Load configuration from file and environment"""
        # Load from file
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from {self.config_path}")
        else:
            logger.warning(f"Config file not found: {self.config_path}")
            
        # Override with environment variables
        self._load_env_vars()
        
    def _load_env_vars(self):
        """Load environment variables"""
        env_mappings = {
            "JARVIS_OPENAI_API_KEY": "ai.openai_api_key",
            "JARVIS_ANTHROPIC_API_KEY": "ai.anthropic_api_key",
            "JARVIS_GOOGLE_API_KEY": "ai.google_api_key",
            "JARVIS_ELEVENLABS_API_KEY": "voice.elevenlabs_api_key",
            "JARVIS_WEATHER_API_KEY": "plugins.weather.api_key",
            "JARVIS_REDIS_HOST": "memory.redis_host",
            "JARVIS_REDIS_PORT": "memory.redis_port",
        }
        
        for env_var, config_path in env_mappings.items():
            if value := os.getenv(env_var):
                self.set(config_path, value)
                logger.debug(f"Loaded {env_var} from environment")
                
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
                
        return value
        
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def save(self):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Saved config to {self.config_path}")