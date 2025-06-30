"""
Unified configuration management for JARVIS
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from copy import deepcopy


class ConfigurationManager:
    """
    Manages configuration for JARVIS components.

    Features:
    - Load from JSON files
    - Environment variable substitution
    - Configuration inheritance
    - Runtime updates
    - Validation
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = config_dir or Path(__file__).parent.parent / "config"
        self.logger = logging.getLogger("jarvis.config")
        self._config = {}
        self._env_config = {}

    def load_config(self, environment: str = "default") -> Dict[str, Any]:
        """
        Load configuration for specified environment.

        Args:
            environment: Environment name (default, development, production)

        Returns:
            Merged configuration dictionary
        """
        # Load base/default config
        base_config = self._load_file("default.json")

        # Load environment-specific config
        env_config = {}
        if environment != "default":
            env_file = f"{environment}.json"
            if (self.config_dir / env_file).exists():
                env_config = self._load_file(env_file)

                # Handle inheritance
                if "extends" in env_config:
                    del env_config["extends"]

        # Merge configurations
        config = self._deep_merge(base_config, env_config)

        # Substitute environment variables
        config = self._substitute_env_vars(config)

        # Validate configuration
        self._validate_config(config)

        self._config = config
        self.logger.info(f"Loaded configuration for environment: {environment}")

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.

        Args:
            key: Configuration key (e.g., "ai_integration.default_model")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value at runtime.

        Args:
            key: Configuration key
            value: New value
        """
        keys = key.split(".")
        config = self._config

        # Navigate to parent
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set value
        config[keys[-1]] = value
        self.logger.debug(f"Configuration updated: {key} = {value}")

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration values.

        Args:
            updates: Dictionary of updates
        """
        self._config = self._deep_merge(self._config, updates)
        self.logger.info("Configuration batch updated")

    def get_component_config(self, component: str) -> Dict[str, Any]:
        """
        Get configuration for specific component.

        Args:
            component: Component name

        Returns:
            Component configuration
        """
        return self._config.get(component, {})

    def _load_file(self, filename: str) -> Dict[str, Any]:
        """Load JSON configuration file"""
        filepath = self.config_dir / filename

        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config file {filename}: {e}")
            return {}

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = deepcopy(base)

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)

        return result

    def _substitute_env_vars(self, config: Any) -> Any:
        """Substitute environment variables in config"""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif (
            isinstance(config, str) and config.startswith("${") and config.endswith("}")
        ):
            # Environment variable reference
            var_name = config[2:-1]
            return os.environ.get(var_name, config)
        else:
            return config

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration"""
        # Store temporarily to use get() method
        temp_config = self._config
        self._config = config
        
        # Check required fields
        required = [
            "jarvis.name",
            "jarvis.version",
            "logging.level",
            "ai_integration.default_model",
        ]

        missing = []
        for key in required:
            if self.get(key) is None:
                missing.append(key)
        
        if missing:
            self._config = temp_config
            raise ValueError(f"Required configuration missing: {', '.join(missing)}")

        # Validate value ranges
        cycle_freq = self.get("consciousness.cycle_frequency", 10)
        if cycle_freq < 1:
            self._config = temp_config
            raise ValueError("consciousness.cycle_frequency must be >= 1")

        neural_cap = self.get("neural.capacity", 2000)
        if neural_cap < 100:
            self._config = temp_config
            raise ValueError("neural.capacity must be >= 100")
            
        # Restore original config
        self._config = temp_config

    def save_config(
        self, filename: str, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save configuration to file.

        Args:
            filename: Output filename
            config: Configuration to save (uses current if None)
        """
        config = config or self._config
        filepath = self.config_dir / filename

        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)

        self.logger.info(f"Configuration saved to {filepath}")

    def __repr__(self) -> str:
        return f"ConfigurationManager(env={self._config.get('jarvis', {}).get('mode', 'unknown')})"


# Global configuration instance
config_manager = ConfigurationManager()


def load_config(environment: str = None) -> Dict[str, Any]:
    """
    Load configuration for environment.

    Args:
        environment: Environment name or None to detect

    Returns:
        Configuration dictionary
    """
    if environment is None:
        # Detect from environment variable
        environment = os.environ.get("JARVIS_ENV", "default")

    return config_manager.load_config(environment)


def get_config(key: str, default: Any = None) -> Any:
    """
    Get configuration value.

    Args:
        key: Configuration key
        default: Default value

    Returns:
        Configuration value
    """
    return config_manager.get(key, default)
