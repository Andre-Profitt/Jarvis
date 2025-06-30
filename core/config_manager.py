#!/usr/bin/env python3
"""
Enhanced Configuration Management for JARVIS
Provides centralized configuration with environment variable support,
validation, and hot-reloading capabilities.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
from functools import lru_cache
import validators
from pydantic import BaseModel, validator
import structlog

logger = structlog.get_logger()


class SynthesisConfig(BaseModel):
    """Configuration for Program Synthesis Engine"""

    # API Configuration
    api_base_url: str = "http://localhost:8000"
    api_timeout: int = 30
    api_retries: int = 3
    api_key: Optional[str] = None

    # Synthesis Settings
    max_synthesis_time: int = 60
    max_concurrent_syntheses: int = 10
    default_synthesis_method: str = "pattern_based"
    synthesis_methods: List[str] = field(
        default_factory=lambda: [
            "pattern_based",
            "template_based",
            "neural",
            "example_based",
            "constraint_based",
        ]
    )

    # Cache Configuration
    enable_cache: bool = True
    cache_ttl: int = 3600
    cache_max_size: int = 1000
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.85

    # Security Settings
    enable_sandbox: bool = True
    sandbox_timeout: int = 10
    sandbox_memory_limit: str = "512MB"
    allowed_imports: List[str] = field(
        default_factory=lambda: [
            "math",
            "datetime",
            "itertools",
            "collections",
            "json",
            "re",
        ]
    )

    # Quality Settings
    min_test_coverage: float = 0.8
    max_complexity_score: int = 10
    enable_type_checking: bool = True

    # Resource Limits
    max_memory_per_synthesis: str = "1GB"
    max_cpu_per_synthesis: float = 1.0
    queue_size_limit: int = 1000

    @validator("api_base_url")
    def validate_url(cls, v):
        if not validators.url(v):
            raise ValueError(f"Invalid URL: {v}")
        return v

    @validator("semantic_cache_threshold")
    def validate_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {v}")
        return v


class ConfigurationManager:
    """Centralized configuration management with hot-reloading"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path or "config.yaml")
        self.env_prefix = "JARVIS_"
        self._config = {}
        self._synthesis_config = None
        self._observers = []
        self._lock = threading.RLock()
        self._file_observer = None

        # Load initial configuration
        self.reload()

        # Start file watcher if enabled
        if self.get("monitoring.enable_config_reload", True):
            self._start_file_watcher()

    def _start_file_watcher(self):
        """Start watching configuration file for changes"""

        class ConfigFileHandler(FileSystemEventHandler):
            def __init__(self, config_manager):
                self.config_manager = config_manager

            def on_modified(self, event):
                if event.src_path == str(self.config_manager.config_path):
                    logger.info("Configuration file modified, reloading...")
                    self.config_manager.reload()

        self._file_observer = Observer()
        handler = ConfigFileHandler(self)
        self._file_observer.schedule(
            handler, path=str(self.config_path.parent), recursive=False
        )
        self._file_observer.start()

    def reload(self):
        """Reload configuration from file and environment"""
        with self._lock:
            try:
                # Load from YAML file
                if self.config_path.exists():
                    with open(self.config_path, "r") as f:
                        self._config = yaml.safe_load(f) or {}
                else:
                    logger.warning(f"Config file not found: {self.config_path}")
                    self._config = {}

                # Override with environment variables
                self._load_env_vars()

                # Validate and create synthesis config
                self._synthesis_config = self._create_synthesis_config()

                # Notify observers
                self._notify_observers()

                logger.info("Configuration loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                raise

    def _load_env_vars(self):
        """Load configuration from environment variables"""
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                # Convert JARVIS_SECTION_KEY to section.key
                config_key = key[len(self.env_prefix) :].lower().replace("_", ".")

                # Parse value
                parsed_value = self._parse_env_value(value)

                # Set in config
                self._set_nested(self._config, config_key, parsed_value)

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value"""
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        # Check for boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Check for number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _set_nested(self, data: Dict, key: str, value: Any):
        """Set nested dictionary value using dot notation"""
        keys = key.split(".")
        current = data

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def _create_synthesis_config(self) -> SynthesisConfig:
        """Create validated synthesis configuration"""
        synthesis_data = self.get("synthesis", {})

        # Map from general config
        synthesis_data.update(
            {
                "enable_cache": self.get("synthesis.cache.enabled", True),
                "cache_ttl": self.get("synthesis.cache.ttl", 3600),
                "cache_max_size": self.get("synthesis.cache.max_size", 1000),
                "semantic_cache_enabled": self.get(
                    "synthesis.cache.semantic_enabled", True
                ),
                "semantic_cache_threshold": self.get(
                    "synthesis.cache.semantic_threshold", 0.85
                ),
                "enable_sandbox": self.get("security.enable_sandbox", True),
                "sandbox_timeout": self.get("security.sandbox_timeout", 10),
                "sandbox_memory_limit": self.get(
                    "security.sandbox_memory_limit", "512MB"
                ),
                "max_concurrent_syntheses": self.get(
                    "resources.max_concurrent_syntheses", 10
                ),
            }
        )

        return SynthesisConfig(**synthesis_data)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        with self._lock:
            keys = key.split(".")
            current = self._config

            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return default

            return current

    def set(self, key: str, value: Any):
        """Set configuration value (runtime only, not persisted)"""
        with self._lock:
            self._set_nested(self._config, key, value)
            self._notify_observers()

    def get_synthesis_config(self) -> SynthesisConfig:
        """Get validated synthesis configuration"""
        return self._synthesis_config

    def register_observer(self, callback):
        """Register callback for configuration changes"""
        self._observers.append(callback)

    def _notify_observers(self):
        """Notify all observers of configuration change"""
        for observer in self._observers:
            try:
                observer(self._config)
            except Exception as e:
                logger.error(f"Observer notification failed: {e}")

    @lru_cache(maxsize=128)
    def get_api_headers(self) -> Dict[str, str]:
        """Get API headers with authentication"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"JARVIS/{self.get('version', '1.0.0')}",
        }

        api_key = self.get("synthesis.api_key") or os.getenv("SYNTHESIS_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        return headers

    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration"""
        return {
            "host": self.get("redis.host", "localhost"),
            "port": self.get("redis.port", 6379),
            "password": self.get("redis.password"),
            "db": self.get("redis.db", 0),
            "decode_responses": self.get("redis.decode_responses", True),
            "max_connections": self.get("redis.max_connections", 50),
        }

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return {
            "enable_metrics": self.get("monitoring.enable_metrics", True),
            "metrics_port": self.get("monitoring.metrics_port", 9090),
            "enable_tracing": self.get("monitoring.enable_tracing", False),
            "otlp_endpoint": self.get("monitoring.otlp_endpoint", "localhost:4317"),
            "log_level": self.get("monitoring.log_level", "INFO"),
            "log_format": self.get("monitoring.log_format", "json"),
        }

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []

        # Check required fields
        required_fields = [
            "redis.host",
            "websocket.port",
            "security.enable_authentication",
        ]

        for field in required_fields:
            if self.get(field) is None:
                issues.append(f"Missing required field: {field}")

        # Validate synthesis config
        try:
            self.get_synthesis_config()
        except Exception as e:
            issues.append(f"Invalid synthesis configuration: {e}")

        # Check file paths
        paths_to_check = ["paths.deployment", "paths.training_data", "paths.logs"]

        for path_key in paths_to_check:
            path_value = self.get(path_key)
            if path_value:
                path = Path(path_value)
                if not path.exists():
                    issues.append(f"Path does not exist: {path_key}={path_value}")

        return issues

    def export_env_template(self, output_path: str = ".env.example"):
        """Export environment variable template"""
        template = []
        template.append("# JARVIS Environment Configuration")
        template.append("# Generated on: " + datetime.now().isoformat())
        template.append("")

        # Flatten configuration for env vars
        def flatten(data, prefix=""):
            for key, value in data.items():
                if isinstance(value, dict):
                    flatten(value, f"{prefix}{key}_")
                else:
                    env_key = f"{self.env_prefix}{prefix}{key}".upper()
                    if isinstance(value, (list, dict)):
                        value = json.dumps(value)
                    template.append(f"{env_key}={value}")

        flatten(self._config)

        with open(output_path, "w") as f:
            f.write("\n".join(template))

        logger.info(f"Environment template exported to {output_path}")

    def __del__(self):
        """Cleanup file observer"""
        if self._file_observer:
            self._file_observer.stop()
            self._file_observer.join()


# Global configuration instance
config_manager = ConfigurationManager()


# Convenience functions
def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    return config_manager.get(key, default)


def get_synthesis_config() -> SynthesisConfig:
    """Get synthesis configuration"""
    return config_manager.get_synthesis_config()


def reload_config():
    """Reload configuration"""
    config_manager.reload()
