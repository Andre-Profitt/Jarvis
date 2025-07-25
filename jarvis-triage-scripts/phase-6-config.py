#!/usr/bin/env python3
"""
Unified Configuration Setup
Agent: Python Modernizer + Structure Architect
Phase: 6 - Single Source of Truth
"""

import os
import json
from pathlib import Path
from typing import Dict, Any

# Configuration template
SETTINGS_TEMPLATE = '''"""
Jarvis Unified Configuration
Single source of truth for all services
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
from functools import lru_cache


class CommonSettings(BaseSettings):
    """Base settings shared across all services"""
    
    # Application
    app_name: str = "Jarvis AI"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = "development"
    
    # API Keys (from environment)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # Database
    database_url: str = "postgresql://jarvis:jarvis@localhost/jarvis"
    redis_url: str = "redis://localhost:6379/0"
    
    # Security
    secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


class OrchestratorSettings(CommonSettings):
    """Settings specific to the orchestrator service"""
    
    # Service configuration
    service_name: str = "orchestrator"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    
    # Task queue
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    
    # Plugins
    plugin_directory: str = "./plugins"
    auto_load_plugins: bool = True


class CoreSettings(CommonSettings):
    """Settings specific to the core library"""
    
    # Model configuration
    model_cache_dir: str = "./models"
    max_model_size_gb: float = 4.0
    enable_gpu: bool = True
    
    # Processing
    batch_size: int = 32
    max_sequence_length: int = 512
    num_workers: int = 4
    
    # Memory limits
    max_memory_mb: int = 8192
    cache_size_mb: int = 1024


class UISettings(CommonSettings):
    """Settings specific to the UI service"""
    
    # Service configuration
    service_name: str = "ui"
    next_public_api_url: str = "http://localhost:8000"
    next_public_ws_url: str = "ws://localhost:8000"
    
    # Analytics
    next_public_analytics_id: Optional[str] = None
    
    # Features
    enable_telemetry: bool = False
    enable_experimental: bool = False


@lru_cache()
def get_settings(service: str = "common") -> BaseSettings:
    """
    Get settings for a specific service
    
    Args:
        service: Service name (orchestrator, core, ui, or common)
        
    Returns:
        Settings instance for the service
    """
    settings_map = {
        "common": CommonSettings,
        "orchestrator": OrchestratorSettings,
        "core": CoreSettings,
        "ui": UISettings,
    }
    
    settings_class = settings_map.get(service, CommonSettings)
    return settings_class()


# Export commonly used settings
settings = get_settings()
'''

ENV_TEMPLATE = '''# Jarvis Environment Configuration
# Copy this file to .env and update with your values

# Environment
ENVIRONMENT=development
DEBUG=true

# API Keys (obtain from respective providers)
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here

# Database
DATABASE_URL=postgresql://jarvis:jarvis@localhost:5432/jarvis
REDIS_URL=redis://localhost:6379/0

# Security (generate new keys for production)
SECRET_KEY=your-secret-key-here-change-in-production
JWT_SECRET_KEY=your-jwt-secret-here-change-in-production

# Service URLs
ORCHESTRATOR_URL=http://localhost:8000
UI_URL=http://localhost:3000

# Feature Flags
ENABLE_TELEMETRY=false
ENABLE_EXPERIMENTAL=false
ENABLE_GPU=true

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Resource Limits
MAX_MEMORY_MB=8192
MAX_MODEL_SIZE_GB=4.0
'''

DOCKER_ENV_TEMPLATE = '''# Docker Compose Environment
# Automatically loaded by docker-compose.yml

# Container Configuration
COMPOSE_PROJECT_NAME=jarvis
COMPOSE_FILE=docker-compose.yml:docker-compose.override.yml

# Service Ports
ORCHESTRATOR_PORT=8000
UI_PORT=3000
POSTGRES_PORT=5432
REDIS_PORT=6379

# Database Credentials
POSTGRES_USER=jarvis
POSTGRES_PASSWORD=jarvis-db-password
POSTGRES_DB=jarvis

# Redis Configuration
REDIS_PASSWORD=jarvis-redis-password

# Volumes
DATA_VOLUME=./data
LOGS_VOLUME=./logs
'''

def create_config_structure():
    """Create unified configuration structure"""
    print("🔧 Creating unified configuration...")
    
    # Create config directory
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Write settings module
    settings_file = Path("services/core/jarvis_core/settings.py")
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    settings_file.write_text(SETTINGS_TEMPLATE)
    print(f"✓ Created: {settings_file}")
    
    # Create environment templates
    env_files = {
        ".env.example": ENV_TEMPLATE,
        ".env.docker": DOCKER_ENV_TEMPLATE,
    }
    
    for filename, content in env_files.items():
        env_file = Path(filename)
        env_file.write_text(content)
        print(f"✓ Created: {env_file}")
    
    # Create service-specific configs
    service_configs = {
        "orchestrator": {
            "service_name": "orchestrator",
            "port": 8000,
            "workers": 4,
            "features": ["api", "websocket", "plugins"]
        },
        "ui": {
            "service_name": "ui",
            "port": 3000,
            "features": ["ssr", "api-routes", "websocket"]
        },
        "core": {
            "service_name": "core",
            "features": ["ml-models", "nlp", "memory"]
        }
    }
    
    for service, config in service_configs.items():
        config_file = config_dir / f"{service}.json"
        config_file.write_text(json.dumps(config, indent=2))
        print(f"✓ Created: {config_file}")
    
    # Create config loader utility
    loader_file = config_dir / "loader.py"
    loader_content = '''"""Configuration loader utilities"""
import os
import json
from pathlib import Path
from typing import Dict, Any

def load_service_config(service: str) -> Dict[str, Any]:
    """Load configuration for a specific service"""
    config_file = Path(__file__).parent / f"{service}.json"
    if config_file.exists():
        return json.loads(config_file.read_text())
    return {}

def get_env_value(key: str, default: Any = None) -> Any:
    """Get environment variable with type conversion"""
    value = os.environ.get(key, default)
    if value is None:
        return None
    
    # Try to parse as JSON for complex types
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        # Try boolean conversion
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        
        # Try numeric conversion
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
'''
    loader_file.write_text(loader_content)
    print(f"✓ Created: {loader_file}")
    
    print("\n✅ Unified configuration structure created!")
    print("\n📋 Next steps:")
    print("1. Copy .env.example to .env")
    print("2. Update .env with your API keys and settings")
    print("3. Import settings in your services:")
    print("   from jarvis_core.settings import get_settings")
    print("   settings = get_settings('orchestrator')")

if __name__ == "__main__":
    create_config_structure()