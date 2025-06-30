"""
JARVIS Production Configuration Manager
Handles environment variables and production settings
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
import yaml
from cryptography.fernet import Fernet
import logging


@dataclass
class JARVISConfig:
    """Main configuration class for JARVIS"""

    # Core Settings
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = ""

    # Database
    DATABASE_URL: str = "sqlite:///jarvis.db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PASSWORD: Optional[str] = None

    # API Keys (encrypted in production)
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    GITHUB_TOKEN: Optional[str] = None
    ELEVENLABS_API_KEY: Optional[str] = None

    # Service URLs
    CONSCIOUSNESS_SERVICE_URL: str = "http://localhost:8001"
    SCHEDULER_SERVICE_URL: str = "http://localhost:8002"
    KNOWLEDGE_SERVICE_URL: str = "http://localhost:8003"
    MONITORING_SERVICE_URL: str = "http://localhost:8004"

    # Feature Flags
    ENABLE_CONSCIOUSNESS: bool = True
    ENABLE_SELF_IMPROVEMENT: bool = True
    ENABLE_MULTI_AI: bool = True
    ENABLE_VOICE: bool = True

    # Security
    JWT_SECRET_KEY: str = ""
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24

    # Monitoring
    SENTRY_DSN: Optional[str] = None
    LOG_LEVEL: str = "INFO"
    METRICS_ENABLED: bool = True

    # Performance
    MAX_WORKERS: int = 4
    REQUEST_TIMEOUT: int = 30
    CACHE_TTL: int = 3600

    # ML/AI Settings
    MODEL_CACHE_DIR: str = "./models"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    LLM_TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2000


class ConfigManager:
    """Manages configuration loading and validation"""

    def __init__(self):
        self.config = JARVISConfig()
        self.encryption_key = None
        self.logger = logging.getLogger(__name__)

    def load_config(self, env: str = "development") -> JARVISConfig:
        """Load configuration based on environment"""
        self.config.ENVIRONMENT = env

        # Load from multiple sources in order of precedence
        self._load_defaults()
        self._load_from_file(f"config/{env}.yaml")
        self._load_from_env()

        # Validate configuration
        self._validate_config()

        # Decrypt sensitive values in production
        if env == "production":
            self._decrypt_secrets()

        return self.config

    def _load_defaults(self):
        """Load default configuration values"""
        defaults = {
            "SECRET_KEY": self._generate_secret_key(),
            "JWT_SECRET_KEY": self._generate_secret_key(),
            "DATABASE_URL": "sqlite:///jarvis.db",
            "REDIS_URL": "redis://localhost:6379/0",
            "LOG_LEVEL": "INFO",
        }

        for key, value in defaults.items():
            if not getattr(self.config, key):
                setattr(self.config, key, value)

    def _load_from_file(self, filepath: str):
        """Load configuration from YAML file"""
        path = Path(filepath)
        if not path.exists():
            self.logger.warning(f"Config file not found: {filepath}")
            return

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if data:
            for key, value in data.items():
                if hasattr(self.config, key.upper()):
                    setattr(self.config, key.upper(), value)

    def _load_from_env(self):
        """Load configuration from environment variables"""
        for field in self.config.__dataclass_fields__:
            env_value = os.getenv(f"JARVIS_{field}")
            if env_value is not None:
                # Convert to appropriate type
                field_type = self.config.__dataclass_fields__[field].type
                if field_type == bool:
                    value = env_value.lower() in ("true", "1", "yes")
                elif field_type == int:
                    value = int(env_value)
                elif field_type == float:
                    value = float(env_value)
                else:
                    value = env_value

                setattr(self.config, field, value)

    def _validate_config(self):
        """Validate configuration values"""
        errors = []

        # Check required fields
        if self.config.ENVIRONMENT == "production":
            required_fields = ["SECRET_KEY", "JWT_SECRET_KEY", "DATABASE_URL"]
            for field in required_fields:
                if not getattr(self.config, field):
                    errors.append(f"Missing required field: {field}")

        # Validate URLs
        for field in ["DATABASE_URL", "REDIS_URL"]:
            url = getattr(self.config, field)
            if url and not self._is_valid_url(url):
                errors.append(f"Invalid URL for {field}: {url}")

        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")

    def _decrypt_secrets(self):
        """Decrypt sensitive configuration values"""
        if not self.encryption_key:
            key_path = Path(".encryption_key")
            if key_path.exists():
                self.encryption_key = key_path.read_text().strip()
            else:
                raise ValueError("Encryption key not found for production")

        fernet = Fernet(self.encryption_key.encode())

        # Decrypt API keys
        encrypted_fields = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "GITHUB_TOKEN",
            "ELEVENLABS_API_KEY",
        ]

        for field in encrypted_fields:
            value = getattr(self.config, field)
            if value and value.startswith("encrypted:"):
                encrypted_data = value[10:]  # Remove 'encrypted:' prefix
                decrypted = fernet.decrypt(encrypted_data.encode()).decode()
                setattr(self.config, field, decrypted)

    def _generate_secret_key(self) -> str:
        """Generate a secure secret key"""
        import secrets

        return secrets.token_urlsafe(32)

    def _is_valid_url(self, url: str) -> bool:
        """Basic URL validation"""
        return url.startswith(
            ("http://", "https://", "redis://", "postgresql://", "sqlite://")
        )

    def save_config_template(self, env: str = "production"):
        """Save configuration template file"""
        template = {
            "environment": env,
            "debug": False if env == "production" else True,
            "database": {
                "url": "postgresql://user:pass@localhost/jarvis",
                "pool_size": 10,
                "max_overflow": 20,
            },
            "redis": {
                "url": "redis://localhost:6379/0",
                "password": "your-redis-password",
            },
            "api_keys": {
                "openai_api_key": "encrypted:...",
                "anthropic_api_key": "encrypted:...",
                "google_api_key": "encrypted:...",
                "github_token": "encrypted:...",
                "elevenlabs_api_key": "encrypted:...",
            },
            "services": {
                "consciousness_service_url": "http://consciousness:8001",
                "scheduler_service_url": "http://scheduler:8002",
                "knowledge_service_url": "http://knowledge:8003",
                "monitoring_service_url": "http://monitoring:8004",
            },
            "features": {
                "enable_consciousness": True,
                "enable_self_improvement": True,
                "enable_multi_ai": True,
                "enable_voice": True,
            },
            "security": {
                "jwt_secret_key": "generate-a-secure-key",
                "jwt_algorithm": "HS256",
                "jwt_expiration_hours": 24,
            },
            "monitoring": {
                "sentry_dsn": "https://your-sentry-dsn",
                "log_level": "INFO",
                "metrics_enabled": True,
            },
            "performance": {"max_workers": 4, "request_timeout": 30, "cache_ttl": 3600},
            "ml_settings": {
                "model_cache_dir": "./models",
                "embedding_model": "text-embedding-ada-002",
                "llm_temperature": 0.7,
                "max_tokens": 2000,
            },
        }

        # Create config directory
        Path("config").mkdir(exist_ok=True)

        # Save as YAML
        with open(f"config/{env}.yaml", "w") as f:
            yaml.dump(template, f, default_flow_style=False)

        print(f"✅ Configuration template saved to config/{env}.yaml")


def create_env_file():
    """Create .env.production template file"""
    env_content = """# JARVIS Production Environment Variables

# Core Settings
JARVIS_ENVIRONMENT=production
JARVIS_DEBUG=false
JARVIS_SECRET_KEY=your-secret-key-here
JARVIS_JWT_SECRET_KEY=your-jwt-secret-here

# Database
JARVIS_DATABASE_URL=postgresql://jarvis:password@localhost:5432/jarvis_prod
JARVIS_DATABASE_POOL_SIZE=20
JARVIS_DATABASE_MAX_OVERFLOW=40

# Redis
JARVIS_REDIS_URL=redis://localhost:6379/0
JARVIS_REDIS_PASSWORD=your-redis-password

# API Keys (use encryption for production)
JARVIS_OPENAI_API_KEY=encrypted:...
JARVIS_ANTHROPIC_API_KEY=encrypted:...
JARVIS_GOOGLE_API_KEY=encrypted:...
JARVIS_GITHUB_TOKEN=encrypted:...
JARVIS_ELEVENLABS_API_KEY=encrypted:...

# Service URLs (Docker service names in production)
JARVIS_CONSCIOUSNESS_SERVICE_URL=http://consciousness:8001
JARVIS_SCHEDULER_SERVICE_URL=http://scheduler:8002
JARVIS_KNOWLEDGE_SERVICE_URL=http://knowledge:8003
JARVIS_MONITORING_SERVICE_URL=http://monitoring:8004

# Feature Flags
JARVIS_ENABLE_CONSCIOUSNESS=true
JARVIS_ENABLE_SELF_IMPROVEMENT=true
JARVIS_ENABLE_MULTI_AI=true
JARVIS_ENABLE_VOICE=true

# Monitoring
JARVIS_SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
JARVIS_LOG_LEVEL=INFO
JARVIS_METRICS_ENABLED=true

# Performance
JARVIS_MAX_WORKERS=8
JARVIS_REQUEST_TIMEOUT=60
JARVIS_CACHE_TTL=7200

# ML/AI Settings
JARVIS_MODEL_CACHE_DIR=/var/lib/jarvis/models
JARVIS_EMBEDDING_MODEL=text-embedding-ada-002
JARVIS_LLM_TEMPERATURE=0.7
JARVIS_MAX_TOKENS=2000
"""

    with open(".env.production", "w") as f:
        f.write(env_content)

    print("✅ Production environment template saved to .env.production")


def create_encryption_utils():
    """Create utility scripts for managing encrypted secrets"""
    encrypt_script = '''#!/usr/bin/env python3
"""Encrypt sensitive configuration values"""

from cryptography.fernet import Fernet
import sys
import getpass

def generate_key():
    """Generate a new encryption key"""
    key = Fernet.generate_key()
    with open('.encryption_key', 'wb') as f:
        f.write(key)
    print("✅ Encryption key saved to .encryption_key")
    print("⚠️  Keep this key secure and never commit it to version control!")

def encrypt_value(value: str) -> str:
    """Encrypt a single value"""
    try:
        with open('.encryption_key', 'rb') as f:
            key = f.read()
    except FileNotFoundError:
        print("❌ Encryption key not found. Run with --generate-key first.")
        sys.exit(1)
    
    fernet = Fernet(key)
    encrypted = fernet.encrypt(value.encode())
    return f"encrypted:{encrypted.decode()}"

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-key":
        generate_key()
    else:
        value = getpass.getpass("Enter value to encrypt: ")
        encrypted = encrypt_value(value)
        print(f"\\nEncrypted value:\\n{encrypted}")
'''

    with open("scripts/encrypt_secrets.py", "w") as f:
        f.write(encrypt_script)

    os.chmod("scripts/encrypt_secrets.py", 0o755)
    print("✅ Encryption utility saved to scripts/encrypt_secrets.py")


if __name__ == "__main__":
    # Create configuration manager
    config_manager = ConfigManager()

    # Save configuration templates
    config_manager.save_config_template("development")
    config_manager.save_config_template("production")

    # Create .env template
    create_env_file()

    # Create encryption utilities
    create_encryption_utils()

    print("\\n✅ Production configuration setup complete!")
    print("📝 Next steps:")
    print("1. Edit .env.production with your actual values")
    print("2. Run: python scripts/encrypt_secrets.py --generate-key")
    print("3. Encrypt sensitive values: python scripts/encrypt_secrets.py")
    print("4. Update config/production.yaml with encrypted values")
