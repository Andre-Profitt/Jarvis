#!/usr/bin/env python3
"""
Secure Configuration Management for JARVIS
Handles API keys and sensitive data securely
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import getpass

logger = logging.getLogger(__name__)


class SecureConfigManager:
    """Secure configuration management with encryption"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path.home() / '.jarvis' / 'config'
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.config_dir / 'config.json'
        self.encrypted_file = self.config_dir / 'secrets.enc'
        self.key_file = self.config_dir / '.key'
        
        self._config = {}
        self._secrets = {}
        self._cipher = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize configuration"""
        # Load or create encryption key
        self._load_or_create_key()
        
        # Load configurations
        self._load_config()
        self._load_secrets()
        
        # Load from environment variables (highest priority)
        self._load_from_env()
    
    def _load_or_create_key(self):
        """Load or create encryption key"""
        if self.key_file.exists():
            # Load existing key
            try:
                with open(self.key_file, 'rb') as f:
                    key = f.read()
                self._cipher = Fernet(key)
            except Exception as e:
                logger.error(f"Failed to load encryption key: {e}")
                self._create_new_key()
        else:
            self._create_new_key()
    
    def _create_new_key(self):
        """Create new encryption key"""
        # Generate key from password
        password = getpass.getpass("Create a password for JARVIS config encryption: ").encode()
        
        # Use PBKDF2 for key derivation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'jarvis-salt-v1',  # In production, use random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        
        # Save key securely (600 permissions)
        self.key_file.touch(mode=0o600)
        with open(self.key_file, 'wb') as f:
            f.write(key)
        
        self._cipher = Fernet(key)
        logger.info("Created new encryption key")
    
    def _load_config(self):
        """Load general configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    self._config = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                self._config = self._get_default_config()
        else:
            self._config = self._get_default_config()
            self._save_config()
    
    def _load_secrets(self):
        """Load encrypted secrets"""
        if self.encrypted_file.exists() and self._cipher:
            try:
                with open(self.encrypted_file, 'rb') as f:
                    encrypted_data = f.read()
                
                decrypted_data = self._cipher.decrypt(encrypted_data)
                self._secrets = json.loads(decrypted_data.decode())
            except Exception as e:
                logger.error(f"Failed to load secrets: {e}")
                self._secrets = {}
        else:
            self._secrets = {}
    
    def _load_from_env(self):
        """Load API keys from environment variables"""
        env_mappings = {
            'OPENAI_API_KEY': 'api_keys.openai',
            'GEMINI_API_KEY': 'api_keys.gemini',
            'ELEVENLABS_API_KEY': 'api_keys.elevenlabs',
            'GITHUB_TOKEN': 'api_keys.github'
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self.set_secret(config_key, value)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'voice': {
                'enabled': True,
                'wake_words': ['jarvis', 'hey jarvis'],
                'language': 'en-US',
                'speech_rate': 175
            },
            'system': {
                'log_level': 'INFO',
                'max_retries': 3,
                'timeout': 30
            },
            'features': {
                'voice_interface': True,
                'memory_persistence': True,
                'autonomous_mode': False,
                'learning_enabled': True
            }
        }
    
    def _save_config(self):
        """Save general configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def _save_secrets(self):
        """Save encrypted secrets"""
        if not self._cipher:
            logger.error("No encryption key available")
            return
        
        try:
            data = json.dumps(self._secrets).encode()
            encrypted_data = self._cipher.encrypt(data)
            
            # Save with restricted permissions
            self.encrypted_file.touch(mode=0o600)
            with open(self.encrypted_file, 'wb') as f:
                f.write(encrypted_data)
        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        # First check secrets
        parts = key.split('.')
        
        # Check if it's an API key
        if parts[0] == 'api_keys':
            return self._get_nested(self._secrets, parts, default)
        
        # Otherwise check general config
        return self._get_nested(self._config, parts, default)
    
    def _get_nested(self, data: Dict, keys: list, default: Any) -> Any:
        """Get nested dictionary value"""
        for key in keys:
            if isinstance(data, dict) and key in data:
                data = data[key]
            else:
                return default
        return data
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        parts = key.split('.')
        self._set_nested(self._config, parts, value)
        self._save_config()
    
    def set_secret(self, key: str, value: str):
        """Set secret value (encrypted)"""
        parts = key.split('.')
        self._set_nested(self._secrets, parts, value)
        self._save_secrets()
    
    def _set_nested(self, data: Dict, keys: list, value: Any):
        """Set nested dictionary value"""
        for key in keys[:-1]:
            if key not in data:
                data[key] = {}
            data = data[key]
        data[keys[-1]] = value
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a service"""
        return self.get(f'api_keys.{service}')
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate which API keys are configured"""
        services = ['openai', 'gemini', 'elevenlabs', 'github']
        status = {}
        
        for service in services:
            key = self.get_api_key(service)
            status[service] = bool(key and len(key) > 10)
        
        return status


# Singleton instance
_config_manager = None


def get_config() -> SecureConfigManager:
    """Get singleton config manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = SecureConfigManager()
    return _config_manager
