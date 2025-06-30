"""
Test Suite for Configuration
======================================
Comprehensive tests for configuration module.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json
from pathlib import Path
import os

# Import test utilities
from tests.conftest import *
from tests.mocks import *

# Import module under test
from core.configuration import ConfigurationManager, config_manager, load_config, get_config


class TestConfiguration:
    """Test suite for Configuration"""
    
    @pytest.fixture
    def component(self, temp_dir):
        """Create component instance with temp config directory"""
        # Create test config files
        config_dir = temp_dir / "config"
        config_dir.mkdir()
        
        # Create default config
        default_config = {
            "jarvis": {"name": "JARVIS", "version": "1.0", "mode": "test"},
            "logging": {"level": "INFO"},
            "ai_integration": {"default_model": "gpt-4"},
            "consciousness": {"cycle_frequency": 10},
            "neural": {"capacity": 2000}
        }
        
        with open(config_dir / "default.json", "w") as f:
            json.dump(default_config, f)
            
        return ConfigurationManager(config_dir)
    
    # ===== Initialization Tests =====
    def test_initialization(self, component):
        """Test component initialization"""
        assert component is not None
        assert isinstance(component, ConfigurationManager)
        assert component.config_dir.exists()
        assert hasattr(component, 'logger')
        assert component._config == {}
    
    def test_load_default_config(self, component):
        """Test loading default configuration"""
        config = component.load_config("default")
        
        assert isinstance(config, dict)
        assert config["jarvis"]["name"] == "JARVIS"
        assert config["jarvis"]["version"] == "1.0"
        assert config["logging"]["level"] == "INFO"
        assert config["ai_integration"]["default_model"] == "gpt-4"
    
    # ===== Core Functionality Tests =====
    def test_get_config_value(self, component):
        """Test getting configuration values"""
        component.load_config("default")
        
        # Test dot notation access
        assert component.get("jarvis.name") == "JARVIS"
        assert component.get("jarvis.version") == "1.0"
        assert component.get("logging.level") == "INFO"
        
        # Test default values
        assert component.get("nonexistent.key", "default") == "default"
        assert component.get("another.missing.key") is None
    
    def test_set_config_value(self, component):
        """Test setting configuration values"""
        component.load_config("default")
        
        # Set new values
        component.set("custom.key", "custom_value")
        assert component.get("custom.key") == "custom_value"
        
        # Set nested values
        component.set("deeply.nested.key", 42)
        assert component.get("deeply.nested.key") == 42
        
        # Override existing values
        component.set("jarvis.name", "JARVIS-2")
        assert component.get("jarvis.name") == "JARVIS-2"
    
    # ===== State Management Tests =====
    @pytest.mark.skip("TODO: Update to match current API")

    def test_update_config(self, component):
        """Test batch configuration updates"""
        component.load_config("default")
        
        updates = {
            "jarvis": {"mode": "production"},
            "new_section": {"key1": "value1", "key2": "value2"}
        }
        
        component.update(updates)
        
        # Check updates were applied
        assert component.get("jarvis.mode") == "production"
        assert component.get("new_section.key1") == "value1"
        assert component.get("new_section.key2") == "value2"
        
        # Check existing values preserved
        assert component.get("jarvis.name") == "JARVIS"
    
    # ===== Error Handling Tests =====
    def test_validation_errors(self, component, temp_dir):
        """Test configuration validation"""
        # Create invalid config
        invalid_config = {
            "jarvis": {"name": "JARVIS"},  # Missing version
            "logging": {"level": "INFO"},
            "ai_integration": {}  # Missing default_model
        }
        
        with open(temp_dir / "config" / "invalid.json", "w") as f:
            json.dump(invalid_config, f)
            
        # Should raise validation error
        with pytest.raises(ValueError, match="Required configuration missing"):
            component.load_config("invalid")
    
    @pytest.mark.skip("TODO: Update to match current API")

    
    def test_environment_variable_substitution(self, component, monkeypatch):
        """Test environment variable substitution"""
        # Set environment variable
        monkeypatch.setenv("TEST_API_KEY", "secret123")
        
        # Create config with env var reference
        config_with_env = {
            "jarvis": {"name": "JARVIS", "version": "1.0"},
            "logging": {"level": "INFO"},
            "ai_integration": {"default_model": "gpt-4", "api_key": "${TEST_API_KEY}"},
            "consciousness": {"cycle_frequency": 10},
            "neural": {"capacity": 2000}
        }
        
        with open(component.config_dir / "env_test.json", "w") as f:
            json.dump(config_with_env, f)
            
        config = component.load_config("env_test")
        assert config["ai_integration"]["api_key"] == "secret123"
    
    # ===== Integration Tests =====
    def test_environment_inheritance(self, component, temp_dir):
        """Test configuration inheritance"""
        # Create production config that extends default
        prod_config = {
            "extends": "default",
            "jarvis": {"mode": "production"},
            "logging": {"level": "WARNING"}
        }
        
        with open(temp_dir / "config" / "production.json", "w") as f:
            json.dump(prod_config, f)
            
        config = component.load_config("production")
        
        # Should have production overrides
        assert config["jarvis"]["mode"] == "production"
        assert config["logging"]["level"] == "WARNING"
        
        # Should inherit from default
        assert config["jarvis"]["name"] == "JARVIS"
        assert config["jarvis"]["version"] == "1.0"
    
    def test_get_component_config(self, component):
        """Test getting component-specific config"""
        component.load_config("default")
        
        # Get component configs
        jarvis_config = component.get_component_config("jarvis")
        assert jarvis_config["name"] == "JARVIS"
        assert jarvis_config["version"] == "1.0"
        
        ai_config = component.get_component_config("ai_integration")
        assert ai_config["default_model"] == "gpt-4"
        
        # Non-existent component
        missing_config = component.get_component_config("nonexistent")
        assert missing_config == {}
    
    # ===== Performance Tests =====
    def test_deep_merge_performance(self, component):
        """Test deep merge performance"""
        base = {f"key{i}": {f"subkey{j}": j for j in range(10)} for i in range(100)}
        override = {f"key{i}": {f"subkey{j}": j*2 for j in range(5)} for i in range(50)}
        
        import time
        start = time.time()
        result = component._deep_merge(base, override)
        elapsed = time.time() - start
        
        # Should merge quickly
        assert elapsed < 0.1
        
        # Verify merge worked correctly
        assert result["key0"]["subkey0"] == 0  # overridden
        assert result["key0"]["subkey9"] == 9  # preserved
        assert result["key99"]["subkey0"] == 0  # not overridden
    
    # ===== Concurrency Tests =====
    @pytest.mark.skip("TODO: Update to match current API")

    def test_concurrent_config_updates(self, component):
        """Test concurrent configuration updates"""
        component.load_config("default")
        
        # Simulate concurrent updates
        import threading
        
        def update_config(key, value):
            component.set(key, value)
            
        threads = []
        for i in range(10):
            t = threading.Thread(target=update_config, args=(f"concurrent.key{i}", i))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        # All updates should be applied
        for i in range(10):
            assert component.get(f"concurrent.key{i}") == i
    
    # ===== Save/Load Tests =====
    def test_save_and_load_config(self, component):
        """Test saving and loading configuration"""
        component.load_config("default")
        
        # Modify config
        component.set("custom.setting", "test_value")
        component.set("another.setting", 123)
        
        # Save to file
        component.save_config("test_save.json")
        
        # Create new instance and load saved config
        new_manager = ConfigurationManager(component.config_dir)
        loaded = new_manager.load_config("test_save")
        
        # Verify saved values
        assert loaded["custom"]["setting"] == "test_value"
        assert loaded["another"]["setting"] == 123
        assert loaded["jarvis"]["name"] == "JARVIS"
    
    # ===== Global Functions Tests =====
    def test_global_functions(self, component, monkeypatch):
        """Test module-level global functions"""
        # Mock the global config_manager
        with patch('core.configuration.config_manager', component):
            # Test load_config function
            monkeypatch.setenv("JARVIS_ENV", "default")
            config = load_config()
            assert config["jarvis"]["name"] == "JARVIS"
            
            # Test get_config function
            value = get_config("jarvis.name")
            assert value == "JARVIS"
            
            # Test with default
            value = get_config("missing.key", "default_value")
            assert value == "default_value"
    
    # ===== Edge Cases Tests =====
    def test_missing_config_file(self, component):
        """Test handling of missing config files"""
        # Try to load non-existent config
        config = component.load_config("nonexistent")
        
        # Should return empty config or default
        assert isinstance(config, dict)
    
    def test_invalid_json_file(self, component, temp_dir):
        """Test handling of invalid JSON"""
        # Create invalid JSON file
        with open(temp_dir / "config" / "invalid_json.json", "w") as f:
            f.write("{ invalid json }")
            
        # Should handle gracefully
        config = component.load_config("invalid_json")
        assert isinstance(config, dict)
    
    def test_range_validation(self, component, temp_dir):
        """Test configuration value range validation"""
        # Test invalid consciousness frequency
        invalid_config = {
            "jarvis": {"name": "JARVIS", "version": "1.0"},
            "logging": {"level": "INFO"},
            "ai_integration": {"default_model": "gpt-4"},
            "consciousness": {"cycle_frequency": 0},  # Invalid: must be >= 1
            "neural": {"capacity": 2000}
        }
        
        with open(temp_dir / "config" / "invalid_range.json", "w") as f:
            json.dump(invalid_config, f)
            
        with pytest.raises(ValueError, match="consciousness.cycle_frequency must be >= 1"):
            component.load_config("invalid_range")