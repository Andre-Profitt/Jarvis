"""
JARVIS Phase 11: Production Deployment Configuration
Optimized settings for production deployment of the integrated system
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import yaml

@dataclass
class Phase11Config:
    """Production configuration for Phase 11 integrated system"""
    
    # General Settings
    environment: str = "production"
    debug: bool = False
    log_level: str = "INFO"
    
    # Performance Settings
    max_concurrent_requests: int = 1000
    request_timeout_ms: int = 5000
    critical_request_timeout_ms: int = 500
    
    # Phase-specific Configurations
    phase1_config: Dict[str, Any] = field(default_factory=lambda: {
        "pipeline": {
            "buffer_size": 10000,
            "priority_queues": 5,
            "batch_processing": True,
            "batch_size": 50
        },
        "state_manager": {
            "smoothing_algorithm": "adaptive",
            "transition_threshold": 0.7,
            "history_size": 1000
        }
    })
    
    phase2_config: Dict[str, Any] = field(default_factory=lambda: {
        "context_memory": {
            "max_memory_size_mb": 500,
            "cache_strategy": "lru",
            "persistence_enabled": True,
            "persistence_interval_seconds": 300
        },
        "pattern_detection": {
            "min_pattern_occurrences": 3,
            "pattern_timeout_hours": 168  # 1 week
        }
    })
    
    phase3_config: Dict[str, Any] = field(default_factory=lambda: {
        "interventions": {
            "min_confidence": 0.8,
            "cooldown_minutes": 15,
            "max_daily_interventions": 10,
            "escalation_enabled": True
        }
    })
    
    phase4_config: Dict[str, Any] = field(default_factory=lambda: {
        "nlp": {
            "model_cache_size": 5,
            "response_cache_size": 10000,
            "max_response_length": 2000,
            "context_window": 4000
        }
    })
    
    phase5_config: Dict[str, Any] = field(default_factory=lambda: {
        "ui": {
            "update_interval_ms": 100,
            "animation_enabled": True,
            "theme": "adaptive",
            "accessibility_mode": True
        }
    })
    
    phase6_config: Dict[str, Any] = field(default_factory=lambda: {
        "cognitive_load": {
            "max_concurrent_notifications": 3,
            "summarization_threshold": 500,
            "progressive_disclosure": True
        }
    })
    
    phase7_config: Dict[str, Any] = field(default_factory=lambda: {
        "performance": {
            "cache_enabled": True,
            "cache_size_mb": 1000,
            "parallel_processing": True,
            "num_workers": os.cpu_count() or 4,
            "lazy_loading": True
        }
    })
    
    phase8_config: Dict[str, Any] = field(default_factory=lambda: {
        "feedback": {
            "learning_rate": 0.01,
            "feedback_buffer_size": 10000,
            "minimum_feedback_for_learning": 10,
            "preference_decay_days": 90
        }
    })
    
    phase9_config: Dict[str, Any] = field(default_factory=lambda: {
        "personalization": {
            "profile_update_interval_hours": 24,
            "min_interactions_for_profile": 50,
            "adaptation_speed": "moderate",
            "privacy_mode": "standard"
        }
    })
    
    phase10_config: Dict[str, Any] = field(default_factory=lambda: {
        "production": {
            "health_check_interval_seconds": 30,
            "metric_collection_interval_seconds": 60,
            "auto_scaling_enabled": True,
            "min_instances": 2,
            "max_instances": 10
        }
    })
    
    # Integration Settings
    integration_config: Dict[str, Any] = field(default_factory=lambda: {
        "cross_phase_timeout_ms": 100,
        "circuit_breaker_threshold": 5,
        "circuit_breaker_timeout_seconds": 60,
        "retry_policy": {
            "max_retries": 3,
            "backoff_multiplier": 2,
            "max_backoff_seconds": 10
        }
    })
    
    # Resource Limits
    resource_limits: Dict[str, Any] = field(default_factory=lambda: {
        "max_memory_gb": 8,
        "max_cpu_percent": 80,
        "max_disk_io_mbps": 100,
        "max_network_bandwidth_mbps": 1000
    })
    
    # Monitoring and Alerting
    monitoring_config: Dict[str, Any] = field(default_factory=lambda: {
        "metrics_enabled": True,
        "metrics_endpoint": "http://localhost:9090/metrics",
        "alerting_enabled": True,
        "alert_channels": ["email", "slack"],
        "critical_alerts": {
            "error_rate_threshold": 0.05,
            "latency_p95_threshold_ms": 1000,
            "memory_usage_threshold_percent": 90
        }
    })
    
    # Security Settings
    security_config: Dict[str, Any] = field(default_factory=lambda: {
        "authentication_required": True,
        "encryption_enabled": True,
        "rate_limiting_enabled": True,
        "rate_limit_per_minute": 1000,
        "ip_whitelist_enabled": False,
        "audit_logging_enabled": True
    })


class ConfigurationManager:
    """Manages configuration for Phase 11 deployment"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "phase11_config.yaml"
        self.config = Phase11Config()
        self._load_config()
        self._apply_environment_overrides()
    
    def _load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
                
                # Update config with loaded data
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides"""
        # General overrides
        self.config.environment = os.getenv('JARVIS_ENV', self.config.environment)
        self.config.debug = os.getenv('JARVIS_DEBUG', '').lower() == 'true'
        self.config.log_level = os.getenv('JARVIS_LOG_LEVEL', self.config.log_level)
        
        # Performance overrides
        if os.getenv('JARVIS_MAX_CONCURRENT'):
            self.config.max_concurrent_requests = int(os.getenv('JARVIS_MAX_CONCURRENT'))
        
        # Resource limit overrides
        if os.getenv('JARVIS_MAX_MEMORY_GB'):
            self.config.resource_limits['max_memory_gb'] = int(os.getenv('JARVIS_MAX_MEMORY_GB'))
    
    def get_phase_config(self, phase_number: int) -> Dict[str, Any]:
        """Get configuration for specific phase"""
        phase_key = f"phase{phase_number}_config"
        return getattr(self.config, phase_key, {})
    
    def get_integration_config(self) -> Dict[str, Any]:
        """Get integration configuration"""
        return self.config.integration_config
    
    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file"""
        save_path = path or self.config_path
        
        config_dict = {
            key: getattr(self.config, key)
            for key in dir(self.config)
            if not key.startswith('_') and not callable(getattr(self.config, key))
        }
        
        if save_path.endswith('.yaml'):
            with open(save_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            with open(save_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration settings"""
        issues = []
        warnings = []
        
        # Check resource limits
        if self.config.resource_limits['max_memory_gb'] < 4:
            warnings.append("Memory limit below recommended 4GB")
        
        # Check performance settings
        if self.config.max_concurrent_requests > 5000:
            warnings.append("Very high concurrent request limit may cause instability")
        
        # Check phase configurations
        if self.config.phase7_config['performance']['cache_size_mb'] > 2000:
            warnings.append("Cache size over 2GB may impact memory usage")
        
        # Check security
        if not self.config.security_config['authentication_required']:
            issues.append("Authentication disabled in production")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def get_optimized_settings(self, workload_type: str) -> Dict[str, Any]:
        """Get optimized settings for specific workload types"""
        optimizations = {
            'high_throughput': {
                'max_concurrent_requests': 2000,
                'phase1_config.pipeline.buffer_size': 20000,
                'phase7_config.performance.num_workers': os.cpu_count() * 2
            },
            'low_latency': {
                'request_timeout_ms': 1000,
                'critical_request_timeout_ms': 100,
                'phase1_config.pipeline.batch_processing': False,
                'phase7_config.performance.cache_enabled': True
            },
            'memory_constrained': {
                'resource_limits.max_memory_gb': 2,
                'phase2_config.context_memory.max_memory_size_mb': 200,
                'phase7_config.performance.cache_size_mb': 200
            },
            'development': {
                'debug': True,
                'log_level': 'DEBUG',
                'max_concurrent_requests': 100,
                'monitoring_config.metrics_enabled': True
            }
        }
        
        return optimizations.get(workload_type, {})


# Production deployment configurations
PRODUCTION_CONFIGS = {
    "aws": {
        "environment": "production",
        "max_concurrent_requests": 5000,
        "resource_limits": {
            "max_memory_gb": 16,
            "max_cpu_percent": 90
        },
        "phase7_config": {
            "performance": {
                "cache_enabled": True,
                "cache_size_mb": 4000,
                "num_workers": 16
            }
        },
        "monitoring_config": {
            "metrics_endpoint": "http://cloudwatch.aws.internal/metrics",
            "alert_channels": ["sns", "email"]
        }
    },
    
    "gcp": {
        "environment": "production", 
        "max_concurrent_requests": 3000,
        "resource_limits": {
            "max_memory_gb": 12,
            "max_cpu_percent": 85
        },
        "monitoring_config": {
            "metrics_endpoint": "http://stackdriver.google.internal/metrics",
            "alert_channels": ["pubsub", "email"]
        }
    },
    
    "on_premise": {
        "environment": "production",
        "max_concurrent_requests": 1000,
        "resource_limits": {
            "max_memory_gb": 8,
            "max_cpu_percent": 80
        },
        "security_config": {
            "ip_whitelist_enabled": True,
            "audit_logging_enabled": True
        }
    }
}


def create_deployment_config(deployment_type: str = "on_premise") -> Phase11Config:
    """Create deployment-specific configuration"""
    config_manager = ConfigurationManager()
    
    # Apply deployment-specific settings
    if deployment_type in PRODUCTION_CONFIGS:
        deployment_settings = PRODUCTION_CONFIGS[deployment_type]
        for key, value in deployment_settings.items():
            if hasattr(config_manager.config, key):
                if isinstance(value, dict) and isinstance(getattr(config_manager.config, key), dict):
                    # Merge dictionaries
                    current = getattr(config_manager.config, key)
                    current.update(value)
                else:
                    setattr(config_manager.config, key, value)
    
    # Validate configuration
    validation = config_manager.validate_config()
    if not validation['valid']:
        raise ValueError(f"Invalid configuration: {validation['issues']}")
    
    if validation['warnings']:
        print(f"Configuration warnings: {validation['warnings']}")
    
    return config_manager.config


# Export functions
def save_production_config(deployment_type: str = "on_premise", path: str = "jarvis_phase11_production.yaml"):
    """Save production configuration to file"""
    config = create_deployment_config(deployment_type)
    config_manager = ConfigurationManager()
    config_manager.config = config
    config_manager.save_config(path)
    print(f"Production configuration saved to {path}")


if __name__ == "__main__":
    # Generate production configurations
    for deployment in ["aws", "gcp", "on_premise"]:
        config = create_deployment_config(deployment)
        save_production_config(deployment, f"jarvis_phase11_{deployment}.yaml")
    
    # Show configuration summary
    config_manager = ConfigurationManager()
    print("\nPhase 11 Production Configuration Summary:")
    print(f"Environment: {config_manager.config.environment}")
    print(f"Max Concurrent Requests: {config_manager.config.max_concurrent_requests}")
    print(f"Memory Limit: {config_manager.config.resource_limits['max_memory_gb']}GB")
    print(f"CPU Limit: {config_manager.config.resource_limits['max_cpu_percent']}%")
    
    # Validate
    validation = config_manager.validate_config()
    print(f"\nValidation: {'PASSED' if validation['valid'] else 'FAILED'}")
    if validation['warnings']:
        print(f"Warnings: {', '.join(validation['warnings'])}")
