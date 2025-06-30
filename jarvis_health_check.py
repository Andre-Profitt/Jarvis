#!/usr/bin/env python3
"""
JARVIS Health Check System
"""

import os
import sys
from pathlib import Path
import importlib
from typing import Dict, Any

class JARVISHealthCheck:
    def __init__(self):
        self.components = {}
        
    def check_environment(self) -> Dict[str, Any]:
        """Check environment setup"""
        env_status = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "env_file": os.path.exists(".env"),
            "virtual_env": hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        }
        
        # Check API keys
        api_keys = ["CLAUDE_API_KEY", "OPENAI_API_KEY"]
        for key in api_keys:
            env_status[key] = os.getenv(key) is not None
            
        return env_status
    
    def check_imports(self) -> Dict[str, bool]:
        """Check if core modules can be imported"""
        modules = [
            "core.minimal_jarvis",
            "core.updated_multi_ai_integration",
            "core.neural_resource_manager",
            "core.self_healing_system"
        ]
        
        import_status = {}
        for module in modules:
            try:
                importlib.import_module(module)
                import_status[module] = True
            except:
                import_status[module] = False
                
        return import_status
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        return {
            "environment": self.check_environment(),
            "imports": self.check_imports()
        }

if __name__ == "__main__":
    checker = JARVISHealthCheck()
    results = checker.run_checks()
    
    print("\n🏥 JARVIS Health Check Results:")
    print("=" * 40)
    
    # Environment
    print("\n📍 Environment:")
    for key, value in results["environment"].items():
        status = "✅" if value else "❌"
        print(f"  {status} {key}: {value}")
    
    # Imports
    print("\n📦 Module Imports:")
    for module, success in results["imports"].items():
        status = "✅" if success else "❌"
        print(f"  {status} {module}")
