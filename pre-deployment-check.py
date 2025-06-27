#!/usr/bin/env python3
"""
Pre-Deployment Verification Script
Ensures JARVIS has the best possible first step!
"""

import subprocess
import sys
import os
from pathlib import Path
import importlib.util
import json
import yaml
import asyncio
from typing import Dict, List, Tuple, Any

class JARVISPreDeploymentCheck:
    """
    Comprehensive checks before JARVIS deployment
    """
    
    def __init__(self):
        self.ecosystem_path = Path("/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM")
        self.check_results = {}
        self.critical_issues = []
        self.warnings = []
        self.ready_for_deployment = True
        
    async def run_all_checks(self):
        """
        Run comprehensive pre-deployment checks
        """
        print("🔍 JARVIS Pre-Deployment Verification")
        print("=" * 50)
        
        # 1. Check Python version
        await self._check_python_version()
        
        # 2. Check required system commands
        await self._check_system_commands()
        
        # 3. Check Python dependencies
        await self._check_python_dependencies()
        
        # 4. Check file structure
        await self._check_file_structure()
        
        # 5. Check configurations
        await self._check_configurations()
        
        # 6. Check ports availability
        await self._check_ports()
        
        # 7. Check disk space
        await self._check_disk_space()
        
        # 8. Import verification
        await self._verify_imports()
        
        # 9. Check MCP setup
        await self._check_mcp_setup()
        
        # Show results
        await self._show_results()
        
    async def _check_python_version(self):
        """Check Python version is 3.8+"""
        print("\n🐍 Checking Python version...")
        
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
            self.check_results["python_version"] = "pass"
        else:
            print(f"  ❌ Python {version.major}.{version.minor} (need 3.8+)")
            self.critical_issues.append("Python 3.8+ required")
            self.check_results["python_version"] = "fail"
            self.ready_for_deployment = False
    
    async def _check_system_commands(self):
        """Check required system commands"""
        print("\n🔧 Checking system commands...")
        
        commands = {
            "git": "--version",
            "redis-cli": "--version",
            "node": "--version",
            "npm": "--version"
        }
        
        for cmd, arg in commands.items():
            try:
                result = subprocess.run(
                    [cmd, arg],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"  ✅ {cmd} installed")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"  ⚠️  {cmd} not found")
                self.warnings.append(f"{cmd} not installed")
    
    async def _check_python_dependencies(self):
        """Check if requirements.txt dependencies are installed"""
        print("\n📦 Checking Python dependencies...")
        
        # Critical dependencies only
        critical_deps = [
            "numpy",
            "torch",
            "transformers",
            "asyncio",
            "aiohttp",
            "redis",
            "websockets"
        ]
        
        missing_deps = []
        
        for dep in critical_deps:
            try:
                importlib.import_module(dep)
                print(f"  ✅ {dep}")
            except ImportError:
                print(f"  ❌ {dep} missing")
                missing_deps.append(dep)
        
        if missing_deps:
            self.warnings.append(f"Missing dependencies: {', '.join(missing_deps)}")
            print(f"\n  💡 Install with: pip install {' '.join(missing_deps)}")
    
    async def _check_file_structure(self):
        """Check required files and directories exist"""
        print("\n📁 Checking file structure...")
        
        required_files = [
            "requirements.txt",
            "config.yaml",
            "missing_components.py",
            "complete-deployment-script.py"
        ]
        
        required_dirs = [
            "logs",
            "models",
            "storage",
            "mcp_servers"
        ]
        
        # Check files
        for file in required_files:
            file_path = self.ecosystem_path / file
            if file_path.exists():
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file} missing")
                self.critical_issues.append(f"Missing file: {file}")
        
        # Create directories if missing
        for dir_name in required_dirs:
            dir_path = self.ecosystem_path / dir_name
            if not dir_path.exists():
                dir_path.mkdir(exist_ok=True)
                print(f"  📂 Created {dir_name}/")
            else:
                print(f"  ✅ {dir_name}/")
    
    async def _check_configurations(self):
        """Check configuration files"""
        print("\n⚙️  Checking configurations...")
        
        config_path = self.ecosystem_path / "config.yaml"
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                
                # Check critical config sections
                required_sections = ["paths", "redis", "websocket", "security"]
                
                for section in required_sections:
                    if section in config:
                        print(f"  ✅ {section} configured")
                    else:
                        print(f"  ⚠️  {section} not configured")
                        self.warnings.append(f"Missing config section: {section}")
                        
            except Exception as e:
                print(f"  ❌ Error reading config: {e}")
                self.critical_issues.append("Invalid config.yaml")
        else:
            print("  ❌ config.yaml not found")
            self.critical_issues.append("Missing config.yaml")
    
    async def _check_ports(self):
        """Check if required ports are available"""
        print("\n🌐 Checking port availability...")
        
        ports_to_check = [
            (6379, "Redis"),
            (8765, "WebSocket"),
            (8080, "API Server")
        ]
        
        for port, service in ports_to_check:
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                if result == 0:
                    print(f"  ⚠️  Port {port} ({service}) in use")
                    self.warnings.append(f"Port {port} already in use")
                else:
                    print(f"  ✅ Port {port} ({service}) available")
            except:
                print(f"  ⚠️  Could not check port {port}")
    
    async def _check_disk_space(self):
        """Check available disk space"""
        print("\n💾 Checking disk space...")
        
        import shutil
        
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (2**30)
        
        if free_gb < 5:
            print(f"  ⚠️  Only {free_gb}GB free (recommend 10GB+)")
            self.warnings.append(f"Low disk space: {free_gb}GB")
        else:
            print(f"  ✅ {free_gb}GB free")
    
    async def _verify_imports(self):
        """Verify critical imports work"""
        print("\n🔌 Verifying imports...")
        
        # Import missing components to ensure they work
        try:
            sys.path.insert(0, str(self.ecosystem_path))
            import missing_components
            print("  ✅ Missing components importable")
        except Exception as e:
            print(f"  ❌ Import error: {e}")
            self.critical_issues.append("Cannot import missing_components")
    
    async def _check_mcp_setup(self):
        """Check MCP configuration"""
        print("\n🔌 Checking MCP setup...")
        
        claude_config = Path.home() / ".config/claude/claude_desktop_config.json"
        
        if claude_config.exists():
            print("  ✅ Claude Desktop config exists")
            print("  ⚠️  Remember to restart Claude Desktop after deployment")
        else:
            print("  ⚠️  Claude Desktop config will be created during deployment")
    
    async def _show_results(self):
        """Show check results and recommendations"""
        print("\n" + "=" * 50)
        print("📊 Pre-Deployment Check Results")
        print("=" * 50)
        
        if self.critical_issues:
            print("\n🔴 CRITICAL ISSUES (must fix):")
            for issue in self.critical_issues:
                print(f"  • {issue}")
        
        if self.warnings:
            print("\n🟡 WARNINGS (recommended to fix):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if not self.critical_issues:
            print("\n✅ JARVIS is ready for deployment!")
            print("\n🚀 Next steps:")
            print("  1. Install missing dependencies:")
            print("     pip install -r requirements.txt")
            print("  2. Start Redis:")
            print("     redis-server --daemonize yes")
            print("  3. Run deployment:")
            print("     python3 complete-deployment-script.py")
            print("\n🌟 Your AI assistant is ready to come to life!")
        else:
            print("\n❌ Deployment blocked due to critical issues")
            print("   Please fix the issues above and run this check again")

# Quick fixes function
async def apply_quick_fixes():
    """Apply quick fixes for common issues"""
    
    print("\n🔧 Applying quick fixes...")
    
    ecosystem_path = Path("/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM")
    
    # Create missing directories
    dirs_to_create = ["logs", "models", "storage", "mcp_servers", "deployment", "training_data"]
    for dir_name in dirs_to_create:
        dir_path = ecosystem_path / dir_name
        dir_path.mkdir(exist_ok=True)
    
    # Create .env file if missing
    env_file = ecosystem_path / ".env"
    if not env_file.exists():
        env_content = """
# JARVIS Environment Variables
OPENAI_API_KEY=your_openai_key_here
# ANTHROPIC_API_KEY not needed - using Claude Desktop MCP!
GITHUB_TOKEN=your_github_token_here
REDIS_PASSWORD=

# Claude Desktop settings
USE_CLAUDE_DESKTOP=true
CLAUDE_SUBSCRIPTION=max_x200
"""
        env_file.write_text(env_content)
        print("  📄 Created .env file (add your API keys)")
    
    # Fix import paths
    init_file = ecosystem_path / "__init__.py"
    init_file.touch()
    
    print("  ✅ Quick fixes applied")

# Main check function
async def check_jarvis_deployment():
    """
    Run pre-deployment checks for JARVIS
    """
    
    print("🌟 JARVIS Pre-Deployment Verification 🌟\n")
    
    checker = JARVISPreDeploymentCheck()
    await checker.run_all_checks()
    
    # Ask if user wants to apply quick fixes
    if checker.warnings or checker.critical_issues:
        response = input("\n🔧 Apply quick fixes? (y/n): ")
        if response.lower() == 'y':
            await apply_quick_fixes()
            print("\n🔄 Re-running checks...")
            checker = JARVISPreDeploymentCheck()
            await checker.run_all_checks()

if __name__ == "__main__":
    asyncio.run(check_jarvis_deployment())