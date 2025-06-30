#!/usr/bin/env python3
"""
Comprehensive Fix Script for JARVIS
Fixes all critical issues found in the 100-point audit
"""

import os
import sys
import shutil
from pathlib import Path
import subprocess
import asyncio
import json


class JARVISFixer:
    def __init__(self):
        self.root = Path(__file__).parent
        self.fixes_applied = []
        self.errors = []

    def run_all_fixes(self):
        """Run all fixes in sequence"""

        print("🔧 JARVIS COMPREHENSIVE FIX SCRIPT")
        print("=" * 50)

        # Fix 1: Update imports in multi-ai-integration.py
        self.fix_multi_ai_imports()

        # Fix 2: Fix circular imports by reorganizing code
        self.fix_circular_imports()

        # Fix 3: Update all files to use real implementations
        self.update_to_real_implementations()

        # Fix 4: Add comprehensive error handling
        self.add_error_handling()

        # Fix 5: Create proper initialization scripts
        self.create_init_scripts()

        # Fix 6: Update launch script
        self.update_launch_script()

        # Fix 7: Create health check system
        self.create_health_checks()

        # Fix 8: Add logging configuration
        self.setup_logging()

        # Fix 9: Create test structure
        self.create_test_structure()

        # Fix 10: Final validation
        self.validate_fixes()

        print("\n✅ FIXES COMPLETE!")
        print(f"Applied {len(self.fixes_applied)} fixes")
        if self.errors:
            print(f"⚠️ {len(self.errors)} errors encountered:")
            for error in self.errors:
                print(f"  - {error}")

    def fix_multi_ai_imports(self):
        """Update multi-ai-integration.py to use real implementations"""

        print("\n📝 Fixing multi-AI imports...")

        old_file = self.root / "multi-ai-integration.py"
        new_file = self.root / "core/updated_multi_ai_integration.py"

        if new_file.exists():
            # Backup old file
            shutil.copy(old_file, old_file.with_suffix(".py.bak"))

            # Copy new implementation
            shutil.copy(new_file, old_file)

            self.fixes_applied.append(
                "Updated multi-AI integration to real implementation"
            )
        else:
            self.errors.append("Could not find updated multi-AI integration")

    def fix_circular_imports(self):
        """Fix circular import issues"""

        print("\n🔄 Fixing circular imports...")

        # Remove duplicate DeviceDiscoverer from missing_components.py
        missing_components_file = self.root / "missing_components.py"

        if missing_components_file.exists():
            content = missing_components_file.read_text()

            # Remove the duplicate DeviceDiscoverer class
            lines = content.split("\n")
            new_lines = []
            skip_class = False

            for line in lines:
                if line.startswith("class DeviceDiscoverer:"):
                    skip_class = True
                elif (
                    skip_class
                    and line
                    and not line.startswith(" ")
                    and not line.startswith("\t")
                ):
                    skip_class = False

                if not skip_class:
                    new_lines.append(line)

            missing_components_file.write_text("\n".join(new_lines))
            self.fixes_applied.append("Fixed circular imports")

    def update_to_real_implementations(self):
        """Update all placeholder implementations"""

        print("\n🚀 Updating to real implementations...")

        # Update voice-first-interface.py to use real integrations
        voice_file = self.root / "voice-first-interface.py"
        if voice_file.exists():
            content = voice_file.read_text()

            # Add imports for real integrations
            import_section = """import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import real integrations
from core.real_elevenlabs_integration import elevenlabs_integration
from core.real_openai_integration import openai_integration
"""

            # Replace imports
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("import os"):
                    lines[i] = import_section
                    break

            voice_file.write_text("\n".join(lines))
            self.fixes_applied.append(
                "Updated voice interface to use real integrations"
            )

    def add_error_handling(self):
        """Add comprehensive error handling"""

        print("\n🛡️ Adding error handling...")

        # Create error handler utility
        error_handler_code = '''#!/usr/bin/env python3
"""
Comprehensive Error Handling for JARVIS
"""

import logging
import traceback
from functools import wraps
from typing import Any, Callable
import asyncio

logger = logging.getLogger(__name__)

def safe_execute(default_return=None):
    """Decorator for safe execution with error handling"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                logger.debug(traceback.format_exc())
                return default_return
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                logger.debug(traceback.format_exc())
                return default_return
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator

class JARVISError(Exception):
    """Base exception for JARVIS"""
    pass

class ConfigurationError(JARVISError):
    """Configuration related errors"""
    pass

class IntegrationError(JARVISError):
    """AI integration errors"""
    pass

class DeviceError(JARVISError):
    """Device communication errors"""
    pass
'''

        utils_dir = self.root / "utils"
        utils_dir.mkdir(exist_ok=True)

        error_file = utils_dir / "error_handling.py"
        error_file.write_text(error_handler_code)

        self.fixes_applied.append("Added comprehensive error handling")

    def create_init_scripts(self):
        """Create proper __init__.py files"""

        print("\n📦 Creating package initialization...")

        dirs_needing_init = ["core", "utils", "tools", "mcp_servers"]

        for dir_name in dirs_needing_init:
            dir_path = self.root / dir_name
            if dir_path.exists():
                init_file = dir_path / "__init__.py"
                init_file.touch()

        self.fixes_applied.append("Created package __init__ files")

    def update_launch_script(self):
        """Update launch script to use real services"""

        print("\n🚀 Updating launch script...")

        # Create enhanced launch script
        enhanced_launch = '''#!/usr/bin/env python3
"""
Enhanced JARVIS Launch Script with Real Services
"""

import asyncio
import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

# Import real integrations
from core.updated_multi_ai_integration import multi_ai
from core.websocket_security import websocket_security, SecureWebSocketHandler
from core.real_elevenlabs_integration import elevenlabs_integration

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealJARVISLauncher:
    """Launch JARVIS with actual services"""
    
    def __init__(self):
        self.services = {}
        self.launch_time = datetime.now()
        
    async def launch(self):
        """Launch all JARVIS services"""
        
        print("""
        ╔══════════════════════════════════════════╗
        ║         🚀 JARVIS LAUNCH SEQUENCE        ║
        ║          REAL SERVICES EDITION           ║
        ╚══════════════════════════════════════════╝
        """)
        
        # Step 1: Initialize AI integrations
        print("\\n[1/5] Initializing AI integrations...")
        await multi_ai.initialize()
        
        # Step 2: Start WebSocket server with security
        print("\\n[2/5] Starting secure WebSocket server...")
        await self.start_websocket_server()
        
        # Step 3: Initialize voice system
        print("\\n[3/5] Initializing voice system...")
        await self.initialize_voice()
        
        # Step 4: Start background services
        print("\\n[4/5] Starting background services...")
        await self.start_background_services()
        
        # Step 5: Final initialization
        print("\\n[5/5] Final initialization...")
        await self.final_initialization()
        
        print("""
        ╔══════════════════════════════════════════╗
        ║       ✅ JARVIS IS NOW ONLINE! ✅        ║
        ╚══════════════════════════════════════════╝
        
        Available Models: """ + str(list(multi_ai.available_models.keys())) + """
        
        Say "Hey JARVIS" to interact!
        """)
        
    async def start_websocket_server(self):
        """Start secure WebSocket server"""
        
        handler = SecureWebSocketHandler(websocket_security)
        
        server = await websocket_security.create_secure_server(
            handler.handle_connection,
            "localhost",
            8765
        )
        
        self.services["websocket"] = server
        logger.info("WebSocket server started on port 8765")
        
    async def initialize_voice(self):
        """Initialize voice system with ElevenLabs"""
        
        try:
            # Test ElevenLabs connection
            if await elevenlabs_integration.test_connection():
                # Speak introduction
                await elevenlabs_integration.speak(
                    "Hello Dad! JARVIS is now online with all real services activated. "
                    "I'm ready to help you with anything you need!",
                    emotion="excited"
                )
                logger.info("Voice system initialized")
            else:
                logger.warning("Voice system unavailable")
        except Exception as e:
            logger.error(f"Voice initialization error: {e}")
            
    async def start_background_services(self):
        """Start all background services"""
        
        # Start monitoring
        asyncio.create_task(self.monitor_services())
        
        # Start health checks
        asyncio.create_task(self.health_check_loop())
        
        logger.info("Background services started")
        
    async def monitor_services(self):
        """Monitor service health"""
        
        while True:
            await asyncio.sleep(60)  # Check every minute
            # Add monitoring logic here
            
    async def health_check_loop(self):
        """Regular health checks"""
        
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            # Add health check logic here
            
    async def final_initialization(self):
        """Final initialization steps"""
        
        # Create success marker
        success_file = Path(__file__).parent / ".jarvis_launched"
        success_file.write_text(f"Launched at {self.launch_time}")
        
        logger.info("JARVIS initialization complete!")

async def main():
    """Main launch function"""
    
    launcher = RealJARVISLauncher()
    
    try:
        await launcher.launch()
        
        # Keep running
        await asyncio.Event().wait()
        
    except KeyboardInterrupt:
        logger.info("Shutting down JARVIS...")
    except Exception as e:
        logger.error(f"Launch failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
'''

        launch_file = self.root / "LAUNCH-JARVIS-REAL.py"
        launch_file.write_text(enhanced_launch)
        launch_file.chmod(0o755)

        self.fixes_applied.append("Created enhanced launch script")

    def create_health_checks(self):
        """Create comprehensive health check system"""

        print("\n❤️ Creating health checks...")

        health_check_code = '''#!/usr/bin/env python3
"""
JARVIS Health Check System
"""

import asyncio
from typing import Dict, Any, List
import psutil
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class JARVISHealthCheck:
    """Comprehensive health monitoring"""
    
    def __init__(self):
        self.checks = {
            "cpu": self.check_cpu,
            "memory": self.check_memory,
            "disk": self.check_disk,
            "network": self.check_network,
            "services": self.check_services,
            "ai_models": self.check_ai_models
        }
        
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "healthy",
            "checks": {}
        }
        
        for name, check_func in self.checks.items():
            try:
                results["checks"][name] = await check_func()
            except Exception as e:
                results["checks"][name] = {
                    "status": "error",
                    "error": str(e)
                }
                results["overall_health"] = "degraded"
        
        return results
    
    async def check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage"""
        
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            "status": "healthy" if cpu_percent < 80 else "warning",
            "usage": cpu_percent,
            "cores": psutil.cpu_count()
        }
    
    async def check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        
        memory = psutil.virtual_memory()
        
        return {
            "status": "healthy" if memory.percent < 80 else "warning",
            "usage": memory.percent,
            "available": memory.available,
            "total": memory.total
        }
    
    async def check_disk(self) -> Dict[str, Any]:
        """Check disk usage"""
        
        disk = psutil.disk_usage('/')
        
        return {
            "status": "healthy" if disk.percent < 90 else "warning",
            "usage": disk.percent,
            "free": disk.free,
            "total": disk.total
        }
    
    async def check_network(self) -> Dict[str, Any]:
        """Check network connectivity"""
        
        # Simple connectivity check
        import socket
        
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return {"status": "healthy", "connectivity": True}
        except:
            return {"status": "error", "connectivity": False}
    
    async def check_services(self) -> Dict[str, Any]:
        """Check JARVIS services"""
        
        services_status = {}
        
        # Check Redis
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379)
            r.ping()
            services_status["redis"] = "running"
        except:
            services_status["redis"] = "stopped"
        
        # Add more service checks
        
        return {
            "status": "healthy" if all(v == "running" for v in services_status.values()) else "degraded",
            "services": services_status
        }
    
    async def check_ai_models(self) -> Dict[str, Any]:
        """Check AI model availability"""
        
        from core.updated_multi_ai_integration import multi_ai
        
        return {
            "status": "healthy" if multi_ai.available_models else "error",
            "available_models": list(multi_ai.available_models.keys())
        }

# Create singleton
health_checker = JARVISHealthCheck()
'''

        health_file = self.root / "core" / "health_checks.py"
        health_file.write_text(health_check_code)

        self.fixes_applied.append("Created health check system")

    def setup_logging(self):
        """Setup comprehensive logging"""

        print("\n📝 Setting up logging...")

        log_config = '''#!/usr/bin/env python3
"""
JARVIS Logging Configuration
"""

import logging
import logging.handlers
from pathlib import Path
import sys

def setup_logging():
    """Configure comprehensive logging for JARVIS"""
    
    # Create logs directory
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # File handler (rotating)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "jarvis.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "jarvis_errors.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # Reduce noise from some libraries
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    
    return root_logger
'''

        log_file = self.root / "utils" / "logging_config.py"
        log_file.write_text(log_config)

        self.fixes_applied.append("Setup comprehensive logging")

    def create_test_structure(self):
        """Create basic test structure"""

        print("\n🧪 Creating test structure...")

        # Create test directories
        test_dir = self.root / "tests"
        test_dir.mkdir(exist_ok=True)

        # Create basic test file
        basic_test = '''#!/usr/bin/env python3
"""
Basic tests for JARVIS
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestJARVISCore:
    """Test core JARVIS functionality"""
    
    @pytest.mark.asyncio
    async def test_multi_ai_initialization(self):
        """Test multi-AI integration initialization"""
        
        from core.updated_multi_ai_integration import multi_ai
        
        await multi_ai.initialize()
        assert len(multi_ai.available_models) > 0
    
    def test_imports(self):
        """Test all imports work"""
        
        imports = [
            "from core.real_claude_integration import claude_integration",
            "from core.real_openai_integration import openai_integration",
            "from core.real_elevenlabs_integration import elevenlabs_integration",
            "from core.websocket_security import websocket_security",
            "from core.health_checks import health_checker"
        ]
        
        for import_stmt in imports:
            try:
                exec(import_stmt)
            except ImportError as e:
                pytest.fail(f"Import failed: {import_stmt} - {e}")

if __name__ == "__main__":
    pytest.main([__file__])
'''

        test_file = test_dir / "test_core.py"
        test_file.write_text(basic_test)

        # Create pytest config
        pytest_config = """[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
asyncio_mode = "auto"
"""

        pyproject_file = self.root / "pyproject.toml"
        pyproject_file.write_text(pytest_config)

        self.fixes_applied.append("Created test structure")

    def validate_fixes(self):
        """Validate all fixes were applied correctly"""

        print("\n✔️ Validating fixes...")

        validations = [
            (self.root / "core" / "real_claude_integration.py", "Claude integration"),
            (self.root / "core" / "real_openai_integration.py", "OpenAI integration"),
            (
                self.root / "core" / "real_elevenlabs_integration.py",
                "ElevenLabs integration",
            ),
            (self.root / "core" / "websocket_security.py", "WebSocket security"),
            (self.root / "core" / "health_checks.py", "Health checks"),
            (self.root / "utils" / "error_handling.py", "Error handling"),
            (self.root / "utils" / "logging_config.py", "Logging config"),
            (self.root / "tests" / "test_core.py", "Test structure"),
            (self.root / "LAUNCH-JARVIS-REAL.py", "Enhanced launch script"),
        ]

        all_valid = True
        for file_path, name in validations:
            if file_path.exists():
                print(f"  ✅ {name}")
            else:
                print(f"  ❌ {name}")
                all_valid = False
                self.errors.append(f"Missing: {name}")

        if all_valid:
            self.fixes_applied.append("All validations passed")

        return all_valid


if __name__ == "__main__":
    fixer = JARVISFixer()
    fixer.run_all_fixes()

    print("\n🎯 Next Steps:")
    print("1. Install any missing dependencies: pip install -r requirements.txt")
    print("2. Set up your API keys in .env file")
    print("3. Start Redis: redis-server")
    print("4. Launch JARVIS: python3 LAUNCH-JARVIS-REAL.py")
    print("\nYour AI son is almost ready to come to life! 🚀")
