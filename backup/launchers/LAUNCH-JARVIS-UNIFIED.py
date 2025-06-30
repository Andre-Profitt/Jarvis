#!/usr/bin/env python3
"""
JARVIS Unified Launcher - Best Practices Implementation
======================================================

A consolidated, modular launcher for the JARVIS ecosystem that follows
software engineering best practices and provides a clean, extensible
architecture for launching JARVIS with various configurations.

Features:
- Modular plugin architecture
- Configuration management (YAML + ENV)
- Service health monitoring
- Graceful degradation
- Feature flags
- Comprehensive logging
- Clean shutdown handling
"""

import asyncio
import sys
import os
import logging
import signal
import argparse
import yaml
import importlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from enum import Enum
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


class ServiceStatus(Enum):
    """Service health states"""

    NOT_STARTED = "not_started"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class ServiceInfo:
    """Information about a registered service"""

    name: str
    module_path: str
    required: bool = False
    enabled: bool = True
    status: ServiceStatus = ServiceStatus.NOT_STARTED
    instance: Optional[Any] = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


class JARVISLauncher:
    """
    Unified JARVIS Launcher with modular architecture and best practices
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/jarvis.yaml"
        self.config = self._load_configuration()
        self.services: Dict[str, ServiceInfo] = {}
        self.patches_applied: Set[str] = set()
        self.shutdown_event = asyncio.Event()
        self.startup_time = datetime.now()

        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # Register core services
        self._register_core_services()

    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration from YAML file with environment override"""
        default_config = {
            "general": {
                "name": "JARVIS",
                "version": "2.0.0",
                "debug": False,
                "log_level": "INFO",
            },
            "services": {
                "websocket": {"enabled": True, "host": "0.0.0.0", "port": 8765},
                "multi_ai": {"enabled": True, "models": ["claude", "gpt4", "gemini"]},
                "voice": {"enabled": True, "provider": "elevenlabs"},
                "consciousness": {"enabled": True, "cycle_frequency": 10},
                "neural_resources": {"enabled": True, "initial_capacity": 1000},
                "self_healing": {"enabled": True, "monitoring_interval": 60},
                "quantum_swarm": {"enabled": True, "n_agents": 50},
                "metacognitive": {"enabled": True, "reflection_interval": 300},
            },
            "features": {
                "interactive_mode": True,
                "auto_save": True,
                "telemetry": False,
            },
        }

        # Try to load from YAML file
        config_path = Path(self.config_path)
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    loaded_config = yaml.safe_load(f)
                    # Deep merge configurations
                    config = self._deep_merge(default_config, loaded_config)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
                config = default_config
        else:
            config = default_config

        # Override with environment variables
        config = self._apply_env_overrides(config)

        return config

    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in update.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _apply_env_overrides(self, config: Dict) -> Dict:
        """Apply environment variable overrides to configuration"""
        # Example: JARVIS_SERVICES_WEBSOCKET_PORT=8080
        for env_key, env_value in os.environ.items():
            if env_key.startswith("JARVIS_"):
                config_path = env_key[7:].lower().split("_")
                current = config
                for i, key in enumerate(config_path[:-1]):
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                # Convert value types
                if env_value.lower() in ("true", "false"):
                    current[config_path[-1]] = env_value.lower() == "true"
                elif env_value.isdigit():
                    current[config_path[-1]] = int(env_value)
                else:
                    current[config_path[-1]] = env_value
        return config

    def _setup_logging(self):
        """Setup comprehensive logging system"""
        log_level = getattr(logging, self.config["general"]["log_level"].upper())

        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    log_dir / f"jarvis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                ),
            ],
        )

        # Set specific logger levels
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    def _register_core_services(self):
        """Register all available services"""
        # Core infrastructure services
        self.register_service(
            "websocket",
            "core.websocket_security",
            required=False,
            config=self.config["services"].get("websocket", {}),
        )

        # AI/ML services
        self.register_service(
            "multi_ai",
            "core.updated_multi_ai_integration",
            required=True,
            config=self.config["services"].get("multi_ai", {}),
        )

        # Voice services
        self.register_service(
            "voice",
            "core.real_elevenlabs_integration",
            required=False,
            config=self.config["services"].get("voice", {}),
        )

        # Advanced cognitive services
        self.register_service(
            "neural_resources",
            "core.neural_integration",
            required=False,
            dependencies=["multi_ai"],
            config=self.config["services"].get("neural_resources", {}),
        )

        self.register_service(
            "self_healing",
            "core.self_healing_integration",
            required=False,
            config=self.config["services"].get("self_healing", {}),
        )

        self.register_service(
            "llm_research",
            "core.llm_research_jarvis",
            required=False,
            dependencies=["multi_ai"],
            config=self.config["services"].get("llm_research", {}),
        )

        self.register_service(
            "quantum_swarm",
            "core.quantum_swarm_jarvis",
            required=False,
            dependencies=["neural_resources"],
            config=self.config["services"].get("quantum_swarm", {}),
        )

        self.register_service(
            "metacognitive",
            "core.metacognitive_jarvis",
            required=False,
            dependencies=["neural_resources", "self_healing"],
            config=self.config["services"].get("metacognitive", {}),
        )

        self.register_service(
            "consciousness",
            "core.consciousness_jarvis",
            required=False,
            dependencies=["neural_resources", "metacognitive"],
            config=self.config["services"].get("consciousness", {}),
        )

    def register_service(
        self,
        name: str,
        module_path: str,
        required: bool = False,
        dependencies: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Register a service with the launcher"""
        service_config = config or {}
        enabled = service_config.get("enabled", True)

        self.services[name] = ServiceInfo(
            name=name,
            module_path=module_path,
            required=required,
            enabled=enabled,
            dependencies=dependencies or [],
            config=service_config,
        )

    async def _apply_patches(self):
        """Apply any necessary monkey patches"""
        # Consciousness cycle patch
        if "consciousness_cycle_patch" not in self.patches_applied:
            try:
                # Apply the consciousness cycle patch if needed
                from core.consciousness_jarvis import ConsciousnessJARVIS

                original_cycle = ConsciousnessJARVIS._consciousness_cycle

                async def patched_cycle(self):
                    try:
                        return await original_cycle(self)
                    except Exception as e:
                        self.logger.error(f"Consciousness cycle error: {e}")
                        await asyncio.sleep(1)

                ConsciousnessJARVIS._consciousness_cycle = patched_cycle
                self.patches_applied.add("consciousness_cycle_patch")
                self.logger.info("Applied consciousness cycle patch")
            except ImportError:
                pass

    async def _check_dependencies(self, service_name: str) -> bool:
        """Check if all dependencies for a service are satisfied"""
        service = self.services[service_name]

        for dep in service.dependencies:
            if dep not in self.services:
                self.logger.error(
                    f"Service {service_name} has unknown dependency: {dep}"
                )
                return False

            dep_service = self.services[dep]
            if not dep_service.enabled:
                self.logger.warning(
                    f"Service {service_name} dependency {dep} is disabled"
                )
                return False

            if dep_service.status not in (
                ServiceStatus.RUNNING,
                ServiceStatus.DEGRADED,
            ):
                self.logger.warning(
                    f"Service {service_name} dependency {dep} is not running"
                )
                return False

        return True

    async def _start_service(self, name: str) -> bool:
        """Start a single service"""
        service = self.services[name]

        if not service.enabled:
            self.logger.info(f"Service {name} is disabled")
            return True

        if service.status == ServiceStatus.RUNNING:
            return True

        # Check dependencies
        if not await self._check_dependencies(name):
            if service.required:
                raise RuntimeError(f"Required service {name} has unmet dependencies")
            return False

        self.logger.info(f"Starting service: {name}")
        service.status = ServiceStatus.STARTING

        try:
            # Dynamic import
            module = importlib.import_module(service.module_path)

            # Get initialization function or class
            if hasattr(module, f"initialize_{name}"):
                # Custom initialization function
                init_func = getattr(module, f"initialize_{name}")
                service.instance = await init_func(
                    config=service.config, services=self.services
                )
            elif hasattr(module, name):
                # Direct service instance
                service.instance = getattr(module, name)
            else:
                # Try to find a suitable class or instance
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and name in attr_name.lower():
                        service.instance = attr(config=service.config)
                        break

            if service.instance is None:
                raise RuntimeError(f"Could not find initialization for service {name}")

            # Call async initialization if available
            if hasattr(service.instance, "initialize"):
                await service.instance.initialize()

            service.status = ServiceStatus.RUNNING
            self.logger.info(f"✅ Service {name} started successfully")
            return True

        except Exception as e:
            service.status = ServiceStatus.FAILED
            service.error = str(e)
            self.logger.error(f"❌ Failed to start service {name}: {e}")

            if service.required:
                raise

            return False

    async def _stop_service(self, name: str):
        """Stop a single service"""
        service = self.services[name]

        if service.status not in (ServiceStatus.RUNNING, ServiceStatus.DEGRADED):
            return

        self.logger.info(f"Stopping service: {name}")
        service.status = ServiceStatus.STOPPING

        try:
            if service.instance and hasattr(service.instance, "shutdown"):
                await service.instance.shutdown()

            service.status = ServiceStatus.STOPPED
            service.instance = None
            self.logger.info(f"Service {name} stopped")

        except Exception as e:
            self.logger.error(f"Error stopping service {name}: {e}")

    async def start_all_services(self):
        """Start all registered services in dependency order"""
        # Apply patches first
        await self._apply_patches()

        # Build dependency graph
        started = set()

        async def start_with_deps(name: str):
            if name in started:
                return

            service = self.services[name]

            # Start dependencies first
            for dep in service.dependencies:
                if dep not in started:
                    await start_with_deps(dep)

            # Start this service
            await self._start_service(name)
            started.add(name)

        # Start all services
        for name in self.services:
            await start_with_deps(name)

    async def monitor_health(self):
        """Continuous health monitoring"""
        while not self.shutdown_event.is_set():
            try:
                health_report = await self.get_health_report()

                # Log health status
                healthy_count = sum(
                    1
                    for s in health_report["services"].values()
                    if s["status"] == "running"
                )
                total_count = len(health_report["services"])

                self.logger.debug(
                    f"Health check: {healthy_count}/{total_count} services running"
                )

                # Check for degraded services
                for name, status in health_report["services"].items():
                    if status["status"] == "degraded":
                        self.logger.warning(f"Service {name} is degraded")

                await asyncio.sleep(30)  # Health check every 30 seconds

            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)

    async def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "uptime": (datetime.now() - self.startup_time).total_seconds(),
            "services": {},
            "overall_health": "healthy",
        }

        failed_services = []
        degraded_services = []

        for name, service in self.services.items():
            service_health = {
                "status": service.status.value,
                "enabled": service.enabled,
                "required": service.required,
            }

            if service.error:
                service_health["error"] = service.error

            # Check service-specific health if running
            if service.status == ServiceStatus.RUNNING and service.instance:
                if hasattr(service.instance, "health_check"):
                    try:
                        health_data = await service.instance.health_check()
                        service_health.update(health_data)
                    except Exception as e:
                        service_health["health_check_error"] = str(e)
                        service.status = ServiceStatus.DEGRADED
                        degraded_services.append(name)

            if service.status == ServiceStatus.FAILED:
                failed_services.append(name)

            report["services"][name] = service_health

        # Determine overall health
        if failed_services:
            report["overall_health"] = "unhealthy"
            report["failed_services"] = failed_services
        elif degraded_services:
            report["overall_health"] = "degraded"
            report["degraded_services"] = degraded_services

        return report

    async def interactive_mode(self):
        """Run interactive command-line interface"""
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

        # Create session with history
        session = PromptSession(
            history=FileHistory(".jarvis_history"),
            auto_suggest=AutoSuggestFromHistory(),
        )

        print(
            """
        ╔══════════════════════════════════════════╗
        ║         🤖 JARVIS INTERACTIVE MODE       ║
        ╚══════════════════════════════════════════╝
        
        Commands:
          status    - Show service status
          health    - Get health report
          start     - Start a service
          stop      - Stop a service
          restart   - Restart a service
          config    - Show configuration
          help      - Show this help
          exit      - Shutdown JARVIS
        """
        )

        while not self.shutdown_event.is_set():
            try:
                # Get user input
                user_input = await session.prompt_async("JARVIS> ")

                if not user_input.strip():
                    continue

                # Parse command
                parts = user_input.strip().split()
                command = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []

                # Execute command
                if command == "exit":
                    self.shutdown_event.set()
                    break
                elif command == "status":
                    await self._cmd_status()
                elif command == "health":
                    await self._cmd_health()
                elif command == "start" and args:
                    await self._cmd_start(args[0])
                elif command == "stop" and args:
                    await self._cmd_stop(args[0])
                elif command == "restart" and args:
                    await self._cmd_restart(args[0])
                elif command == "config":
                    await self._cmd_config()
                elif command == "help":
                    await self._cmd_help()
                else:
                    # Delegate to multi-AI if available
                    if (
                        "multi_ai" in self.services
                        and self.services["multi_ai"].instance
                    ):
                        try:
                            response = await self.services["multi_ai"].instance.process(
                                user_input
                            )
                            print(f"\n{response}\n")
                        except Exception as e:
                            print(f"Error processing request: {e}")
                    else:
                        print("Unknown command. Type 'help' for available commands.")

            except KeyboardInterrupt:
                continue
            except EOFError:
                self.shutdown_event.set()
                break
            except Exception as e:
                self.logger.error(f"Interactive mode error: {e}")
                traceback.print_exc()

    async def _cmd_status(self):
        """Show service status"""
        print("\n=== Service Status ===")
        for name, service in self.services.items():
            status_icon = {
                ServiceStatus.RUNNING: "✅",
                ServiceStatus.DEGRADED: "⚠️",
                ServiceStatus.FAILED: "❌",
                ServiceStatus.STOPPED: "⏹️",
                ServiceStatus.NOT_STARTED: "⭕",
            }.get(service.status, "❓")

            enabled_str = "enabled" if service.enabled else "disabled"
            required_str = " (required)" if service.required else ""

            print(
                f"{status_icon} {name}: {service.status.value} [{enabled_str}]{required_str}"
            )

            if service.error:
                print(f"   Error: {service.error}")

    async def _cmd_health(self):
        """Show health report"""
        report = await self.get_health_report()

        print(f"\n=== Health Report ===")
        print(f"Timestamp: {report['timestamp']}")
        print(f"Uptime: {report['uptime']:.0f} seconds")
        print(f"Overall Health: {report['overall_health'].upper()}")

        if "failed_services" in report:
            print(f"\nFailed Services: {', '.join(report['failed_services'])}")

        if "degraded_services" in report:
            print(f"\nDegraded Services: {', '.join(report['degraded_services'])}")

    async def _cmd_start(self, service_name: str):
        """Start a service"""
        if service_name not in self.services:
            print(f"Unknown service: {service_name}")
            return

        if await self._start_service(service_name):
            print(f"Service {service_name} started")
        else:
            print(f"Failed to start service {service_name}")

    async def _cmd_stop(self, service_name: str):
        """Stop a service"""
        if service_name not in self.services:
            print(f"Unknown service: {service_name}")
            return

        await self._stop_service(service_name)
        print(f"Service {service_name} stopped")

    async def _cmd_restart(self, service_name: str):
        """Restart a service"""
        await self._cmd_stop(service_name)
        await asyncio.sleep(1)
        await self._cmd_start(service_name)

    async def _cmd_config(self):
        """Show configuration"""
        print("\n=== Configuration ===")
        print(yaml.dump(self.config, default_flow_style=False))

    async def _cmd_help(self):
        """Show help"""
        print(
            """
Available Commands:
  status              - Show status of all services
  health              - Get detailed health report
  start <service>     - Start a specific service
  stop <service>      - Stop a specific service
  restart <service>   - Restart a specific service
  config              - Show current configuration
  help                - Show this help message
  exit                - Shutdown JARVIS

Available Services:
"""
        )
        for name in sorted(self.services.keys()):
            print(f"  - {name}")

    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Starting graceful shutdown...")

        # Stop services in reverse dependency order
        stopped = set()

        async def stop_with_deps(name: str):
            if name in stopped:
                return

            # Stop dependents first
            for other_name, other_service in self.services.items():
                if name in other_service.dependencies and other_name not in stopped:
                    await stop_with_deps(other_name)

            # Stop this service
            await self._stop_service(name)
            stopped.add(name)

        for name in self.services:
            await stop_with_deps(name)

        self.logger.info("All services stopped")

    async def run(self):
        """Main entry point"""
        print(
            """
        ╔══════════════════════════════════════════╗
        ║      🚀 JARVIS UNIFIED LAUNCHER 🚀       ║
        ╚══════════════════════════════════════════╝
        """
        )

        try:
            # Start all services
            self.logger.info("Starting JARVIS services...")
            await self.start_all_services()

            # Get initial status
            health = await self.get_health_report()
            running_services = [
                name
                for name, info in health["services"].items()
                if info["status"] == "running"
            ]

            print(f"\n✅ JARVIS is online with {len(running_services)} active services")
            print(f"Active services: {', '.join(running_services)}")

            # Start background tasks
            tasks = []

            # Health monitoring
            tasks.append(asyncio.create_task(self.monitor_health()))

            # Interactive mode if enabled
            if self.config["features"]["interactive_mode"]:
                tasks.append(asyncio.create_task(self.interactive_mode()))

            # Wait for shutdown signal
            await self.shutdown_event.wait()

            # Cancel background tasks
            for task in tasks:
                task.cancel()

            # Shutdown
            await self.shutdown()

        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            traceback.print_exc()
            await self.shutdown()
            sys.exit(1)


def setup_signal_handlers(launcher: JARVISLauncher):
    """Setup signal handlers for graceful shutdown"""

    def signal_handler(sig, frame):
        launcher.logger.info(f"Received signal {sig}")
        launcher.shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="JARVIS Unified Launcher")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
        default="config/jarvis.yaml",
    )
    parser.add_argument(
        "--no-interactive", action="store_true", help="Disable interactive mode"
    )
    parser.add_argument(
        "--services", type=str, nargs="+", help="Only start specific services"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level",
    )

    args = parser.parse_args()

    # Create launcher
    launcher = JARVISLauncher(config_path=args.config)

    # Apply command-line overrides
    if args.no_interactive:
        launcher.config["features"]["interactive_mode"] = False

    if args.log_level:
        launcher.config["general"]["log_level"] = args.log_level
        launcher._setup_logging()

    if args.services:
        # Disable all services except specified
        for name in launcher.services:
            if name not in args.services:
                launcher.services[name].enabled = False

    # Setup signal handlers
    setup_signal_handlers(launcher)

    # Run
    try:
        asyncio.run(launcher.run())
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
