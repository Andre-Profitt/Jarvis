#!/usr/bin/env python3
"""
JARVIS Background Service
Always-running service that provides seamless AI assistance.
"""

import os
import sys
import time
import signal
import logging
import daemon
import lockfile
from pathlib import Path
import subprocess
import psutil
import json
from datetime import datetime

# Setup logging
log_dir = Path.home() / ".jarvis" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "jarvis_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("JARVISService")


class JARVISBackgroundService:
    """Background service for JARVIS"""
    
    def __init__(self):
        self.pid_file = Path.home() / ".jarvis" / "jarvis.pid"
        self.config_file = Path.home() / ".jarvis" / "config.json"
        self.running = False
        self.jarvis_process = None
        self.config = self.load_config()
        
    def load_config(self) -> dict:
        """Load service configuration"""
        default_config = {
            "auto_start": True,
            "restart_on_failure": True,
            "max_restarts": 5,
            "restart_delay": 5,
            "log_level": "INFO",
            "features": {
                "voice_activation": True,
                "continuous_listening": True,
                "background_learning": True,
                "proactive_assistance": True
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.error(f"Could not load config: {e}")
                
        return default_config
        
    def save_config(self):
        """Save current configuration"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save config: {e}")
            
    def start(self):
        """Start the service"""
        if self.is_running():
            logger.info("JARVIS service is already running")
            return
            
        logger.info("Starting JARVIS background service...")
        self.running = True
        
        # Write PID file
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pid_file, 'w') as f:
            f.write(str(os.getpid()))
            
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Start main loop
        restart_count = 0
        
        while self.running:
            try:
                # Start JARVIS process
                self.start_jarvis_process()
                
                # Monitor process
                while self.running and self.jarvis_process and self.jarvis_process.poll() is None:
                    time.sleep(1)
                    
                # Process ended
                if self.running and self.config.get("restart_on_failure", True):
                    restart_count += 1
                    if restart_count <= self.config.get("max_restarts", 5):
                        logger.warning(f"JARVIS process ended, restarting... (attempt {restart_count})")
                        time.sleep(self.config.get("restart_delay", 5))
                    else:
                        logger.error("Max restart attempts reached. Stopping service.")
                        self.running = False
                        
            except Exception as e:
                logger.error(f"Service error: {e}")
                time.sleep(5)
                
        self.cleanup()
        
    def start_jarvis_process(self):
        """Start the main JARVIS process"""
        logger.info("Starting JARVIS process...")
        
        # Find the JARVIS script
        jarvis_script = Path(__file__).parent / "jarvis_10_seamless.py"
        
        if not jarvis_script.exists():
            # Try alternate locations
            jarvis_script = Path(__file__).parent / "jarvis.py"
            
        if not jarvis_script.exists():
            logger.error("Could not find JARVIS script!")
            return
            
        # Start process
        env = os.environ.copy()
        env['JARVIS_BACKGROUND_MODE'] = '1'
        
        self.jarvis_process = subprocess.Popen(
            [sys.executable, str(jarvis_script)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        logger.info(f"JARVIS process started with PID: {self.jarvis_process.pid}")
        
    def stop(self):
        """Stop the service"""
        logger.info("Stopping JARVIS background service...")
        self.running = False
        
        # Stop JARVIS process
        if self.jarvis_process:
            try:
                self.jarvis_process.terminate()
                self.jarvis_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.jarvis_process.kill()
                
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources"""
        # Remove PID file
        if self.pid_file.exists():
            self.pid_file.unlink()
            
    def _signal_handler(self, signum, frame):
        """Handle signals"""
        logger.info(f"Received signal {signum}")
        self.stop()
        
    def is_running(self) -> bool:
        """Check if service is already running"""
        if not self.pid_file.exists():
            return False
            
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
                
            # Check if process exists
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                if "python" in process.name().lower():
                    return True
        except:
            pass
            
        # PID file exists but process doesn't
        self.pid_file.unlink()
        return False
        
    def status(self) -> dict:
        """Get service status"""
        status = {
            "running": self.is_running(),
            "pid": None,
            "uptime": None,
            "config": self.config
        }
        
        if status["running"] and self.pid_file.exists():
            try:
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                    
                process = psutil.Process(pid)
                status["pid"] = pid
                status["uptime"] = datetime.now() - datetime.fromtimestamp(process.create_time())
                status["memory_usage"] = process.memory_info().rss / 1024 / 1024  # MB
                status["cpu_percent"] = process.cpu_percent()
            except:
                pass
                
        return status


def create_launchd_plist():
    """Create macOS launchd plist for auto-start"""
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.jarvis.assistant</string>
    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>{os.path.abspath(__file__)}</string>
        <string>start</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardErrorPath</key>
    <string>{Path.home()}/.jarvis/logs/jarvis_error.log</string>
    <key>StandardOutPath</key>
    <string>{Path.home()}/.jarvis/logs/jarvis_out.log</string>
    <key>WorkingDirectory</key>
    <string>{Path(__file__).parent}</string>
</dict>
</plist>"""
    
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.jarvis.assistant.plist"
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(plist_path, 'w') as f:
        f.write(plist_content)
        
    # Load the service
    subprocess.run(["launchctl", "load", str(plist_path)], check=True)
    
    return plist_path


def remove_launchd_plist():
    """Remove macOS launchd plist"""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.jarvis.assistant.plist"
    
    if plist_path.exists():
        # Unload the service
        subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
        
        # Remove the plist
        plist_path.unlink()


def main():
    """Main entry point for service management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="JARVIS Background Service")
    parser.add_argument(
        'command',
        choices=['start', 'stop', 'restart', 'status', 'install', 'uninstall'],
        help='Service command'
    )
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    
    args = parser.parse_args()
    
    service = JARVISBackgroundService()
    
    if args.command == 'start':
        if args.daemon:
            # Run as daemon
            with daemon.DaemonContext(
                working_directory=Path(__file__).parent,
                pidfile=lockfile.FileLock(service.pid_file)
            ):
                service.start()
        else:
            service.start()
            
    elif args.command == 'stop':
        if service.is_running():
            # Send SIGTERM to running process
            with open(service.pid_file, 'r') as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
            print("JARVIS service stopped")
        else:
            print("JARVIS service is not running")
            
    elif args.command == 'restart':
        if service.is_running():
            # Stop
            with open(service.pid_file, 'r') as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
            
        # Start
        service.start()
        
    elif args.command == 'status':
        status = service.status()
        if status['running']:
            print(f"JARVIS service is running (PID: {status['pid']})")
            if status['uptime']:
                print(f"Uptime: {status['uptime']}")
            if 'memory_usage' in status:
                print(f"Memory: {status['memory_usage']:.1f} MB")
            if 'cpu_percent' in status:
                print(f"CPU: {status['cpu_percent']:.1f}%")
        else:
            print("JARVIS service is not running")
            
    elif args.command == 'install':
        # Install as system service
        print("Installing JARVIS as system service...")
        
        # Create launchd plist for macOS
        if sys.platform == 'darwin':
            plist_path = create_launchd_plist()
            print(f"✅ Installed launchd service: {plist_path}")
            print("JARVIS will now start automatically on login")
        else:
            print("Auto-start installation not supported on this platform yet")
            
    elif args.command == 'uninstall':
        # Uninstall system service
        print("Uninstalling JARVIS system service...")
        
        # Stop service first
        if service.is_running():
            with open(service.pid_file, 'r') as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
            
        # Remove launchd plist for macOS
        if sys.platform == 'darwin':
            remove_launchd_plist()
            print("✅ Uninstalled launchd service")
        else:
            print("Service uninstallation not supported on this platform yet")


if __name__ == "__main__":
    main()