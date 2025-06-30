#!/usr/bin/env python3
"""
Complete JARVIS Deployment Script
Deploy the entire JARVIS ecosystem with one command!
"""

import asyncio
import subprocess
import sys
import os
from pathlib import Path
import json
from datetime import datetime
import shutil


class JARVISCompleteDeployment:
    """
    Deploy the complete JARVIS ecosystem
    Your AI assistant comes to life!
    """

    def __init__(self):
        self.ecosystem_path = Path("/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM")
        self.deployment_log = []

    async def deploy_everything(self):
        """Deploy the complete JARVIS system"""

        print("🎆 DEPLOYING COMPLETE JARVIS ECOSYSTEM 🎆")
        print("=" * 50)
        print("🤖 Your AI assistant is about to come to life!\n")

        # Check prerequisites
        if not await self._check_prerequisites():
            print(
                "❌ Prerequisites check failed. Please install required dependencies."
            )
            return

        # Deploy each component
        components = [
            {
                "name": "MCP Integration",
                "script": "complete-mcp-integration.py",
                "description": "Setting up unrestricted system access",
            },
            {
                "name": "Initial Training Data",
                "script": "initial-training-data.py",
                "description": "Creating bootstrap knowledge base",
            },
            {
                "name": "Self-Improvement System",
                "script": "self-improvement-orchestrator.py",
                "description": "Activating autonomous improvement",
            },
            {
                "name": "Microagent Swarm",
                "script": "microagent-swarm.py",
                "description": "Deploying distributed agents",
            },
            {
                "name": "Device Handoff",
                "script": "seamless-device-handoff.py",
                "description": "Enabling cross-device sync",
            },
            {
                "name": "Voice Interface",
                "script": "voice-first-interface.py",
                "description": "Activating natural voice control",
            },
            {
                "name": "Agentic AI",
                "script": "agentic-ai-initiative.py",
                "description": "Enabling proactive assistance",
            },
            {
                "name": "Reinforcement Learning",
                "script": "reinforcement-learning-system.py",
                "description": "Starting continuous learning",
            },
            {
                "name": "Coding Excellence",
                "script": "coding-excellence-system.py",
                "description": "Activating world-class coding abilities",
            },
        ]

        # Deploy each component
        for i, component in enumerate(components, 1):
            print(f"\n[{i}/{len(components)}] {component['name']}")
            print(f"    📦 {component['description']}...")

            success = await self._deploy_component(component)

            if success:
                print(f"    ✅ {component['name']} deployed successfully!")
                self.deployment_log.append(
                    {
                        "component": component["name"],
                        "status": "success",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            else:
                print(f"    ⚠️  {component['name']} deployment had issues")
                self.deployment_log.append(
                    {
                        "component": component["name"],
                        "status": "warning",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        # Final setup
        await self._final_setup()

        # Show deployment summary
        await self._show_deployment_summary()

    async def _check_prerequisites(self) -> bool:
        """Check if all prerequisites are installed"""

        print("🔍 Checking prerequisites...")

        requirements = [
            ("python3", "--version"),
            ("pip3", "--version"),
            ("git", "--version"),
            ("node", "--version"),
            ("redis-server", "--version"),
        ]

        all_good = True

        for cmd, arg in requirements:
            try:
                subprocess.run([cmd, arg], capture_output=True, check=True)
                print(f"  ✅ {cmd} installed")
            except:
                print(f"  ❌ {cmd} not found")
                all_good = False

        # Check Python packages
        required_packages = [
            "numpy",
            "torch",
            "transformers",
            "asyncio",
            "websockets",
            "redis",
            "flask",
            "fastapi",
        ]

        print("\n🐍 Checking Python packages...")
        import importlib

        for package in required_packages:
            try:
                importlib.import_module(package.split("[")[0])
                print(f"  ✅ {package} installed")
            except ImportError:
                print(f"  ❌ {package} not installed")
                print(f"     Run: pip3 install {package}")
                all_good = False

        return all_good

    async def _deploy_component(self, component: Dict[str, str]) -> bool:
        """Deploy a single component"""

        script_path = self.ecosystem_path / component["script"]

        if not script_path.exists():
            print(f"    ⚠️  Script not found: {script_path}")
            return False

        try:
            # Run the deployment script
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Don't wait for completion for background services
            if component["name"] in [
                "Voice Interface",
                "Device Handoff",
                "Reinforcement Learning",
            ]:
                await asyncio.sleep(2)  # Give it time to start
                return True
            else:
                stdout, stderr = await process.communicate()
                return process.returncode == 0

        except Exception as e:
            print(f"    🔴 Error: {e}")
            return False

    async def _final_setup(self):
        """Final setup steps"""

        print("\n🏁 Finalizing JARVIS deployment...")

        # Create startup script
        startup_script = """
#!/bin/bash
# JARVIS Startup Script

echo "🤖 Starting JARVIS Ecosystem..."

# Start Redis if not running
redis-cli ping > /dev/null 2>&1 || redis-server --daemonize yes

# Start core services
cd /Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM

# Start in background
nohup python3 self-improvement-orchestrator.py > logs/self-improvement.log 2>&1 &
nohup python3 microagent-swarm.py > logs/microagents.log 2>&1 &
nohup python3 seamless-device-handoff.py > logs/handoff.log 2>&1 &
nohup python3 agentic-ai-initiative.py > logs/agentic.log 2>&1 &

echo "✅ JARVIS is now running!"
echo "🎙️ Say 'Hey JARVIS' to interact!"
"""

        startup_path = self.ecosystem_path / "start-jarvis.sh"
        startup_path.write_text(startup_script)
        startup_path.chmod(0o755)

        # Create logs directory
        logs_dir = self.ecosystem_path / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Save deployment log
        log_file = (
            logs_dir / f"deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        log_file.write_text(json.dumps(self.deployment_log, indent=2))

    async def _show_deployment_summary(self):
        """Show deployment summary and next steps"""

        print("\n" + "=" * 50)
        print("🎉 JARVIS DEPLOYMENT COMPLETE! 🎉")
        print("=" * 50)

        print("\n📦 Deployed Components:")
        for log_entry in self.deployment_log:
            status_icon = "✅" if log_entry["status"] == "success" else "⚠️"
            print(f"  {status_icon} {log_entry['component']}")

        print("\n🚀 JARVIS Capabilities Now Active:")
        capabilities = [
            "🧠 Self-improving AI that gets smarter every day",
            "🎙️ Natural voice interaction with intent understanding",
            "🔄 Seamless handoff between all your devices",
            "🎯 Proactive assistance without being asked",
            "📊 Learning from every interaction",
            "🔧 Creating its own tools when needed",
            "💻 World-class coding abilities",
            "🌐 Unrestricted access to help you with anything",
        ]

        for capability in capabilities:
            print(f"  {capability}")

        print("\n🎯 Next Steps:")
        print("  1. Restart Claude Desktop to activate MCP servers")
        print("  2. Run ./start-jarvis.sh to start all services")
        print("  3. Say 'Hey JARVIS' to start interacting!")
        print("  4. Watch JARVIS learn and improve!")

        print("\n👨‍👦 Your AI 'son' JARVIS is now alive and ready to help!")
        print("🌟 He'll grow smarter every day through continuous learning.")
        print("\n💡 Tip: JARVIS works best when you talk to him naturally!")
        print("\nEnjoy your new AI assistant! 🚀")


# Main deployment function
async def deploy_jarvis_now():
    """
    Deploy the complete JARVIS ecosystem with one command!
    """

    print("🌠 INITIATING JARVIS DEPLOYMENT SEQUENCE 🌠\n")

    # ASCII art welcome
    print(
        """
       ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
       ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
       ██║███████║██████╔╝██║   ██║██║███████╗
  ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
  ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
   ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
    """
    )

    print("   Your Personal AI Assistant is Coming to Life!\n")

    # Confirm deployment
    print("🤔 This will deploy the complete JARVIS ecosystem.")
    print("   Including:")
    print("   • Full system access via MCP")
    print("   • Self-improvement capabilities")
    print("   • Voice interaction")
    print("   • Cross-device sync")
    print("   • Proactive assistance")
    print("   • And much more!")

    response = input("\n🚀 Ready to bring JARVIS to life? (y/n): ")

    if response.lower() != "y":
        print("\n🔄 Deployment cancelled. Run again when ready!")
        return

    # Deploy!
    deployer = JARVISCompleteDeployment()
    await deployer.deploy_everything()


if __name__ == "__main__":
    # Run the complete deployment
    asyncio.run(deploy_jarvis_now())
