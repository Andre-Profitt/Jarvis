#!/usr/bin/env python3
"""
JARVIS Phase 1 Quick Setup & Demo
One-click setup and demonstration of all Phase 1 features
"""

import asyncio
import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

class Phase1Setup:
    """Quick setup for JARVIS Phase 1"""
    
    def __init__(self):
        self.processes = []
        
    async def run(self):
        """Run complete Phase 1 setup and demo"""
        print("\n" + "="*60)
        print("🚀 JARVIS PHASE 1 QUICK SETUP")
        print("="*60)
        print("This will:")
        print("  1. Run tests to verify installation")
        print("  2. Start the monitoring server")
        print("  3. Open the dashboard in your browser")
        print("  4. Launch JARVIS with Phase 1 enhancements")
        print("  5. Run interactive demos")
        print("="*60)
        
        # Step 1: Run tests
        print("\n1️⃣ Running tests...")
        test_success = await self.run_tests()
        
        if not test_success:
            print("\n❌ Some tests failed. Continue anyway? (y/n): ", end="")
            if input().lower() != 'y':
                return
                
        # Step 2: Start monitoring server
        print("\n2️⃣ Starting monitoring server...")
        monitor_process = self.start_monitoring_server()
        if monitor_process:
            self.processes.append(monitor_process)
            await asyncio.sleep(2)  # Give server time to start
            
        # Step 3: Open dashboard
        print("\n3️⃣ Opening dashboard in browser...")
        self.open_dashboard()
        
        # Step 4: Launch JARVIS
        print("\n4️⃣ Launching JARVIS with Phase 1...")
        await self.launch_jarvis()
        
        print("\n✅ Phase 1 is running!")
        print("\n📝 Quick Commands:")
        print("  • Type 'help' for available commands")
        print("  • Type 'status' to see current state")
        print("  • Type 'predict' for future state prediction")
        print("  • Type 'quit' to exit")
        print("\n💡 Try these demos:")
        print("  • 'I'm feeling stressed' - See stress response")
        print("  • 'I'm in deep focus' - See flow state protection")
        print("  • 'I'm tired' - See energy management")
        
    async def run_tests(self):
        """Run Phase 1 tests"""
        try:
            # Import and run tests
            from test_jarvis_phase1 import TestPhase1
            
            tester = TestPhase1()
            await tester.run_all_tests()
            
            # Check if all passed
            total = tester.passed_tests + tester.failed_tests
            if total > 0:
                success_rate = (tester.passed_tests / total) * 100
                return success_rate >= 80  # 80% pass rate minimum
            return False
            
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return False
            
    def start_monitoring_server(self):
        """Start the monitoring server in background"""
        try:
            # Start server as subprocess
            process = subprocess.Popen(
                [sys.executable, 'jarvis_monitoring_server.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            print("✅ Monitoring server started (PID: {})".format(process.pid))
            return process
            
        except Exception as e:
            print(f"⚠️  Could not start monitoring server: {e}")
            print("   You can start it manually: python jarvis_monitoring_server.py")
            return None
            
    def open_dashboard(self):
        """Open dashboard in default browser"""
        try:
            dashboard_path = Path(__file__).parent / 'jarvis-phase1-monitor.html'
            if dashboard_path.exists():
                webbrowser.open(f'file://{dashboard_path.absolute()}')
                print("✅ Dashboard opened in browser")
            else:
                print("⚠️  Dashboard file not found")
        except Exception as e:
            print(f"⚠️  Could not open dashboard: {e}")
            
    async def launch_jarvis(self):
        """Launch JARVIS with Phase 1 enhancements"""
        try:
            # Import and run
            from launch_jarvis_phase1 import JARVISPhase1Launcher
            
            launcher = JARVISPhase1Launcher()
            await launcher.initialize()
            
            # Run demo then interactive mode
            print("\n" + "-"*60)
            print("Running quick demo...")
            print("-"*60)
            await launcher.run_demo()
            
            print("\n" + "-"*60)
            print("Entering interactive mode...")
            print("-"*60)
            await launcher.run_interactive()
            
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        except Exception as e:
            print(f"❌ Error launching JARVIS: {e}")
            import traceback
            traceback.print_exc()
            
    def cleanup(self):
        """Clean up processes"""
        print("\n🧹 Cleaning up...")
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        print("✅ Cleanup complete")

async def main():
    """Main entry point"""
    setup = Phase1Setup()
    
    try:
        await setup.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        setup.cleanup()

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║            🤖 JARVIS PHASE 1 ENHANCED 🤖                  ║
    ║                                                           ║
    ║    Unified Input Pipeline + Fluid State Management       ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())