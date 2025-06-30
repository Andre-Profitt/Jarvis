#!/usr/bin/env python3
"""
JARVIS Phase 11 Launcher - Complete System Integration
Launches the fully integrated JARVIS system with all 10 phases working together
"""

import asyncio
import argparse
import logging
import os
import sys
import time
import webbrowser
from datetime import datetime
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase11_integration.system_integration_orchestrator import SystemIntegrationOrchestrator
from phase11_integration.production_config import ConfigurationManager, create_deployment_config

# ASCII Art Banner
BANNER = """
╔═══════════════════════════════════════════════════════════════════╗
║     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗    ██████╗  ██╗     ║
║     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝    ██╔══██╗███║     ║
║     ██║███████║██████╔╝██║   ██║██║███████╗    ██████╔╝╚██║     ║
║██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║    ██╔═══╝  ██║     ║
║╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║    ██║      ██║     ║
║ ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝    ╚═╝      ╚═╝     ║
║                                                                   ║
║           Phase 11: Complete System Integration 🚀                ║
║                All 10 Phases Working As One                       ║
╚═══════════════════════════════════════════════════════════════════╝
"""

class Phase11Launcher:
    """Launcher for the fully integrated JARVIS system"""
    
    def __init__(self, config_type: str = "production", debug: bool = False):
        self.config_type = config_type
        self.debug = debug
        self.orchestrator = None
        self.logger = self._setup_logging()
        self.start_time = time.time()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_level = logging.DEBUG if self.debug else logging.INFO
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler(f'logs/jarvis_phase11_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return logging.getLogger(__name__)
    
    def print_banner(self):
        """Print the startup banner"""
        print("\033[96m" + BANNER + "\033[0m")  # Cyan color
        print(f"\n🔧 Configuration: {self.config_type}")
        print(f"🐛 Debug Mode: {'ON' if self.debug else 'OFF'}")
        print(f"🕐 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 70 + "\n")
    
    async def initialize_system(self) -> bool:
        """Initialize the complete JARVIS system"""
        try:
            self.logger.info("Initializing JARVIS Phase 11 System Integration...")
            
            # Create orchestrator
            self.orchestrator = SystemIntegrationOrchestrator()
            
            # Load configuration
            if self.config_type != "default":
                config = create_deployment_config(self.config_type)
                self.logger.info(f"Loaded {self.config_type} configuration")
            
            # Initialize all phases
            print("📦 Initializing all phases...")
            success = await self.orchestrator.initialize_all_phases()
            
            if success:
                self.logger.info("✅ All phases initialized successfully!")
                
                # Show phase status
                print("\n📊 Phase Status:")
                for phase_num in range(1, 11):
                    phase_key = f"phase{phase_num}"
                    components = len(self.orchestrator.phases.get(phase_key, {}))
                    print(f"   Phase {phase_num}: ✓ Active ({components} components)")
                
                return True
            else:
                self.logger.error("❌ Failed to initialize some phases")
                return False
                
        except Exception as e:
            self.logger.error(f"Initialization error: {e}", exc_info=True)
            return False
    
    async def run_startup_tests(self) -> bool:
        """Run startup integration tests"""
        print("\n🧪 Running startup tests...")
        
        try:
            # Quick health check
            health = await self.orchestrator.generate_health_report()
            
            if health['overall_health'] == 'HEALTHY':
                self.logger.info("✅ System health check passed")
                
                # Run basic integration test
                test_request = {
                    'voice': {'text': 'System startup test'},
                    'context': {'type': 'startup_test'}
                }
                
                result = await self.orchestrator.process_request(test_request)
                
                if result['status'] == 'success':
                    self.logger.info("✅ Basic integration test passed")
                    return True
                else:
                    self.logger.warning("⚠️  Basic integration test failed")
                    return False
            else:
                self.logger.warning(f"⚠️  System health: {health['overall_health']}")
                return False
                
        except Exception as e:
            self.logger.error(f"Startup test error: {e}")
            return False
    
    async def start_monitoring(self):
        """Start the monitoring dashboard"""
        try:
            # Start monitoring server
            dashboard_path = os.path.join(
                os.path.dirname(__file__), 
                'phase11_dashboard.html'
            )
            
            if os.path.exists(dashboard_path):
                print("\n🖥️  Opening monitoring dashboard...")
                webbrowser.open(f'file://{os.path.abspath(dashboard_path)}')
            else:
                self.logger.warning("Dashboard file not found")
                
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
    
    async def main_loop(self):
        """Main processing loop"""
        self.logger.info("🎯 JARVIS Phase 11 is now active!")
        print("\n✨ JARVIS is ready! All systems operational.\n")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._performance_monitor()),
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._optimization_loop())
        ]
        
        try:
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested...")
            
            # Cancel background tasks
            for task in tasks:
                task.cancel()
            
            await self.shutdown()
    
    async def _performance_monitor(self):
        """Monitor system performance"""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                
                # Get performance metrics
                benchmarks = await self.orchestrator.run_performance_benchmarks()
                score = benchmarks.get('overall_score', 0)
                
                if score < 0.7:
                    self.logger.warning(f"Performance degradation detected: {score:.2f}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Performance monitor error: {e}")
    
    async def _health_monitor(self):
        """Monitor system health"""
        while True:
            try:
                await asyncio.sleep(30)  # Every 30 seconds
                
                # Check health
                health = await self.orchestrator.generate_health_report()
                
                if health['overall_health'] != 'HEALTHY':
                    self.logger.warning(f"Health issue: {health['overall_health']}")
                    
                    # Log recommendations
                    for rec in health.get('recommendations', []):
                        self.logger.info(f"Recommendation: {rec}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
    
    async def _optimization_loop(self):
        """Continuous optimization loop"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Run optimization
                optimizations = await self.orchestrator.optimize_system_configuration()
                
                if optimizations['recommended_changes']:
                    self.logger.info(f"Found {len(optimizations['recommended_changes'])} optimizations")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Optimization error: {e}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        print("\n🛑 Shutting down JARVIS Phase 11...")
        
        # Save final metrics
        if self.orchestrator:
            try:
                # Generate final report
                report = await self.orchestrator.generate_health_report()
                
                # Calculate uptime
                uptime = (time.time() - self.start_time) / 3600
                
                print(f"\n📊 Final Statistics:")
                print(f"   Uptime: {uptime:.1f} hours")
                print(f"   Total Requests: {self.orchestrator.metrics.total_requests}")
                print(f"   Success Rate: {self.orchestrator.metrics.calculate_success_rate():.1%}")
                print(f"   Errors: {self.orchestrator.metrics.error_count}")
                
            except Exception as e:
                self.logger.error(f"Error generating final report: {e}")
        
        print("\n👋 JARVIS Phase 11 shutdown complete. Goodbye!")
    
    async def run(self):
        """Run the complete Phase 11 system"""
        self.print_banner()
        
        # Initialize
        if not await self.initialize_system():
            self.logger.error("Failed to initialize system")
            return 1
        
        # Run tests
        if not await self.run_startup_tests():
            self.logger.warning("Startup tests failed, but continuing...")
        
        # Start monitoring
        await self.start_monitoring()
        
        # Run main loop
        try:
            await self.main_loop()
            return 0
        except Exception as e:
            self.logger.error(f"Fatal error: {e}", exc_info=True)
            return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='JARVIS Phase 11 - Complete System Integration'
    )
    
    parser.add_argument(
        '--config',
        choices=['default', 'production', 'aws', 'gcp', 'on_premise'],
        default='production',
        help='Configuration type to use'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Run tests only, then exit'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run performance benchmarks'
    )
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = Phase11Launcher(
        config_type=args.config,
        debug=args.debug
    )
    
    # Run appropriate mode
    if args.test_only:
        # Test mode
        async def test_mode():
            launcher.print_banner()
            if await launcher.initialize_system():
                test_results = await launcher.orchestrator.run_integration_tests()
                print(f"\nTest Results: {test_results['overall_status']}")
                for test in test_results['tests']:
                    status_icon = "✅" if test['status'] == 'PASSED' else "❌"
                    print(f"{status_icon} {test['name']}: {test['status']}")
                return 0 if test_results['overall_status'] == 'PASSED' else 1
            return 1
        
        sys.exit(asyncio.run(test_mode()))
        
    elif args.benchmark:
        # Benchmark mode
        async def benchmark_mode():
            launcher.print_banner()
            if await launcher.initialize_system():
                print("\n📊 Running performance benchmarks...")
                benchmarks = await launcher.orchestrator.run_performance_benchmarks()
                
                print(f"\nOverall Score: {benchmarks['overall_score']:.2f}")
                print("\nPhase Benchmarks:")
                for phase, data in benchmarks['phase_benchmarks'].items():
                    print(f"\n{phase}:")
                    for comp, metrics in data['benchmarks'].items():
                        print(f"  {comp}: {metrics['avg_latency_ms']:.1f}ms avg")
                
                return 0
            return 1
        
        sys.exit(asyncio.run(benchmark_mode()))
        
    else:
        # Normal mode
        try:
            sys.exit(asyncio.run(launcher.run()))
        except KeyboardInterrupt:
            print("\n\n⚡ Interrupted by user")
            sys.exit(0)


if __name__ == "__main__":
    main()
