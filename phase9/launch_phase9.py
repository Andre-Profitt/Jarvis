#!/usr/bin/env python3
"""
JARVIS Phase 9: Launch Script
=============================
Launches JARVIS with Phase 9 performance optimizations
"""

import asyncio
import argparse
import sys
import os
import time
from datetime import datetime
import subprocess
import signal
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase9.jarvis_phase9_integration import JARVISOptimizedCore
from phase9.monitoring_server import PerformanceMonitoringServer


class JARVISPhase9Launcher:
    """
    Launcher for JARVIS with Phase 9 optimizations
    """
    
    def __init__(self):
        self.jarvis_core = None
        self.monitor_server = None
        self.monitor_process = None
        self.running = False
    
    async def start_monitoring_server(self, standalone: bool = False):
        """Start the performance monitoring server"""
        try:
            # Start as subprocess for better isolation
            cmd = [
                sys.executable,
                'phase9/monitoring_server.py',
                '--port', '8765'
            ]
            
            if standalone:
                cmd.append('--standalone')
            
            self.monitor_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Give it time to start
            await asyncio.sleep(2)
            
            print("✅ Performance monitoring server started")
            print("📊 Dashboard: http://localhost:8765")
            
        except Exception as e:
            print(f"⚠️  Could not start monitoring server: {e}")
            print("   Continuing without real-time monitoring...")
    
    async def initialize_jarvis(self, optimization_level: str = 'balanced'):
        """Initialize JARVIS with Phase 9 optimizations"""
        print("\n🚀 Initializing JARVIS with Phase 9 Performance Optimizations...")
        
        self.jarvis_core = JARVISOptimizedCore()
        await self.jarvis_core.initialize()
        
        # Set optimization level
        self.jarvis_core.optimizer.set_optimization_level(optimization_level)
        
        print(f"✅ JARVIS initialized with {optimization_level} optimization")
        
        # Show initial stats
        report = self.jarvis_core.get_optimization_report()
        print(f"\n📊 Initial Performance Stats:")
        print(f"   Cache Size: {report['cache_stats'].get('memory_saved_mb', 0):.1f} MB")
        print(f"   Workers: {report['parallel_stats'].get('current_workers', 'N/A')}")
        print(f"   Modules Loaded: {report['lazy_loading_stats'].get('loaded_modules', 0)}")
    
    async def run_interactive_mode(self):
        """Run JARVIS in interactive mode"""
        print("\n💬 JARVIS Phase 9 - Interactive Mode")
        print("=" * 50)
        print("Commands:")
        print("  'help' - Show this help")
        print("  'stats' - Show performance statistics")
        print("  'optimize' - Run auto-optimization")
        print("  'dashboard' - Open performance dashboard")
        print("  'process <text>' - Process input text")
        print("  'batch <n>' - Process n test inputs")
        print("  'level <mode>' - Set optimization level")
        print("  'exit' - Quit JARVIS")
        print("=" * 50)
        
        while self.running:
            try:
                # Get user input
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, input, "\n🤖 JARVIS> "
                )
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.strip().split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                # Handle commands
                if command == 'exit' or command == 'quit':
                    break
                
                elif command == 'help':
                    print("\nCommands:")
                    print("  'stats' - Show performance statistics")
                    print("  'optimize' - Run auto-optimization")
                    print("  'dashboard' - Open performance dashboard")
                    print("  'process <text>' - Process input text")
                    print("  'batch <n>' - Process n test inputs")
                    print("  'level <mode>' - Set optimization level (conservative/balanced/aggressive)")
                    print("  'clear' - Clear screen")
                    print("  'exit' - Quit JARVIS")
                
                elif command == 'stats':
                    await self.show_performance_stats()
                
                elif command == 'optimize':
                    print("🔧 Running auto-optimization...")
                    await self.jarvis_core.run_auto_optimization()
                    print("✅ Optimization complete!")
                
                elif command == 'dashboard':
                    # Open dashboard in browser
                    dashboard_path = os.path.join(
                        os.path.dirname(__file__),
                        'performance_monitor.html'
                    )
                    import webbrowser
                    webbrowser.open(f'file://{dashboard_path}')
                    print("📊 Dashboard opened in browser")
                
                elif command == 'process':
                    if args:
                        result = await self.process_input(args)
                        print(f"\n📝 Result: {json.dumps(result, indent=2)}")
                    else:
                        print("❌ Please provide text to process")
                
                elif command == 'batch':
                    try:
                        n = int(args) if args else 10
                        await self.run_batch_test(n)
                    except ValueError:
                        print("❌ Please provide a valid number")
                
                elif command == 'level':
                    if args in ['conservative', 'balanced', 'aggressive']:
                        self.jarvis_core.optimizer.set_optimization_level(args)
                        print(f"✅ Optimization level set to: {args}")
                    else:
                        print("❌ Valid levels: conservative, balanced, aggressive")
                
                elif command == 'clear':
                    os.system('clear' if os.name != 'nt' else 'cls')
                
                else:
                    # Process as regular input
                    result = await self.process_input(user_input)
                    print(f"\n📝 Result: {json.dumps(result, indent=2)}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Shutting down...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def process_input(self, text: str) -> dict:
        """Process a single input"""
        start_time = time.time()
        
        result = await self.jarvis_core.process_batch([{
            'type': 'text',
            'data': text,
            'timestamp': datetime.now().isoformat()
        }])
        
        elapsed = time.time() - start_time
        
        return {
            'input': text,
            'output': result[0] if result else None,
            'processing_time': f"{elapsed:.3f}s"
        }
    
    async def run_batch_test(self, n: int):
        """Run a batch processing test"""
        print(f"\n🧪 Running batch test with {n} inputs...")
        
        # Generate test inputs
        inputs = [
            {
                'type': 'text',
                'data': f'Test message {i}: Analyze this text for sentiment and entities',
                'timestamp': datetime.now().isoformat()
            }
            for i in range(n)
        ]
        
        # Process batch
        start_time = time.time()
        results = await self.jarvis_core.process_batch(inputs)
        elapsed = time.time() - start_time
        
        # Show results
        print(f"\n✅ Processed {len(results)} inputs in {elapsed:.2f}s")
        print(f"   Average: {elapsed/n:.3f}s per input")
        print(f"   Throughput: {n/elapsed:.1f} inputs/second")
        
        # Show cache performance
        cache_stats = self.jarvis_core.optimizer.cache.stats.get_stats()
        print(f"\n📊 Cache Performance:")
        print(f"   Hit Rate: {cache_stats['hit_rate']:.1%}")
        print(f"   Time Saved: {cache_stats['time_saved_seconds']:.1f}s")
    
    async def show_performance_stats(self):
        """Display current performance statistics"""
        report = self.jarvis_core.get_optimization_report()
        
        print("\n" + "="*60)
        print("JARVIS Phase 9 - Performance Report")
        print("="*60)
        
        # Cache Statistics
        cache = report['cache_stats']
        print(f"\n📦 Cache Performance:")
        print(f"   Hit Rate: {cache['hit_rate']:.1%}")
        print(f"   Hits/Misses: {cache['hits']}/{cache['misses']}")
        print(f"   Memory Saved: {cache['memory_saved_mb']:.1f} MB")
        print(f"   Time Saved: {cache['time_saved_seconds']:.1f}s")
        
        # Parallel Processing
        parallel = report['parallel_stats']
        if parallel:
            print(f"\n⚡ Parallel Processing:")
            print(f"   Current Workers: {parallel.get('current_workers', 'N/A')}")
            print(f"   Average Task Time: {parallel.get('average_time', 0):.3f}s")
            sys_stats = parallel.get('system_stats', {})
            print(f"   CPU Usage: {sys_stats.get('cpu_percent', 0):.1f}%")
            print(f"   Memory Usage: {sys_stats.get('memory_percent', 0):.1f}%")
        
        # Lazy Loading
        lazy = report['lazy_loading_stats']
        print(f"\n💤 Lazy Loading:")
        print(f"   Loaded Modules: {lazy['loaded_modules']}")
        print(f"   Total Loading Time: {lazy['total_loading_time']:.2f}s")
        if lazy.get('hot_modules'):
            print(f"   Hot Modules: {', '.join([m[0] for m in lazy['hot_modules'][:3]])}")
        
        # Current Performance
        current = report['monitor_dashboard']['current_metrics']
        print(f"\n📈 Current Performance:")
        print(f"   CPU: {current['cpu']:.1f}%")
        print(f"   Memory: {current['memory']:.1f}%")
        print(f"   Avg Response: {current['avg_response_time']:.1f}ms")
        
        # Optimization Impact
        if 'optimization_impact' in report:
            impact = report['optimization_impact']
            print(f"\n🎯 Optimization Impact:")
            for metric, value in impact.items():
                print(f"   {metric.replace('_', ' ').title()}: {value}")
        
        # Recommendations
        recs = report['monitor_dashboard']['recommendations']
        if recs:
            print(f"\n💡 Recommendations:")
            for rec in recs[:3]:
                print(f"   - {rec['suggestion']}")
                print(f"     Impact: {rec['expected_impact']}")
        
        print("\n" + "="*60)
    
    async def run(self, args):
        """Main run method"""
        self.running = True
        
        try:
            # Start monitoring server if requested
            if args.monitor:
                await self.start_monitoring_server(args.standalone)
            
            # Initialize JARVIS
            await self.initialize_jarvis(args.optimization)
            
            # Run in requested mode
            if args.batch:
                # Run batch test
                await self.run_batch_test(args.batch)
                
            elif args.demo:
                # Run demo
                print("\n🎭 Running Phase 9 Demo...")
                
                # Demo different features
                demos = [
                    ("Cached Processing", self.demo_caching),
                    ("Parallel Batch", self.demo_parallel),
                    ("Auto-Optimization", self.demo_optimization)
                ]
                
                for name, demo_func in demos:
                    print(f"\n{'='*50}")
                    print(f"Demo: {name}")
                    print('='*50)
                    await demo_func()
                    await asyncio.sleep(1)
                
            else:
                # Interactive mode
                await self.run_interactive_mode()
                
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
        finally:
            await self.shutdown()
    
    async def demo_caching(self):
        """Demo caching performance"""
        text = "Analyze this text for sentiment and extract key entities"
        
        # First call
        start = time.time()
        result1 = await self.process_input(text)
        time1 = time.time() - start
        
        # Second call (cached)
        start = time.time()
        result2 = await self.process_input(text)
        time2 = time.time() - start
        
        print(f"First call: {time1:.3f}s")
        print(f"Cached call: {time2:.3f}s")
        print(f"Speed improvement: {time1/time2:.1f}x")
    
    async def demo_parallel(self):
        """Demo parallel processing"""
        await self.run_batch_test(50)
    
    async def demo_optimization(self):
        """Demo auto-optimization"""
        print("Running auto-optimization...")
        await self.jarvis_core.run_auto_optimization()
        
        # Show optimization results
        level = self.jarvis_core.optimizer.optimization_level
        print(f"Optimization level: {level}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.running = False
        
        print("\n🛑 Shutting down JARVIS Phase 9...")
        
        # Shutdown JARVIS
        if self.jarvis_core:
            await self.jarvis_core.shutdown()
        
        # Stop monitoring server
        if self.monitor_process:
            self.monitor_process.terminate()
            self.monitor_process.wait(timeout=5)
        
        print("✅ Shutdown complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='JARVIS Phase 9 - Performance Optimized Launch'
    )
    
    parser.add_argument(
        '--optimization', '-o',
        choices=['conservative', 'balanced', 'aggressive'],
        default='balanced',
        help='Optimization level (default: balanced)'
    )
    
    parser.add_argument(
        '--monitor', '-m',
        action='store_true',
        help='Start performance monitoring server'
    )
    
    parser.add_argument(
        '--standalone', '-s',
        action='store_true',
        help='Run monitor in standalone mode (no JARVIS required)'
    )
    
    parser.add_argument(
        '--batch', '-b',
        type=int,
        metavar='N',
        help='Run batch test with N inputs'
    )
    
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run performance optimization demos'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                   JARVIS PHASE 9                         ║
    ║              Performance Optimization Suite              ║
    ╠══════════════════════════════════════════════════════════╣
    ║                                                          ║
    ║  🚀 Intelligent Caching      ⚡ Parallel Processing      ║
    ║  💤 Lazy Loading            📊 Real-time Monitoring     ║
    ║  🤖 Auto-optimization       🎯 Pattern Recognition      ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Create and run launcher
    launcher = JARVISPhase9Launcher()
    
    # Handle shutdown signals
    def signal_handler(sig, frame):
        print("\n\nReceived shutdown signal...")
        launcher.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run
    asyncio.run(launcher.run(args))


if __name__ == "__main__":
    main()
