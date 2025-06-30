"""
JARVIS Phase 10: Ultra Performance Launcher
Start JARVIS with blazing fast performance optimizations
"""

import asyncio
import sys
import argparse
import logging
from typing import Optional

# Add parent directory to path
sys.path.append('.')

from core.jarvis_ultra_core import JARVISUltraCore, PerformanceConfig
from monitoring.performance_dashboard import start_performance_monitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def launch_jarvis_ultra(args):
    """Launch JARVIS with Phase 10 optimizations"""
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║       JARVIS ULTRA - Phase 10: Performance Optimizations      ║
    ║                                                               ║
    ║                    🚀 BLAZING FAST 🚀                         ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Configure performance settings
    config = PerformanceConfig(
        enable_parallel_processing=not args.no_parallel,
        enable_intelligent_caching=not args.no_cache,
        enable_lazy_loading=not args.no_lazy,
        enable_jit_compilation=not args.no_jit,
        cache_size_mb=args.cache_size,
        memory_threshold=args.memory_threshold
    )
    
    # Create JARVIS Ultra instance
    logger.info("Initializing JARVIS Ultra Core...")
    jarvis = JARVISUltraCore(config)
    
    # Initialize
    await jarvis.initialize()
    logger.info("JARVIS Ultra Core initialized successfully")
    
    # Optimize for workload if specified
    if args.workload:
        logger.info(f"Optimizing for {args.workload} workload...")
        await jarvis.optimize_for_workload(args.workload)
    
    # Run benchmark if requested
    if args.benchmark:
        print("\n📊 Running Performance Benchmark...")
        results = await jarvis.run_performance_benchmark()
        print("\nBenchmark Results:")
        for test, time_taken in results.items():
            print(f"  {test.replace('_', ' ').title()}: {time_taken*1000:.1f}ms")
    
    # Start performance monitor if requested
    monitor = None
    if args.monitor:
        logger.info(f"Starting performance monitor on port {args.monitor_port}...")
        monitor = await start_performance_monitor(jarvis, args.monitor_port)
        print(f"\n🌐 Performance Dashboard: http://localhost:{args.monitor_port}")
    
    # Demo mode
    if args.demo:
        await run_demo(jarvis)
    
    # Interactive mode
    if args.interactive:
        await interactive_mode(jarvis)
    
    # Keep running if monitor is active
    if monitor:
        print("\n✨ JARVIS Ultra is running with performance monitoring")
        print("Press Ctrl+C to stop...")
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            pass
    
    # Shutdown
    logger.info("Shutting down JARVIS Ultra...")
    await jarvis.shutdown()


async def run_demo(jarvis):
    """Run performance demonstration"""
    print("\n🎭 JARVIS Ultra Performance Demo")
    print("=" * 50)
    
    # Test 1: Multi-modal parallel processing
    print("\n1️⃣ Testing Multi-modal Parallel Processing...")
    input_data = {
        'vision': {'description': 'Person at computer'},
        'audio': {'transcription': 'Hello JARVIS, how are you?'},
        'biometric': {'heart_rate': 72, 'stress_level': 0.3}
    }
    
    import time
    start = time.time()
    result = await jarvis.process_input(input_data, 'demo')
    elapsed = time.time() - start
    
    print(f"   ✓ Processed 3 modalities in {elapsed:.3f}s")
    print(f"   ✓ Current state: {result.get('state', 'unknown')}")
    
    # Test 2: Cache performance
    print("\n2️⃣ Testing Intelligent Cache...")
    
    # First call (miss)
    start = time.time()
    await jarvis.get_user_preferences("demo_user")
    miss_time = time.time() - start
    
    # Second call (hit)
    start = time.time()
    await jarvis.get_user_preferences("demo_user")
    hit_time = time.time() - start
    
    speedup = miss_time / hit_time if hit_time > 0 else 1
    print(f"   ✓ Cache speedup: {speedup:.1f}x faster")
    
    # Test 3: Module management
    print("\n3️⃣ Testing Lazy Loading...")
    status = jarvis.lazy_loader.get_status()
    print(f"   ✓ Loaded: {status['loaded_modules']}/{status['total_modules']} modules")
    print(f"   ✓ Memory saved: {status['memory_usage']['available_memory']}")
    
    # Test 4: JIT compilation
    print("\n4️⃣ Testing JIT Compilation...")
    vec1 = list(range(100))
    vec2 = list(range(100, 200))
    
    # Warmup JIT
    for _ in range(200):
        jarvis.compute_similarity_score(vec1, vec2)
    
    jit_stats = jarvis.jit_compiler.get_compilation_stats()
    print(f"   ✓ JIT speedup: {jit_stats['average_speedup']:.1f}x")
    
    # Performance summary
    print("\n📈 Performance Summary:")
    report = jarvis.get_performance_report()
    print(f"   • Average response time: {report['metrics']['average_response_time']*1000:.1f}ms")
    print(f"   • Cache hit rate: {report['cache']['hit_rate']:.1%}")
    print(f"   • CPU usage: {report['system']['cpu_usage']:.1f}%")
    print(f"   • Memory usage: {report['system']['memory_usage']:.1f}%")


async def interactive_mode(jarvis):
    """Interactive JARVIS session"""
    print("\n💬 Interactive JARVIS Ultra Session")
    print("Type 'help' for commands or 'exit' to quit")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n🗣️  You: ").strip()
            
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'help':
                print_help()
            elif user_input.lower() == 'status':
                print_status(jarvis)
            elif user_input.lower() == 'benchmark':
                results = await jarvis.run_performance_benchmark()
                for test, time_taken in results.items():
                    print(f"  {test}: {time_taken*1000:.1f}ms")
            else:
                # Process as normal input
                result = await jarvis.process_input({'text': user_input}, 'interactive')
                print(f"\n🤖 JARVIS: Processing complete")
                print(f"   State: {result.get('state', 'unknown')}")
                print(f"   Mode: {result.get('response_mode', 'unknown')}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def print_help():
    """Print help for interactive mode"""
    print("""
Available commands:
  help      - Show this help message
  status    - Show current performance status
  benchmark - Run performance benchmark
  exit      - Exit interactive mode
  
Or type any message to interact with JARVIS
""")


def print_status(jarvis):
    """Print current status"""
    report = jarvis.get_performance_report()
    
    print("\n📊 Current Status:")
    print(f"  Requests processed: {report['metrics']['requests_processed']}")
    print(f"  Average response: {report['metrics']['average_response_time']*1000:.1f}ms")
    print(f"  Cache hit rate: {report['cache']['hit_rate']:.1%}")
    print(f"  Loaded modules: {report['modules']['loaded_modules']}/{report['modules']['total_modules']}")
    print(f"  CPU usage: {report['system']['cpu_usage']:.1f}%")
    print(f"  Memory usage: {report['system']['memory_usage']:.1f}%")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="JARVIS Ultra - Phase 10 Performance Optimized AI Assistant"
    )
    
    # Performance options
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable intelligent caching')
    parser.add_argument('--no-lazy', action='store_true',
                       help='Disable lazy loading')
    parser.add_argument('--no-jit', action='store_true',
                       help='Disable JIT compilation')
    
    # Configuration options
    parser.add_argument('--cache-size', type=int, default=100,
                       help='Cache size in MB (default: 100)')
    parser.add_argument('--memory-threshold', type=float, default=0.8,
                       help='Memory threshold for lazy loading (default: 0.8)')
    parser.add_argument('--workload', choices=['real_time', 'batch_processing', 
                                               'memory_constrained', 'gpu_intensive'],
                       help='Optimize for specific workload')
    
    # Runtime options
    parser.add_argument('--demo', action='store_true',
                       help='Run performance demonstration')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive session')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark on startup')
    parser.add_argument('--monitor', action='store_true',
                       help='Start performance monitoring dashboard')
    parser.add_argument('--monitor-port', type=int, default=8889,
                       help='Port for monitoring dashboard (default: 8889)')
    
    args = parser.parse_args()
    
    # Default to demo + monitor if no mode specified
    if not any([args.demo, args.interactive, args.monitor]):
        args.demo = True
        args.monitor = True
    
    # Run
    try:
        asyncio.run(launch_jarvis_ultra(args))
    except KeyboardInterrupt:
        print("\n\n👋 JARVIS Ultra shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()