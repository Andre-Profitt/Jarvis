"""
JARVIS Phase 10: Enhanced Core with Performance Optimizations
Integrates all performance optimization components
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import psutil

# Import all Phase 10 components
from .performance_optimizer import PerformanceOptimizer, optimize
from .parallel_processor import ParallelProcessor, ParallelTask, ParallelStrategy
from .intelligent_cache import IntelligentCache, cached, CacheTier
from .lazy_loader import LazyLoader, lazy_load
from .jit_compiler import JITCompiler, auto_compile

# Import previous phase components (1-9)
from .unified_input_pipeline import UnifiedInputPipeline
from .fluid_state_management import FluidStateManager
from .jarvis_enhanced_core import JARVISEnhancedCore as JARVISBaseCore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations"""
    enable_parallel_processing: bool = True
    enable_intelligent_caching: bool = True
    enable_lazy_loading: bool = True
    enable_jit_compilation: bool = True
    cache_size_mb: int = 100
    parallel_workers: Optional[int] = None
    memory_threshold: float = 0.8
    jit_threshold: int = 100


class JARVISUltraCore(JARVISBaseCore):
    """
    JARVIS Enhanced Core with Phase 10 Performance Optimizations
    Blazing fast with intelligent resource management
    """
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        # Initialize base functionality
        super().__init__()
        
        # Performance configuration
        self.perf_config = config or PerformanceConfig()
        
        # Initialize performance components
        self.performance_optimizer = PerformanceOptimizer(self)
        
        if self.perf_config.enable_parallel_processing:
            self.parallel_processor = ParallelProcessor(
                max_workers=self.perf_config.parallel_workers
            )
        
        if self.perf_config.enable_intelligent_caching:
            self.cache = IntelligentCache(
                max_memory_size=self.perf_config.cache_size_mb * 1024 * 1024
            )
        
        if self.perf_config.enable_lazy_loading:
            self.lazy_loader = LazyLoader(
                memory_threshold=self.perf_config.memory_threshold
            )
        
        if self.perf_config.enable_jit_compilation:
            self.jit_compiler = JITCompiler()
            # Apply JIT to hot functions
            self._apply_jit_optimization()
        
        # Performance metrics
        self.performance_metrics = {
            'requests_processed': 0,
            'average_response_time': 0.0,
            'cache_hits': 0,
            'parallel_tasks': 0,
            'jit_compilations': 0
        }
        
        logger.info("JARVIS Ultra Core initialized with Phase 10 optimizations")
    
    @optimize  # Apply performance optimization decorator
    async def process_input(self, input_data: Dict[str, Any], 
                          source: str = "unknown") -> Dict[str, Any]:
        """
        Ultra-optimized input processing
        """
        start_time = time.time()
        
        # Check cache first
        if self.perf_config.enable_intelligent_caching:
            cache_key = self._generate_cache_key(input_data, source)
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                self.performance_metrics['cache_hits'] += 1
                return cached_result
        
        # Determine if parallel processing would help
        if self._should_use_parallel(input_data):
            result = await self._process_parallel(input_data, source)
        else:
            # Use base processing
            result = await super().process_input(input_data, source)
        
        # Cache result
        if self.perf_config.enable_intelligent_caching:
            await self.cache.set(cache_key, result, ttl=300)  # 5 min TTL
        
        # Update metrics
        response_time = time.time() - start_time
        self._update_performance_metrics(response_time)
        
        return result
    
    async def _process_parallel(self, input_data: Dict[str, Any], 
                              source: str) -> Dict[str, Any]:
        """
        Process input using parallel processing
        """
        # Use parallel processor for multi-modal input
        if hasattr(self, 'parallel_processor'):
            result = await self.parallel_processor.process_multimodal_input(input_data)
            self.performance_metrics['parallel_tasks'] += 1
            
            # Merge with state processing
            state_result = await self.state_manager.process_input({
                'modalities': result['modalities']
            })
            
            return {
                **result,
                'state': state_result['current_state'],
                'response_mode': state_result['response_mode']
            }
        else:
            return await super().process_input(input_data, source)
    
    @lazy_load("vision.advanced_vision")
    async def process_vision_heavy(self, image_data: Any) -> Dict[str, Any]:
        """
        Heavy vision processing with lazy loading
        """
        # Vision module will be loaded only when needed
        vision_module = self.lazy_loader.get_module("vision.advanced_vision")
        return await vision_module.process_image(image_data)
    
    @cached(ttl=3600, tier=CacheTier.L2_REDIS)  # Cache for 1 hour
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get user preferences with intelligent caching
        """
        # This would normally be a database call
        # Cached to avoid repeated lookups
        return {
            'user_id': user_id,
            'preferences': {
                'response_style': 'concise',
                'proactive_level': 0.7,
                'notification_threshold': 0.8
            }
        }
    
    @auto_compile()  # Apply JIT compilation
    def compute_similarity_score(self, vec1: List[float], 
                                vec2: List[float]) -> float:
        """
        Compute similarity score with JIT optimization
        """
        # This function will be JIT compiled after threshold
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0
    
    def _should_use_parallel(self, input_data: Dict[str, Any]) -> bool:
        """
        Determine if input should be processed in parallel
        """
        if not self.perf_config.enable_parallel_processing:
            return False
        
        # Use parallel for multi-modal input
        modality_count = sum(1 for k in input_data.keys() 
                           if k in ['vision', 'audio', 'language', 'biometric'])
        
        return modality_count >= 2
    
    def _generate_cache_key(self, input_data: Dict[str, Any], 
                          source: str) -> str:
        """
        Generate cache key for input
        """
        # Simplified key generation
        import hashlib
        import json
        
        key_data = {
            'source': source,
            'data_keys': sorted(input_data.keys()),
            'timestamp': int(time.time() / 60)  # 1-minute buckets
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _update_performance_metrics(self, response_time: float):
        """
        Update performance metrics
        """
        self.performance_metrics['requests_processed'] += 1
        
        # Update average response time (exponential moving average)
        alpha = 0.1
        current_avg = self.performance_metrics['average_response_time']
        self.performance_metrics['average_response_time'] = (
            alpha * response_time + (1 - alpha) * current_avg
        )
    
    def _apply_jit_optimization(self):
        """
        Apply JIT optimization to hot methods
        """
        # List of methods to optimize
        hot_methods = [
            self.compute_similarity_score,
            # Add more methods that are compute-intensive
        ]
        
        for method in hot_methods:
            optimized = self.jit_compiler.profile_and_compile(method)
            # Replace with optimized version
            setattr(self, method.__name__, optimized)
    
    async def load_feature(self, feature_name: str) -> bool:
        """
        Load a feature with all its required modules
        """
        if hasattr(self, 'lazy_loader'):
            return await self.lazy_loader.load_feature(feature_name)
        return True
    
    async def optimize_for_workload(self, workload_type: str):
        """
        Optimize JARVIS for specific workload types
        """
        logger.info(f"Optimizing for {workload_type} workload")
        
        if workload_type == "real_time":
            # Optimize for low latency
            self.perf_config.enable_intelligent_caching = True
            self.perf_config.enable_jit_compilation = True
            # Preload critical modules
            await self.lazy_loader.preload_essential()
            
        elif workload_type == "batch_processing":
            # Optimize for throughput
            self.perf_config.enable_parallel_processing = True
            # Increase worker pool
            self.parallel_processor = ParallelProcessor(max_workers=psutil.cpu_count() * 2)
            
        elif workload_type == "memory_constrained":
            # Optimize for memory efficiency
            self.perf_config.enable_lazy_loading = True
            self.perf_config.memory_threshold = 0.6
            # Reduce cache size
            self.cache.max_memory_size = 50 * 1024 * 1024  # 50MB
            
        elif workload_type == "gpu_intensive":
            # Optimize for GPU workloads
            if self.performance_optimizer.gpu_available:
                logger.info("GPU optimization enabled")
                # GPU-specific optimizations would go here
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report
        """
        report = {
            'metrics': self.performance_metrics,
            'system': {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'active_threads': threading.active_count()
            }
        }
        
        # Add component-specific reports
        if hasattr(self, 'performance_optimizer'):
            report['optimizer'] = self.performance_optimizer.get_optimization_report()
        
        if hasattr(self, 'parallel_processor'):
            report['parallel'] = self.parallel_processor.get_performance_report()
        
        if hasattr(self, 'cache'):
            report['cache'] = self.cache.get_stats()
        
        if hasattr(self, 'lazy_loader'):
            report['modules'] = self.lazy_loader.get_status()
        
        if hasattr(self, 'jit_compiler'):
            report['jit'] = self.jit_compiler.get_compilation_stats()
        
        return report
    
    async def run_performance_benchmark(self) -> Dict[str, Any]:
        """
        Run performance benchmark
        """
        logger.info("Running performance benchmark...")
        
        benchmarks = {}
        
        # Test 1: Single request latency
        start = time.time()
        await self.process_input({'text': 'Hello JARVIS'}, 'benchmark')
        benchmarks['single_request_latency'] = time.time() - start
        
        # Test 2: Parallel processing
        if hasattr(self, 'parallel_processor'):
            start = time.time()
            tasks = []
            for i in range(10):
                task = ParallelTask(
                    task_id=f"bench_{i}",
                    function=lambda x: x * x,
                    args=(i,),
                    kwargs={},
                    strategy=ParallelStrategy.THREAD
                )
                tasks.append(task)
            
            await self.parallel_processor.execute_parallel_tasks(tasks)
            benchmarks['parallel_10_tasks'] = time.time() - start
        
        # Test 3: Cache performance
        if hasattr(self, 'cache'):
            # First call (miss)
            start = time.time()
            await self.get_user_preferences("test_user")
            benchmarks['cache_miss_latency'] = time.time() - start
            
            # Second call (hit)
            start = time.time()
            await self.get_user_preferences("test_user")
            benchmarks['cache_hit_latency'] = time.time() - start
        
        # Test 4: JIT compilation benefit
        if hasattr(self, 'jit_compiler'):
            vec1 = list(range(1000))
            vec2 = list(range(1000, 2000))
            
            # Warmup
            for _ in range(200):  # Exceed JIT threshold
                self.compute_similarity_score(vec1, vec2)
            
            # Benchmark
            start = time.time()
            for _ in range(1000):
                self.compute_similarity_score(vec1, vec2)
            benchmarks['jit_1000_computations'] = time.time() - start
        
        return benchmarks
    
    async def shutdown(self):
        """
        Graceful shutdown with cleanup
        """
        logger.info("Shutting down JARVIS Ultra Core")
        
        # Shutdown all components
        if hasattr(self, 'performance_optimizer'):
            await self.performance_optimizer.shutdown()
        
        if hasattr(self, 'parallel_processor'):
            await self.parallel_processor.shutdown()
        
        if hasattr(self, 'cache'):
            await self.cache.shutdown()
        
        if hasattr(self, 'lazy_loader'):
            await self.lazy_loader.shutdown()
        
        await super().shutdown()


# Global instance management
_jarvis_instance = None

def get_jarvis_instance() -> JARVISUltraCore:
    """Get global JARVIS instance"""
    global _jarvis_instance
    if _jarvis_instance is None:
        _jarvis_instance = JARVISUltraCore()
    return _jarvis_instance


# Example usage
async def demo_ultra_performance():
    """Demonstrate Ultra Core performance"""
    jarvis = JARVISUltraCore()
    
    # Initialize
    await jarvis.initialize()
    
    # Run benchmark
    print("Running performance benchmark...")
    benchmark_results = await jarvis.run_performance_benchmark()
    
    print("\nBenchmark Results:")
    for test, result in benchmark_results.items():
        print(f"{test}: {result:.3f}s")
    
    # Get performance report
    report = jarvis.get_performance_report()
    
    print("\nPerformance Report:")
    print(f"Requests processed: {report['metrics']['requests_processed']}")
    print(f"Average response time: {report['metrics']['average_response_time']:.3f}s")
    print(f"Cache hit rate: {report['cache']['hit_rate']:.2%}")
    
    # Cleanup
    await jarvis.shutdown()


if __name__ == "__main__":
    asyncio.run(demo_ultra_performance())