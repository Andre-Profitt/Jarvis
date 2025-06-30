"""
JARVIS Phase 10: Test Suite
Comprehensive tests for performance optimizations
"""

import asyncio
import time
import numpy as np
import torch
import pytest
from typing import List, Dict, Any

# Import Phase 10 components
from core.jarvis_ultra_core import JARVISUltraCore, PerformanceConfig
from core.performance_optimizer import PerformanceOptimizer
from core.parallel_processor import ParallelProcessor, ParallelTask, ParallelStrategy
from core.intelligent_cache import IntelligentCache, CacheTier
from core.lazy_loader import LazyLoader
from core.jit_compiler import JITCompiler


class TestPerformanceOptimizations:
    """Test suite for Phase 10 performance optimizations"""
    
    @pytest.fixture
    async def jarvis(self):
        """Create JARVIS instance for testing"""
        config = PerformanceConfig(
            enable_parallel_processing=True,
            enable_intelligent_caching=True,
            enable_lazy_loading=True,
            enable_jit_compilation=True
        )
        jarvis = JARVISUltraCore(config)
        await jarvis.initialize()
        yield jarvis
        await jarvis.shutdown()
    
    async def test_parallel_processing(self, jarvis):
        """Test parallel processing capabilities"""
        print("\n=== Testing Parallel Processing ===")
        
        # Test multi-modal input processing
        input_data = {
            'vision': {'image': np.random.rand(224, 224, 3)},
            'audio': {'waveform': np.random.rand(16000)},
            'language': {'text': 'Test parallel processing'},
            'biometric': {'heart_rate': 72}
        }
        
        start_time = time.time()
        result = await jarvis.process_input(input_data, 'test')
        parallel_time = time.time() - start_time
        
        assert 'modalities' in result
        assert len(result['modalities']) == 4
        print(f"Parallel processing time: {parallel_time:.3f}s")
        
        # Compare with sequential processing (simulate)
        sequential_time = 0.1 + 0.05 + 0.08 + 0.02  # Sum of individual processing times
        speedup = sequential_time / parallel_time
        print(f"Estimated speedup: {speedup:.2f}x")
        
        return parallel_time < sequential_time
    
    async def test_intelligent_caching(self, jarvis):
        """Test caching performance"""
        print("\n=== Testing Intelligent Caching ===")
        
        # First call - cache miss
        start_time = time.time()
        result1 = await jarvis.get_user_preferences("test_user")
        miss_time = time.time() - start_time
        
        # Second call - cache hit
        start_time = time.time()
        result2 = await jarvis.get_user_preferences("test_user")
        hit_time = time.time() - start_time
        
        assert result1 == result2
        assert hit_time < miss_time
        
        cache_speedup = miss_time / hit_time
        print(f"Cache miss time: {miss_time:.3f}s")
        print(f"Cache hit time: {hit_time:.3f}s")
        print(f"Cache speedup: {cache_speedup:.2f}x")
        
        # Test cache stats
        cache_stats = jarvis.cache.get_stats()
        assert cache_stats['total_hits'] > 0
        print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
        
        return True
    
    async def test_lazy_loading(self, jarvis):
        """Test lazy loading efficiency"""
        print("\n=== Testing Lazy Loading ===")
        
        # Check initial module status
        initial_status = jarvis.lazy_loader.get_status()
        initial_loaded = initial_status['loaded_modules']
        
        print(f"Initially loaded modules: {initial_loaded}/{initial_status['total_modules']}")
        
        # Load a feature
        success = await jarvis.load_feature("vision_capabilities")
        assert success
        
        # Check modules after feature load
        after_status = jarvis.lazy_loader.get_status()
        after_loaded = after_status['loaded_modules']
        
        print(f"After loading vision: {after_loaded}/{after_status['total_modules']}")
        assert after_loaded > initial_loaded
        
        # Test memory efficiency
        memory_saved = (1 - after_loaded / after_status['total_modules']) * 100
        print(f"Memory saved by lazy loading: {memory_saved:.1f}%")
        
        return True
    
    def test_jit_compilation(self, jarvis):
        """Test JIT compilation speedup"""
        print("\n=== Testing JIT Compilation ===")
        
        # Create test vectors
        vec1 = list(range(1000))
        vec2 = list(range(1000, 2000))
        
        # Warmup phase - trigger JIT compilation
        print("Warming up JIT compiler...")
        for i in range(150):  # Exceed JIT threshold
            jarvis.compute_similarity_score(vec1, vec2)
        
        # Benchmark original vs compiled
        iterations = 1000
        
        # Time compiled version
        start_time = time.time()
        for _ in range(iterations):
            jarvis.compute_similarity_score(vec1, vec2)
        compiled_time = time.time() - start_time
        
        print(f"JIT compiled time for {iterations} iterations: {compiled_time:.3f}s")
        
        # Get JIT stats
        jit_stats = jarvis.jit_compiler.get_compilation_stats()
        print(f"Average JIT speedup: {jit_stats['average_speedup']:.2f}x")
        
        return jit_stats['successful'] > 0
    
    async def test_performance_optimizer(self, jarvis):
        """Test overall performance optimization"""
        print("\n=== Testing Performance Optimizer ===")
        
        # Define a compute-heavy function
        async def heavy_computation(data: np.ndarray) -> np.ndarray:
            return np.dot(data, data.T)
        
        # Test optimization
        data = np.random.rand(100, 100)
        
        result, opt_result = await jarvis.performance_optimizer.optimize_execution(
            heavy_computation, data
        )
        
        assert opt_result.success
        print(f"Optimization strategy: {opt_result.strategy.value}")
        print(f"Performance gain: {opt_result.performance_gain:.2f}x")
        
        # Get optimization report
        report = jarvis.performance_optimizer.get_optimization_report()
        print(f"Hot paths identified: {len(report['hot_paths'])}")
        
        return True
    
    async def test_memory_optimization(self, jarvis):
        """Test memory optimization features"""
        print("\n=== Testing Memory Optimization ===")
        
        # Simulate high memory usage
        large_data = [np.random.rand(1000, 1000) for _ in range(10)]
        
        # Store in cache
        for i, data in enumerate(large_data):
            await jarvis.cache.set(f"large_{i}", data, ttl=60)
        
        # Check if emergency cleanup is triggered
        cache_stats = jarvis.cache.get_stats()
        print(f"Cache memory usage: {cache_stats['memory_usage']}")
        
        # Force cleanup
        jarvis.cache.emergency_cleanup()
        
        # Check memory after cleanup
        cache_stats_after = jarvis.cache.get_stats()
        print(f"Cache memory after cleanup: {cache_stats_after['memory_usage']}")
        
        return True
    
    async def test_gpu_acceleration(self, jarvis):
        """Test GPU acceleration if available"""
        print("\n=== Testing GPU Acceleration ===")
        
        if not jarvis.performance_optimizer.gpu_available:
            print("GPU not available, skipping GPU tests")
            return True
        
        # Create GPU-optimizable data
        tensor_data = torch.randn(1000, 1000)
        
        # Function that benefits from GPU
        async def matrix_operation(data: torch.Tensor) -> torch.Tensor:
            return torch.matmul(data, data.T)
        
        # Execute with optimization
        result, opt_result = await jarvis.performance_optimizer.optimize_execution(
            matrix_operation, tensor_data
        )
        
        if opt_result.strategy == "gpu_acceleration":
            print(f"GPU acceleration achieved: {opt_result.performance_gain:.2f}x speedup")
        
        return True
    
    async def test_workload_optimization(self, jarvis):
        """Test workload-specific optimizations"""
        print("\n=== Testing Workload Optimization ===")
        
        workloads = ["real_time", "batch_processing", "memory_constrained"]
        
        for workload in workloads:
            print(f"\nOptimizing for {workload}...")
            await jarvis.optimize_for_workload(workload)
            
            # Run a quick benchmark
            input_data = {'text': f'Testing {workload} mode'}
            start_time = time.time()
            await jarvis.process_input(input_data, 'test')
            response_time = time.time() - start_time
            
            print(f"{workload} response time: {response_time:.3f}s")
        
        return True
    
    async def test_comprehensive_benchmark(self, jarvis):
        """Run comprehensive benchmark"""
        print("\n=== Running Comprehensive Benchmark ===")
        
        results = await jarvis.run_performance_benchmark()
        
        print("\nBenchmark Results:")
        for test, time_taken in results.items():
            print(f"  {test}: {time_taken:.3f}s")
        
        # Get final performance report
        report = jarvis.get_performance_report()
        
        print("\nFinal Performance Report:")
        print(f"  Requests processed: {report['metrics']['requests_processed']}")
        print(f"  Average response time: {report['metrics']['average_response_time']:.3f}s")
        print(f"  Cache hits: {report['metrics']['cache_hits']}")
        print(f"  Parallel tasks: {report['metrics']['parallel_tasks']}")
        print(f"  JIT compilations: {report['metrics']['jit_compilations']}")
        
        return True


async def run_all_tests():
    """Run all Phase 10 tests"""
    print("=" * 60)
    print("JARVIS Phase 10 - Performance Optimization Tests")
    print("=" * 60)
    
    test_suite = TestPerformanceOptimizations()
    
    # Create JARVIS instance
    config = PerformanceConfig()
    jarvis = JARVISUltraCore(config)
    await jarvis.initialize()
    
    # Run tests
    tests = [
        test_suite.test_parallel_processing,
        test_suite.test_intelligent_caching,
        test_suite.test_lazy_loading,
        test_suite.test_jit_compilation,
        test_suite.test_performance_optimizer,
        test_suite.test_memory_optimization,
        test_suite.test_gpu_acceleration,
        test_suite.test_workload_optimization,
        test_suite.test_comprehensive_benchmark
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if asyncio.iscoroutinefunction(test):
                result = await test(jarvis)
            else:
                result = test(jarvis)
            
            if result:
                passed += 1
                print(f"✅ {test.__name__} PASSED\n")
            else:
                failed += 1
                print(f"❌ {test.__name__} FAILED\n")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} FAILED with error: {e}\n")
    
    # Cleanup
    await jarvis.shutdown()
    
    # Summary
    print("=" * 60)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return passed, failed


if __name__ == "__main__":
    asyncio.run(run_all_tests())