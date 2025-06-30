"""
JARVIS Phase 9: Test Suite
==========================
Comprehensive tests for performance optimization components
"""

import asyncio
import pytest
import time
import numpy as np
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase9.performance_optimizer import (
    IntelligentCache,
    ParallelProcessor,
    LazyLoader,
    PerformanceMonitor,
    JARVISPerformanceOptimizer,
    cached
)
from phase9.jarvis_phase9_integration import (
    JARVISOptimizedCore,
    optimize_jarvis_function
)


# ==================== Cache Tests ====================

class TestIntelligentCache:
    """Test intelligent caching system"""
    
    @pytest.mark.asyncio
    async def test_basic_caching(self):
        """Test basic cache operations"""
        cache = IntelligentCache(memory_limit_mb=10)
        
        # Test set and get
        await cache.set("test_key", {"data": "test_value"}, ttl_seconds=60)
        result = await cache.get("test_key")
        
        assert result is not None
        assert result["data"] == "test_value"
        
        # Test cache miss
        miss_result = await cache.get("non_existent_key")
        assert miss_result is None
        
        # Test stats
        stats = cache.stats.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        """Test LRU eviction"""
        cache = IntelligentCache(memory_limit_mb=1)  # Very small cache
        
        # Fill cache
        for i in range(100):
            await cache.set(f"key_{i}", f"value_{i}" * 1000)  # Large values
        
        # Check evictions occurred
        stats = cache.stats.get_stats()
        assert stats['evictions'] > 0
        assert len(cache.memory_cache) < 100
    
    @pytest.mark.asyncio
    async def test_cache_compression(self):
        """Test data compression"""
        cache = IntelligentCache(compression_enabled=True)
        
        # Large data that should be compressed
        large_data = "x" * 10000
        await cache.set("large_key", large_data)
        
        # Verify it's stored and retrieved correctly
        result = await cache.get("large_key")
        assert result == large_data
    
    @pytest.mark.asyncio
    async def test_cache_decorator(self):
        """Test caching decorator"""
        call_count = 0
        
        @cached(ttl_seconds=60)
        async def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return x * 2
        
        # First call
        result1 = await expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call (should be cached)
        result2 = await expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented
        
        # Different argument
        result3 = await expensive_function(10)
        assert result3 == 20
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_pattern_analysis(self):
        """Test access pattern analysis"""
        cache = IntelligentCache()
        
        # Generate access pattern
        for _ in range(20):
            await cache.get("hot_key")
            await cache.set("hot_key", "value")
        
        for i in range(5):
            await cache.get(f"cold_key_{i}")
        
        patterns = cache.analyze_patterns()
        
        assert 'hot_keys' in patterns
        assert patterns['cache_efficiency'] >= 0
        assert patterns['total_unique_keys'] > 0


# ==================== Parallel Processing Tests ====================

class TestParallelProcessor:
    """Test parallel processing system"""
    
    @pytest.mark.asyncio
    async def test_parallel_map(self):
        """Test parallel mapping"""
        processor = ParallelProcessor(max_workers=4)
        
        def square(x):
            return x * x
        
        items = list(range(100))
        
        # Parallel processing
        start = time.time()
        results = await processor.map_async(square, items)
        parallel_time = time.time() - start
        
        assert len(results) == 100
        assert results[10] == 100  # 10 * 10
        
        processor.shutdown()
    
    @pytest.mark.asyncio
    async def test_async_parallel_processing(self):
        """Test async function parallel processing"""
        processor = ParallelProcessor()
        
        async def async_square(x):
            await asyncio.sleep(0.01)
            return x * x
        
        items = list(range(20))
        results = await processor.map_async(async_square, items)
        
        assert len(results) == 20
        assert results[5] == 25
        
        processor.shutdown()
    
    @pytest.mark.asyncio
    async def test_pipeline_processing(self):
        """Test pipeline execution"""
        processor = ParallelProcessor()
        
        def add_one(x):
            return x + 1
        
        def multiply_two(x):
            return x * 2
        
        def square(x):
            return x * x
        
        # Pipeline: (5 + 1) * 2 ^ 2 = 12 ^ 2 = 144
        result = await processor.pipeline([add_one, multiply_two, square], 5)
        assert result == 144
        
        processor.shutdown()
    
    @pytest.mark.asyncio
    async def test_adaptive_scaling(self):
        """Test adaptive worker scaling"""
        processor = ParallelProcessor(adaptive_scaling=True)
        
        # Get initial workers
        initial_workers = processor.current_workers
        
        # Simulate high load
        async def heavy_task(x):
            await asyncio.sleep(0.1)
            return x
        
        # Process many items
        await processor.map_async(heavy_task, list(range(50)))
        
        # Check performance stats
        stats = processor.get_performance_stats()
        assert 'average_time' in stats
        assert 'current_workers' in stats
        assert stats['current_workers'] > 0
        
        processor.shutdown()


# ==================== Lazy Loading Tests ====================

class TestLazyLoader:
    """Test lazy loading system"""
    
    @pytest.mark.asyncio
    async def test_basic_loading(self):
        """Test basic module loading"""
        loader = LazyLoader()
        await loader.start_prefetcher()
        
        # Load module
        module = await loader.load('jarvis.nlp')
        assert module is not None
        assert 'analyzer' in module
        
        # Second load should be faster (cached)
        start = time.time()
        module2 = await loader.load('jarvis.nlp')
        load_time = time.time() - start
        
        assert load_time < 0.05  # Should be very fast
        assert module2 == module
        
        # Check stats
        stats = loader.get_stats()
        assert stats['loaded_modules'] == 1
        assert loader.access_counts['jarvis.nlp'] == 2
    
    @pytest.mark.asyncio
    async def test_prefetching(self):
        """Test predictive prefetching"""
        loader = LazyLoader()
        await loader.start_prefetcher()
        
        # Load module that triggers prefetching
        await loader.load('jarvis.nlp')
        
        # Give prefetcher time to work
        await asyncio.sleep(0.2)
        
        # Related modules should be loading/loaded
        assert loader.prefetch_queue.qsize() > 0 or len(loader.loaded_modules) > 1
    
    @pytest.mark.asyncio
    async def test_module_unloading(self):
        """Test module unloading"""
        loader = LazyLoader()
        
        # Load and then unload
        await loader.load('jarvis.vision')
        assert 'jarvis.vision' in loader.loaded_modules
        
        await loader.unload('jarvis.vision')
        assert 'jarvis.vision' not in loader.loaded_modules


# ==================== Performance Monitor Tests ====================

class TestPerformanceMonitor:
    """Test performance monitoring"""
    
    @pytest.mark.asyncio
    async def test_monitoring_start_stop(self):
        """Test monitor lifecycle"""
        monitor = PerformanceMonitor()
        
        await monitor.start_monitoring()
        assert monitor._monitoring is True
        
        # Let it collect some data
        await asyncio.sleep(1)
        
        await monitor.stop_monitoring()
        assert monitor._monitoring is False
    
    @pytest.mark.asyncio
    async def test_metric_recording(self):
        """Test metric recording"""
        monitor = PerformanceMonitor()
        
        # Record metrics
        monitor.record_response_time('test_op', 150.5)
        monitor.record_cache_performance({'hit_rate': 0.85})
        monitor.record_parallel_efficiency(0.92)
        
        # Get dashboard data
        dashboard = monitor.get_dashboard_data()
        
        assert 'current_metrics' in dashboard
        assert 'trends' in dashboard
        assert 'recommendations' in dashboard
    
    @pytest.mark.asyncio
    async def test_alert_generation(self):
        """Test alert generation"""
        monitor = PerformanceMonitor()
        await monitor.start_monitoring()
        
        # Simulate high CPU
        for _ in range(15):
            monitor.metrics['cpu_usage'].append(85.0)
        
        # Trigger analysis
        monitor._analyze_performance({'cpu_percent': 85.0})
        
        # Check alerts
        assert len(monitor.alerts) > 0
        assert any(alert['type'] == 'high_cpu' for alert in monitor.alerts)
        
        await monitor.stop_monitoring()


# ==================== Integration Tests ====================

class TestJARVISPerformanceOptimizer:
    """Test main performance optimizer"""
    
    @pytest.mark.asyncio
    async def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        optimizer = JARVISPerformanceOptimizer()
        await optimizer.initialize()
        
        assert optimizer._running is True
        assert optimizer.cache is not None
        assert optimizer.parallel_processor is not None
        assert optimizer.lazy_loader is not None
        assert optimizer.monitor is not None
        
        await optimizer.shutdown()
    
    @pytest.mark.asyncio
    async def test_optimize_operation(self):
        """Test operation optimization"""
        optimizer = JARVISPerformanceOptimizer()
        await optimizer.initialize()
        
        call_count = 0
        
        async def test_operation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return x * 3
        
        # First call
        result1 = await optimizer.optimize_operation(test_operation, 7)
        assert result1 == 21
        assert call_count == 1
        
        # Second call (should be cached)
        result2 = await optimizer.optimize_operation(test_operation, 7)
        assert result2 == 21
        assert call_count == 1  # Not incremented
        
        await optimizer.shutdown()
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing optimization"""
        optimizer = JARVISPerformanceOptimizer()
        await optimizer.initialize()
        
        async def process_item(item: int) -> int:
            await asyncio.sleep(0.01)
            return item * 2
        
        items = list(range(50))
        
        start = time.time()
        results = await optimizer.batch_process(items, process_item)
        elapsed = time.time() - start
        
        assert len(results) == 50
        assert results[25] == 50
        assert elapsed < 0.5  # Should be much faster than sequential (0.5s)
        
        await optimizer.shutdown()
    
    @pytest.mark.asyncio
    async def test_optimization_levels(self):
        """Test different optimization levels"""
        optimizer = JARVISPerformanceOptimizer()
        await optimizer.initialize()
        
        # Test level changes
        optimizer.set_optimization_level('aggressive')
        assert optimizer.optimization_level == 'aggressive'
        assert optimizer.cache.memory_limit > 1024 * 1024 * 1024
        
        optimizer.set_optimization_level('conservative')
        assert optimizer.optimization_level == 'conservative'
        assert optimizer.cache.memory_limit < 1024 * 1024 * 1024
        
        await optimizer.shutdown()
    
    @pytest.mark.asyncio
    async def test_performance_report(self):
        """Test performance reporting"""
        optimizer = JARVISPerformanceOptimizer()
        await optimizer.initialize()
        
        # Generate some activity
        for i in range(10):
            await optimizer.optimize_operation(lambda x: x + 1, i)
        
        report = optimizer.get_performance_report()
        
        assert 'cache_stats' in report
        assert 'parallel_stats' in report
        assert 'lazy_loading_stats' in report
        assert 'monitor_dashboard' in report
        assert 'optimization_level' in report
        
        await optimizer.shutdown()


# ==================== JARVIS Integration Tests ====================

class TestJARVISOptimizedCore:
    """Test JARVIS core integration"""
    
    @pytest.mark.asyncio
    async def test_optimized_core_initialization(self):
        """Test optimized core initialization"""
        core = JARVISOptimizedCore()
        await core.initialize()
        
        assert core._initialized is True
        assert core.optimizer is not None
        
        await core.shutdown()
    
    @pytest.mark.asyncio
    async def test_batch_processing_integration(self):
        """Test batch processing with JARVIS core"""
        core = JARVISOptimizedCore()
        await core.initialize()
        
        inputs = [
            {'type': 'test', 'data': f'Input {i}'}
            for i in range(20)
        ]
        
        results = await core.process_batch(inputs)
        
        assert len(results) == 20
        assert all('processed' in r or 'error' in r for r in results)
        
        # Check metrics
        assert core.performance_metrics['parallel_operations'] > 0
        
        await core.shutdown()
    
    @pytest.mark.asyncio
    async def test_optimization_report_integration(self):
        """Test optimization reporting with JARVIS metrics"""
        core = JARVISOptimizedCore()
        await core.initialize()
        
        # Generate some activity
        await core.process_batch([{'test': i} for i in range(10)])
        
        report = core.get_optimization_report()
        
        assert 'jarvis_metrics' in report
        assert 'optimization_impact' in report
        
        await core.shutdown()


# ==================== Performance Benchmarks ====================

class BenchmarkTests:
    """Performance benchmark tests"""
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Benchmark cache performance"""
        cache = IntelligentCache()
        
        # Write performance
        start = time.time()
        for i in range(1000):
            await cache.set(f"key_{i}", f"value_{i}")
        write_time = time.time() - start
        
        # Read performance (all hits)
        start = time.time()
        for i in range(1000):
            await cache.get(f"key_{i}")
        read_hit_time = time.time() - start
        
        # Read performance (all misses)
        start = time.time()
        for i in range(1000):
            await cache.get(f"missing_key_{i}")
        read_miss_time = time.time() - start
        
        print(f"\nCache Performance:")
        print(f"  Writes: {1000/write_time:.0f} ops/sec")
        print(f"  Read hits: {1000/read_hit_time:.0f} ops/sec")
        print(f"  Read misses: {1000/read_miss_time:.0f} ops/sec")
        
        # Assert reasonable performance
        assert write_time < 1.0  # Should handle 1000+ writes/sec
        assert read_hit_time < 0.5  # Should handle 2000+ reads/sec
    
    @pytest.mark.asyncio
    async def test_parallel_speedup(self):
        """Benchmark parallel processing speedup"""
        processor = ParallelProcessor()
        
        def cpu_bound_task(x):
            # Simulate CPU work
            result = 0
            for i in range(10000):
                result += x * i
            return result
        
        items = list(range(100))
        
        # Sequential baseline
        start = time.time()
        sequential_results = [cpu_bound_task(x) for x in items]
        sequential_time = time.time() - start
        
        # Parallel processing
        start = time.time()
        parallel_results = await processor.map_async(cpu_bound_task, items)
        parallel_time = time.time() - start
        
        speedup = sequential_time / parallel_time
        
        print(f"\nParallel Processing Speedup:")
        print(f"  Sequential: {sequential_time:.2f}s")
        print(f"  Parallel: {parallel_time:.2f}s")
        print(f"  Speedup: {speedup:.1f}x")
        
        assert speedup > 1.5  # At least 1.5x speedup
        assert sequential_results == parallel_results  # Same results
        
        processor.shutdown()


# ==================== Test Runner ====================

async def run_all_tests():
    """Run all tests and display results"""
    print("\n" + "="*60)
    print("JARVIS Phase 9 - Test Suite")
    print("="*60 + "\n")
    
    test_classes = [
        TestIntelligentCache,
        TestParallelProcessor,
        TestLazyLoader,
        TestPerformanceMonitor,
        TestJARVISPerformanceOptimizer,
        TestJARVISOptimizedCore,
        BenchmarkTests
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 40)
        
        test_instance = test_class()
        test_methods = [
            method for method in dir(test_instance)
            if method.startswith('test_')
        ]
        
        for method_name in test_methods:
            total_tests += 1
            method = getattr(test_instance, method_name)
            
            try:
                if asyncio.iscoroutinefunction(method):
                    await method()
                else:
                    method()
                
                print(f"  ✅ {method_name}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ❌ {method_name}: {str(e)}")
                failed_tests.append((test_class.__name__, method_name, str(e)))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFailed Tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")
    else:
        print("\n✅ All tests passed!")
    
    print("\n" + "="*60)
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_all_tests())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
