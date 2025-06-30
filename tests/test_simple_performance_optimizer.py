#!/usr/bin/env python3
"""
Comprehensive test suite for Simple Performance Optimizer
Tests caching, monitoring, connection pooling, and decorators
"""

import pytest
import asyncio
import time
import threading
import sqlite3
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import hashlib
import json

from core.simple_performance_optimizer import (
    RequestMetrics,
    PerformanceMonitor,
    MemoryCache,
    DatabasePool,
    OptimizedRequestProcessor,
    cached,
    track_performance,
    monitor,
    cache,
    warm_cache,
    clear_cache,
    get_performance_stats
)


# Test fixtures
@pytest.fixture
def perf_monitor():
    """Create a clean PerformanceMonitor instance"""
    return PerformanceMonitor()


@pytest.fixture
def memory_cache():
    """Create a clean MemoryCache instance"""
    return MemoryCache()


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        db_path = f.name
    
    # Initialize database with test table
    conn = sqlite3.connect(db_path)
    conn.execute('''
        CREATE TABLE test_data (
            id INTEGER PRIMARY KEY,
            name TEXT,
            value INTEGER
        )
    ''')
    conn.execute("INSERT INTO test_data (name, value) VALUES ('test1', 100)")
    conn.execute("INSERT INTO test_data (name, value) VALUES ('test2', 200)")
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup
    try:
        os.unlink(db_path)
    except:
        pass


@pytest.fixture
def db_pool(temp_db):
    """Create a DatabasePool instance"""
    pool = DatabasePool(temp_db, pool_size=3)
    yield pool
    pool.close()


@pytest.fixture
def request_processor(temp_db):
    """Create an OptimizedRequestProcessor instance"""
    return OptimizedRequestProcessor(temp_db)


# Test RequestMetrics
class TestRequestMetrics:
    def test_metrics_creation(self):
        """Test RequestMetrics creation and completion"""
        metric = RequestMetrics(
            request_id="test123",
            request_type="api_call",
            start_time=time.time()
        )
        
        assert metric.request_id == "test123"
        assert metric.request_type == "api_call"
        assert metric.duration is None
        assert metric.cache_hit is False
        
        # Complete the metric
        time.sleep(0.1)
        metric.complete()
        
        assert metric.end_time is not None
        assert metric.duration is not None
        assert metric.duration >= 0.1
    
    def test_metrics_with_error(self):
        """Test RequestMetrics with error tracking"""
        metric = RequestMetrics(
            request_id="error_test",
            request_type="database",
            start_time=time.time()
        )
        
        metric.error = "Connection failed"
        metric.complete()
        
        assert metric.error == "Connection failed"
        assert metric.duration is not None


# Test PerformanceMonitor
class TestPerformanceMonitor:
    def test_request_tracking(self, perf_monitor):
        """Test basic request tracking"""
        # Start a request
        metric = perf_monitor.start_request("req1", "api")
        
        assert "req1" in perf_monitor.active_requests
        assert metric.request_type == "api"
        
        # Complete the request
        time.sleep(0.05)
        perf_monitor.complete_request("req1")
        
        assert "req1" not in perf_monitor.active_requests
        assert len(perf_monitor.metrics) == 1
        assert perf_monitor.metrics[0].duration >= 0.05
    
    def test_concurrent_requests(self, perf_monitor):
        """Test tracking multiple concurrent requests"""
        # Start multiple requests
        metrics = []
        for i in range(5):
            metric = perf_monitor.start_request(f"req{i}", f"type{i%2}")
            metrics.append(metric)
        
        assert len(perf_monitor.active_requests) == 5
        
        # Complete them in different order
        perf_monitor.complete_request("req2")
        perf_monitor.complete_request("req0")
        perf_monitor.complete_request("req4")
        
        assert len(perf_monitor.active_requests) == 2
        assert len(perf_monitor.metrics) == 3
    
    def test_get_stats_empty(self, perf_monitor):
        """Test getting stats with no metrics"""
        stats = perf_monitor.get_stats()
        
        assert stats["message"] == "No metrics available"
    
    def test_get_stats_with_data(self, perf_monitor):
        """Test getting stats with metrics"""
        # Create some test metrics
        for i in range(10):
            metric = perf_monitor.start_request(f"req{i}", "fast" if i < 5 else "slow")
            time.sleep(0.01 if i < 5 else 0.05)
            
            if i == 7:
                metric.cache_hit = True
            
            perf_monitor.complete_request(f"req{i}")
        
        stats = perf_monitor.get_stats()
        
        assert stats["total_requests"] == 10
        assert stats["average_duration"] > 0
        assert stats["median_duration"] > 0
        assert stats["cache_hit_rate"] == 0.1  # 1 out of 10
        assert "by_type" in stats
        assert "fast" in stats["by_type"]
        assert "slow" in stats["by_type"]
        assert stats["by_type"]["fast"]["count"] == 5
        assert stats["by_type"]["slow"]["count"] == 5
    
    def test_thread_safety(self, perf_monitor):
        """Test thread safety of performance monitor"""
        def worker(worker_id):
            for i in range(10):
                req_id = f"worker{worker_id}_req{i}"
                perf_monitor.start_request(req_id, "concurrent")
                time.sleep(0.001)
                perf_monitor.complete_request(req_id)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(perf_monitor.metrics) == 50  # 5 workers * 10 requests each


# Test MemoryCache
class TestMemoryCache:
    def test_basic_cache_operations(self, memory_cache):
        """Test basic get/set operations"""
        # Set value
        memory_cache.set("key1", "value1")
        
        # Get value
        result = memory_cache.get("key1")
        assert result == "value1"
        
        # Get non-existent key
        result = memory_cache.get("nonexistent")
        assert result is None
    
    def test_cache_ttl(self, memory_cache):
        """Test cache TTL functionality"""
        # Set with short TTL
        memory_cache.set("expires_fast", "data", ttl=0.1)  # 100ms
        
        # Should exist immediately
        assert memory_cache.get("expires_fast") == "data"
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Should be expired
        assert memory_cache.get("expires_fast") is None
    
    def test_cache_ttl_types(self, memory_cache):
        """Test different TTL type settings"""
        # String TTL
        memory_cache.set("short_ttl", "data1", ttl="short")
        memory_cache.set("medium_ttl", "data2", ttl="medium")
        memory_cache.set("long_ttl", "data3", ttl="long")
        
        assert memory_cache.get("short_ttl") == "data1"
        assert memory_cache.get("medium_ttl") == "data2"
        assert memory_cache.get("long_ttl") == "data3"
        
        # Integer TTL
        memory_cache.set("custom_ttl", "data4", ttl=3600)
        assert memory_cache.get("custom_ttl") == "data4"
    
    def test_cache_hit_tracking(self, memory_cache):
        """Test cache hit tracking"""
        memory_cache.set("tracked", "value")
        
        # Multiple accesses
        for _ in range(5):
            memory_cache.get("tracked")
        
        # Check hit count
        entry = memory_cache.cache["tracked"]
        assert entry["hits"] == 5
        assert "last_accessed" in entry
    
    def test_cache_invalidation(self, memory_cache):
        """Test cache invalidation"""
        # Set multiple keys
        memory_cache.set("user:1", "data1")
        memory_cache.set("user:2", "data2")
        memory_cache.set("post:1", "data3")
        
        # Invalidate by pattern
        memory_cache.invalidate("user:")
        
        assert memory_cache.get("user:1") is None
        assert memory_cache.get("user:2") is None
        assert memory_cache.get("post:1") == "data3"
        
        # Invalidate all
        memory_cache.invalidate()
        assert memory_cache.get("post:1") is None
    
    def test_cache_stats(self, memory_cache):
        """Test cache statistics"""
        # Empty cache stats
        stats = memory_cache.get_stats()
        assert stats["total_entries"] == 0
        assert stats["total_hits"] == 0
        
        # Add data and access
        memory_cache.set("key1", "value1")
        memory_cache.set("key2", "value2" * 100)  # Larger value
        
        memory_cache.get("key1")
        memory_cache.get("key1")
        memory_cache.get("key2")
        
        stats = memory_cache.get_stats()
        assert stats["total_entries"] == 2
        assert stats["total_hits"] == 3
        assert stats["memory_usage"] > 0
    
    def test_thread_safety(self, memory_cache):
        """Test thread safety of cache operations"""
        def writer(thread_id):
            for i in range(100):
                memory_cache.set(f"thread{thread_id}_key{i}", f"value{i}")
        
        def reader(thread_id):
            for i in range(100):
                memory_cache.get(f"thread{thread_id}_key{i}")
        
        # Start multiple threads
        threads = []
        for i in range(5):
            t1 = threading.Thread(target=writer, args=(i,))
            t2 = threading.Thread(target=reader, args=(i,))
            threads.extend([t1, t2])
            t1.start()
            t2.start()
        
        for t in threads:
            t.join()
        
        # Verify cache integrity
        stats = memory_cache.get_stats()
        assert stats["total_entries"] > 0


# Test DatabasePool
class TestDatabasePool:
    def test_pool_initialization(self, db_pool):
        """Test database pool initialization"""
        assert db_pool.pool_size == 3
        assert not db_pool.connections.empty()
    
    def test_connection_acquisition(self, db_pool):
        """Test getting connections from pool"""
        with db_pool.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM test_data")
            count = cursor.fetchone()[0]
            assert count == 2
    
    def test_concurrent_connections(self, db_pool):
        """Test concurrent connection usage"""
        results = []
        
        def worker():
            with db_pool.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM test_data")
                results.extend(cursor.fetchall())
                time.sleep(0.01)  # Simulate work
        
        threads = []
        for _ in range(10):  # More threads than pool size
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Each thread should get results
        assert len(results) == 20  # 10 threads * 2 rows each
    
    def test_execute_query(self, db_pool):
        """Test execute_query method"""
        # Without caching
        results = db_pool.execute_query("SELECT * FROM test_data WHERE value > ?", (100,))
        assert len(results) == 1
        assert results[0]["name"] == "test2"
    
    def test_execute_query_with_cache(self, db_pool):
        """Test execute_query with caching"""
        # First query (cache miss)
        results1 = db_pool.execute_query(
            "SELECT * FROM test_data",
            cache_key="all_data"
        )
        assert len(results1) == 2
        
        # Second query (cache hit)
        results2 = db_pool.execute_query(
            "SELECT * FROM test_data",
            cache_key="all_data"
        )
        assert results2 == results1
        
        # Verify it was from cache
        assert cache.get("all_data") is not None
    
    def test_pool_cleanup(self):
        """Test pool cleanup"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
            db_path = f.name
        
        pool = DatabasePool(db_path, pool_size=2)
        
        # Verify pool is initialized
        assert pool.connections.qsize() == 2
        
        # Close pool
        pool.close()
        
        # Verify connections are closed
        assert pool.connections.empty()
        
        # Cleanup
        os.unlink(db_path)


# Test decorators
class TestDecorators:
    @pytest.mark.asyncio
    async def test_cached_decorator_async(self):
        """Test cached decorator with async function"""
        call_count = 0
        
        @cached(ttl="short")
        async def expensive_operation(x, y):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x + y
        
        # First call
        result1 = await expensive_operation(5, 3)
        assert result1 == 8
        assert call_count == 1
        
        # Second call (cached)
        result2 = await expensive_operation(5, 3)
        assert result2 == 8
        assert call_count == 1  # Not called again
        
        # Different args
        result3 = await expensive_operation(10, 5)
        assert result3 == 15
        assert call_count == 2
    
    def test_cached_decorator_sync(self):
        """Test cached decorator with sync function"""
        call_count = 0
        
        @cached(ttl=1, key_prefix="test")
        def calculate(x):
            nonlocal call_count
            call_count += 1
            return x * x
        
        # First call
        result1 = calculate(4)
        assert result1 == 16
        assert call_count == 1
        
        # Second call (cached)
        result2 = calculate(4)
        assert result2 == 16
        assert call_count == 1
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Third call (expired)
        result3 = calculate(4)
        assert result3 == 16
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_track_performance_decorator_async(self):
        """Test track_performance decorator with async function"""
        @track_performance("test_operation")
        async def async_operation():
            await asyncio.sleep(0.05)
            return "done"
        
        result = await async_operation()
        assert result == "done"
        
        # Check that metric was recorded
        stats = monitor.get_stats()
        assert stats["total_requests"] >= 1
    
    def test_track_performance_decorator_sync(self):
        """Test track_performance decorator with sync function"""
        @track_performance("sync_op")
        def sync_operation():
            time.sleep(0.05)
            return "completed"
        
        result = sync_operation()
        assert result == "completed"
        
        # Check metrics
        stats = monitor.get_stats()
        assert any(m.request_type == "sync_op" for m in monitor.metrics)
    
    def test_track_performance_with_error(self):
        """Test track_performance with function that raises error"""
        @track_performance("error_op")
        def failing_operation():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_operation()
        
        # Check that error was recorded
        error_metrics = [m for m in monitor.metrics if m.error is not None]
        assert len(error_metrics) >= 1
        assert "Test error" in error_metrics[-1].error


# Test OptimizedRequestProcessor
class TestOptimizedRequestProcessor:
    @pytest.mark.asyncio
    async def test_request_type_identification(self, request_processor):
        """Test request type identification"""
        test_cases = [
            ("What's the weather today?", "weather"),
            ("What time is it?", "time"),
            ("Search for Python tutorials", "search"),
            ("Calculate 5 + 3", "calculate"),
            ("Remind me to call mom", "reminder"),
            ("Hello there", "general")
        ]
        
        for request, expected_type in test_cases:
            req_type = request_processor._identify_request_type(request)
            assert req_type == expected_type
    
    @pytest.mark.asyncio
    async def test_process_request_basic(self, request_processor):
        """Test basic request processing"""
        result = await request_processor.process_request("Test request")
        
        assert "request" in result
        assert "type" in result
        assert "result" in result
        assert "timestamp" in result
        assert result["request"] == "Test request"
        assert result["type"] == "general"
    
    @pytest.mark.asyncio
    async def test_process_request_with_cache(self, request_processor):
        """Test request processing with caching"""
        # First request
        result1 = await request_processor.process_request("What time is it?")
        assert result1["type"] == "time"
        
        # Second request (should be cached)
        result2 = await request_processor.process_request("What time is it?")
        assert result2 == result1
    
    @pytest.mark.asyncio
    async def test_register_handler(self, request_processor):
        """Test registering custom handlers"""
        custom_called = False
        
        async def custom_handler(request, context):
            nonlocal custom_called
            custom_called = True
            return {"custom": True, "request": request}
        
        request_processor.register_handler("custom", custom_handler)
        
        # Override type identification for testing
        original_identify = request_processor._identify_request_type
        request_processor._identify_request_type = lambda r: "custom"
        
        result = await request_processor.process_request("Custom request")
        
        assert custom_called
        assert result["result"]["custom"] is True
        
        # Restore original
        request_processor._identify_request_type = original_identify
    
    @pytest.mark.asyncio
    async def test_get_performance_report(self, request_processor):
        """Test performance report generation"""
        # Process some requests
        await request_processor.process_request("Test 1")
        await request_processor.process_request("Test 2")
        
        report = request_processor.get_performance_report()
        
        assert "request_stats" in report
        assert "cache_stats" in report
        assert "optimization_status" in report
        
        assert report["optimization_status"]["caching_enabled"] is True
        assert report["optimization_status"]["connection_pooling"] is True
        assert report["optimization_status"]["performance_tracking"] is True


# Test global functions
class TestGlobalFunctions:
    def test_warm_cache(self):
        """Test cache warming functionality"""
        test_data = [
            {"id": 1, "name": "Item 1"},
            {"id": 2, "name": "Item 2"},
            {"id": 3, "name": "Item 3"}
        ]
        
        warm_cache(test_data)
        
        # Verify data is in cache
        assert cache.get("warm:1") == test_data[0]
        assert cache.get("warm:2") == test_data[1]
        assert cache.get("warm:3") == test_data[2]
    
    def test_clear_cache(self):
        """Test cache clearing"""
        # Add test data
        cache.set("test:1", "value1")
        cache.set("test:2", "value2")
        cache.set("other:1", "value3")
        
        # Clear by pattern
        clear_cache("test:")
        
        assert cache.get("test:1") is None
        assert cache.get("test:2") is None
        assert cache.get("other:1") == "value3"
        
        # Clear all
        clear_cache()
        assert cache.get("other:1") is None
    
    def test_get_performance_stats(self):
        """Test global performance stats"""
        # Generate some activity
        cache.set("stat_test", "value")
        cache.get("stat_test")
        
        monitor.start_request("stat_req", "test")
        monitor.complete_request("stat_req")
        
        stats = get_performance_stats()
        
        assert "monitor" in stats
        assert "cache" in stats
        assert stats["cache"]["total_entries"] >= 1
        assert stats["monitor"]["total_requests"] >= 1


# Integration tests
class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_optimization_pipeline(self, temp_db):
        """Test complete optimization pipeline"""
        processor = OptimizedRequestProcessor(temp_db)
        
        # Simulate multiple requests
        requests = [
            "What's the weather?",
            "Calculate 10 + 20",
            "What's the weather?",  # Cached
            "Search for optimization",
            "What time is it?",
            "Calculate 10 + 20",  # Cached
        ]
        
        start_time = time.time()
        results = []
        
        for req in requests:
            result = await processor.process_request(req)
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Verify results
        assert len(results) == 6
        assert results[0]["type"] == "weather"
        assert results[2] == results[0]  # Cached result
        assert results[5] == results[1]  # Cached result
        
        # Check performance
        report = processor.get_performance_report()
        cache_stats = report["cache_stats"]
        
        assert cache_stats["total_entries"] >= 3  # At least 3 unique requests
        assert cache_stats["total_hits"] >= 2  # At least 2 cache hits
        
        # Verify optimization worked (should be fast due to caching)
        assert total_time < 1.0  # Should complete quickly
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, temp_db):
        """Test handling concurrent requests"""
        processor = OptimizedRequestProcessor(temp_db)
        
        async def make_request(req_id):
            return await processor.process_request(f"Request {req_id}")
        
        # Launch concurrent requests
        tasks = [make_request(i) for i in range(20)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 20
        
        # Check performance stats
        stats = get_performance_stats()
        assert stats["monitor"]["total_requests"] >= 20
    
    def test_memory_efficiency(self):
        """Test memory efficiency of cache"""
        # Fill cache with data
        for i in range(1000):
            cache.set(f"key{i}", f"value{i}" * 10)
        
        stats = cache.get_stats()
        assert stats["total_entries"] >= 1000  # May have more due to other tests
        
        # Clear old entries
        cache.invalidate()
        
        stats = cache.get_stats()
        assert stats["total_entries"] == 0
        assert stats["memory_usage"] == 0


# Performance benchmarks
class TestPerformance:
    def test_cache_performance(self, memory_cache):
        """Benchmark cache performance"""
        num_operations = 10000
        
        # Write performance
        start = time.time()
        for i in range(num_operations):
            memory_cache.set(f"perf_key_{i}", f"value_{i}")
        write_time = time.time() - start
        
        # Read performance
        start = time.time()
        for i in range(num_operations):
            memory_cache.get(f"perf_key_{i}")
        read_time = time.time() - start
        
        # Performance assertions
        assert write_time < 1.0  # Should handle 10k writes in < 1s
        assert read_time < 0.5   # Reads should be faster
        
        print(f"\nCache Performance:")
        print(f"  Writes: {num_operations/write_time:.0f} ops/sec")
        print(f"  Reads: {num_operations/read_time:.0f} ops/sec")
    
    @pytest.mark.asyncio
    async def test_decorator_overhead(self):
        """Test overhead of performance decorators"""
        # Clear cache before test
        cache.invalidate()
        
        # Baseline function with some work
        async def baseline(x):
            # Do some actual work so timing is meaningful
            await asyncio.sleep(0.001)
            return x * 2
        
        # Decorated function
        @cached(ttl=1)
        @track_performance("test_overhead")
        async def decorated(x):
            await asyncio.sleep(0.001)
            return x * 2
        
        # Warm up
        await baseline(1)
        await decorated(1)
        
        # Measure baseline (different values to avoid cache)
        start = time.time()
        for i in range(10):
            await baseline(i)
        baseline_time = time.time() - start
        
        # Measure decorated (different values to test real overhead)
        start = time.time()
        for i in range(10, 20):
            await decorated(i)
        decorated_time = time.time() - start
        
        # Calculate overhead
        overhead = (decorated_time - baseline_time) / baseline_time if baseline_time > 0 else 0
        
        # With actual work, overhead should be minimal
        assert overhead < 0.5  # Less than 50% overhead
        
        print(f"\nDecorator Overhead: {overhead*100:.1f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])