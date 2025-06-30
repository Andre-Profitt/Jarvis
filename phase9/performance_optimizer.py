"""
JARVIS Phase 9: Performance Optimizer
=====================================
Advanced performance optimization system with parallel processing,
intelligent caching, and lazy loading capabilities.
"""

import asyncio
import functools
import hashlib
import json
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Set, Tuple
from threading import Lock
import numpy as np
import redis
import pickle
import lz4.frame
import psutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Caching System ====================

class CacheStats:
    """Track cache performance metrics"""
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.memory_saved = 0
        self.time_saved = 0.0
        self._lock = Lock()
    
    def record_hit(self, time_saved_ms: float):
        with self._lock:
            self.hits += 1
            self.time_saved += time_saved_ms
    
    def record_miss(self):
        with self._lock:
            self.misses += 1
    
    def record_eviction(self, memory_freed: int):
        with self._lock:
            self.evictions += 1
            self.memory_saved += memory_freed
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': self.hit_rate,
                'evictions': self.evictions,
                'memory_saved_mb': self.memory_saved / (1024 * 1024),
                'time_saved_seconds': self.time_saved / 1000
            }


class IntelligentCache:
    """
    Multi-tiered caching system with compression and TTL
    """
    def __init__(self, 
                 memory_limit_mb: int = 512,
                 redis_enabled: bool = True,
                 compression_enabled: bool = True):
        # L1 Cache: In-memory LRU
        self.memory_cache = OrderedDict()
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes
        self.current_memory = 0
        
        # L2 Cache: Redis
        self.redis_enabled = redis_enabled
        if redis_enabled:
            try:
                self.redis_client = redis.Redis(
                    host='localhost', 
                    port=6379, 
                    decode_responses=False
                )
                self.redis_client.ping()
            except:
                logger.warning("Redis not available, using memory cache only")
                self.redis_enabled = False
        
        # Configuration
        self.compression_enabled = compression_enabled
        self.stats = CacheStats()
        self._lock = Lock()
        
        # Pattern recognition for predictive caching
        self.access_patterns = deque(maxlen=1000)
        self.prefetch_candidates = set()
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function call"""
        key_data = {
            'func': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data for storage"""
        serialized = pickle.dumps(data)
        if self.compression_enabled and len(serialized) > 1024:  # Only compress if > 1KB
            return lz4.frame.compress(serialized, compression_level=1)
        return serialized
    
    def _decompress_data(self, data: bytes) -> Any:
        """Decompress stored data"""
        if self.compression_enabled and data.startswith(b'\x04"M\x18'):  # LZ4 magic bytes
            decompressed = lz4.frame.decompress(data)
            return pickle.loads(decompressed)
        return pickle.loads(data)
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data"""
        try:
            return len(pickle.dumps(data))
        except:
            return 1024  # Default estimate
    
    def _evict_lru(self, required_space: int):
        """Evict least recently used items"""
        with self._lock:
            while self.current_memory + required_space > self.memory_limit and self.memory_cache:
                key, (data, _, _) = self.memory_cache.popitem(last=False)
                size = self._estimate_size(data)
                self.current_memory -= size
                self.stats.record_eviction(size)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        start_time = time.time()
        
        # Check L1 (memory)
        with self._lock:
            if key in self.memory_cache:
                data, expiry, size = self.memory_cache[key]
                if expiry is None or expiry > datetime.now():
                    # Move to end (most recently used)
                    self.memory_cache.move_to_end(key)
                    elapsed = (time.time() - start_time) * 1000
                    self.stats.record_hit(elapsed)
                    self.access_patterns.append((key, datetime.now()))
                    return data
                else:
                    # Expired
                    del self.memory_cache[key]
                    self.current_memory -= size
        
        # Check L2 (Redis)
        if self.redis_enabled:
            try:
                compressed = self.redis_client.get(f"jarvis:cache:{key}")
                if compressed:
                    data = self._decompress_data(compressed)
                    elapsed = (time.time() - start_time) * 1000
                    self.stats.record_hit(elapsed)
                    
                    # Promote to L1
                    await self.set(key, data, ttl_seconds=3600, l1_only=True)
                    self.access_patterns.append((key, datetime.now()))
                    return data
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        self.stats.record_miss()
        return None
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None, 
                  l1_only: bool = False) -> bool:
        """Set value in cache"""
        try:
            size = self._estimate_size(value)
            expiry = datetime.now() + timedelta(seconds=ttl_seconds) if ttl_seconds else None
            
            # Store in L1
            if size <= self.memory_limit:
                self._evict_lru(size)
                with self._lock:
                    self.memory_cache[key] = (value, expiry, size)
                    self.current_memory += size
            
            # Store in L2
            if self.redis_enabled and not l1_only:
                try:
                    compressed = self._compress_data(value)
                    self.redis_client.set(
                        f"jarvis:cache:{key}",
                        compressed,
                        ex=ttl_seconds
                    )
                except Exception as e:
                    logger.error(f"Redis set error: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze access patterns for optimization"""
        if not self.access_patterns:
            return {}
        
        # Find frequently accessed keys
        key_counts = {}
        for key, _ in self.access_patterns:
            key_counts[key] = key_counts.get(key, 0) + 1
        
        # Identify hot keys
        total_accesses = len(self.access_patterns)
        hot_keys = {
            k: v/total_accesses 
            for k, v in key_counts.items() 
            if v/total_accesses > 0.05  # Accessed >5% of time
        }
        
        return {
            'hot_keys': hot_keys,
            'total_unique_keys': len(key_counts),
            'cache_efficiency': self.stats.hit_rate
        }


def cached(ttl_seconds: int = 3600, cache_instance: Optional[IntelligentCache] = None):
    """Decorator for automatic caching"""
    def decorator(func):
        nonlocal cache_instance
        if cache_instance is None:
            cache_instance = IntelligentCache()
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_instance._generate_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = await cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache_instance.set(cache_key, result, ttl_seconds)
            
            return result
        
        # Add cache stats method
        wrapper.get_cache_stats = lambda: cache_instance.stats.get_stats()
        wrapper.clear_cache = lambda: cache_instance.memory_cache.clear()
        
        return wrapper
    return decorator


# ==================== Parallel Processing System ====================

class ParallelProcessor:
    """
    Advanced parallel processing with dynamic worker allocation
    """
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 use_processes: bool = False,
                 adaptive_scaling: bool = True):
        
        self.max_workers = max_workers or mp.cpu_count()
        self.use_processes = use_processes
        self.adaptive_scaling = adaptive_scaling
        
        # Create executor
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Performance tracking
        self.task_times = deque(maxlen=100)
        self.current_workers = self.max_workers
        self._lock = Lock()
        
        # System monitoring
        self.cpu_threshold = 80  # CPU usage threshold
        self.memory_threshold = 85  # Memory usage threshold
    
    def _monitor_system(self) -> Dict[str, float]:
        """Monitor system resources"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'active_workers': self.executor._threads.__len__() if hasattr(self.executor, '_threads') else 0
        }
    
    def _adjust_workers(self):
        """Dynamically adjust worker count based on system load"""
        if not self.adaptive_scaling:
            return
        
        stats = self._monitor_system()
        
        with self._lock:
            if stats['cpu_percent'] > self.cpu_threshold or stats['memory_percent'] > self.memory_threshold:
                # Reduce workers
                new_workers = max(1, self.current_workers - 1)
                if new_workers < self.current_workers:
                    self.current_workers = new_workers
                    if hasattr(self.executor, '_max_workers'):
                        self.executor._max_workers = new_workers
                    logger.info(f"Reduced workers to {new_workers} due to high system load")
            
            elif stats['cpu_percent'] < 50 and stats['memory_percent'] < 50:
                # Increase workers
                new_workers = min(self.max_workers, self.current_workers + 1)
                if new_workers > self.current_workers:
                    self.current_workers = new_workers
                    if hasattr(self.executor, '_max_workers'):
                        self.executor._max_workers = new_workers
                    logger.info(f"Increased workers to {new_workers} due to low system load")
    
    async def map_async(self, func: Callable, items: List[Any], 
                        chunk_size: Optional[int] = None) -> List[Any]:
        """
        Parallel map with automatic chunking and load balancing
        """
        if not items:
            return []
        
        # Adjust workers based on system load
        self._adjust_workers()
        
        # Determine optimal chunk size
        if chunk_size is None:
            chunk_size = max(1, len(items) // (self.current_workers * 4))
        
        # Create chunks
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        # Process chunks in parallel
        loop = asyncio.get_event_loop()
        start_time = time.time()
        
        if asyncio.iscoroutinefunction(func):
            # Async function
            tasks = []
            for chunk in chunks:
                task = loop.create_task(self._process_chunk_async(func, chunk))
                tasks.append(task)
            
            chunk_results = await asyncio.gather(*tasks)
        else:
            # Sync function
            futures = []
            for chunk in chunks:
                future = loop.run_in_executor(self.executor, self._process_chunk_sync, func, chunk)
                futures.append(future)
            
            chunk_results = await asyncio.gather(*futures)
        
        # Flatten results
        results = [item for chunk in chunk_results for item in chunk]
        
        # Track performance
        elapsed = time.time() - start_time
        self.task_times.append(elapsed)
        
        return results
    
    async def _process_chunk_async(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """Process chunk with async function"""
        tasks = [func(item) for item in chunk]
        return await asyncio.gather(*tasks)
    
    def _process_chunk_sync(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """Process chunk with sync function"""
        return [func(item) for item in chunk]
    
    async def pipeline(self, stages: List[Callable], data: Any) -> Any:
        """
        Execute pipeline of functions with parallel stages
        """
        result = data
        
        for stage in stages:
            if isinstance(result, list) and len(result) > 1:
                # Process list in parallel
                result = await self.map_async(stage, result)
            else:
                # Single item
                if asyncio.iscoroutinefunction(stage):
                    result = await stage(result)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(self.executor, stage, result)
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.task_times:
            return {}
        
        times = list(self.task_times)
        return {
            'average_time': np.mean(times),
            'median_time': np.median(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'current_workers': self.current_workers,
            'system_stats': self._monitor_system()
        }
    
    def shutdown(self):
        """Shutdown executor"""
        self.executor.shutdown(wait=True)


# ==================== Lazy Loading System ====================

class LazyLoader:
    """
    Intelligent lazy loading with predictive prefetching
    """
    def __init__(self, cache: Optional[IntelligentCache] = None):
        self.cache = cache or IntelligentCache()
        self.loaded_modules = {}
        self.loading_times = {}
        self.access_counts = {}
        self.prefetch_queue = asyncio.Queue()
        self._lock = Lock()
        
        # Start prefetch worker
        self.prefetch_task = None
    
    async def start_prefetcher(self):
        """Start background prefetch worker"""
        self.prefetch_task = asyncio.create_task(self._prefetch_worker())
    
    async def _prefetch_worker(self):
        """Background worker for predictive prefetching"""
        while True:
            try:
                # Get next module to prefetch
                module_name = await asyncio.wait_for(
                    self.prefetch_queue.get(), 
                    timeout=60.0
                )
                
                # Check if already loaded
                if module_name not in self.loaded_modules:
                    logger.info(f"Prefetching module: {module_name}")
                    await self.load(module_name, priority='low')
                
            except asyncio.TimeoutError:
                # No prefetch requests, sleep
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Prefetch error: {e}")
                await asyncio.sleep(1)
    
    async def load(self, module_name: str, priority: str = 'normal') -> Any:
        """
        Lazy load module with caching
        """
        # Check if already loaded
        with self._lock:
            if module_name in self.loaded_modules:
                self.access_counts[module_name] = self.access_counts.get(module_name, 0) + 1
                return self.loaded_modules[module_name]
        
        start_time = time.time()
        
        # Try cache first
        cached_module = await self.cache.get(f"module:{module_name}")
        if cached_module:
            with self._lock:
                self.loaded_modules[module_name] = cached_module
                self.access_counts[module_name] = self.access_counts.get(module_name, 0) + 1
            return cached_module
        
        # Load module
        try:
            if module_name.startswith('jarvis.'):
                # Internal JARVIS module
                module = await self._load_jarvis_module(module_name)
            else:
                # External module
                module = await self._load_external_module(module_name)
            
            # Cache module
            await self.cache.set(f"module:{module_name}", module, ttl_seconds=3600)
            
            # Track loading time
            load_time = time.time() - start_time
            with self._lock:
                self.loaded_modules[module_name] = module
                self.loading_times[module_name] = load_time
                self.access_counts[module_name] = 1
            
            # Predict related modules for prefetching
            await self._predict_related_modules(module_name)
            
            logger.info(f"Loaded {module_name} in {load_time:.2f}s")
            return module
            
        except Exception as e:
            logger.error(f"Failed to load {module_name}: {e}")
            raise
    
    async def _load_jarvis_module(self, module_name: str) -> Any:
        """Load internal JARVIS module"""
        # Simulate loading JARVIS components
        components = {
            'jarvis.nlp': {'analyzer': 'NLPAnalyzer', 'memory': '512MB'},
            'jarvis.vision': {'processor': 'VisionProcessor', 'memory': '1GB'},
            'jarvis.memory': {'store': 'MemoryStore', 'memory': '2GB'},
            'jarvis.reasoning': {'engine': 'ReasoningEngine', 'memory': '768MB'}
        }
        
        # Simulate loading delay
        await asyncio.sleep(0.1)
        
        return components.get(module_name, {})
    
    async def _load_external_module(self, module_name: str) -> Any:
        """Load external module"""
        # In real implementation, would use importlib
        await asyncio.sleep(0.05)
        return {'module': module_name, 'loaded': True}
    
    async def _predict_related_modules(self, module_name: str):
        """Predict and queue related modules for prefetching"""
        # Define module relationships
        relationships = {
            'jarvis.nlp': ['jarvis.memory', 'jarvis.reasoning'],
            'jarvis.vision': ['jarvis.memory'],
            'jarvis.reasoning': ['jarvis.memory', 'jarvis.nlp']
        }
        
        related = relationships.get(module_name, [])
        for related_module in related:
            if related_module not in self.loaded_modules:
                await self.prefetch_queue.put(related_module)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get lazy loading statistics"""
        with self._lock:
            total_modules = len(self.loaded_modules)
            total_time = sum(self.loading_times.values())
            
            hot_modules = sorted(
                self.access_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            return {
                'loaded_modules': total_modules,
                'total_loading_time': total_time,
                'average_loading_time': total_time / total_modules if total_modules > 0 else 0,
                'hot_modules': hot_modules,
                'cache_stats': self.cache.stats.get_stats()
            }
    
    async def unload(self, module_name: str):
        """Unload module to free memory"""
        with self._lock:
            if module_name in self.loaded_modules:
                del self.loaded_modules[module_name]
                logger.info(f"Unloaded module: {module_name}")


# ==================== Performance Monitor ====================

class PerformanceMonitor:
    """
    Real-time performance monitoring and optimization recommendations
    """
    def __init__(self):
        self.metrics = {
            'response_times': deque(maxlen=1000),
            'memory_usage': deque(maxlen=100),
            'cpu_usage': deque(maxlen=100),
            'cache_performance': deque(maxlen=100),
            'parallel_efficiency': deque(maxlen=100)
        }
        self.alerts = []
        self.recommendations = []
        self._monitoring = False
        self._monitor_task = None
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self._monitoring = False
        if self._monitor_task:
            await self._monitor_task
        logger.info("Performance monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                
                # Analyze performance
                self._analyze_performance(metrics)
                
                # Generate recommendations
                self._generate_recommendations()
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        metrics = {
            'timestamp': datetime.now(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_io': psutil.disk_io_counters(),
            'network_io': psutil.net_io_counters()
        }
        
        # Store metrics
        self.metrics['cpu_usage'].append(metrics['cpu_percent'])
        self.metrics['memory_usage'].append(metrics['memory_percent'])
        
        return metrics
    
    def _analyze_performance(self, current_metrics: Dict[str, Any]):
        """Analyze performance and detect issues"""
        # CPU analysis
        if len(self.metrics['cpu_usage']) > 10:
            avg_cpu = np.mean(list(self.metrics['cpu_usage'])[-10:])
            if avg_cpu > 80:
                self.alerts.append({
                    'type': 'high_cpu',
                    'severity': 'warning',
                    'message': f'High CPU usage: {avg_cpu:.1f}%',
                    'timestamp': datetime.now()
                })
        
        # Memory analysis
        if len(self.metrics['memory_usage']) > 10:
            avg_memory = np.mean(list(self.metrics['memory_usage'])[-10:])
            if avg_memory > 85:
                self.alerts.append({
                    'type': 'high_memory',
                    'severity': 'warning',
                    'message': f'High memory usage: {avg_memory:.1f}%',
                    'timestamp': datetime.now()
                })
    
    def _generate_recommendations(self):
        """Generate performance optimization recommendations"""
        self.recommendations.clear()
        
        # CPU recommendations
        if self.metrics['cpu_usage'] and np.mean(list(self.metrics['cpu_usage'])) > 70:
            self.recommendations.append({
                'category': 'cpu_optimization',
                'priority': 'high',
                'suggestion': 'Enable more aggressive caching to reduce CPU load',
                'expected_impact': '20-30% CPU reduction'
            })
        
        # Memory recommendations
        if self.metrics['memory_usage'] and np.mean(list(self.metrics['memory_usage'])) > 75:
            self.recommendations.append({
                'category': 'memory_optimization',
                'priority': 'high',
                'suggestion': 'Increase cache eviction rate and enable compression',
                'expected_impact': '15-25% memory reduction'
            })
        
        # Cache recommendations
        if self.metrics['cache_performance']:
            recent_hit_rates = [m['hit_rate'] for m in list(self.metrics['cache_performance'])[-10:]]
            if recent_hit_rates and np.mean(recent_hit_rates) < 0.7:
                self.recommendations.append({
                    'category': 'cache_optimization',
                    'priority': 'medium',
                    'suggestion': 'Analyze access patterns and implement predictive caching',
                    'expected_impact': '30-40% cache hit rate improvement'
                })
    
    def record_response_time(self, operation: str, duration_ms: float):
        """Record operation response time"""
        self.metrics['response_times'].append({
            'operation': operation,
            'duration_ms': duration_ms,
            'timestamp': datetime.now()
        })
    
    def record_cache_performance(self, stats: Dict[str, Any]):
        """Record cache performance metrics"""
        self.metrics['cache_performance'].append({
            **stats,
            'timestamp': datetime.now()
        })
    
    def record_parallel_efficiency(self, efficiency: float):
        """Record parallel processing efficiency"""
        self.metrics['parallel_efficiency'].append({
            'efficiency': efficiency,
            'timestamp': datetime.now()
        })
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get performance dashboard data"""
        return {
            'current_metrics': {
                'cpu': self.metrics['cpu_usage'][-1] if self.metrics['cpu_usage'] else 0,
                'memory': self.metrics['memory_usage'][-1] if self.metrics['memory_usage'] else 0,
                'avg_response_time': np.mean([
                    m['duration_ms'] for m in list(self.metrics['response_times'])[-100:]
                ]) if self.metrics['response_times'] else 0
            },
            'trends': {
                'cpu_history': list(self.metrics['cpu_usage'])[-50:],
                'memory_history': list(self.metrics['memory_usage'])[-50:],
                'response_times': [
                    m['duration_ms'] for m in list(self.metrics['response_times'])[-100:]
                ]
            },
            'alerts': self.alerts[-10:],  # Last 10 alerts
            'recommendations': self.recommendations,
            'timestamp': datetime.now().isoformat()
        }


# ==================== Main Performance Optimizer ====================

class JARVISPerformanceOptimizer:
    """
    Central performance optimization system for JARVIS
    """
    def __init__(self):
        # Initialize components
        self.cache = IntelligentCache(
            memory_limit_mb=1024,
            redis_enabled=True,
            compression_enabled=True
        )
        
        self.parallel_processor = ParallelProcessor(
            max_workers=mp.cpu_count() * 2,
            use_processes=False,
            adaptive_scaling=True
        )
        
        self.lazy_loader = LazyLoader(cache=self.cache)
        self.monitor = PerformanceMonitor()
        
        # Performance settings
        self.optimization_level = 'balanced'  # 'aggressive', 'balanced', 'conservative'
        self._running = False
    
    async def initialize(self):
        """Initialize performance optimizer"""
        logger.info("Initializing JARVIS Performance Optimizer...")
        
        # Start components
        await self.lazy_loader.start_prefetcher()
        await self.monitor.start_monitoring()
        
        self._running = True
        logger.info("Performance Optimizer initialized successfully")
    
    async def optimize_operation(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Optimize any operation with caching and parallel processing
        """
        start_time = time.time()
        
        # Check if operation is cacheable
        if hasattr(operation, '__name__'):
            cache_key = self.cache._generate_key(operation.__name__, args, kwargs)
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                duration = (time.time() - start_time) * 1000
                self.monitor.record_response_time(operation.__name__, duration)
                return cached_result
        
        # Execute operation
        if asyncio.iscoroutinefunction(operation):
            result = await operation(*args, **kwargs)
        else:
            # Run in parallel processor
            result = await self.parallel_processor.pipeline([operation], (args, kwargs))
        
        # Cache result
        if hasattr(operation, '__name__'):
            await self.cache.set(cache_key, result, ttl_seconds=3600)
        
        # Record metrics
        duration = (time.time() - start_time) * 1000
        self.monitor.record_response_time(
            operation.__name__ if hasattr(operation, '__name__') else 'anonymous',
            duration
        )
        
        return result
    
    async def batch_process(self, items: List[Any], processor: Callable, 
                           batch_size: Optional[int] = None) -> List[Any]:
        """
        Process items in optimized batches
        """
        if not items:
            return []
        
        # Determine optimal batch size based on system resources
        if batch_size is None:
            system_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
            if system_memory > 16:
                batch_size = 1000
            elif system_memory > 8:
                batch_size = 500
            else:
                batch_size = 100
        
        # Process in parallel
        results = await self.parallel_processor.map_async(
            processor, 
            items, 
            chunk_size=batch_size
        )
        
        # Update efficiency metrics
        efficiency = len(items) / (len(items) + batch_size) if batch_size > 0 else 1.0
        self.monitor.record_parallel_efficiency(efficiency)
        
        return results
    
    async def preload_modules(self, module_names: List[str]):
        """
        Preload modules for faster access
        """
        for module in module_names:
            await self.lazy_loader.prefetch_queue.put(module)
    
    def set_optimization_level(self, level: str):
        """
        Set optimization aggressiveness
        """
        if level in ['aggressive', 'balanced', 'conservative']:
            self.optimization_level = level
            
            if level == 'aggressive':
                self.cache.memory_limit = 2048 * 1024 * 1024  # 2GB
                self.parallel_processor.max_workers = mp.cpu_count() * 3
            elif level == 'conservative':
                self.cache.memory_limit = 512 * 1024 * 1024  # 512MB
                self.parallel_processor.max_workers = mp.cpu_count()
            
            logger.info(f"Optimization level set to: {level}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report
        """
        return {
            'cache_stats': self.cache.stats.get_stats(),
            'cache_patterns': self.cache.analyze_patterns(),
            'parallel_stats': self.parallel_processor.get_performance_stats(),
            'lazy_loading_stats': self.lazy_loader.get_stats(),
            'monitor_dashboard': self.monitor.get_dashboard_data(),
            'optimization_level': self.optimization_level
        }
    
    async def auto_optimize(self):
        """
        Automatically optimize based on current performance
        """
        report = self.get_performance_report()
        
        # Auto-adjust based on metrics
        cache_hit_rate = report['cache_stats']['hit_rate']
        if cache_hit_rate < 0.5:
            # Poor cache performance - analyze patterns
            patterns = report['cache_patterns']
            if patterns.get('hot_keys'):
                # Prefetch hot keys
                for key in patterns['hot_keys']:
                    logger.info(f"Auto-optimizing: Prefetching hot key {key}")
        
        # Adjust parallel workers based on CPU
        cpu_usage = report['monitor_dashboard']['current_metrics']['cpu']
        if cpu_usage > 80:
            self.set_optimization_level('conservative')
        elif cpu_usage < 30:
            self.set_optimization_level('aggressive')
    
    async def shutdown(self):
        """
        Gracefully shutdown optimizer
        """
        logger.info("Shutting down Performance Optimizer...")
        
        self._running = False
        
        # Stop components
        await self.monitor.stop_monitoring()
        self.parallel_processor.shutdown()
        
        # Save cache stats
        stats = self.get_performance_report()
        logger.info(f"Final performance report: {json.dumps(stats, indent=2)}")
        
        logger.info("Performance Optimizer shutdown complete")


# ==================== Demo and Testing ====================

async def demo_performance_optimizer():
    """Demonstrate performance optimization capabilities"""
    
    print("\n🚀 JARVIS Performance Optimizer Demo\n")
    
    # Initialize optimizer
    optimizer = JARVISPerformanceOptimizer()
    await optimizer.initialize()
    
    print("✅ Performance Optimizer initialized\n")
    
    # Demo 1: Cached operations
    print("📊 Demo 1: Intelligent Caching")
    print("-" * 50)
    
    @cached(ttl_seconds=3600, cache_instance=optimizer.cache)
    async def expensive_calculation(n: int) -> int:
        """Simulate expensive calculation"""
        await asyncio.sleep(1)  # Simulate work
        return n ** 2
    
    # First call - will be slow
    start = time.time()
    result1 = await expensive_calculation(42)
    time1 = time.time() - start
    print(f"First call: {result1} (took {time1:.2f}s)")
    
    # Second call - will be fast (cached)
    start = time.time()
    result2 = await expensive_calculation(42)
    time2 = time.time() - start
    print(f"Second call: {result2} (took {time2:.2f}s)")
    print(f"Speed improvement: {time1/time2:.1f}x faster\n")
    
    # Demo 2: Parallel processing
    print("⚡ Demo 2: Parallel Processing")
    print("-" * 50)
    
    async def process_item(item: int) -> int:
        """Process single item"""
        await asyncio.sleep(0.1)  # Simulate work
        return item * 2
    
    items = list(range(20))
    
    # Sequential processing
    start = time.time()
    sequential_results = []
    for item in items:
        result = await process_item(item)
        sequential_results.append(result)
    sequential_time = time.time() - start
    
    # Parallel processing
    start = time.time()
    parallel_results = await optimizer.batch_process(items, process_item)
    parallel_time = time.time() - start
    
    print(f"Sequential: {sequential_time:.2f}s")
    print(f"Parallel: {parallel_time:.2f}s")
    print(f"Speed improvement: {sequential_time/parallel_time:.1f}x faster\n")
    
    # Demo 3: Lazy loading
    print("💤 Demo 3: Lazy Loading with Prefetching")
    print("-" * 50)
    
    # Preload predicted modules
    await optimizer.preload_modules(['jarvis.nlp', 'jarvis.vision'])
    
    # Load module (will be fast if prefetched)
    start = time.time()
    nlp_module = await optimizer.lazy_loader.load('jarvis.nlp')
    load_time = time.time() - start
    print(f"Loaded jarvis.nlp in {load_time:.2f}s")
    
    # Show lazy loading stats
    stats = optimizer.lazy_loader.get_stats()
    print(f"Loaded modules: {stats['loaded_modules']}")
    print(f"Total loading time: {stats['total_loading_time']:.2f}s\n")
    
    # Demo 4: Performance monitoring
    print("📈 Demo 4: Real-time Performance Monitoring")
    print("-" * 50)
    
    # Get performance dashboard
    dashboard = optimizer.monitor.get_dashboard_data()
    print(f"CPU Usage: {dashboard['current_metrics']['cpu']:.1f}%")
    print(f"Memory Usage: {dashboard['current_metrics']['memory']:.1f}%")
    print(f"Avg Response Time: {dashboard['current_metrics']['avg_response_time']:.1f}ms")
    
    # Show recommendations
    if dashboard['recommendations']:
        print("\n🎯 Optimization Recommendations:")
        for rec in dashboard['recommendations']:
            print(f"- {rec['suggestion']} (Impact: {rec['expected_impact']})")
    
    # Demo 5: Auto-optimization
    print("\n🤖 Demo 5: Auto-Optimization")
    print("-" * 50)
    
    await optimizer.auto_optimize()
    print(f"Current optimization level: {optimizer.optimization_level}")
    
    # Get final performance report
    print("\n📊 Final Performance Report:")
    print("-" * 50)
    report = optimizer.get_performance_report()
    print(f"Cache Hit Rate: {report['cache_stats']['hit_rate']:.1%}")
    print(f"Time Saved: {report['cache_stats']['time_saved_seconds']:.1f}s")
    print(f"Memory Saved: {report['cache_stats']['memory_saved_mb']:.1f}MB")
    
    # Shutdown
    await optimizer.shutdown()
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_performance_optimizer())
