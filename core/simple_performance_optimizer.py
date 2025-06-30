#!/usr/bin/env python3
"""
Performance Optimizer for JARVIS
Integrates caching, connection pooling, and performance monitoring
"""

import asyncio
import time
import sqlite3
import json
import hashlib
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
import logging
from collections import defaultdict
import statistics
import threading
from queue import Queue

logger = logging.getLogger(__name__)

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    request_id: str
    request_type: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    cache_hit: bool = False
    db_queries: int = 0
    error: Optional[str] = None
    
    def complete(self):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

class PerformanceMonitor:
    """Global performance monitoring system"""
    
    def __init__(self):
        self.metrics: List[RequestMetrics] = []
        self.active_requests: Dict[str, RequestMetrics] = {}
        self._lock = threading.Lock()
        
    def start_request(self, request_id: str, request_type: str) -> RequestMetrics:
        """Start tracking a request"""
        metric = RequestMetrics(
            request_id=request_id,
            request_type=request_type,
            start_time=time.time()
        )
        
        with self._lock:
            self.active_requests[request_id] = metric
            
        return metric
    
    def complete_request(self, request_id: str):
        """Complete tracking a request"""
        with self._lock:
            if request_id in self.active_requests:
                metric = self.active_requests.pop(request_id)
                metric.complete()
                self.metrics.append(metric)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.metrics:
            return {"message": "No metrics available"}
        
        durations = [m.duration for m in self.metrics if m.duration]
        cache_hits = sum(1 for m in self.metrics if m.cache_hit)
        
        stats_by_type = defaultdict(list)
        for m in self.metrics:
            if m.duration:
                stats_by_type[m.request_type].append(m.duration)
        
        return {
            "total_requests": len(self.metrics),
            "average_duration": statistics.mean(durations) if durations else 0,
            "median_duration": statistics.median(durations) if durations else 0,
            "cache_hit_rate": cache_hits / len(self.metrics) if self.metrics else 0,
            "by_type": {
                req_type: {
                    "count": len(times),
                    "average": statistics.mean(times),
                    "median": statistics.median(times)
                }
                for req_type, times in stats_by_type.items()
            }
        }

# Global monitor instance
monitor = PerformanceMonitor()

# ============================================================================
# INTELLIGENT MEMORY CACHE
# ============================================================================

class MemoryCache:
    """Thread-safe in-memory cache with TTL support"""
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.ttl_settings = {
            "short": 300,    # 5 minutes
            "medium": 3600,  # 1 hour
            "long": 86400    # 24 hours
        }
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        if "expires_at" not in entry:
            return False
        return time.time() > entry["expires_at"]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if not self._is_expired(entry):
                    entry["hits"] = entry.get("hits", 0) + 1
                    entry["last_accessed"] = time.time()
                    return entry["value"]
                else:
                    del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Union[str, int] = "medium"):
        """Set value in cache with TTL"""
        with self.lock:
            if isinstance(ttl, str):
                ttl_seconds = self.ttl_settings.get(ttl, 3600)
            else:
                ttl_seconds = ttl
            
            self.cache[key] = {
                "value": value,
                "created_at": time.time(),
                "expires_at": time.time() + ttl_seconds,
                "hits": 0,
                "last_accessed": time.time()
            }
    
    def invalidate(self, pattern: str = None):
        """Invalidate cache entries"""
        with self.lock:
            if pattern:
                keys_to_remove = [k for k in self.cache if pattern in k]
                for key in keys_to_remove:
                    del self.cache[key]
            else:
                self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_entries = len(self.cache)
            total_hits = sum(entry.get("hits", 0) for entry in self.cache.values())
            
            return {
                "total_entries": total_entries,
                "total_hits": total_hits,
                "memory_usage": sum(len(str(v)) for v in self.cache.values())
            }

# ============================================================================
# DATABASE CONNECTION POOL
# ============================================================================

class DatabasePool:
    """SQLite connection pool with optimizations"""
    
    def __init__(self, database_path: str, pool_size: int = 5):
        self.database_path = database_path
        self.pool_size = pool_size
        self.connections: Queue = Queue(maxsize=pool_size)
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        for _ in range(self.pool_size):
            conn = sqlite3.connect(self.database_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            
            # Enable optimizations
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            
            self.connections.put(conn)
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        conn = self.connections.get()
        try:
            yield conn
        finally:
            self.connections.put(conn)
    
    def execute_query(self, query: str, params: tuple = (), cache_key: str = None) -> List[Dict]:
        """Execute a query with optional caching"""
        # Check cache first if key provided
        if cache_key and cache:
            cached_result = cache.get(cache_key)
            if cached_result:
                return cached_result
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]
        
        # Cache results if key provided
        if cache_key and cache:
            cache.set(cache_key, results)
        
        return results
    
    def close(self):
        """Close all connections"""
        while not self.connections.empty():
            conn = self.connections.get()
            conn.close()

# ============================================================================
# PERFORMANCE DECORATORS
# ============================================================================

def cached(ttl: Union[str, int] = "medium", key_prefix: str = ""):
    """Decorator for automatic caching"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__, str(args), str(sorted(kwargs.items()))]
            cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Check cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Call function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result
            cache.set(cache_key, result, ttl)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__, str(args), str(sorted(kwargs.items()))]
            cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Check cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            cache.set(cache_key, result, ttl)
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def track_performance(request_type: str = "general"):
    """Decorator to track function performance"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            request_id = hashlib.md5(f"{time.time()}:{func.__name__}".encode()).hexdigest()[:8]
            metric = monitor.start_request(request_id, request_type)
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                metric.error = str(e)
                raise
            finally:
                monitor.complete_request(request_id)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            request_id = hashlib.md5(f"{time.time()}:{func.__name__}".encode()).hexdigest()[:8]
            metric = monitor.start_request(request_id, request_type)
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                metric.error = str(e)
                raise
            finally:
                monitor.complete_request(request_id)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# ============================================================================
# OPTIMIZED REQUEST PROCESSOR
# ============================================================================

class OptimizedRequestProcessor:
    """Optimized request processing with caching and pooling"""
    
    def __init__(self, database_path: str = "jarvis.db"):
        self.db_pool = DatabasePool(database_path)
        self.cache = cache
        self.request_handlers: Dict[str, Callable] = {}
    
    def register_handler(self, request_type: str, handler: Callable):
        """Register a request handler"""
        self.request_handlers[request_type] = handler
    
    @track_performance("request")
    @cached(ttl="short")
    async def process_request(self, request: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a request with optimizations"""
        request_type = self._identify_request_type(request)
        
        if request_type in self.request_handlers:
            handler = self.request_handlers[request_type]
            if asyncio.iscoroutinefunction(handler):
                result = await handler(request, context)
            else:
                result = handler(request, context)
        else:
            result = await self._default_handler(request, context)
        
        return {
            "request": request,
            "type": request_type,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "cached": False  # Will be True on cache hits
        }
    
    def _identify_request_type(self, request: str) -> str:
        """Identify request type for optimized routing"""
        request_lower = request.lower()
        
        # Quick pattern matching
        patterns = {
            "weather": ["weather", "temperature", "forecast"],
            "time": ["time", "date", "clock", "when"],
            "search": ["search", "find", "look for", "query"],
            "calculate": ["calculate", "compute", "math", "solve"],
            "reminder": ["remind", "reminder", "schedule", "alarm"]
        }
        
        for req_type, keywords in patterns.items():
            if any(keyword in request_lower for keyword in keywords):
                return req_type
        
        return "general"
    
    async def _default_handler(self, request: str, context: Dict[str, Any]) -> Any:
        """Default request handler"""
        return {
            "response": f"Processed request: {request}",
            "context": context
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            "request_stats": monitor.get_stats(),
            "cache_stats": cache.get_stats(),
            "optimization_status": {
                "caching_enabled": True,
                "connection_pooling": True,
                "performance_tracking": True
            }
        }

# ============================================================================
# INITIALIZATION
# ============================================================================

# Global instances
cache = MemoryCache()

# Performance optimization utilities
def warm_cache(data: List[Dict[str, Any]], key_pattern: str = "warm:{id}"):
    """Pre-warm cache with frequently accessed data"""
    for item in data:
        key = key_pattern.format(**item)
        cache.set(key, item, ttl="long")

def clear_cache(pattern: str = None):
    """Clear cache entries"""
    cache.invalidate(pattern)

def get_performance_stats() -> Dict[str, Any]:
    """Get global performance statistics"""
    return {
        "monitor": monitor.get_stats(),
        "cache": cache.get_stats()
    }