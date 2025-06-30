"""
JARVIS Phase 10: Intelligent Cache
Multi-tier caching system with predictive warming
"""

import asyncio
import time
import pickle
import json
import hashlib
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict, defaultdict
import redis
import aioredis
import numpy as np
from functools import wraps
import threading
import sqlite3
import lz4.frame
import msgpack
from cachetools import TTLCache, LRUCache, LFUCache
import logging
from datetime import datetime, timedelta
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheTier(Enum):
    """Cache tier levels"""
    L1_MEMORY = "memory"  # In-memory, fastest
    L2_REDIS = "redis"    # Redis, fast
    L3_DISK = "disk"      # SQLite, slower but persistent
    

class CacheStrategy(Enum):
    """Caching strategies"""
    LRU = "least_recently_used"
    LFU = "least_frequently_used"
    TTL = "time_to_live"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"


@dataclass
class CacheEntry:
    """Single cache entry"""
    key: str
    value: Any
    size: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None
    tier: CacheTier = CacheTier.L1_MEMORY
    compression: bool = False
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def update_access(self):
        """Update access statistics"""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    tier_hits: Dict[CacheTier, int] = field(default_factory=lambda: defaultdict(int))
    tier_misses: Dict[CacheTier, int] = field(default_factory=lambda: defaultdict(int))
    average_latency: Dict[CacheTier, float] = field(default_factory=lambda: defaultdict(float))
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class PredictiveModel:
    """
    Simple predictive model for cache warming
    """
    
    def __init__(self):
        self.access_patterns = defaultdict(list)
        self.sequence_length = 10
        self.prediction_threshold = 0.7
        
    def record_access(self, key: str, context: Dict[str, Any]):
        """Record access pattern"""
        pattern = {
            'time': time.time(),
            'hour': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'context': context
        }
        self.access_patterns[key].append(pattern)
        
        # Keep only recent patterns
        if len(self.access_patterns[key]) > 100:
            self.access_patterns[key] = self.access_patterns[key][-100:]
    
    def predict_next_access(self, current_context: Dict[str, Any]) -> List[str]:
        """Predict which keys will be accessed next"""
        predictions = []
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()
        
        for key, patterns in self.access_patterns.items():
            if len(patterns) < 5:
                continue
            
            # Simple time-based prediction
            hourly_matches = sum(1 for p in patterns[-20:] if p['hour'] == current_hour)
            daily_matches = sum(1 for p in patterns[-20:] if p['day_of_week'] == current_day)
            
            score = (hourly_matches + daily_matches) / 40
            
            if score > self.prediction_threshold:
                predictions.append(key)
        
        return predictions[:10]  # Top 10 predictions


class IntelligentCache:
    """
    Multi-tier intelligent caching system with predictive warming
    """
    
    def __init__(self, 
                 max_memory_size: int = 1024 * 1024 * 100,  # 100MB
                 redis_url: str = "redis://localhost:6379",
                 disk_path: str = "/tmp/jarvis_cache.db"):
        
        # Cache configuration
        self.max_memory_size = max_memory_size
        self.current_memory_size = 0
        
        # L1: In-memory caches
        self.memory_lru = LRUCache(maxsize=1000)
        self.memory_lfu = LFUCache(maxsize=500)
        self.memory_ttl = TTLCache(maxsize=500, ttl=300)  # 5 min TTL
        
        # L2: Redis connection
        self.redis_url = redis_url
        self.redis_client = None
        self.redis_available = False
        self._init_redis()
        
        # L3: Disk cache (SQLite)
        self.disk_path = disk_path
        self.disk_conn = None
        self._init_disk_cache()
        
        # Statistics
        self.stats = CacheStats()
        
        # Predictive model
        self.predictor = PredictiveModel()
        
        # Background tasks
        self.background_tasks = []
        self.running = True
        self._start_background_tasks()
        
        # Lock for thread safety
        self.lock = threading.RLock()
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis.from_url(self.redis_url)
            self.redis_client.ping()
            self.redis_available = True
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_available = False
    
    def _init_disk_cache(self):
        """Initialize disk-based cache"""
        try:
            self.disk_conn = sqlite3.connect(self.disk_path, check_same_thread=False)
            self.disk_conn.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at REAL,
                    last_accessed REAL,
                    access_count INTEGER,
                    ttl REAL,
                    size INTEGER
                )
            ''')
            self.disk_conn.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache(last_accessed)')
            self.disk_conn.commit()
            logger.info("Disk cache initialized")
        except Exception as e:
            logger.error(f"Failed to initialize disk cache: {e}")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Eviction task
        eviction_task = threading.Thread(target=self._eviction_loop)
        eviction_task.daemon = True
        eviction_task.start()
        self.background_tasks.append(eviction_task)
        
        # Predictive warming task
        warming_task = threading.Thread(target=self._warming_loop)
        warming_task.daemon = True
        warming_task.start()
        self.background_tasks.append(warming_task)
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function and arguments"""
        # Create a hashable representation
        key_data = {
            'func': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def get(self, key: str, tier: Optional[CacheTier] = None) -> Optional[Any]:
        """
        Get value from cache
        """
        start_time = time.time()
        
        with self.lock:
            # Try each tier in order
            tiers = [tier] if tier else [CacheTier.L1_MEMORY, CacheTier.L2_REDIS, CacheTier.L3_DISK]
            
            for cache_tier in tiers:
                value = await self._get_from_tier(key, cache_tier)
                if value is not None:
                    # Update statistics
                    self.stats.hits += 1
                    self.stats.tier_hits[cache_tier] += 1
                    
                    # Update latency
                    latency = time.time() - start_time
                    self._update_latency(cache_tier, latency)
                    
                    # Promote to higher tiers if accessed from lower tier
                    if cache_tier != CacheTier.L1_MEMORY:
                        await self._promote_to_memory(key, value)
                    
                    return value
            
            # Cache miss
            self.stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None, 
                  tier: CacheTier = CacheTier.L1_MEMORY) -> bool:
        """
        Set value in cache
        """
        try:
            # Calculate size
            size = self._estimate_size(value)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                size=size,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=ttl,
                tier=tier
            )
            
            # Store in appropriate tier
            success = await self._set_in_tier(entry, tier)
            
            # Record access pattern
            self.predictor.record_access(key, {'operation': 'set'})
            
            return success
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def _get_from_tier(self, key: str, tier: CacheTier) -> Optional[Any]:
        """Get value from specific tier"""
        try:
            if tier == CacheTier.L1_MEMORY:
                # Check all memory caches
                for cache in [self.memory_lru, self.memory_lfu, self.memory_ttl]:
                    if key in cache:
                        return cache[key]
                        
            elif tier == CacheTier.L2_REDIS and self.redis_available:
                data = self.redis_client.get(key)
                if data:
                    return self._deserialize(data)
                    
            elif tier == CacheTier.L3_DISK and self.disk_conn:
                cursor = self.disk_conn.execute(
                    'SELECT value FROM cache WHERE key = ?', (key,)
                )
                row = cursor.fetchone()
                if row:
                    # Update access time
                    self.disk_conn.execute(
                        'UPDATE cache SET last_accessed = ?, access_count = access_count + 1 WHERE key = ?',
                        (time.time(), key)
                    )
                    return self._deserialize(row[0])
                    
        except Exception as e:
            logger.error(f"Error getting from {tier}: {e}")
            
        return None
    
    async def _set_in_tier(self, entry: CacheEntry, tier: CacheTier) -> bool:
        """Set value in specific tier"""
        try:
            if tier == CacheTier.L1_MEMORY:
                # Check memory constraints
                if self.current_memory_size + entry.size > self.max_memory_size:
                    await self._evict_memory(entry.size)
                
                # Store in appropriate memory cache
                if entry.ttl:
                    self.memory_ttl[entry.key] = entry.value
                else:
                    self.memory_lru[entry.key] = entry.value
                
                self.current_memory_size += entry.size
                return True
                
            elif tier == CacheTier.L2_REDIS and self.redis_available:
                data = self._serialize(entry.value)
                if entry.ttl:
                    self.redis_client.setex(entry.key, int(entry.ttl), data)
                else:
                    self.redis_client.set(entry.key, data)
                return True
                
            elif tier == CacheTier.L3_DISK and self.disk_conn:
                data = self._serialize(entry.value)
                self.disk_conn.execute(
                    '''INSERT OR REPLACE INTO cache 
                    (key, value, created_at, last_accessed, access_count, ttl, size) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (entry.key, data, entry.created_at, entry.last_accessed, 
                     entry.access_count, entry.ttl, entry.size)
                )
                self.disk_conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error setting in {tier}: {e}")
            
        return False
    
    async def _promote_to_memory(self, key: str, value: Any):
        """Promote value to memory cache"""
        size = self._estimate_size(value)
        if size < self.max_memory_size * 0.1:  # Only promote if < 10% of memory
            entry = CacheEntry(
                key=key,
                value=value,
                size=size,
                created_at=time.time(),
                last_accessed=time.time()
            )
            await self._set_in_tier(entry, CacheTier.L1_MEMORY)
    
    async def _evict_memory(self, required_size: int):
        """Evict entries from memory to make space"""
        # Simple eviction: remove least recently used
        evicted_size = 0
        
        while evicted_size < required_size and len(self.memory_lru) > 0:
            # Get least recently used item
            key = next(iter(self.memory_lru))
            value = self.memory_lru.pop(key)
            
            size = self._estimate_size(value)
            evicted_size += size
            self.current_memory_size -= size
            self.stats.evictions += 1
            
            # Optionally demote to lower tier
            await self.set(key, value, tier=CacheTier.L2_REDIS)
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        try:
            # Try msgpack first (faster)
            return msgpack.packb(value)
        except:
            # Fallback to pickle
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        try:
            # Try msgpack first
            return msgpack.unpackb(data)
        except:
            # Fallback to pickle
            return pickle.loads(data)
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) 
                         for k, v in value.items())
            elif isinstance(value, np.ndarray):
                return value.nbytes
            elif isinstance(value, torch.Tensor):
                return value.element_size() * value.nelement()
            else:
                # Fallback: serialize and measure
                return len(self._serialize(value))
        except:
            return 1000  # Default size
    
    def _update_latency(self, tier: CacheTier, latency: float):
        """Update average latency for tier"""
        current = self.stats.average_latency[tier]
        # Exponential moving average
        self.stats.average_latency[tier] = 0.9 * current + 0.1 * latency
    
    def _eviction_loop(self):
        """Background eviction task"""
        while self.running:
            try:
                time.sleep(60)  # Run every minute
                
                # Clean expired entries
                self._clean_expired()
                
                # Balance tiers
                self._balance_tiers()
                
            except Exception as e:
                logger.error(f"Eviction loop error: {e}")
    
    def _warming_loop(self):
        """Background predictive warming task"""
        while self.running:
            try:
                time.sleep(30)  # Run every 30 seconds
                
                # Get predictions
                predictions = self.predictor.predict_next_access({})
                
                # Warm cache with predictions
                for key in predictions:
                    # Check if already in L1
                    if key not in self.memory_lru:
                        # Try to promote from lower tiers
                        asyncio.run(self.get(key))
                
            except Exception as e:
                logger.error(f"Warming loop error: {e}")
    
    def _clean_expired(self):
        """Clean expired entries from all tiers"""
        # Clean memory TTL cache (handled automatically by TTLCache)
        
        # Clean Redis
        if self.redis_available:
            # Redis handles TTL automatically
            pass
        
        # Clean disk
        if self.disk_conn:
            current_time = time.time()
            self.disk_conn.execute(
                'DELETE FROM cache WHERE ttl IS NOT NULL AND created_at + ttl < ?',
                (current_time,)
            )
            self.disk_conn.commit()
    
    def _balance_tiers(self):
        """Balance data across tiers based on access patterns"""
        # Move frequently accessed disk items to Redis
        if self.disk_conn and self.redis_available:
            cursor = self.disk_conn.execute(
                '''SELECT key, value, access_count FROM cache 
                WHERE access_count > 10 
                ORDER BY access_count DESC LIMIT 10'''
            )
            for row in cursor:
                key, value, _ = row
                self.redis_client.set(key, value)
    
    def cache_decorator(self, ttl: Optional[float] = None, 
                       tier: CacheTier = CacheTier.L1_MEMORY):
        """
        Decorator for caching function results
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                key = self._generate_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                cached = await self.get(key)
                if cached is not None:
                    return cached
                
                # Execute function
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                # Store in cache
                await self.set(key, result, ttl=ttl, tier=tier)
                
                return result
            
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'hit_rate': self.stats.hit_rate,
            'total_hits': self.stats.hits,
            'total_misses': self.stats.misses,
            'evictions': self.stats.evictions,
            'memory_usage': f"{self.current_memory_size / 1024 / 1024:.2f} MB",
            'tier_performance': {
                tier.value: {
                    'hits': self.stats.tier_hits[tier],
                    'avg_latency_ms': self.stats.average_latency[tier] * 1000
                }
                for tier in CacheTier
            }
        }
    
    def emergency_cleanup(self):
        """Emergency cleanup to free memory"""
        logger.warning("Emergency cache cleanup triggered")
        
        # Clear half of memory cache
        to_remove = len(self.memory_lru) // 2
        for _ in range(to_remove):
            if len(self.memory_lru) > 0:
                self.memory_lru.popitem()
        
        # Update size estimate
        self.current_memory_size = self.current_memory_size // 2
        
        # Force garbage collection
        import gc
        gc.collect()
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Intelligent Cache")
        self.running = False
        
        # Wait for background tasks
        for task in self.background_tasks:
            task.join(timeout=5)
        
        # Close connections
        if self.redis_client:
            self.redis_client.close()
        
        if self.disk_conn:
            self.disk_conn.close()


# Global cache instance
_cache_instance = None

def get_cache() -> IntelligentCache:
    """Get global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = IntelligentCache()
    return _cache_instance


# Convenience decorators
def cached(ttl: Optional[float] = None, tier: CacheTier = CacheTier.L1_MEMORY):
    """Convenience decorator for caching"""
    cache = get_cache()
    return cache.cache_decorator(ttl, tier)