#!/usr/bin/env python3
"""
Resource Management and Rate Limiting for JARVIS
Controls resource allocation, prevents overload, and ensures fair usage
"""

import asyncio
import time
import psutil
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from contextlib import asynccontextmanager
import structlog
from functools import wraps

from .config_manager import config_manager
from .monitoring import monitoring_service, monitor_performance

logger = structlog.get_logger()


@dataclass
class ResourceLimit:
    """Resource limit configuration"""

    max_memory_mb: int
    max_cpu_percent: float
    max_concurrent_tasks: int
    max_file_handles: int = 100
    max_network_connections: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_mb": self.max_memory_mb,
            "cpu_percent": self.max_cpu_percent,
            "concurrent_tasks": self.max_concurrent_tasks,
            "file_handles": self.max_file_handles,
            "network_connections": self.max_network_connections,
        }


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""

    requests_per_minute: int
    burst_size: int
    cooldown_seconds: int = 60

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rpm": self.requests_per_minute,
            "burst": self.burst_size,
            "cooldown": self.cooldown_seconds,
        }


class TokenBucket:
    """Token bucket for rate limiting"""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens"""
        with self.lock:
            # Refill bucket
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            # Try to consume
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def wait_time(self, tokens: int = 1) -> float:
        """Time to wait for tokens to be available"""
        with self.lock:
            if self.tokens >= tokens:
                return 0.0
            needed = tokens - self.tokens
            return needed / self.refill_rate


class ResourceMonitor:
    """Monitors system resource usage"""

    def __init__(self):
        self.process = psutil.Process()
        self.history = deque(maxlen=60)  # Last 60 samples
        self.monitoring = False
        self._monitor_task = None

    async def start_monitoring(self):
        """Start monitoring resources"""
        self.monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Resource monitoring started")

    async def stop_monitoring(self):
        """Stop monitoring resources"""
        self.monitoring = False
        if self._monitor_task:
            await self._monitor_task
        logger.info("Resource monitoring stopped")

    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Collect metrics
                metrics = self.get_current_usage()
                self.history.append((time.time(), metrics))

                # Update Prometheus metrics
                monitoring_service.memory_usage.set(metrics["memory_bytes"])

                # Log if high usage
                if metrics["memory_percent"] > 80:
                    logger.warning(
                        f"High memory usage: {metrics['memory_percent']:.1f}%"
                    )
                if metrics["cpu_percent"] > 80:
                    logger.warning(f"High CPU usage: {metrics['cpu_percent']:.1f}%")

            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")

            await asyncio.sleep(1)  # Sample every second

    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        memory = psutil.virtual_memory()
        cpu_percent = self.process.cpu_percent(interval=0.1)

        return {
            "memory_bytes": self.process.memory_info().rss,
            "memory_mb": self.process.memory_info().rss / 1024 / 1024,
            "memory_percent": self.process.memory_percent(),
            "system_memory_percent": memory.percent,
            "cpu_percent": cpu_percent,
            "system_cpu_percent": psutil.cpu_percent(interval=0.1),
            "open_files": len(self.process.open_files()),
            "num_threads": self.process.num_threads(),
            "connections": len(self.process.connections()),
        }

    def get_average_usage(self, window_seconds: int = 60) -> Dict[str, float]:
        """Get average usage over time window"""
        if not self.history:
            return self.get_current_usage()

        cutoff_time = time.time() - window_seconds
        recent_samples = [(t, m) for t, m in self.history if t > cutoff_time]

        if not recent_samples:
            return self.get_current_usage()

        # Calculate averages
        avg_metrics = {}
        keys = recent_samples[0][1].keys()

        for key in keys:
            values = [m[key] for _, m in recent_samples]
            avg_metrics[f"avg_{key}"] = sum(values) / len(values)

        return avg_metrics


class ResourceManager:
    """Manages resource allocation and rate limiting"""

    def __init__(self):
        # Load configuration
        self.limits = self._load_limits()
        self.rate_limits = self._load_rate_limits()

        # Resource tracking
        self.active_tasks = set()
        self.task_resources = {}  # task_id -> resources used
        self.semaphores = {}  # resource_type -> semaphore

        # Rate limiting
        self.rate_limiters = {}  # endpoint -> TokenBucket
        self.request_history = defaultdict(deque)  # user -> timestamps

        # Monitoring
        self.monitor = ResourceMonitor()
        self.enforcement_enabled = config_manager.get(
            "resources.enforcement_enabled", True
        )

        # Initialize semaphores
        self._init_semaphores()

        logger.info("Resource manager initialized")

    def _load_limits(self) -> ResourceLimit:
        """Load resource limits from config"""
        return ResourceLimit(
            max_memory_mb=int(config_manager.get("resources.max_memory_mb", 2048)),
            max_cpu_percent=config_manager.get("resources.max_cpu_percent", 80.0),
            max_concurrent_tasks=config_manager.get(
                "resources.max_concurrent_tasks", 20
            ),
            max_file_handles=config_manager.get("resources.max_file_handles", 100),
            max_network_connections=config_manager.get(
                "resources.max_network_connections", 50
            ),
        )

    def _load_rate_limits(self) -> Dict[str, RateLimitConfig]:
        """Load rate limit configurations"""
        default_config = RateLimitConfig(
            requests_per_minute=60, burst_size=10, cooldown_seconds=60
        )

        # Load endpoint-specific limits
        limits = {
            "synthesis": RateLimitConfig(
                requests_per_minute=config_manager.get("rate_limits.synthesis.rpm", 30),
                burst_size=config_manager.get("rate_limits.synthesis.burst", 5),
            ),
            "ai_query": RateLimitConfig(
                requests_per_minute=config_manager.get("rate_limits.ai_query.rpm", 60),
                burst_size=config_manager.get("rate_limits.ai_query.burst", 10),
            ),
            "default": default_config,
        }

        return limits

    def _init_semaphores(self):
        """Initialize resource semaphores"""
        self.semaphores = {
            "tasks": asyncio.Semaphore(self.limits.max_concurrent_tasks),
            "files": asyncio.Semaphore(self.limits.max_file_handles),
            "network": asyncio.Semaphore(self.limits.max_network_connections),
        }

    async def start(self):
        """Start resource manager"""
        await self.monitor.start_monitoring()

    async def stop(self):
        """Stop resource manager"""
        await self.monitor.stop_monitoring()

    @asynccontextmanager
    async def acquire_resources(
        self,
        task_id: str,
        resource_types: List[str] = ["tasks"],
        estimated_memory_mb: float = 100,
    ):
        """Acquire resources for a task"""
        acquired = []

        try:
            # Check if we have capacity
            if not await self.check_capacity(estimated_memory_mb):
                raise ResourceLimitExceeded("Insufficient system resources")

            # Acquire semaphores
            for resource_type in resource_types:
                if resource_type in self.semaphores:
                    await self.semaphores[resource_type].acquire()
                    acquired.append(resource_type)

            # Track task
            self.active_tasks.add(task_id)
            self.task_resources[task_id] = {
                "start_time": time.time(),
                "resource_types": resource_types,
                "estimated_memory_mb": estimated_memory_mb,
            }

            logger.info(f"Resources acquired for task {task_id}")

            yield

        finally:
            # Release resources
            for resource_type in acquired:
                self.semaphores[resource_type].release()

            # Clean up tracking
            self.active_tasks.discard(task_id)
            if task_id in self.task_resources:
                duration = time.time() - self.task_resources[task_id]["start_time"]
                logger.info(f"Task {task_id} completed in {duration:.2f}s")
                del self.task_resources[task_id]

    async def check_capacity(self, required_memory_mb: float) -> bool:
        """Check if we have capacity for new task"""
        if not self.enforcement_enabled:
            return True

        usage = self.monitor.get_current_usage()

        # Check memory
        current_mb = usage["memory_mb"]
        if current_mb + required_memory_mb > self.limits.max_memory_mb:
            logger.warning(
                f"Memory limit would be exceeded: {current_mb + required_memory_mb}MB > {self.limits.max_memory_mb}MB"
            )
            return False

        # Check CPU
        if usage["cpu_percent"] > self.limits.max_cpu_percent:
            logger.warning(f"CPU usage too high: {usage['cpu_percent']}%")
            return False

        return True

    def check_rate_limit(
        self, endpoint: str, user_id: Optional[str] = None
    ) -> Tuple[bool, Optional[float]]:
        """Check if request is rate limited"""
        if not self.enforcement_enabled:
            return True, None

        # Get rate limit config
        config = self.rate_limits.get(endpoint, self.rate_limits["default"])

        # Get or create rate limiter
        key = f"{endpoint}:{user_id or 'global'}"
        if key not in self.rate_limiters:
            self.rate_limiters[key] = TokenBucket(
                capacity=config.burst_size,
                refill_rate=config.requests_per_minute / 60.0,
            )

        limiter = self.rate_limiters[key]

        # Try to consume token
        if limiter.consume():
            # Track request
            if user_id:
                self.request_history[user_id].append(time.time())
                # Keep only recent history
                cutoff = time.time() - 3600  # 1 hour
                while (
                    self.request_history[user_id]
                    and self.request_history[user_id][0] < cutoff
                ):
                    self.request_history[user_id].popleft()

            return True, None
        else:
            wait_time = limiter.wait_time()
            logger.warning(f"Rate limit exceeded for {key}, wait {wait_time:.1f}s")
            return False, wait_time

    def rate_limit_decorator(self, endpoint: str):
        """Decorator for rate limiting functions"""

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract user_id if available
                user_id = kwargs.get("user_id")

                # Check rate limit
                allowed, wait_time = self.check_rate_limit(endpoint, user_id)
                if not allowed:
                    raise ResourceLimitExceeded(
                        f"Rate limit exceeded. Please wait {wait_time:.1f} seconds."
                    )

                return await func(*args, **kwargs)

            return wrapper

        return decorator

    def get_status(self) -> Dict[str, Any]:
        """Get resource manager status"""
        usage = self.monitor.get_current_usage()
        avg_usage = self.monitor.get_average_usage(60)

        return {
            "limits": self.limits.to_dict(),
            "current_usage": usage,
            "average_usage": avg_usage,
            "active_tasks": len(self.active_tasks),
            "task_details": list(self.active_tasks),
            "rate_limiters": {
                key: {"tokens": limiter.tokens, "capacity": limiter.capacity}
                for key, limiter in self.rate_limiters.items()
            },
            "enforcement_enabled": self.enforcement_enabled,
        }

    def get_task_resources(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get resources used by a task"""
        return self.task_resources.get(task_id)

    async def wait_for_capacity(
        self, required_memory_mb: float, timeout: float = 30.0
    ) -> bool:
        """Wait for capacity to become available"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if await self.check_capacity(required_memory_mb):
                return True

            await asyncio.sleep(1)

        return False

    def cleanup_stale_tasks(self, max_age_seconds: int = 3600):
        """Clean up stale task records"""
        now = time.time()
        stale_tasks = []

        for task_id, info in self.task_resources.items():
            if now - info["start_time"] > max_age_seconds:
                stale_tasks.append(task_id)

        for task_id in stale_tasks:
            logger.warning(f"Cleaning up stale task: {task_id}")
            self.active_tasks.discard(task_id)
            del self.task_resources[task_id]

        return len(stale_tasks)


# Global resource manager instance
resource_manager = ResourceManager()


# Convenience decorators and functions
def with_resource_limit(
    resource_types: List[str] = ["tasks"], estimated_memory_mb: float = 100
):
    """Decorator to apply resource limits to a function"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            task_id = f"{func.__name__}_{time.time()}"

            async with resource_manager.acquire_resources(
                task_id, resource_types, estimated_memory_mb
            ):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def rate_limited(endpoint: str):
    """Decorator for rate limiting"""
    return resource_manager.rate_limit_decorator(endpoint)


async def check_system_resources() -> Dict[str, Any]:
    """Check current system resource status"""
    return resource_manager.get_status()
