#!/usr/bin/env python3
"""
Real-time Performance Monitor for JARVIS
Advanced monitoring with metrics collection, bottleneck detection, and auto-optimization
"""

import asyncio
import time
import psutil
import threading
import json
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
import cProfile
import pstats
import io
import tracemalloc
import gc
import sys
import os

# Visualization support
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.backends.backend_pdf import PdfPages
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Point-in-time performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read: float
    disk_io_write: float
    network_sent: float
    network_recv: float
    thread_count: int
    open_files: int
    response_time_avg: float
    response_time_p95: float
    active_tasks: int
    queued_tasks: int
    cache_hit_rate: float
    error_rate: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class BottleneckReport:
    """Detected performance bottleneck"""
    component: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    metric: str
    current_value: float
    threshold: float
    description: str
    suggested_action: str
    timestamp: datetime


class MetricsCollector:
    """Collect system and application metrics"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.last_disk_io = self.process.io_counters()
        self.last_net_io = psutil.net_io_counters()
        self.last_collect_time = time.time()
        
    def collect(self) -> Dict[str, float]:
        """Collect current metrics"""
        current_time = time.time()
        time_delta = current_time - self.last_collect_time
        
        # CPU metrics
        cpu_percent = self.process.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Disk I/O
        try:
            disk_io = self.process.io_counters()
            disk_read_rate = (disk_io.read_bytes - self.last_disk_io.read_bytes) / time_delta
            disk_write_rate = (disk_io.write_bytes - self.last_disk_io.write_bytes) / time_delta
            self.last_disk_io = disk_io
        except:
            disk_read_rate = 0
            disk_write_rate = 0
        
        # Network I/O
        net_io = psutil.net_io_counters()
        net_sent_rate = (net_io.bytes_sent - self.last_net_io.bytes_sent) / time_delta
        net_recv_rate = (net_io.bytes_recv - self.last_net_io.bytes_recv) / time_delta
        self.last_net_io = net_io
        
        # Process info
        try:
            thread_count = self.process.num_threads()
            open_files = len(self.process.open_files())
        except:
            thread_count = 0
            open_files = 0
        
        self.last_collect_time = current_time
        
        return {
            'cpu_percent': cpu_percent,
            'cpu_count': cpu_count,
            'memory_percent': memory_percent,
            'memory_mb': memory_mb,
            'disk_read_rate': disk_read_rate,
            'disk_write_rate': disk_write_rate,
            'net_sent_rate': net_sent_rate,
            'net_recv_rate': net_recv_rate,
            'thread_count': thread_count,
            'open_files': open_files
        }


class PerformanceProfiler:
    """Advanced performance profiling"""
    
    def __init__(self):
        self.profiles = {}
        self.call_counts = defaultdict(int)
        self.total_times = defaultdict(float)
        self.memory_snapshots = []
        self.profiling_enabled = True
        
    def profile(self, name: str):
        """Decorator for profiling functions"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.profiling_enabled:
                    return func(*args, **kwargs)
                
                # CPU profiling
                pr = cProfile.Profile()
                pr.enable()
                
                # Memory profiling
                tracemalloc.start()
                snapshot1 = tracemalloc.take_snapshot()
                
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    pr.disable()
                    tracemalloc.stop()
                    raise e
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Stop profiling
                pr.disable()
                snapshot2 = tracemalloc.take_snapshot()
                tracemalloc.stop()
                
                # Store results
                self.call_counts[name] += 1
                self.total_times[name] += execution_time
                
                # Store profile
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
                ps.print_stats(10)  # Top 10 functions
                self.profiles[name] = s.getvalue()
                
                # Memory diff
                top_stats = snapshot2.compare_to(snapshot1, 'lineno')
                memory_growth = sum(stat.size_diff for stat in top_stats)
                
                if memory_growth > 1024 * 1024:  # 1MB threshold
                    logger.warning(f"{name} allocated {memory_growth / 1024 / 1024:.2f} MB")
                
                return result
            
            return wrapper
        return decorator
    
    def get_top_functions(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top time-consuming functions"""
        avg_times = {
            name: self.total_times[name] / self.call_counts[name]
            for name in self.call_counts
        }
        return sorted(avg_times.items(), key=lambda x: x[1], reverse=True)[:n]


class BottleneckDetector:
    """Detect and analyze performance bottlenecks"""
    
    def __init__(self):
        self.thresholds = {
            'cpu_percent': {'warning': 70, 'critical': 90},
            'memory_percent': {'warning': 70, 'critical': 85},
            'response_time_p95': {'warning': 1.0, 'critical': 5.0},
            'error_rate': {'warning': 0.05, 'critical': 0.1},
            'queue_depth': {'warning': 100, 'critical': 500},
            'cache_hit_rate': {'warning': 0.7, 'critical': 0.5}  # Lower is worse
        }
        
        self.bottleneck_history = deque(maxlen=1000)
        
    def analyze(self, metrics: MetricSnapshot) -> List[BottleneckReport]:
        """Analyze metrics for bottlenecks"""
        bottlenecks = []
        
        # CPU bottleneck
        if metrics.cpu_percent > self.thresholds['cpu_percent']['critical']:
            bottlenecks.append(BottleneckReport(
                component='CPU',
                severity='critical',
                metric='cpu_percent',
                current_value=metrics.cpu_percent,
                threshold=self.thresholds['cpu_percent']['critical'],
                description=f'CPU usage critically high at {metrics.cpu_percent:.1f}%',
                suggested_action='Consider scaling horizontally or optimizing CPU-intensive operations',
                timestamp=metrics.timestamp
            ))
        elif metrics.cpu_percent > self.thresholds['cpu_percent']['warning']:
            bottlenecks.append(BottleneckReport(
                component='CPU',
                severity='medium',
                metric='cpu_percent',
                current_value=metrics.cpu_percent,
                threshold=self.thresholds['cpu_percent']['warning'],
                description=f'CPU usage elevated at {metrics.cpu_percent:.1f}%',
                suggested_action='Monitor CPU usage and prepare for scaling',
                timestamp=metrics.timestamp
            ))
        
        # Memory bottleneck
        if metrics.memory_percent > self.thresholds['memory_percent']['critical']:
            bottlenecks.append(BottleneckReport(
                component='Memory',
                severity='critical',
                metric='memory_percent',
                current_value=metrics.memory_percent,
                threshold=self.thresholds['memory_percent']['critical'],
                description=f'Memory usage critical at {metrics.memory_percent:.1f}%',
                suggested_action='Immediate memory optimization needed. Check for memory leaks.',
                timestamp=metrics.timestamp
            ))
        
        # Response time bottleneck
        if metrics.response_time_p95 > self.thresholds['response_time_p95']['critical']:
            bottlenecks.append(BottleneckReport(
                component='Response Time',
                severity='high',
                metric='response_time_p95',
                current_value=metrics.response_time_p95,
                threshold=self.thresholds['response_time_p95']['critical'],
                description=f'P95 response time very high at {metrics.response_time_p95:.2f}s',
                suggested_action='Profile slow operations and optimize critical paths',
                timestamp=metrics.timestamp
            ))
        
        # Cache performance
        if metrics.cache_hit_rate < self.thresholds['cache_hit_rate']['critical']:
            bottlenecks.append(BottleneckReport(
                component='Cache',
                severity='medium',
                metric='cache_hit_rate',
                current_value=metrics.cache_hit_rate,
                threshold=self.thresholds['cache_hit_rate']['critical'],
                description=f'Cache hit rate low at {metrics.cache_hit_rate:.1%}',
                suggested_action='Review cache strategy and increase cache size if needed',
                timestamp=metrics.timestamp
            ))
        
        # Store in history
        for bottleneck in bottlenecks:
            self.bottleneck_history.append(bottleneck)
        
        return bottlenecks


class AutoOptimizer:
    """Automatic performance optimization"""
    
    def __init__(self, jarvis_core):
        self.jarvis = jarvis_core
        self.optimization_history = []
        self.applied_optimizations = set()
        
    async def apply_optimizations(self, bottlenecks: List[BottleneckReport]):
        """Apply automatic optimizations based on bottlenecks"""
        for bottleneck in bottlenecks:
            optimization_key = f"{bottleneck.component}:{bottleneck.metric}"
            
            if optimization_key in self.applied_optimizations:
                continue
            
            try:
                if bottleneck.component == 'Memory' and bottleneck.severity == 'critical':
                    await self._optimize_memory()
                    
                elif bottleneck.component == 'CPU' and bottleneck.severity in ['high', 'critical']:
                    await self._optimize_cpu()
                    
                elif bottleneck.component == 'Cache':
                    await self._optimize_cache()
                    
                elif bottleneck.component == 'Response Time':
                    await self._optimize_response_time()
                
                self.applied_optimizations.add(optimization_key)
                self.optimization_history.append({
                    'timestamp': datetime.now(),
                    'bottleneck': bottleneck,
                    'success': True
                })
                
            except Exception as e:
                logger.error(f"Optimization failed: {e}")
                self.optimization_history.append({
                    'timestamp': datetime.now(),
                    'bottleneck': bottleneck,
                    'success': False,
                    'error': str(e)
                })
    
    async def _optimize_memory(self):
        """Emergency memory optimization"""
        logger.warning("Applying emergency memory optimization")
        
        # Force garbage collection
        gc.collect()
        
        # Clear caches
        if hasattr(self.jarvis, 'cache'):
            # Keep only recent items
            cache_size = len(self.jarvis.cache.cache)
            if cache_size > 1000:
                # Clear 50% of cache
                keys_to_remove = list(self.jarvis.cache.cache.keys())[:cache_size // 2]
                for key in keys_to_remove:
                    del self.jarvis.cache.cache[key]
        
        # Reduce thread pool size
        if hasattr(self.jarvis, 'thread_pool'):
            self.jarvis.thread_pool._max_workers = max(2, self.jarvis.thread_pool._max_workers - 1)
    
    async def _optimize_cpu(self):
        """CPU optimization"""
        logger.warning("Applying CPU optimization")
        
        # Adjust task priorities
        if hasattr(self.jarvis, 'task_queue'):
            self.jarvis.task_queue.batch_size = max(5, self.jarvis.task_queue.batch_size - 2)
            self.jarvis.task_queue.batch_timeout = min(0.5, self.jarvis.task_queue.batch_timeout * 2)
    
    async def _optimize_cache(self):
        """Cache optimization"""
        logger.info("Optimizing cache configuration")
        
        if hasattr(self.jarvis, 'cache'):
            # Increase cache size
            self.jarvis.cache.max_size = int(self.jarvis.cache.max_size * 1.5)
            
            # Adjust TTL based on hit rate
            if self.jarvis.cache.hit_rate < 0.5:
                self.jarvis.cache.ttl = self.jarvis.cache.ttl * 2
    
    async def _optimize_response_time(self):
        """Response time optimization"""
        logger.info("Optimizing response time")
        
        # Increase parallelism
        if hasattr(self.jarvis, 'thread_pool'):
            current_workers = self.jarvis.thread_pool._max_workers
            cpu_count = psutil.cpu_count()
            if current_workers < cpu_count * 2:
                self.jarvis.thread_pool._max_workers = min(cpu_count * 2, current_workers + 2)


class RealtimePerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self, jarvis_core=None, update_interval: float = 1.0):
        self.jarvis = jarvis_core
        self.update_interval = update_interval
        
        # Components
        self.collector = MetricsCollector()
        self.profiler = PerformanceProfiler()
        self.detector = BottleneckDetector()
        self.optimizer = AutoOptimizer(jarvis_core) if jarvis_core else None
        
        # Data storage
        self.metrics_history = deque(maxlen=3600)  # 1 hour at 1s intervals
        self.current_metrics = None
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_task = None
        
        # Custom metric providers
        self.metric_providers = {}
        
        # Alerts
        self.alert_callbacks = []
        
        logger.info("Realtime Performance Monitor initialized")
    
    def register_metric_provider(self, name: str, provider: Callable[[], Dict[str, float]]):
        """Register custom metric provider"""
        self.metric_providers[name] = provider
    
    def register_alert_callback(self, callback: Callable[[BottleneckReport], None]):
        """Register alert callback"""
        self.alert_callbacks.append(callback)
    
    async def start(self):
        """Start monitoring"""
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
    
    async def stop(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitor_task:
            await self.monitor_task
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                self.current_metrics = metrics
                self.metrics_history.append(metrics)
                
                # Detect bottlenecks
                bottlenecks = self.detector.analyze(metrics)
                
                # Send alerts
                for bottleneck in bottlenecks:
                    if bottleneck.severity in ['high', 'critical']:
                        await self._send_alert(bottleneck)
                
                # Apply auto-optimizations
                if self.optimizer and bottlenecks:
                    await self.optimizer.apply_optimizations(bottlenecks)
                
                # Wait for next cycle
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.update_interval * 2)
    
    async def _collect_metrics(self) -> MetricSnapshot:
        """Collect all metrics"""
        # System metrics
        system_metrics = self.collector.collect()
        
        # Application metrics
        app_metrics = await self._collect_app_metrics()
        
        # Custom metrics
        custom_metrics = {}
        for name, provider in self.metric_providers.items():
            try:
                custom_metrics.update(provider())
            except Exception as e:
                logger.error(f"Custom metric provider {name} failed: {e}")
        
        return MetricSnapshot(
            timestamp=datetime.now(),
            cpu_percent=system_metrics['cpu_percent'],
            memory_percent=system_metrics['memory_percent'],
            memory_mb=system_metrics['memory_mb'],
            disk_io_read=system_metrics['disk_read_rate'],
            disk_io_write=system_metrics['disk_write_rate'],
            network_sent=system_metrics['net_sent_rate'],
            network_recv=system_metrics['net_recv_rate'],
            thread_count=system_metrics['thread_count'],
            open_files=system_metrics['open_files'],
            response_time_avg=app_metrics.get('response_time_avg', 0),
            response_time_p95=app_metrics.get('response_time_p95', 0),
            active_tasks=app_metrics.get('active_tasks', 0),
            queued_tasks=app_metrics.get('queued_tasks', 0),
            cache_hit_rate=app_metrics.get('cache_hit_rate', 0),
            error_rate=app_metrics.get('error_rate', 0),
            custom_metrics=custom_metrics
        )
    
    async def _collect_app_metrics(self) -> Dict[str, float]:
        """Collect application-specific metrics"""
        metrics = {}
        
        if self.jarvis:
            # Response times
            if hasattr(self.jarvis, 'monitor') and hasattr(self.jarvis.monitor, 'response_times'):
                times = list(self.jarvis.monitor.response_times)
                if times:
                    metrics['response_time_avg'] = np.mean(times)
                    metrics['response_time_p95'] = np.percentile(times, 95)
            
            # Cache metrics
            if hasattr(self.jarvis, 'cache'):
                metrics['cache_hit_rate'] = self.jarvis.cache.hit_rate
            
            # Task queue metrics
            if hasattr(self.jarvis, 'task_queue'):
                metrics['queued_tasks'] = self.jarvis.task_queue.queue.qsize()
                metrics['active_tasks'] = len(self.jarvis.task_queue.workers)
        
        return metrics
    
    async def _send_alert(self, bottleneck: BottleneckReport):
        """Send alert for bottleneck"""
        logger.warning(f"BOTTLENECK ALERT: {bottleneck.description}")
        
        for callback in self.alert_callbacks:
            try:
                await asyncio.create_task(callback(bottleneck))
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def get_current_metrics(self) -> Optional[MetricSnapshot]:
        """Get current metrics snapshot"""
        return self.current_metrics
    
    def get_metrics_summary(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """Get metrics summary for specified duration"""
        if not self.metrics_history:
            return {}
        
        # Get recent metrics
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate summary statistics
        summary = {
            'duration_minutes': duration_minutes,
            'sample_count': len(recent_metrics),
            'cpu': {
                'avg': np.mean([m.cpu_percent for m in recent_metrics]),
                'max': np.max([m.cpu_percent for m in recent_metrics]),
                'p95': np.percentile([m.cpu_percent for m in recent_metrics], 95)
            },
            'memory': {
                'avg_mb': np.mean([m.memory_mb for m in recent_metrics]),
                'max_mb': np.max([m.memory_mb for m in recent_metrics]),
                'avg_percent': np.mean([m.memory_percent for m in recent_metrics])
            },
            'response_time': {
                'avg': np.mean([m.response_time_avg for m in recent_metrics]),
                'p95': np.mean([m.response_time_p95 for m in recent_metrics])
            },
            'cache_hit_rate': np.mean([m.cache_hit_rate for m in recent_metrics]),
            'error_rate': np.mean([m.error_rate for m in recent_metrics]),
            'bottlenecks': len(self.detector.bottleneck_history)
        }
        
        return summary
    
    def export_metrics(self, filepath: str):
        """Export metrics history to file"""
        data = {
            'metrics': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'cpu_percent': m.cpu_percent,
                    'memory_mb': m.memory_mb,
                    'response_time_avg': m.response_time_avg,
                    'cache_hit_rate': m.cache_hit_rate
                }
                for m in self.metrics_history
            ],
            'bottlenecks': [
                {
                    'timestamp': b.timestamp.isoformat(),
                    'component': b.component,
                    'severity': b.severity,
                    'description': b.description
                }
                for b in self.detector.bottleneck_history
            ],
            'summary': self.get_metrics_summary(60)  # 1 hour summary
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")


# Example usage
if __name__ == "__main__":
    async def test_monitor():
        # Create mock JARVIS core
        class MockJarvis:
            def __init__(self):
                self.cache = type('obj', (object,), {'hit_rate': 0.85, 'max_size': 1000})()
                self.monitor = type('obj', (object,), {'response_times': deque([0.1, 0.2, 0.15, 0.3])})()
        
        jarvis = MockJarvis()
        monitor = RealtimePerformanceMonitor(jarvis)
        
        # Register custom metric
        monitor.register_metric_provider(
            'custom_metric',
            lambda: {'custom_value': np.random.random()}
        )
        
        # Register alert
        async def alert_handler(bottleneck):
            print(f"ALERT: {bottleneck.description}")
        
        monitor.register_alert_callback(alert_handler)
        
        # Start monitoring
        await monitor.start()
        
        # Run for 10 seconds
        await asyncio.sleep(10)
        
        # Get summary
        summary = monitor.get_metrics_summary(1)
        print(f"Performance Summary: {json.dumps(summary, indent=2)}")
        
        # Export metrics
        monitor.export_metrics("performance_report.json")
        
        # Stop
        await monitor.stop()
    
    asyncio.run(test_monitor())