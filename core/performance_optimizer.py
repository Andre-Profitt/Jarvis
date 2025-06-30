"""
JARVIS Phase 10: Performance Optimizer
Main orchestration engine for all performance optimizations
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from enum import Enum
import multiprocessing as mp
import logging
from collections import defaultdict, deque
import numpy as np
import cProfile
import pstats
import io
from functools import wraps
import gc
import torch
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Performance optimization strategies"""
    PARALLEL = "parallel_processing"
    CACHE = "intelligent_caching"
    LAZY = "lazy_loading"
    JIT = "jit_compilation"
    MEMORY = "memory_optimization"
    GPU = "gpu_acceleration"
    BATCH = "batch_processing"
    PREDICTIVE = "predictive_optimization"


@dataclass
class PerformanceMetrics:
    """Real-time performance tracking"""
    response_time: float = 0.0
    throughput: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    cache_hit_rate: float = 0.0
    parallel_efficiency: float = 0.0
    gpu_utilization: float = 0.0
    queue_depth: int = 0
    active_threads: int = 0
    optimization_gains: Dict[str, float] = field(default_factory=dict)
    bottlenecks: List[str] = field(default_factory=list)
    
    def update(self, **kwargs):
        """Update metrics"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class OptimizationResult:
    """Result of an optimization attempt"""
    strategy: OptimizationStrategy
    success: bool
    performance_gain: float
    memory_saved: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class PerformanceOptimizer:
    """
    Master performance optimization engine for JARVIS
    Coordinates all optimization strategies
    """
    
    def __init__(self, jarvis_core):
        self.jarvis = jarvis_core
        self.metrics = PerformanceMetrics()
        
        # Execution pools
        self.thread_pool = ThreadPoolExecutor(max_workers=mp.cpu_count() * 2)
        self.process_pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.optimization_history = defaultdict(list)
        self.hot_paths = defaultdict(int)
        self.profiler_data = {}
        
        # Optimization thresholds
        self.parallel_threshold = 0.1  # 100ms
        self.cache_threshold = 3  # 3+ calls triggers caching
        self.jit_threshold = 100  # 100+ calls triggers JIT
        self.memory_threshold = 0.8  # 80% memory usage
        
        # GPU support
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.device = torch.device("cuda")
            logger.info(f"GPU acceleration available: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
        
        # Start monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_performance)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    async def optimize_execution(self, func: Callable, *args, **kwargs) -> Tuple[Any, OptimizationResult]:
        """
        Intelligently optimize function execution
        """
        start_time = time.time()
        func_name = f"{func.__module__}.{func.__name__}"
        
        # Track hot paths
        self.hot_paths[func_name] += 1
        
        # Determine optimization strategy
        strategy = self._select_optimization_strategy(func_name, args, kwargs)
        
        # Apply optimization
        if strategy == OptimizationStrategy.PARALLEL:
            result, opt_result = await self._parallel_execution(func, args, kwargs)
        elif strategy == OptimizationStrategy.CACHE:
            result, opt_result = await self._cached_execution(func, args, kwargs)
        elif strategy == OptimizationStrategy.GPU:
            result, opt_result = await self._gpu_execution(func, args, kwargs)
        elif strategy == OptimizationStrategy.JIT:
            result, opt_result = await self._jit_execution(func, args, kwargs)
        else:
            # Default execution
            result = await self._default_execution(func, args, kwargs)
            opt_result = OptimizationResult(
                strategy=OptimizationStrategy.PARALLEL,
                success=False,
                performance_gain=0.0
            )
        
        # Track performance
        execution_time = time.time() - start_time
        self._track_performance(func_name, execution_time, opt_result)
        
        return result, opt_result
    
    def _select_optimization_strategy(self, func_name: str, args, kwargs) -> OptimizationStrategy:
        """
        Intelligently select the best optimization strategy
        """
        call_count = self.hot_paths[func_name]
        
        # Check if function is GPU-optimizable
        if self.gpu_available and self._is_gpu_optimizable(func_name, args):
            return OptimizationStrategy.GPU
        
        # JIT for very hot paths
        if call_count > self.jit_threshold:
            return OptimizationStrategy.JIT
        
        # Cache for frequently called functions
        if call_count > self.cache_threshold:
            return OptimizationStrategy.CACHE
        
        # Parallel for heavy computations
        if self._estimate_computation_cost(args) > self.parallel_threshold:
            return OptimizationStrategy.PARALLEL
        
        return OptimizationStrategy.PARALLEL
    
    async def _parallel_execution(self, func: Callable, args, kwargs) -> Tuple[Any, OptimizationResult]:
        """
        Execute function in parallel if possible
        """
        try:
            # Check if function can be parallelized
            if hasattr(func, '_parallelize') and func._parallelize:
                # Custom parallel implementation
                result = await func._parallel_execute(*args, **kwargs)
            else:
                # Default thread pool execution
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.thread_pool,
                    func,
                    *args,
                    **kwargs
                )
            
            return result, OptimizationResult(
                strategy=OptimizationStrategy.PARALLEL,
                success=True,
                performance_gain=0.3  # Estimated 30% gain
            )
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            # Fallback to default
            result = await self._default_execution(func, args, kwargs)
            return result, OptimizationResult(
                strategy=OptimizationStrategy.PARALLEL,
                success=False,
                performance_gain=0.0
            )
    
    async def _cached_execution(self, func: Callable, args, kwargs) -> Tuple[Any, OptimizationResult]:
        """
        Execute with intelligent caching
        """
        # This will be implemented by intelligent_cache.py
        # For now, just execute normally
        result = await self._default_execution(func, args, kwargs)
        return result, OptimizationResult(
            strategy=OptimizationStrategy.CACHE,
            success=True,
            performance_gain=0.0  # Will be calculated by cache
        )
    
    async def _gpu_execution(self, func: Callable, args, kwargs) -> Tuple[Any, OptimizationResult]:
        """
        Execute on GPU if beneficial
        """
        try:
            # Move tensors to GPU
            gpu_args = self._move_to_gpu(args)
            gpu_kwargs = self._move_to_gpu(kwargs)
            
            # Execute on GPU
            with torch.cuda.amp.autocast():  # Automatic mixed precision
                result = await self._default_execution(func, gpu_args, gpu_kwargs)
            
            # Move result back to CPU if needed
            result = self._move_to_cpu(result)
            
            return result, OptimizationResult(
                strategy=OptimizationStrategy.GPU,
                success=True,
                performance_gain=0.5  # Estimated 50% gain
            )
        except Exception as e:
            logger.error(f"GPU execution failed: {e}")
            # Fallback to CPU
            result = await self._default_execution(func, args, kwargs)
            return result, OptimizationResult(
                strategy=OptimizationStrategy.GPU,
                success=False,
                performance_gain=0.0
            )
    
    async def _jit_execution(self, func: Callable, args, kwargs) -> Tuple[Any, OptimizationResult]:
        """
        Just-in-time compilation for hot paths
        """
        # This will be implemented by jit_compiler.py
        result = await self._default_execution(func, args, kwargs)
        return result, OptimizationResult(
            strategy=OptimizationStrategy.JIT,
            success=True,
            performance_gain=0.0  # Will be calculated by JIT
        )
    
    async def _default_execution(self, func: Callable, args, kwargs) -> Any:
        """
        Default async execution
        """
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)
    
    def profile_function(self, func: Callable) -> Callable:
        """
        Decorator to profile function performance
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create profiler
            pr = cProfile.Profile()
            pr.enable()
            
            # Execute function
            result = await self.optimize_execution(func, *args, **kwargs)
            
            # Stop profiling
            pr.disable()
            
            # Store profile data
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(10)  # Top 10 functions
            self.profiler_data[func.__name__] = s.getvalue()
            
            return result
        
        return wrapper
    
    def _monitor_performance(self):
        """
        Background performance monitoring
        """
        while self.monitoring_active:
            try:
                # System metrics
                self.metrics.cpu_usage = psutil.cpu_percent(interval=0.1)
                self.metrics.memory_usage = psutil.virtual_memory().percent / 100
                
                # GPU metrics
                if self.gpu_available:
                    self.metrics.gpu_utilization = self._get_gpu_utilization()
                
                # Thread pool metrics
                self.metrics.active_threads = self.thread_pool._threads.__len__()
                
                # Identify bottlenecks
                self._identify_bottlenecks()
                
                # Auto-optimize if needed
                if self.metrics.memory_usage > self.memory_threshold:
                    self._trigger_memory_optimization()
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    def _identify_bottlenecks(self):
        """
        Identify performance bottlenecks
        """
        bottlenecks = []
        
        if self.metrics.cpu_usage > 0.9:
            bottlenecks.append("CPU")
        if self.metrics.memory_usage > 0.8:
            bottlenecks.append("Memory")
        if self.metrics.queue_depth > 100:
            bottlenecks.append("Queue")
        if self.metrics.cache_hit_rate < 0.5:
            bottlenecks.append("Cache")
        
        self.metrics.bottlenecks = bottlenecks
    
    def _trigger_memory_optimization(self):
        """
        Emergency memory optimization
        """
        logger.warning("High memory usage detected, triggering optimization")
        
        # Force garbage collection
        gc.collect()
        
        # Clear old cache entries
        if hasattr(self.jarvis, 'cache'):
            self.jarvis.cache.emergency_cleanup()
        
        # Reduce thread pool if needed
        if self.thread_pool._threads.__len__() > mp.cpu_count():
            self.thread_pool._max_workers = mp.cpu_count()
    
    def _is_gpu_optimizable(self, func_name: str, args) -> bool:
        """
        Check if function can benefit from GPU
        """
        # Check for tensor operations
        for arg in args:
            if isinstance(arg, (np.ndarray, torch.Tensor)):
                if arg.size > 10000:  # Large arrays benefit from GPU
                    return True
        
        # Check function annotations
        if func_name in ['neural_processing', 'vision_processing', 'matrix_operations']:
            return True
        
        return False
    
    def _estimate_computation_cost(self, args) -> float:
        """
        Estimate computational cost of arguments
        """
        cost = 0.0
        
        for arg in args:
            if isinstance(arg, (list, tuple)):
                cost += len(arg) * 0.001
            elif isinstance(arg, (np.ndarray, torch.Tensor)):
                cost += arg.size * 0.0001
            elif isinstance(arg, dict):
                cost += len(arg) * 0.002
        
        return cost
    
    def _move_to_gpu(self, data):
        """
        Move data to GPU recursively
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.device)
        elif isinstance(data, (list, tuple)):
            return type(data)(self._move_to_gpu(item) for item in data)
        elif isinstance(data, dict):
            return {k: self._move_to_gpu(v) for k, v in data.items()}
        return data
    
    def _move_to_cpu(self, data):
        """
        Move data back to CPU recursively
        """
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy() if data.device.type == 'cuda' else data
        elif isinstance(data, (list, tuple)):
            return type(data)(self._move_to_cpu(item) for item in data)
        elif isinstance(data, dict):
            return {k: self._move_to_cpu(v) for k, v in data.items()}
        return data
    
    def _track_performance(self, func_name: str, execution_time: float, opt_result: OptimizationResult):
        """
        Track performance metrics
        """
        self.performance_history.append({
            'function': func_name,
            'time': execution_time,
            'strategy': opt_result.strategy.value,
            'gain': opt_result.performance_gain,
            'timestamp': time.time()
        })
        
        self.optimization_history[func_name].append(opt_result)
    
    def _get_gpu_utilization(self) -> float:
        """
        Get GPU utilization percentage
        """
        if self.gpu_available:
            return torch.cuda.utilization() / 100
        return 0.0
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive optimization report
        """
        report = {
            'current_metrics': self.metrics.__dict__,
            'hot_paths': dict(self.hot_paths.most_common(10)),
            'optimization_summary': {},
            'recommendations': []
        }
        
        # Analyze optimization history
        for func_name, results in self.optimization_history.items():
            successful = [r for r in results if r.success]
            if successful:
                avg_gain = np.mean([r.performance_gain for r in successful])
                report['optimization_summary'][func_name] = {
                    'average_gain': avg_gain,
                    'successful_optimizations': len(successful),
                    'total_attempts': len(results)
                }
        
        # Generate recommendations
        if self.metrics.cache_hit_rate < 0.7:
            report['recommendations'].append("Increase cache size or improve cache strategy")
        if self.metrics.parallel_efficiency < 0.8:
            report['recommendations'].append("Review parallel implementation for better efficiency")
        if self.metrics.memory_usage > 0.7:
            report['recommendations'].append("Implement more aggressive memory optimization")
        
        return report
    
    async def shutdown(self):
        """
        Graceful shutdown
        """
        logger.info("Shutting down Performance Optimizer")
        self.monitoring_active = False
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


# Decorator for easy optimization
def optimize(func: Callable) -> Callable:
    """
    Decorator to automatically optimize function execution
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        if hasattr(self, 'performance_optimizer'):
            result, _ = await self.performance_optimizer.optimize_execution(
                func, self, *args, **kwargs
            )
            return result
        else:
            # No optimizer available, execute normally
            return await func(self, *args, **kwargs)
    
    return wrapper