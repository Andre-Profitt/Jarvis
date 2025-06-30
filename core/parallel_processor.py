"""
JARVIS Phase 10: Parallel Processor
Advanced parallel execution framework for multi-modal processing
"""

import asyncio
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import threading
import queue
import time
import logging
from collections import defaultdict
import ray
import dask
from dask import delayed
from dask.distributed import Client, as_completed as dask_completed
import torch
import torch.multiprocessing as torch_mp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParallelStrategy(Enum):
    """Parallel execution strategies"""
    THREAD = "thread_based"
    PROCESS = "process_based"
    ASYNC = "async_coroutines"
    DISTRIBUTED = "distributed_computing"
    GPU_PARALLEL = "gpu_parallel"
    HYBRID = "hybrid_approach"


@dataclass
class ParallelTask:
    """Task for parallel execution"""
    task_id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    dependencies: List[str] = None
    strategy: ParallelStrategy = ParallelStrategy.THREAD
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ParallelResult:
    """Result from parallel execution"""
    task_id: str
    result: Any
    execution_time: float
    strategy_used: ParallelStrategy
    success: bool = True
    error: Optional[Exception] = None


class ModalityProcessor:
    """
    Process different modalities in parallel
    """
    
    def __init__(self):
        self.processors = {
            'vision': self._process_vision,
            'audio': self._process_audio,
            'language': self._process_language,
            'biometric': self._process_biometric,
            'temporal': self._process_temporal,
            'spatial': self._process_spatial
        }
    
    async def _process_vision(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process visual input"""
        # Simulate vision processing
        await asyncio.sleep(0.1)
        return {'vision_features': np.random.rand(512), 'objects_detected': ['person', 'computer']}
    
    async def _process_audio(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio input"""
        await asyncio.sleep(0.05)
        return {'audio_features': np.random.rand(256), 'transcription': 'Hello JARVIS'}
    
    async def _process_language(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process language input"""
        await asyncio.sleep(0.08)
        return {'intent': 'query', 'entities': ['weather', 'tomorrow']}
    
    async def _process_biometric(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process biometric data"""
        await asyncio.sleep(0.02)
        return {'heart_rate': 72, 'stress_level': 0.3}
    
    async def _process_temporal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process temporal patterns"""
        await asyncio.sleep(0.03)
        return {'time_patterns': ['morning_routine'], 'anomalies': []}
    
    async def _process_spatial(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process spatial information"""
        await asyncio.sleep(0.04)
        return {'location': 'office', 'movement_pattern': 'stationary'}


class ParallelProcessor:
    """
    Advanced parallel processing engine for JARVIS
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        # Determine optimal worker count
        self.cpu_count = mp.cpu_count()
        self.max_workers = max_workers or self.cpu_count * 2
        
        # Thread pool for I/O bound tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Process pool for CPU bound tasks
        self.process_pool = ProcessPoolExecutor(max_workers=self.cpu_count)
        
        # Async event loop
        self.loop = asyncio.get_event_loop()
        
        # Task queues
        self.task_queue = asyncio.Queue()
        self.priority_queue = asyncio.PriorityQueue()
        
        # Results storage
        self.results = {}
        self.pending_tasks = {}
        
        # Modality processor
        self.modality_processor = ModalityProcessor()
        
        # GPU support
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            torch_mp.set_start_method('spawn', force=True)
            self.gpu_streams = [torch.cuda.Stream() for _ in range(4)]
        
        # Distributed computing support (optional)
        self.dask_client = None
        self.ray_initialized = False
        
        # Performance tracking
        self.execution_times = defaultdict(list)
        self.strategy_performance = defaultdict(lambda: {'total_time': 0, 'count': 0})
    
    async def process_multimodal_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process multiple input modalities in parallel
        """
        start_time = time.time()
        
        # Create tasks for each modality present in input
        tasks = []
        for modality, data in input_data.items():
            if modality in self.modality_processor.processors:
                processor = self.modality_processor.processors[modality]
                task = asyncio.create_task(processor(data))
                tasks.append((modality, task))
        
        # Process all modalities in parallel
        results = {}
        for modality, task in tasks:
            try:
                results[modality] = await task
            except Exception as e:
                logger.error(f"Error processing {modality}: {e}")
                results[modality] = {'error': str(e)}
        
        # Combine results
        combined_result = {
            'timestamp': time.time(),
            'processing_time': time.time() - start_time,
            'modalities': results
        }
        
        return combined_result
    
    async def execute_parallel_tasks(self, tasks: List[ParallelTask]) -> List[ParallelResult]:
        """
        Execute multiple tasks using optimal parallel strategies
        """
        # Group tasks by strategy
        strategy_groups = defaultdict(list)
        for task in tasks:
            strategy_groups[task.strategy].append(task)
        
        # Execute each group with its optimal strategy
        all_results = []
        
        for strategy, task_group in strategy_groups.items():
            if strategy == ParallelStrategy.THREAD:
                results = await self._execute_threaded(task_group)
            elif strategy == ParallelStrategy.PROCESS:
                results = await self._execute_multiprocess(task_group)
            elif strategy == ParallelStrategy.ASYNC:
                results = await self._execute_async(task_group)
            elif strategy == ParallelStrategy.GPU_PARALLEL:
                results = await self._execute_gpu_parallel(task_group)
            elif strategy == ParallelStrategy.DISTRIBUTED:
                results = await self._execute_distributed(task_group)
            else:  # HYBRID
                results = await self._execute_hybrid(task_group)
            
            all_results.extend(results)
        
        return all_results
    
    async def _execute_threaded(self, tasks: List[ParallelTask]) -> List[ParallelResult]:
        """
        Execute tasks using thread pool
        """
        futures = {}
        results = []
        
        # Submit all tasks
        for task in tasks:
            future = self.thread_pool.submit(
                self._execute_single_task,
                task
            )
            futures[future] = task
        
        # Collect results
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(ParallelResult(
                    task_id=task.task_id,
                    result=None,
                    execution_time=0,
                    strategy_used=ParallelStrategy.THREAD,
                    success=False,
                    error=e
                ))
        
        return results
    
    async def _execute_multiprocess(self, tasks: List[ParallelTask]) -> List[ParallelResult]:
        """
        Execute CPU-intensive tasks using process pool
        """
        futures = {}
        results = []
        
        for task in tasks:
            # Note: Functions must be pickleable for multiprocessing
            future = self.process_pool.submit(
                self._execute_single_task,
                task
            )
            futures[future] = task
        
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(ParallelResult(
                    task_id=task.task_id,
                    result=None,
                    execution_time=0,
                    strategy_used=ParallelStrategy.PROCESS,
                    success=False,
                    error=e
                ))
        
        return results
    
    async def _execute_async(self, tasks: List[ParallelTask]) -> List[ParallelResult]:
        """
        Execute I/O-bound tasks using asyncio
        """
        coroutines = []
        
        for task in tasks:
            coro = self._execute_async_task(task)
            coroutines.append(coro)
        
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        return [
            result if isinstance(result, ParallelResult) else 
            ParallelResult(
                task_id=tasks[i].task_id,
                result=None,
                execution_time=0,
                strategy_used=ParallelStrategy.ASYNC,
                success=False,
                error=result
            )
            for i, result in enumerate(results)
        ]
    
    async def _execute_gpu_parallel(self, tasks: List[ParallelTask]) -> List[ParallelResult]:
        """
        Execute tasks on GPU in parallel
        """
        if not self.gpu_available:
            # Fallback to CPU
            return await self._execute_threaded(tasks)
        
        results = []
        
        # Distribute tasks across GPU streams
        stream_tasks = [[] for _ in self.gpu_streams]
        for i, task in enumerate(tasks):
            stream_tasks[i % len(self.gpu_streams)].append(task)
        
        # Execute on each stream
        stream_futures = []
        for stream_idx, stream_task_list in enumerate(stream_tasks):
            if stream_task_list:
                future = self._execute_on_gpu_stream(
                    stream_task_list, 
                    self.gpu_streams[stream_idx]
                )
                stream_futures.append(future)
        
        # Collect results
        for future in asyncio.as_completed(stream_futures):
            stream_results = await future
            results.extend(stream_results)
        
        return results
    
    async def _execute_distributed(self, tasks: List[ParallelTask]) -> List[ParallelResult]:
        """
        Execute tasks using distributed computing
        """
        # Initialize Dask if not already done
        if self.dask_client is None:
            self.dask_client = Client(n_workers=4, threads_per_worker=2)
        
        results = []
        futures = []
        
        # Submit tasks to Dask
        for task in tasks:
            future = self.dask_client.submit(
                self._execute_single_task,
                task
            )
            futures.append((task, future))
        
        # Collect results
        for task, future in futures:
            try:
                result = await future
                results.append(result)
            except Exception as e:
                results.append(ParallelResult(
                    task_id=task.task_id,
                    result=None,
                    execution_time=0,
                    strategy_used=ParallelStrategy.DISTRIBUTED,
                    success=False,
                    error=e
                ))
        
        return results
    
    async def _execute_hybrid(self, tasks: List[ParallelTask]) -> List[ParallelResult]:
        """
        Use hybrid approach - automatically select best strategy per task
        """
        # Analyze tasks and select optimal strategy
        optimized_tasks = []
        
        for task in tasks:
            # Estimate task characteristics
            is_io_bound = self._is_io_bound(task)
            is_cpu_intensive = self._is_cpu_intensive(task)
            is_gpu_optimizable = self._is_gpu_optimizable(task)
            
            # Select strategy
            if is_gpu_optimizable and self.gpu_available:
                task.strategy = ParallelStrategy.GPU_PARALLEL
            elif is_cpu_intensive:
                task.strategy = ParallelStrategy.PROCESS
            elif is_io_bound:
                task.strategy = ParallelStrategy.ASYNC
            else:
                task.strategy = ParallelStrategy.THREAD
            
            optimized_tasks.append(task)
        
        # Execute with optimized strategies
        return await self.execute_parallel_tasks(optimized_tasks)
    
    def _execute_single_task(self, task: ParallelTask) -> ParallelResult:
        """
        Execute a single task (for thread/process pools)
        """
        start_time = time.time()
        
        try:
            result = task.function(*task.args, **task.kwargs)
            execution_time = time.time() - start_time
            
            return ParallelResult(
                task_id=task.task_id,
                result=result,
                execution_time=execution_time,
                strategy_used=task.strategy,
                success=True
            )
        except Exception as e:
            return ParallelResult(
                task_id=task.task_id,
                result=None,
                execution_time=time.time() - start_time,
                strategy_used=task.strategy,
                success=False,
                error=e
            )
    
    async def _execute_async_task(self, task: ParallelTask) -> ParallelResult:
        """
        Execute an async task
        """
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(task.function):
                result = await task.function(*task.args, **task.kwargs)
            else:
                # Run in executor if not async
                result = await self.loop.run_in_executor(
                    self.thread_pool,
                    task.function,
                    *task.args,
                    **task.kwargs
                )
            
            execution_time = time.time() - start_time
            
            return ParallelResult(
                task_id=task.task_id,
                result=result,
                execution_time=execution_time,
                strategy_used=ParallelStrategy.ASYNC,
                success=True
            )
        except Exception as e:
            return ParallelResult(
                task_id=task.task_id,
                result=None,
                execution_time=time.time() - start_time,
                strategy_used=ParallelStrategy.ASYNC,
                success=False,
                error=e
            )
    
    async def _execute_on_gpu_stream(self, tasks: List[ParallelTask], stream) -> List[ParallelResult]:
        """
        Execute tasks on a specific GPU stream
        """
        results = []
        
        with torch.cuda.stream(stream):
            for task in tasks:
                result = self._execute_single_task(task)
                results.append(result)
        
        return results
    
    def _is_io_bound(self, task: ParallelTask) -> bool:
        """
        Heuristic to determine if task is I/O bound
        """
        func_name = task.function.__name__
        io_indicators = ['read', 'write', 'fetch', 'download', 'request', 'query']
        return any(indicator in func_name.lower() for indicator in io_indicators)
    
    def _is_cpu_intensive(self, task: ParallelTask) -> bool:
        """
        Heuristic to determine if task is CPU intensive
        """
        func_name = task.function.__name__
        cpu_indicators = ['compute', 'calculate', 'process', 'analyze', 'transform']
        return any(indicator in func_name.lower() for indicator in cpu_indicators)
    
    def _is_gpu_optimizable(self, task: ParallelTask) -> bool:
        """
        Heuristic to determine if task can benefit from GPU
        """
        func_name = task.function.__name__
        gpu_indicators = ['neural', 'matrix', 'tensor', 'vision', 'deep']
        return any(indicator in func_name.lower() for indicator in gpu_indicators)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get performance statistics
        """
        report = {
            'total_tasks_executed': sum(s['count'] for s in self.strategy_performance.values()),
            'strategy_performance': {}
        }
        
        for strategy, stats in self.strategy_performance.items():
            if stats['count'] > 0:
                report['strategy_performance'][strategy.value] = {
                    'average_time': stats['total_time'] / stats['count'],
                    'total_tasks': stats['count']
                }
        
        return report
    
    async def shutdown(self):
        """
        Graceful shutdown
        """
        logger.info("Shutting down Parallel Processor")
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Shutdown process pool
        self.process_pool.shutdown(wait=True)
        
        # Close Dask client if exists
        if self.dask_client:
            await self.dask_client.close()
        
        # Shutdown Ray if initialized
        if self.ray_initialized:
            ray.shutdown()


# Example usage functions
async def example_cpu_task(data: np.ndarray) -> np.ndarray:
    """Example CPU-intensive task"""
    return np.dot(data, data.T)


async def example_io_task(url: str) -> str:
    """Example I/O-bound task"""
    await asyncio.sleep(0.1)  # Simulate network delay
    return f"Data from {url}"


async def example_gpu_task(tensor: torch.Tensor) -> torch.Tensor:
    """Example GPU-optimizable task"""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.matmul(tensor, tensor.T)