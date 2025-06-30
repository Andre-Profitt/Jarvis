"""
JARVIS Phase 10: JIT Compiler
Just-in-time compilation for hot code paths
"""

import ast
import inspect
import time
import dis
import types
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import numba
from numba import jit, njit, vectorize, cuda
import torch
import torch.jit as torch_jit
from functools import wraps
import logging
from collections import defaultdict
import cython
import pyximport
pyximport.install(language_level=3)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompilationTier(Enum):
    """JIT compilation tiers"""
    INTERPRETER = 0      # Normal Python
    TIER1_QUICK = 1     # Quick optimization
    TIER2_OPTIMIZED = 2 # Full optimization
    TIER3_NATIVE = 3    # Native code


@dataclass
class FunctionProfile:
    """Profile data for a function"""
    name: str
    call_count: int = 0
    total_time: float = 0.0
    average_time: float = 0.0
    last_compilation_tier: CompilationTier = CompilationTier.INTERPRETER
    compilation_time: float = 0.0
    speedup: float = 1.0
    arg_types: List[type] = field(default_factory=list)
    hot_loops: List[str] = field(default_factory=list)
    compilation_failures: int = 0


@dataclass 
class OptimizationResult:
    """Result of JIT optimization"""
    success: bool
    tier: CompilationTier
    speedup: float
    compilation_time: float
    compiled_func: Optional[Callable] = None
    error: Optional[str] = None


class JITCompiler:
    """
    Adaptive JIT compiler for JARVIS
    """
    
    def __init__(self):
        # Compilation thresholds
        self.tier1_threshold = 100      # Calls before tier 1
        self.tier2_threshold = 1000     # Calls before tier 2
        self.tier3_threshold = 10000    # Calls before tier 3
        
        # Function profiles
        self.profiles: Dict[str, FunctionProfile] = {}
        
        # Compiled function cache
        self.compiled_cache: Dict[str, Dict[CompilationTier, Callable]] = defaultdict(dict)
        
        # Performance tracking
        self.compilation_stats = {
            'total_compilations': 0,
            'successful_compilations': 0,
            'failed_compilations': 0,
            'total_speedup': 0.0
        }
        
        # GPU availability
        self.cuda_available = cuda.is_available()
        
        # Supported types for numba
        self.numba_types = (int, float, np.ndarray, list, tuple)
    
    def profile_and_compile(self, func: Callable) -> Callable:
        """
        Decorator to profile and JIT compile functions
        """
        func_name = f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create profile
            if func_name not in self.profiles:
                self.profiles[func_name] = FunctionProfile(name=func_name)
            
            profile = self.profiles[func_name]
            
            # Check if we should compile
            compiled_func = self._get_compiled_function(func, profile, args)
            
            # Execute and time
            start_time = time.time()
            
            try:
                if compiled_func:
                    result = compiled_func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
            except Exception as e:
                # Fallback to original on compilation errors
                logger.warning(f"Compiled function failed, falling back: {e}")
                result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Update profile
            profile.call_count += 1
            profile.total_time += execution_time
            profile.average_time = profile.total_time / profile.call_count
            
            # Record arg types for better compilation
            if not profile.arg_types and args:
                profile.arg_types = [type(arg) for arg in args]
            
            return result
        
        # Store original function reference
        wrapper._original_func = func
        
        return wrapper
    
    def _get_compiled_function(self, func: Callable, profile: FunctionProfile, 
                              args: tuple) -> Optional[Callable]:
        """
        Get appropriate compiled version based on profile
        """
        # Determine compilation tier
        tier = self._determine_tier(profile)
        
        # Check cache
        if tier in self.compiled_cache[profile.name]:
            return self.compiled_cache[profile.name][tier]
        
        # Compile if threshold reached
        if tier != CompilationTier.INTERPRETER:
            result = self._compile_function(func, profile, tier, args)
            if result.success and result.compiled_func:
                self.compiled_cache[profile.name][tier] = result.compiled_func
                profile.last_compilation_tier = tier
                profile.speedup = result.speedup
                return result.compiled_func
        
        return None
    
    def _determine_tier(self, profile: FunctionProfile) -> CompilationTier:
        """
        Determine appropriate compilation tier
        """
        if profile.call_count < self.tier1_threshold:
            return CompilationTier.INTERPRETER
        elif profile.call_count < self.tier2_threshold:
            return CompilationTier.TIER1_QUICK
        elif profile.call_count < self.tier3_threshold:
            return CompilationTier.TIER2_OPTIMIZED
        else:
            return CompilationTier.TIER3_NATIVE
    
    def _compile_function(self, func: Callable, profile: FunctionProfile,
                         tier: CompilationTier, args: tuple) -> OptimizationResult:
        """
        Compile function to specified tier
        """
        start_time = time.time()
        
        try:
            if tier == CompilationTier.TIER1_QUICK:
                compiled = self._tier1_compile(func, args)
            elif tier == CompilationTier.TIER2_OPTIMIZED:
                compiled = self._tier2_compile(func, args)
            elif tier == CompilationTier.TIER3_NATIVE:
                compiled = self._tier3_compile(func, args)
            else:
                return OptimizationResult(False, tier, 1.0, 0.0, error="Invalid tier")
            
            compilation_time = time.time() - start_time
            
            # Benchmark speedup
            speedup = self._benchmark_speedup(func, compiled, args)
            
            self.compilation_stats['total_compilations'] += 1
            self.compilation_stats['successful_compilations'] += 1
            self.compilation_stats['total_speedup'] += speedup
            
            return OptimizationResult(
                success=True,
                tier=tier,
                speedup=speedup,
                compilation_time=compilation_time,
                compiled_func=compiled
            )
            
        except Exception as e:
            compilation_time = time.time() - start_time
            self.compilation_stats['total_compilations'] += 1
            self.compilation_stats['failed_compilations'] += 1
            profile.compilation_failures += 1
            
            logger.error(f"JIT compilation failed for {profile.name}: {e}")
            
            return OptimizationResult(
                success=False,
                tier=tier,
                speedup=1.0,
                compilation_time=compilation_time,
                error=str(e)
            )
    
    def _tier1_compile(self, func: Callable, args: tuple) -> Callable:
        """
        Tier 1: Quick compilation with basic optimizations
        """
        # Check if function is suitable for numba
        if self._is_numba_compatible(func, args):
            # Use numba with minimal optimization
            return jit(nopython=False, cache=True)(func)
        
        # Try torch JIT for tensor operations
        if self._has_tensor_operations(func):
            return torch_jit.script(func)
        
        # Fallback: Python bytecode optimization
        return self._optimize_bytecode(func)
    
    def _tier2_compile(self, func: Callable, args: tuple) -> Callable:
        """
        Tier 2: Optimized compilation
        """
        # Aggressive numba compilation
        if self._is_numba_compatible(func, args):
            # Analyze function for parallel opportunities
            if self._has_parallel_loops(func):
                return jit(nopython=True, parallel=True, cache=True)(func)
            else:
                return jit(nopython=True, cache=True)(func)
        
        # CUDA compilation for suitable functions
        if self.cuda_available and self._is_cuda_suitable(func):
            return cuda.jit(func)
        
        # Torch optimization
        if self._has_tensor_operations(func):
            traced = torch_jit.trace(func, args)
            return torch_jit.optimize_for_inference(traced)
        
        return self._optimize_bytecode(func)
    
    def _tier3_compile(self, func: Callable, args: tuple) -> Callable:
        """
        Tier 3: Native code generation
        """
        # Try Cython compilation
        cython_func = self._compile_with_cython(func)
        if cython_func:
            return cython_func
        
        # Fallback to tier 2
        return self._tier2_compile(func, args)
    
    def _is_numba_compatible(self, func: Callable, args: tuple) -> bool:
        """
        Check if function can be compiled with numba
        """
        # Check argument types
        for arg in args:
            if not isinstance(arg, self.numba_types):
                return False
        
        # Check function source
        try:
            source = inspect.getsource(func)
            # Look for incompatible patterns
            incompatible = ['yield', 'async', 'await', 'with', 'try', 'except']
            return not any(pattern in source for pattern in incompatible)
        except:
            return False
    
    def _has_tensor_operations(self, func: Callable) -> bool:
        """
        Check if function uses tensor operations
        """
        try:
            source = inspect.getsource(func)
            tensor_ops = ['torch', 'tensor', 'numpy', 'np.']
            return any(op in source for op in tensor_ops)
        except:
            return False
    
    def _has_parallel_loops(self, func: Callable) -> bool:
        """
        Check if function has parallelizable loops
        """
        try:
            source = inspect.getsource(func)
            # Simple heuristic - look for loops
            return 'for' in source and ('range' in source or 'enumerate' in source)
        except:
            return False
    
    def _is_cuda_suitable(self, func: Callable) -> bool:
        """
        Check if function is suitable for CUDA
        """
        try:
            source = inspect.getsource(func)
            # Look for array operations
            cuda_ops = ['dot', 'matmul', 'conv', 'sum', 'mean']
            return any(op in source for op in cuda_ops)
        except:
            return False
    
    def _optimize_bytecode(self, func: Callable) -> Callable:
        """
        Optimize Python bytecode
        """
        # Get bytecode
        code = func.__code__
        
        # Create optimized code object
        optimized_code = types.CodeType(
            code.co_argcount,
            code.co_posonlyargcount,
            code.co_kwonlyargcount,
            code.co_nlocals,
            code.co_stacksize,
            code.co_flags | 0x00000010,  # CO_OPTIMIZED flag
            code.co_code,
            code.co_consts,
            code.co_names,
            code.co_varnames,
            code.co_filename,
            code.co_name,
            code.co_firstlineno,
            code.co_lnotab,
            code.co_freevars,
            code.co_cellvars
        )
        
        # Create new function with optimized code
        return types.FunctionType(
            optimized_code,
            func.__globals__,
            func.__name__,
            func.__defaults__,
            func.__closure__
        )
    
    def _compile_with_cython(self, func: Callable) -> Optional[Callable]:
        """
        Compile function with Cython
        """
        # This would require generating .pyx file and compiling
        # For now, return None
        return None
    
    def _benchmark_speedup(self, original: Callable, compiled: Callable, 
                          args: tuple, iterations: int = 100) -> float:
        """
        Benchmark speedup of compiled function
        """
        # Warmup
        for _ in range(10):
            original(*args)
            compiled(*args)
        
        # Benchmark original
        start = time.time()
        for _ in range(iterations):
            original(*args)
        original_time = time.time() - start
        
        # Benchmark compiled
        start = time.time()
        for _ in range(iterations):
            compiled(*args)
        compiled_time = time.time() - start
        
        # Calculate speedup
        speedup = original_time / compiled_time if compiled_time > 0 else 1.0
        
        return speedup
    
    def get_hot_functions(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get hottest functions for optimization
        """
        # Sort by total time spent
        sorted_profiles = sorted(
            self.profiles.items(),
            key=lambda x: x[1].total_time,
            reverse=True
        )
        
        hot_functions = []
        for func_name, profile in sorted_profiles[:top_n]:
            hot_functions.append({
                'name': func_name,
                'call_count': profile.call_count,
                'total_time': profile.total_time,
                'average_time': profile.average_time,
                'tier': profile.last_compilation_tier.name,
                'speedup': profile.speedup
            })
        
        return hot_functions
    
    def adaptive_compile(self, module):
        """
        Adaptively compile all functions in a module
        """
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) or inspect.ismethod(obj):
                # Wrap with JIT compiler
                compiled = self.profile_and_compile(obj)
                setattr(module, name, compiled)
    
    def get_compilation_stats(self) -> Dict[str, Any]:
        """
        Get JIT compilation statistics
        """
        avg_speedup = (
            self.compilation_stats['total_speedup'] / 
            self.compilation_stats['successful_compilations']
            if self.compilation_stats['successful_compilations'] > 0 
            else 1.0
        )
        
        return {
            'total_compilations': self.compilation_stats['total_compilations'],
            'successful': self.compilation_stats['successful_compilations'],
            'failed': self.compilation_stats['failed_compilations'],
            'average_speedup': avg_speedup,
            'tier_distribution': self._get_tier_distribution(),
            'hot_functions': self.get_hot_functions()
        }
    
    def _get_tier_distribution(self) -> Dict[str, int]:
        """
        Get distribution of functions by compilation tier
        """
        distribution = defaultdict(int)
        
        for profile in self.profiles.values():
            distribution[profile.last_compilation_tier.name] += 1
        
        return dict(distribution)


# Specialized JIT decorators
def jit_compile(nopython=True, parallel=False, cache=True):
    """
    Decorator for Numba JIT compilation
    """
    def decorator(func):
        return jit(nopython=nopython, parallel=parallel, cache=cache)(func)
    return decorator


def torch_compile(optimize=True):
    """
    Decorator for PyTorch JIT compilation
    """
    def decorator(func):
        compiled = torch_jit.script(func)
        if optimize:
            compiled = torch_jit.optimize_for_inference(compiled)
        return compiled
    return decorator


def auto_compile(compiler: Optional[JITCompiler] = None):
    """
    Decorator for automatic JIT compilation
    """
    def decorator(func):
        if compiler is None:
            # Use global compiler instance
            global_compiler = JITCompiler()
            return global_compiler.profile_and_compile(func)
        else:
            return compiler.profile_and_compile(func)
    
    return decorator


# Example optimized functions
@jit_compile(nopython=True, parallel=True)
def fast_matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Example of JIT-compiled matrix multiplication"""
    return np.dot(a, b)


@torch_compile()
def fast_neural_forward(x: torch.Tensor, weight: torch.Tensor, 
                       bias: torch.Tensor) -> torch.Tensor:
    """Example of torch JIT compiled function"""
    return torch.nn.functional.linear(x, weight, bias)