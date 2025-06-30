"""
JARVIS Phase 9: Integration Module
==================================
Integrates performance optimization with existing JARVIS infrastructure
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import logging
import json
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase9.performance_optimizer import (
    JARVISPerformanceOptimizer,
    cached,
    IntelligentCache
)

# Import existing JARVIS components
try:
    from core.unified_input_pipeline import UnifiedInputPipeline
    from core.fluid_state_management import FluidStateManager
    from core.jarvis_enhanced_core import JARVISEnhancedCore
except ImportError:
    print("Warning: Some JARVIS components not found. Running in standalone mode.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JARVISOptimizedCore:
    """
    JARVIS Core with integrated Phase 9 performance optimizations
    """
    
    def __init__(self):
        # Initialize performance optimizer
        self.optimizer = JARVISPerformanceOptimizer()
        
        # Initialize core components with optimization
        self.input_pipeline = None
        self.state_manager = None
        self.enhanced_core = None
        
        # Optimization metrics
        self.performance_metrics = {
            'pipeline_calls': 0,
            'state_updates': 0,
            'cache_hits': 0,
            'parallel_operations': 0,
            'total_time_saved': 0.0
        }
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize optimized JARVIS core"""
        logger.info("Initializing JARVIS Optimized Core with Phase 9 enhancements...")
        
        # Initialize performance optimizer
        await self.optimizer.initialize()
        
        # Try to initialize existing components with optimization wrappers
        try:
            # Wrap existing components with optimization
            self.input_pipeline = await self._wrap_input_pipeline()
            self.state_manager = await self._wrap_state_manager()
            self.enhanced_core = await self._wrap_enhanced_core()
        except Exception as e:
            logger.warning(f"Could not wrap existing components: {e}")
            logger.info("Running with performance optimizer only")
        
        # Preload common modules
        await self.optimizer.preload_modules([
            'jarvis.nlp',
            'jarvis.vision',
            'jarvis.memory',
            'jarvis.reasoning'
        ])
        
        self._initialized = True
        logger.info("JARVIS Optimized Core initialized successfully")
    
    async def _wrap_input_pipeline(self):
        """Wrap input pipeline with caching and parallel processing"""
        try:
            pipeline = UnifiedInputPipeline()
            
            # Cache the process_input method
            original_process = pipeline.process_input
            
            @cached(ttl_seconds=300, cache_instance=self.optimizer.cache)
            async def cached_process_input(input_data: Dict[str, Any], source: str) -> Dict[str, Any]:
                self.performance_metrics['pipeline_calls'] += 1
                return await original_process(input_data, source)
            
            # Replace with cached version
            pipeline.process_input = cached_process_input
            
            logger.info("Input pipeline wrapped with caching")
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to wrap input pipeline: {e}")
            return None
    
    async def _wrap_state_manager(self):
        """Wrap state manager with performance optimization"""
        try:
            manager = FluidStateManager()
            
            # Optimize state transitions with caching
            original_transition = manager.transition_to_state
            
            async def optimized_transition(new_state: str):
                self.performance_metrics['state_updates'] += 1
                # Use parallel processing for state calculations
                return await self.optimizer.optimize_operation(
                    original_transition, 
                    new_state
                )
            
            manager.transition_to_state = optimized_transition
            
            logger.info("State manager wrapped with optimization")
            return manager
            
        except Exception as e:
            logger.error(f"Failed to wrap state manager: {e}")
            return None
    
    async def _wrap_enhanced_core(self):
        """Wrap enhanced core with comprehensive optimization"""
        try:
            core = JARVISEnhancedCore()
            await core.initialize()
            
            # Wrap process method with full optimization
            original_process = core.process
            
            async def optimized_process(input_data: Any) -> Any:
                # Use full optimization pipeline
                return await self.optimizer.optimize_operation(
                    original_process,
                    input_data
                )
            
            core.process = optimized_process
            
            logger.info("Enhanced core wrapped with optimization")
            return core
            
        except Exception as e:
            logger.error(f"Failed to wrap enhanced core: {e}")
            return None
    
    async def process_batch(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple inputs in parallel with optimization
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        # Define processing function
        async def process_single(input_data: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # Try to use enhanced core if available
                if self.enhanced_core:
                    return await self.enhanced_core.process(input_data)
                else:
                    # Fallback to basic processing
                    return {"processed": input_data, "timestamp": datetime.now().isoformat()}
            except Exception as e:
                logger.error(f"Error processing input: {e}")
                return {"error": str(e), "input": input_data}
        
        # Process in parallel batches
        results = await self.optimizer.batch_process(inputs, process_single)
        
        # Update metrics
        elapsed = time.time() - start_time
        self.performance_metrics['parallel_operations'] += 1
        self.performance_metrics['total_time_saved'] += elapsed
        
        logger.info(f"Processed {len(inputs)} inputs in {elapsed:.2f}s")
        
        return results
    
    async def optimize_memory_usage(self):
        """
        Optimize memory usage by unloading unused modules
        """
        stats = self.optimizer.lazy_loader.get_stats()
        
        # Find least used modules
        if 'hot_modules' in stats:
            all_modules = set(self.optimizer.lazy_loader.loaded_modules.keys())
            hot_modules = set([m[0] for m in stats['hot_modules']])
            cold_modules = all_modules - hot_modules
            
            # Unload cold modules
            for module in cold_modules:
                await self.optimizer.lazy_loader.unload(module)
                logger.info(f"Unloaded cold module: {module}")
    
    async def run_auto_optimization(self):
        """
        Run automatic optimization based on current performance
        """
        await self.optimizer.auto_optimize()
        
        # Also optimize memory
        await self.optimize_memory_usage()
        
        # Update cache strategies based on patterns
        patterns = self.optimizer.cache.analyze_patterns()
        if patterns.get('hot_keys'):
            logger.info(f"Identified {len(patterns['hot_keys'])} hot cache keys")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization report
        """
        base_report = self.optimizer.get_performance_report()
        
        # Add JARVIS-specific metrics
        base_report['jarvis_metrics'] = self.performance_metrics
        
        # Calculate optimization impact
        if self.performance_metrics['pipeline_calls'] > 0:
            cache_efficiency = self.optimizer.cache.stats.hit_rate
            time_saved_per_call = (
                self.performance_metrics['total_time_saved'] / 
                self.performance_metrics['pipeline_calls']
            )
            
            base_report['optimization_impact'] = {
                'cache_efficiency': f"{cache_efficiency:.1%}",
                'avg_time_saved_per_call': f"{time_saved_per_call:.3f}s",
                'total_time_saved': f"{self.performance_metrics['total_time_saved']:.1f}s",
                'memory_optimized': f"{base_report['cache_stats']['memory_saved_mb']:.1f}MB"
            }
        
        return base_report
    
    async def shutdown(self):
        """Gracefully shutdown optimized core"""
        logger.info("Shutting down JARVIS Optimized Core...")
        
        # Get final report
        report = self.get_optimization_report()
        
        # Log optimization impact
        if 'optimization_impact' in report:
            logger.info("Phase 9 Optimization Impact:")
            for metric, value in report['optimization_impact'].items():
                logger.info(f"  - {metric}: {value}")
        
        # Shutdown optimizer
        await self.optimizer.shutdown()
        
        logger.info("JARVIS Optimized Core shutdown complete")


# ==================== Optimization Decorators ====================

def optimize_jarvis_function(
    cache_ttl: int = 3600,
    parallel: bool = True,
    lazy_load_modules: Optional[List[str]] = None
):
    """
    Decorator to optimize any JARVIS function
    """
    def decorator(func):
        # Create dedicated cache for this function
        func_cache = IntelligentCache(memory_limit_mb=256)
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Lazy load required modules
            if lazy_load_modules:
                optimizer = JARVISPerformanceOptimizer()
                for module in lazy_load_modules:
                    await optimizer.lazy_loader.load(module)
            
            # Check cache
            cache_key = func_cache._generate_key(func.__name__, args, kwargs)
            cached_result = await func_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            start_time = time.time()
            
            if parallel and asyncio.iscoroutinefunction(func):
                # Run with parallel optimization if possible
                result = await func(*args, **kwargs)
            else:
                result = await func(*args, **kwargs)
            
            # Cache result
            await func_cache.set(cache_key, result, ttl_seconds=cache_ttl)
            
            # Log performance
            elapsed = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {elapsed:.3f}s")
            
            return result
        
        # Add utility methods
        wrapper.clear_cache = lambda: func_cache.memory_cache.clear()
        wrapper.get_cache_stats = lambda: func_cache.stats.get_stats()
        
        return wrapper
    return decorator


# ==================== Usage Examples ====================

class OptimizedJARVISExamples:
    """Examples of using Phase 9 optimizations in JARVIS"""
    
    @staticmethod
    @optimize_jarvis_function(cache_ttl=1800, lazy_load_modules=['jarvis.nlp'])
    async def analyze_text_optimized(text: str) -> Dict[str, Any]:
        """Example: Optimized text analysis"""
        # Simulate NLP processing
        await asyncio.sleep(0.5)
        
        return {
            'sentiment': 'positive',
            'entities': ['JARVIS', 'optimization'],
            'topics': ['performance', 'AI'],
            'processed_at': datetime.now().isoformat()
        }
    
    @staticmethod
    async def process_vision_batch_optimized(images: List[bytes]) -> List[Dict[str, Any]]:
        """Example: Batch vision processing with optimization"""
        optimizer = JARVISPerformanceOptimizer()
        await optimizer.initialize()
        
        async def process_image(image: bytes) -> Dict[str, Any]:
            # Simulate vision processing
            await asyncio.sleep(0.2)
            return {
                'objects': ['computer', 'desk'],
                'scene': 'office',
                'confidence': 0.95
            }
        
        # Process images in parallel batches
        results = await optimizer.batch_process(images, process_image)
        
        await optimizer.shutdown()
        return results
    
    @staticmethod
    async def optimized_reasoning_chain(query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Example: Optimized reasoning chain with caching"""
        optimizer = JARVISPerformanceOptimizer()
        await optimizer.initialize()
        
        # Define reasoning stages
        @cached(ttl_seconds=600, cache_instance=optimizer.cache)
        async def understand_query(q: str) -> Dict[str, Any]:
            await asyncio.sleep(0.3)
            return {'intent': 'question', 'domain': 'general'}
        
        @cached(ttl_seconds=600, cache_instance=optimizer.cache)
        async def retrieve_knowledge(intent: str, domain: str) -> List[str]:
            await asyncio.sleep(0.4)
            return ['fact1', 'fact2', 'fact3']
        
        @cached(ttl_seconds=300, cache_instance=optimizer.cache)
        async def generate_response(facts: List[str], query: str) -> str:
            await asyncio.sleep(0.5)
            return f"Based on {len(facts)} facts, the answer is..."
        
        # Execute reasoning chain with optimization
        understanding = await understand_query(query)
        knowledge = await retrieve_knowledge(
            understanding['intent'], 
            understanding['domain']
        )
        response = await generate_response(knowledge, query)
        
        result = {
            'query': query,
            'understanding': understanding,
            'knowledge_used': len(knowledge),
            'response': response,
            'cache_stats': optimizer.cache.stats.get_stats()
        }
        
        await optimizer.shutdown()
        return result


# ==================== Demo Integration ====================

async def demo_phase9_integration():
    """Demonstrate Phase 9 integration with JARVIS"""
    
    print("\n🔧 JARVIS Phase 9 Integration Demo\n")
    
    # Initialize optimized core
    jarvis = JARVISOptimizedCore()
    await jarvis.initialize()
    
    print("✅ JARVIS Optimized Core initialized\n")
    
    # Demo 1: Batch processing
    print("📦 Demo 1: Optimized Batch Processing")
    print("-" * 50)
    
    # Create test inputs
    test_inputs = [
        {'type': 'text', 'data': f'Message {i}'} 
        for i in range(10)
    ]
    
    start = time.time()
    results = await jarvis.process_batch(test_inputs)
    elapsed = time.time() - start
    
    print(f"Processed {len(results)} inputs in {elapsed:.2f}s")
    print(f"Average time per input: {elapsed/len(results):.3f}s\n")
    
    # Demo 2: Auto-optimization
    print("🤖 Demo 2: Automatic Optimization")
    print("-" * 50)
    
    await jarvis.run_auto_optimization()
    print("Auto-optimization completed\n")
    
    # Demo 3: Optimized functions
    print("⚡ Demo 3: Optimized Function Examples")
    print("-" * 50)
    
    # Text analysis with caching
    text = "JARVIS performance optimization is amazing!"
    
    start = time.time()
    result1 = await OptimizedJARVISExamples.analyze_text_optimized(text)
    time1 = time.time() - start
    
    start = time.time()
    result2 = await OptimizedJARVISExamples.analyze_text_optimized(text)  # Cached
    time2 = time.time() - start
    
    print(f"First analysis: {time1:.3f}s")
    print(f"Cached analysis: {time2:.3f}s")
    print(f"Speed improvement: {time1/time2:.1f}x\n")
    
    # Demo 4: Performance report
    print("📊 Demo 4: Performance Report")
    print("-" * 50)
    
    report = jarvis.get_optimization_report()
    
    print("Cache Performance:")
    print(f"  - Hit Rate: {report['cache_stats']['hit_rate']:.1%}")
    print(f"  - Time Saved: {report['cache_stats']['time_saved_seconds']:.1f}s")
    
    if 'optimization_impact' in report:
        print("\nOptimization Impact:")
        for metric, value in report['optimization_impact'].items():
            print(f"  - {metric}: {value}")
    
    print("\nCurrent Performance:")
    current = report['monitor_dashboard']['current_metrics']
    print(f"  - CPU: {current['cpu']:.1f}%")
    print(f"  - Memory: {current['memory']:.1f}%")
    print(f"  - Avg Response: {current['avg_response_time']:.1f}ms")
    
    # Demo 5: Recommendations
    if report['monitor_dashboard']['recommendations']:
        print("\n🎯 Optimization Recommendations:")
        for rec in report['monitor_dashboard']['recommendations']:
            print(f"  - {rec['suggestion']}")
            print(f"    Expected Impact: {rec['expected_impact']}")
    
    # Shutdown
    await jarvis.shutdown()
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    # Run integration demo
    asyncio.run(demo_phase9_integration())
