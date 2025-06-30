"""
JARVIS Phase 10: Lazy Loader
Dynamic module loading system for memory efficiency
"""

import importlib
import importlib.util
import sys
import os
import time
import asyncio
import weakref
import gc
from typing import Dict, List, Any, Optional, Callable, Type, Union
from dataclasses import dataclass
from enum import Enum
import psutil
import logging
from collections import defaultdict
from functools import wraps
import inspect
import threading
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoadPriority(Enum):
    """Module loading priority"""
    CRITICAL = 1     # Always loaded
    HIGH = 2        # Load on startup
    MEDIUM = 3      # Load when needed
    LOW = 4         # Load only if memory available
    OPTIONAL = 5    # May not load at all


@dataclass
class ModuleInfo:
    """Information about a loadable module"""
    name: str
    path: str
    priority: LoadPriority
    estimated_size: int  # Estimated memory usage in bytes
    dependencies: List[str]
    features: List[str]
    load_time: Optional[float] = None
    last_accessed: Optional[float] = None
    access_count: int = 0
    loaded: bool = False
    module_ref: Optional[weakref.ref] = None


@dataclass
class FeatureInfo:
    """Information about a feature"""
    name: str
    required_modules: List[str]
    optional_modules: List[str]
    memory_estimate: int
    enabled: bool = False


class LazyLoader:
    """
    Dynamic module loading system for JARVIS
    Loads modules on-demand to save memory
    """
    
    def __init__(self, memory_threshold: float = 0.8):
        self.memory_threshold = memory_threshold  # Load modules if memory < 80%
        
        # Module registry
        self.modules: Dict[str, ModuleInfo] = {}
        self.loaded_modules: Dict[str, Any] = {}
        
        # Feature registry
        self.features: Dict[str, FeatureInfo] = {}
        
        # Performance tracking
        self.load_times: Dict[str, List[float]] = defaultdict(list)
        self.memory_usage: Dict[str, int] = {}
        
        # Module usage patterns
        self.usage_patterns = defaultdict(list)
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Background monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Initialize module registry
        self._register_modules()
    
    def _register_modules(self):
        """Register all JARVIS modules"""
        # Core modules (always loaded)
        self.register_module(
            "core.unified_input_pipeline",
            LoadPriority.CRITICAL,
            estimated_size=1024*1024,  # 1MB
            features=["input_processing"]
        )
        
        self.register_module(
            "core.fluid_state_management", 
            LoadPriority.CRITICAL,
            estimated_size=512*1024,  # 512KB
            features=["state_management"]
        )
        
        # High priority modules
        self.register_module(
            "core.proactive_context_engine",
            LoadPriority.HIGH,
            estimated_size=2*1024*1024,  # 2MB
            features=["context_awareness", "proactive_responses"]
        )
        
        # Medium priority modules
        self.register_module(
            "vision.advanced_vision",
            LoadPriority.MEDIUM,
            estimated_size=50*1024*1024,  # 50MB
            features=["vision_processing", "object_detection"],
            dependencies=["torch", "torchvision"]
        )
        
        self.register_module(
            "nlp.language_processor",
            LoadPriority.MEDIUM,
            estimated_size=100*1024*1024,  # 100MB
            features=["language_understanding", "sentiment_analysis"],
            dependencies=["transformers"]
        )
        
        # Low priority modules
        self.register_module(
            "creative.music_generator",
            LoadPriority.LOW,
            estimated_size=200*1024*1024,  # 200MB
            features=["music_generation"],
            dependencies=["music21", "tensorflow"]
        )
        
        self.register_module(
            "analysis.data_analyzer",
            LoadPriority.LOW,
            estimated_size=30*1024*1024,  # 30MB
            features=["data_analysis", "visualization"],
            dependencies=["pandas", "matplotlib"]
        )
        
        # Optional modules
        self.register_module(
            "experimental.quantum_processor",
            LoadPriority.OPTIONAL,
            estimated_size=500*1024*1024,  # 500MB
            features=["quantum_simulation"],
            dependencies=["qiskit"]
        )
        
        # Register features
        self._register_features()
    
    def _register_features(self):
        """Register JARVIS features and their module requirements"""
        self.register_feature(
            "basic_interaction",
            required_modules=["core.unified_input_pipeline", "core.fluid_state_management"],
            optional_modules=[],
            memory_estimate=2*1024*1024
        )
        
        self.register_feature(
            "vision_capabilities",
            required_modules=["vision.advanced_vision"],
            optional_modules=["vision.face_recognition", "vision.scene_understanding"],
            memory_estimate=100*1024*1024
        )
        
        self.register_feature(
            "advanced_language",
            required_modules=["nlp.language_processor"],
            optional_modules=["nlp.translation", "nlp.summarization"],
            memory_estimate=150*1024*1024
        )
        
        self.register_feature(
            "creative_tasks",
            required_modules=["creative.music_generator"],
            optional_modules=["creative.art_generator", "creative.story_writer"],
            memory_estimate=300*1024*1024
        )
    
    def register_module(self, name: str, priority: LoadPriority, 
                       estimated_size: int, features: List[str],
                       dependencies: Optional[List[str]] = None):
        """Register a module"""
        module_path = name.replace('.', '/')
        
        self.modules[name] = ModuleInfo(
            name=name,
            path=module_path,
            priority=priority,
            estimated_size=estimated_size,
            dependencies=dependencies or [],
            features=features
        )
    
    def register_feature(self, name: str, required_modules: List[str],
                        optional_modules: List[str], memory_estimate: int):
        """Register a feature"""
        self.features[name] = FeatureInfo(
            name=name,
            required_modules=required_modules,
            optional_modules=optional_modules,
            memory_estimate=memory_estimate
        )
    
    async def load_module(self, module_name: str, force: bool = False) -> Any:
        """
        Load a module dynamically
        """
        with self.lock:
            # Check if already loaded
            if module_name in self.loaded_modules and not force:
                module_info = self.modules.get(module_name)
                if module_info:
                    module_info.last_accessed = time.time()
                    module_info.access_count += 1
                return self.loaded_modules[module_name]
            
            # Check if module registered
            if module_name not in self.modules:
                raise ValueError(f"Module {module_name} not registered")
            
            module_info = self.modules[module_name]
            
            # Check memory availability
            if not self._check_memory_available(module_info.estimated_size):
                # Try to free memory
                await self._free_memory(module_info.estimated_size)
                
                # Check again
                if not self._check_memory_available(module_info.estimated_size):
                    raise MemoryError(f"Insufficient memory to load {module_name}")
            
            # Load dependencies first
            for dep in module_info.dependencies:
                try:
                    if dep not in sys.modules:
                        importlib.import_module(dep)
                except ImportError:
                    logger.warning(f"Optional dependency {dep} not available")
            
            # Load module
            start_time = time.time()
            
            try:
                # Try to import the module
                module = importlib.import_module(module_name)
                
                # Store weak reference
                module_info.module_ref = weakref.ref(module)
                self.loaded_modules[module_name] = module
                
                # Update info
                load_time = time.time() - start_time
                module_info.load_time = load_time
                module_info.loaded = True
                module_info.last_accessed = time.time()
                module_info.access_count = 1
                
                # Track performance
                self.load_times[module_name].append(load_time)
                
                # Estimate actual memory usage
                self._update_memory_usage(module_name)
                
                logger.info(f"Loaded module {module_name} in {load_time:.2f}s")
                
                return module
                
            except Exception as e:
                logger.error(f"Failed to load module {module_name}: {e}")
                raise
    
    async def load_feature(self, feature_name: str) -> bool:
        """
        Load all modules required for a feature
        """
        if feature_name not in self.features:
            raise ValueError(f"Feature {feature_name} not registered")
        
        feature = self.features[feature_name]
        
        # Check total memory requirement
        total_memory = feature.memory_estimate
        if not self._check_memory_available(total_memory):
            logger.warning(f"Insufficient memory for feature {feature_name}")
            return False
        
        try:
            # Load required modules
            for module_name in feature.required_modules:
                await self.load_module(module_name)
            
            # Try to load optional modules
            for module_name in feature.optional_modules:
                try:
                    await self.load_module(module_name)
                except Exception as e:
                    logger.warning(f"Could not load optional module {module_name}: {e}")
            
            feature.enabled = True
            logger.info(f"Feature {feature_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load feature {feature_name}: {e}")
            return False
    
    def unload_module(self, module_name: str, force: bool = False):
        """
        Unload a module to free memory
        """
        with self.lock:
            if module_name not in self.loaded_modules:
                return
            
            module_info = self.modules.get(module_name)
            if not module_info:
                return
            
            # Don't unload critical modules unless forced
            if module_info.priority == LoadPriority.CRITICAL and not force:
                return
            
            # Check if module is used by enabled features
            if not force:
                for feature in self.features.values():
                    if feature.enabled and module_name in feature.required_modules:
                        logger.warning(f"Cannot unload {module_name}, required by {feature.name}")
                        return
            
            # Remove from loaded modules
            module = self.loaded_modules.pop(module_name, None)
            
            # Clear module from sys.modules
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # Update info
            module_info.loaded = False
            module_info.module_ref = None
            
            # Force garbage collection
            del module
            gc.collect()
            
            logger.info(f"Unloaded module {module_name}")
    
    def get_module(self, module_name: str) -> Any:
        """
        Get a loaded module (non-async version)
        """
        if module_name in self.loaded_modules:
            module_info = self.modules.get(module_name)
            if module_info:
                module_info.last_accessed = time.time()
                module_info.access_count += 1
            return self.loaded_modules[module_name]
        
        # Module not loaded
        raise RuntimeError(f"Module {module_name} not loaded. Use load_module() first.")
    
    def lazy_import(self, module_name: str) -> 'LazyModule':
        """
        Create a lazy module proxy
        """
        return LazyModule(module_name, self)
    
    def _check_memory_available(self, required_bytes: int) -> bool:
        """Check if enough memory is available"""
        memory = psutil.virtual_memory()
        available = memory.available
        
        # Check against threshold
        if memory.percent / 100 > self.memory_threshold:
            return False
        
        return available > required_bytes * 1.5  # 50% buffer
    
    async def _free_memory(self, required_bytes: int):
        """Free memory by unloading modules"""
        # Sort modules by priority and last access time
        loaded = [
            (name, info) for name, info in self.modules.items()
            if info.loaded and info.priority != LoadPriority.CRITICAL
        ]
        
        # Sort by priority (descending) and last access (ascending)
        loaded.sort(key=lambda x: (x[1].priority.value, -x[1].last_accessed))
        
        freed = 0
        for module_name, module_info in loaded:
            if freed >= required_bytes:
                break
            
            # Unload module
            self.unload_module(module_name)
            freed += module_info.estimated_size
    
    def _update_memory_usage(self, module_name: str):
        """Update actual memory usage of module"""
        # This is an estimate - real measurement would be complex
        before = psutil.Process().memory_info().rss
        
        # Touch all module attributes to load them
        module = self.loaded_modules.get(module_name)
        if module:
            for attr_name in dir(module):
                try:
                    getattr(module, attr_name)
                except:
                    pass
        
        after = psutil.Process().memory_info().rss
        self.memory_usage[module_name] = max(0, after - before)
    
    def _monitor_memory(self):
        """Background memory monitoring"""
        while self.monitoring_active:
            try:
                memory = psutil.virtual_memory()
                
                # Check if we need to free memory
                if memory.percent / 100 > self.memory_threshold:
                    # Find least used modules to unload
                    loaded = [
                        (name, info) for name, info in self.modules.items()
                        if info.loaded and info.priority == LoadPriority.LOW
                    ]
                    
                    if loaded:
                        # Unload least recently used
                        loaded.sort(key=lambda x: x[1].last_accessed)
                        self.unload_module(loaded[0][0])
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get loader status"""
        memory = psutil.virtual_memory()
        
        loaded_size = sum(
            info.estimated_size for info in self.modules.values()
            if info.loaded
        )
        
        return {
            'total_modules': len(self.modules),
            'loaded_modules': len(self.loaded_modules),
            'enabled_features': [f.name for f in self.features.values() if f.enabled],
            'memory_usage': {
                'system_percent': memory.percent,
                'estimated_module_memory': f"{loaded_size / 1024 / 1024:.2f} MB",
                'available_memory': f"{memory.available / 1024 / 1024 / 1024:.2f} GB"
            },
            'module_details': {
                name: {
                    'loaded': info.loaded,
                    'priority': info.priority.name,
                    'access_count': info.access_count,
                    'last_accessed': time.time() - info.last_accessed if info.last_accessed else None
                }
                for name, info in self.modules.items()
            }
        }
    
    async def preload_essential(self):
        """Preload essential modules"""
        for module_name, module_info in self.modules.items():
            if module_info.priority in [LoadPriority.CRITICAL, LoadPriority.HIGH]:
                try:
                    await self.load_module(module_name)
                except Exception as e:
                    logger.error(f"Failed to preload {module_name}: {e}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Lazy Loader")
        self.monitoring_active = False


class LazyModule:
    """
    Proxy object for lazy module loading
    """
    
    def __init__(self, module_name: str, loader: LazyLoader):
        self._module_name = module_name
        self._loader = loader
        self._module = None
    
    def __getattr__(self, name):
        """Load module on first attribute access"""
        if self._module is None:
            # Load module synchronously
            loop = asyncio.new_event_loop()
            self._module = loop.run_until_complete(
                self._loader.load_module(self._module_name)
            )
        
        return getattr(self._module, name)


# Decorator for lazy loading
def lazy_load(module_name: str):
    """
    Decorator to lazy load dependencies
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get loader instance
            from .jarvis_enhanced_core import get_jarvis_instance
            jarvis = get_jarvis_instance()
            
            if hasattr(jarvis, 'lazy_loader'):
                # Load required module
                await jarvis.lazy_loader.load_module(module_name)
            
            # Execute function
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Auto-unload decorator
def auto_unload(priority: LoadPriority = LoadPriority.LOW):
    """
    Decorator to mark functions that can have their modules auto-unloaded
    """
    def decorator(func):
        func._auto_unload_priority = priority
        return func
    
    return decorator